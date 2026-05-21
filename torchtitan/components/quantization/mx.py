# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields
from importlib.util import find_spec
from typing import Literal

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import swap_token_dispatcher


def _patch_torchao_fsdp_pre_all_gather() -> None:
    """Patch TorchAO's MXFP8/Float8 wrapper base class to pad along dim 0 in
    ``fsdp_pre_all_gather`` when the parameter is unevenly sharded by FSDP.

    Background: upstream ``TrainingWeightWrapperBaseTensor.fsdp_pre_all_gather``
    in ``torchao/prototype/moe_training/tensor.py`` simply returns
    ``self._data.to(mp_policy.param_dtype)`` without any padding. When the
    outer (un-sharded) param size along dim 0 is not divisible by the FSDP
    world size, FSDP pads the global tensor so every rank's local shard has
    ``ceildiv(outer_size[0], world_size)`` rows. Tail ranks therefore end up
    with shards that contain *only padding* (``self._data.shape[0] == 0``).
    FSDP's foreach_all_gather asserts that ``fsdp_pre_all_gather`` returns
    inputs with the padded sharded size and crashes (PyTorch
    ``_fully_shard/_fsdp_param.py:811``) when a wrapper returns a length-0
    tensor instead. This affects e.g. DeepSeek-V3's ``kv_a_proj_with_mqa``
    (out=576) on FSDP world size 256 -> padded shard 3, tail rank shard 0.

    We previously carried a similar patch on a torchao fork for the grouped-
    experts path, but since the recipe was rebased onto upstream torchao and
    ``MXFP8Linear`` now hits the same base-class hook, the right fix is to
    patch the base. Idempotent / safe to call multiple times.
    """
    try:
        from torchao.prototype.moe_training.tensor import (
            TrainingWeightWrapperBaseTensor,
        )
    except ImportError:
        return

    if getattr(
        TrainingWeightWrapperBaseTensor.fsdp_pre_all_gather,
        "_llmb_uneven_shard_padded",
        False,
    ):
        return

    import torch
    import torch.nn.functional as F

    def fsdp_pre_all_gather_padded(self, mesh, outer_size, outer_stride, module, mp_policy):
        data = self._data.to(mp_policy.param_dtype)
        world_size = mesh.size()
        # ceildiv(outer_size[0], world_size) is what FSDP expects on every rank.
        padded_dim0 = -(-int(outer_size[0]) // world_size)
        if data.shape[0] < padded_dim0:
            pad = [0] * (2 * data.ndim)
            pad[-1] = padded_dim0 - data.shape[0]  # pad dim 0 (bottom)
            data = F.pad(data, tuple(pad))
        elif data.shape[0] > padded_dim0:
            # Should not happen for correctly-sharded FSDP params; defend anyway.
            data = data[:padded_dim0]
        return (data,), ()

    fsdp_pre_all_gather_padded._llmb_uneven_shard_padded = True
    TrainingWeightWrapperBaseTensor.fsdp_pre_all_gather = fsdp_pre_all_gather_padded
    logger.info(
        "Patched torchao TrainingWeightWrapperBaseTensor.fsdp_pre_all_gather "
        "to handle uneven FSDP sharding (pads dim 0 to ceildiv(outer_size[0], world_size))."
    )


_patch_torchao_fsdp_pre_all_gather()


class MXFP8Linear(Linear):
    """Linear that applies MXFP8 quantization in its constructor."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for Linear.Config that builds MXFP8Linear."""

        _recipe_name: str = "mxfp8_rceil"

    def __init__(self, config: Config):
        super().__init__(config)
        from torchao.prototype.moe_training.config import (
            MXFP8TrainingOpConfig,
            MXFP8TrainingRecipe,
        )
        from torchao.quantization.quant_api import quantize_

        recipe = MXFP8TrainingRecipe(config._recipe_name)
        mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
        quantize_(self, config=mxfp8_op_config)


class MXFP8LinearConverter(QuantizationConverter):
    """Apply MXFP8 quantization to modules matching FQNs (e.g. Flux blocks)."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        # Must match a value of ``MXFP8TrainingRecipe`` in torchao
        # (``prototype/moe_training/config.py``). cuBLAS dispatch lives inside
        # ``mxfp8_rceil`` -> ``KernelPreference.AUTO`` on SM100+; there is no
        # separate ``mxfp8_cublas`` recipe in this API.
        recipe_name: Literal[
            "mxfp8_rceil", "mxfp8_rceil_wgrad_with_hp", "mxfp8_emulated_rceil"
        ] = "mxfp8_rceil"
        """
        Quantization recipe name. Options: ["mxfp8_rceil", "mxfp8_rceil_wgrad_with_hp", "mxfp8_emulated_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode when
          computing the e8m0 scale factors; dispatches to ``torch._scaled_grouped_mm``
          (cuBLAS-native scaled grouped GEMM) on SM100+.
        - mxfp8_rceil_wgrad_with_hp: same as mxfp8_rceil but computes the weight
          gradient in high precision.
        - mxfp8_emulated_rceil: emulated MXFP8 (fp32 dequant + bf16 gemm); for
          correctness debugging / non-SM100 hardware.
        """

        fqns: list[str] = field(default_factory=list)
        """
        *Prototype feature, performance optimization still in progress*
        Comma-separated list of fully qualified names of MoE modules to apply MXFP8 dynamic quantization
        on grouped GEMM operations.
        This is a prototype feature that requires the torchao nightly build.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )

        # Can be removed if we enable the emulated versions
        assert has_cuda_capability(
            10, 0
        ), "MXFP8 is only supported on SM100 or later architectures"

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config) -> None:
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = MXFP8Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                    _recipe_name=self.config.recipe_name,
                )
                if isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info(
            f"Converted modules to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm and linear ops"
        )


_mxfp8_experts_cache: dict[type, type] = {}


def _get_mxfp8_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create an MXFP8-quantized subclass of *parent_cls*.

    Works for any ``GroupedExperts`` subclass (e.g. gpt-oss variants).
    The returned class has a proper ``_owner`` set by ``__init_subclass__``.
    """
    if parent_cls in _mxfp8_experts_cache:
        return _mxfp8_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXFP8GroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            recipe_name: str = "mxfp8_rceil"

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import (
                MXFP8TrainingOpConfig,
                MXFP8TrainingRecipe,
            )
            from torchao.quantization.quant_api import quantize_

            recipe = MXFP8TrainingRecipe(config.recipe_name)
            mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
            quantize_(
                self,
                config=mxfp8_op_config,
                filter_fn=lambda mod, _fqn: isinstance(mod, GroupedExperts),
            )

    MXFP8GroupedExperts.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8GroupedExperts.__qualname__ = f"MXFP8{parent_cls.__name__}"
    _mxfp8_experts_cache[parent_cls] = MXFP8GroupedExperts
    return MXFP8GroupedExperts


class MXFP8GroupedExpertsConverter(QuantizationConverter):
    """Apply MXFP8 quantization to MoE expert grouped GEMMs."""

    # MXFP8: scaling block size is (1 x 32), so contracting dim must be divisible by 32.
    PAD_MULTIPLE = 32

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        # Must match a value of ``MXFP8TrainingRecipe`` in torchao.
        # See MXFP8LinearConverter.Config above for why ``mxfp8_cublas`` is not
        # in this list.
        recipe_name: Literal[
            "mxfp8_rceil", "mxfp8_rceil_wgrad_with_hp", "mxfp8_emulated_rceil"
        ] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs.
        Options: ["mxfp8_rceil", "mxfp8_rceil_wgrad_with_hp", "mxfp8_emulated_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode when
          computing the e8m0 scale factors; dispatches to ``torch._scaled_grouped_mm``
          (cuBLAS-native scaled grouped GEMM) on SM100+ (B200/B300/GB200/GB300).
        - mxfp8_rceil_wgrad_with_hp: same as mxfp8_rceil but computes the weight
          gradient in high precision.
        - mxfp8_emulated_rceil: emulated MXFP8 for correctness debugging.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 MoE training."
            )

        assert has_cuda_capability(
            10, 0
        ), "MXFP8 is only supported on SM100 or later architectures"

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config) -> None:
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            swap_token_dispatcher(config, self.PAD_MULTIPLE)
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                recipe_name=self.config.recipe_name,
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            f"Converted GroupedExperts to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm and linear ops"
        )
