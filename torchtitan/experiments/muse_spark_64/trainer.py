# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Source-pinned runtime for the experimental 64-GPU Muse Spark recipe."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from torchtitan.distributed.accelerator_graph import accelerator_graph_manager
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine

logger = logging.getLogger(__name__)


class ExperimentalTrainingEngine(TrainingEngine):
    """Own packing callbacks and the final EP-barrier stream for this recipe."""

    def initialize(self, **kwargs: Any) -> None:
        self._w13_overlap = None
        super().initialize(**kwargs)
        from torch.distributed.fsdp import FSDPModule

        from . import packing

        for part in self.model_parts:
            packing.install(part)
            # The measured recipe uses the native single-buffer RS policy.
            for module in part.modules():
                if isinstance(module, FSDPModule):
                    module.set_reduce_scatter_max_input_buffers(1, recurse=False)

    def load_checkpoint(self) -> bool:
        if self._w13_overlap is not None:
            raise RuntimeError(
                "This experiment permits one checkpoint load per process."
            )
        loaded = super().load_checkpoint()
        if self.num_completed_steps != 0:
            raise ValueError(
                "This experimental recipe has not validated checkpoint resume."
            )
        from torchtitan.models.common.dist_moe.routed_experts import (
            DistMoeRoutedExperts,
        )

        from .w13.install import install

        runtime = self._dist_moe_runtime
        if runtime is None or len(self.model_parts) != 1:
            raise ValueError(
                "This experiment requires one full-model Dist-MoE context."
            )
        if torch.distributed.get_world_size() != 64:
            raise ValueError("This experiment requires 64 real GPUs.")
        experts = tuple(
            module
            for module in self.model_parts[0].modules()
            if isinstance(module, DistMoeRoutedExperts)
        )
        if len(experts) != 62 or tuple(runtime._modules) != experts:
            raise ValueError(
                "This experiment requires all 62 experts on one shared context."
            )
        callbacks = tuple(module.output_postprocess for module in experts)
        manifest = Path(__file__).with_name("w13") / "SOURCE_MANIFEST.json"
        self._w13_overlap = install(
            runtime.context,
            callbacks,
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            screen_only=True,
        )
        logger.warning(
            "Experimental Muse Spark bundle enabled: DP32/CP2, scratch10/save0, "
            "fused packing and W13 barrier overlap. Numerical equivalence and "
            "long-run training quality remain unqualified."
        )
        return loaded

    def _register_overlap_graphs(self) -> None:
        # Do not retain strong graph references across native graph teardown.
        if accelerator_graph_manager._dummy_graph is not None:
            self._w13_overlap.register_graph(accelerator_graph_manager._dummy_graph)
        for wrapper in accelerator_graph_manager._wrappers:
            if wrapper._graph is not None:
                self._w13_overlap.register_graph(wrapper._graph)

    def close(self) -> None:
        overlap = getattr(self, "_w13_overlap", None)
        if overlap is not None:
            if overlap.owner.poisoned or overlap.owner.active:
                raise RuntimeError(
                    "Failed experimental backward is terminal; terminate this process."
                )
            self._register_overlap_graphs()
        super().close()
        if overlap is not None:
            overlap.restore_after_native_teardown()
            self._w13_overlap = None


class ExperimentalTrainer(Trainer):
    """Select the experimental engine through the existing trainer extension."""

    engine_cls = ExperimentalTrainingEngine

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        def __post_init__(self) -> None:
            Trainer.Config.__post_init__(self)
            from .data import OriginalWorkDataLoader

            if not isinstance(self.dataloader, OriginalWorkDataLoader.Config):
                raise ValueError(
                    "This experiment requires the original DP8 input mapping."
                )
            p = self.parallelism
            geometry = (
                p.data_parallel_replicate_degree,
                p.data_parallel_shard_degree,
                p.context_parallel_degree,
                p.expert_parallel_degree,
                p.tensor_parallel_degree,
                p.pipeline_parallel_degree,
            )
            if geometry != (1, 32, 2, 64, 1, 1):
                raise ValueError("This experiment requires DP32/CP2/EP64/TP1/PP1.")
            if (
                self.comm.backend == "fake"
                or self.training.disable_accelerator_graphs
                or not self.training.accelerator_graph_per_microbatch_group
                or self.optim.enable_accelerator_graph
                or self.validator is not None
                or self.create_seed_checkpoint
            ):
                raise ValueError(
                    "This experiment requires real process groups, per-group model "
                    "graphs, an eager optimizer and training-only execution."
                )
            if (
                self.training.num_tokens_per_microbatch_per_dp_rank != 32768
                or self.training.num_tokens_per_train_step != 64 * 32768
                or self.model.max_context_length != 32768
                or self.precision.param_dtype != "bfloat16"
                or self.precision.dtype != "float32"
                or self.precision.reduce_dtype != "float32"
            ):
                raise ValueError(
                    "This experiment requires the measured 32K BF16 workload."
                )
            runtime = self.dist_moe_runtime
            if (
                runtime is None
                or runtime.scratch_capacity_factor != 10.0
                or runtime.activation_slot_capacity_factor != 0.0
                or runtime.num_sms is not None
                or runtime.vmm_capacity_factor is not None
            ):
                raise ValueError(
                    "This experiment requires scratch10/save0 and default SMs."
                )
            if self.checkpointer is None or not self.checkpointer.load_only:
                raise ValueError(
                    "This experiment requires training-only checkpoint loading."
                )
