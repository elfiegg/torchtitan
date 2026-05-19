# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.config import CompileConfig
from torchtitan.tools.logging import logger


# TODO: Remove this monkeypatch once FakeTensorMode.__init__ is decorated with
# @torch.compiler.disable(recursive=True) upstream.
# See https://github.com/pytorch/pytorch/issues/178887
FakeTensorMode.__init__ = torch.compiler.disable(  # type: ignore[method-assign]
    FakeTensorMode.__init__, recursive=True
)


def apply_compile(model: nn.Module, compile_config: CompileConfig) -> None:
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # Needed for torch.compile to handle data-dependent dynamic shapes in
    # token-choice MoE dispatch. Harmless for dense models.
    torch._dynamo.config.capture_scalar_outputs = True
    # Skip replaying forward side effects (e.g. RoPE cache updates) during
    # the AC recompute in backward. Eager AC replays the forward python
    # side-effects in backward, but torch.compile has no easy way to reapply
    # python mutations in the backward. Setting this flag accepts this eager
    # and compile divergence by skipping reapplication of side effects.
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = (
        True  # pyrefly: ignore [bad-assignment]
    )

    # We need fullgraph=False (not the upstream default True) because the
    # DeepEP MoE all-to-all path used by all of our 671B benchmark variants
    # invokes ``torch.utils._python_dispatch._disable_current_modes()`` from
    # inside ``torchtitan.distributed.deepep.deepep.dispatch_tokens``. The
    # context manager's ``__init__`` evaluates
    # ``_len_torch_dispatch_stack()``, which is a torch.* op that returns
    # ``int`` (not Tensor). Under ``fullgraph=True``, Dynamo refuses to
    # include this in the FX output graph and raises
    # ``torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor`` --
    # the failure surfaces specifically when ``apply_ac`` (mode='full') has
    # wrapped the block in ``checkpoint_wrapper`` and compile is tracing
    # through the checkpoint HOP, but it would happen with either
    # apply_ac/apply_compile ordering since the offending call lives inside
    # the block.forward path that compile traces.
    #
    # Upstream's ``skip_fwd_side_effects_in_bwd_under_checkpoint = True``
    # (above) takes care of the AC-side issues; this fullgraph=False knob
    # is independent and only matters when MoE comm_backend in
    # {'deepep', 'hybridep'}. We keep it on globally for simplicity --
    # graph breaks at the dispatch boundary are unavoidable today anyway.
    #
    # TODO: upstream a fix that either marks ``_len_torch_dispatch_stack``
    # as a Dynamo "constant scalar" or refactors ``dispatch_tokens`` to
    # avoid ``_disable_current_modes`` on the hot path.
    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block.compile(backend=compile_config.backend, fullgraph=False)

    logger.info("Compiling each TransformerBlock with torch.compile")
