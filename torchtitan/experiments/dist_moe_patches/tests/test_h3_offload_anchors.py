# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU anchor check for h3_offload against the installed dist_moe.

Catches upstream drift of ``dist_moe.api._Bf16Autograd.forward`` and
``_backward_impl`` before a GPU job does: every edit must match exactly once,
alone and on top of swiglu_recompute, and the patched sources must compile
against ``dist_moe.api``'s globals. Also checks that the wrong import order
(h3_offload before swiglu_recompute) fails loudly. Nothing is installed. Skips
when dist_moe is not importable; the stacked cases skip when swiglu_recompute
is not in this checkout.
"""

import inspect
import textwrap

import pytest

from torchtitan.experiments.dist_moe_patches import h3_offload

api = pytest.importorskip("dist_moe.api")

_EDITS = {
    "forward": h3_offload._FORWARD_EDITS,
    "_backward_impl": h3_offload._BACKWARD_EDITS,
}


def _upstream_source(name: str) -> str:
    return inspect.getsource(api._Bf16Autograd.__dict__[name].__func__)


def _compile(name: str, src: str) -> None:
    namespace: dict = {}
    exec(compile(src, "<test>", "exec"), api.__dict__, namespace)
    assert isinstance(namespace[name], staticmethod)


@pytest.mark.parametrize("name", sorted(_EDITS))
def test_edits_apply_to_upstream(name):
    _compile(name, h3_offload.patch_source(_upstream_source(name), _EDITS[name], name))


def test_edits_apply_on_top_of_swiglu_recompute():
    swiglu_recompute = pytest.importorskip(
        "torchtitan.experiments.dist_moe_patches.swiglu_recompute"
    )
    name = "_backward_impl"
    src = _upstream_source(name)
    stacked = textwrap.dedent(swiglu_recompute.patch_source(src))
    _compile(name, h3_offload.patch_source(stacked, _EDITS[name], name))
    # Wrong order: SwiGLU's anchors are written against the original layout.
    h3_first = h3_offload.patch_source(src, _EDITS[name], name)
    with pytest.raises(RuntimeError, match="anchor matched 0 times"):
        swiglu_recompute.patch_source(h3_first)
