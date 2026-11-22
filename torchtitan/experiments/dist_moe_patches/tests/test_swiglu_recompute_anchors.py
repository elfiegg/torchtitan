# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU anchor check for swiglu_recompute against the installed dist_moe.

Catches upstream drift of ``dist_moe.api._Bf16Autograd._backward_impl`` before
a GPU job does: every edit must match exactly once, and the patched source must
compile against ``dist_moe.api``'s globals. Nothing is installed. Skips when
dist_moe is not importable.
"""

import inspect
import textwrap

import pytest

from torchtitan.experiments.dist_moe_patches import swiglu_recompute

api = pytest.importorskip("dist_moe.api")


def test_edits_apply_to_upstream_backward_impl():
    func = api._Bf16Autograd.__dict__["_backward_impl"].__func__
    new_src = textwrap.dedent(swiglu_recompute.patch_source(inspect.getsource(func)))
    namespace: dict = {}
    exec(compile(new_src, "<test>", "exec"), api.__dict__, namespace)
    assert isinstance(namespace["_backward_impl"], staticmethod)
