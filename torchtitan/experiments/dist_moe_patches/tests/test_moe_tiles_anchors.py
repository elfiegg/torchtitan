# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for moe_tiles and the measured spec it ships with.

``test_measured_spec_is_valid`` is pure Python. ``test_apply_and_remove`` needs
dist_moe (skips otherwise): it runs ``apply()`` with the measured spec, which
checks every exact-text anchor in ``dist_moe/api.py``, ``kernels/config.py``
and ``kernels/grouped_gemm_kernel.py`` and registers the configs, then confirms
that exactly the spec's launchers are wrapped and that the fc13_fprop config can
still emit the fused SwiGLU forward, and restores the launchers.
"""

from pathlib import Path

import pytest

from torchtitan.experiments.dist_moe_patches import moe_tiles

SPEC_PATH = Path(__file__).parents[1] / "moe_tiles_muse_spark_816b.json"
INTERMEDIATE_DIM = 4096  # expert_hidden_dim the spec was measured at


def test_measured_spec_is_valid():
    spec = moe_tiles.load_spec(str(SPEC_PATH))
    assert spec, "measured spec is empty"
    for gemm, (config, expect) in spec.items():
        assert isinstance(config, dict) and expect is not None, gemm
        assert moe_tiles.check_config(config, gemm) == [], gemm


def test_apply_and_remove():
    dgg = pytest.importorskip("dist_moe.kernels.dist_grouped_gemm")
    spec = moe_tiles.load_spec(str(SPEC_PATH))
    moe_tiles.apply(spec)
    try:
        for launcher, gemm in moe_tiles.LAUNCHERS.items():
            wrapped = hasattr(getattr(dgg, launcher), "_moe_tiles_original")
            assert wrapped == (gemm in spec), launcher
        fc13_name = moe_tiles.registered_name(spec["fc13_fprop"][0])
        assert dgg._native_swiglu_fwd_config_is_compatible(fc13_name, INTERMEDIATE_DIM)
    finally:
        moe_tiles.remove()
