"""Opt-in TorchTitan recipe for the phase-preserving Muse Spark C4 replay."""

import json
import os
from pathlib import Path

from gb300_config import fit_host_memory
from h3_campaign import host_h3_fields
from torchtitan_recipes.models._muse_spark_indexed_replay import (
    COMPACT_CAPACITY,
    ORIGINAL_CAPACITY,
)
from torchtitan_recipes.models.muse_spark import (
    muse_spark_816b_a42b_dist_moe_bf16_spmd_64_compact_benchmark,
)


def bf16_repacked():
    """Recipe entry point for the existing campaign launcher."""
    source_dir = os.environ["MUSE_REPACK_SOURCE"]
    plan_dir = os.environ["MUSE_REPACK_PLAN"]
    config = muse_spark_816b_a42b_dist_moe_bf16_spmd_64_compact_benchmark(
        source_dir=source_dir,
        plan_dir=plan_dir,
        slot_gib=int(os.environ.get("MUSE_SLOT_GIB", "49")),
        save_moe_projections=os.environ.get("MUSE_SAVE_PROJ", "1") == "1",
    )
    if os.environ.get("MUSE_PLATFORM") == "gb300":
        config = fit_host_memory(config)
    if os.environ.get("RANK") == "0" and (output_dir := os.environ.get("REPRO_OUTPUT")):
        fields = {
            "variant": os.environ.get("VARIANT", "phase-preserving-repack"),
            "bf16_compute": True,
            "source_dir": source_dir,
            "plan_dir": plan_dir,
            "phase_preserved": True,
            "input_plan_sha256": json.loads((Path(plan_dir) / "manifest.json").read_text())["input_plan_sha256"],
            "norm_fusion": os.environ.get("EXP_NORM_FUSION", "0"),
            "swiglu_recompute": os.environ.get("EXP_SWIGLU_RECOMPUTE", "0"),
            "learned_route": os.environ.get("EXP_LEARNED_ROUTE", "0"),
            "host_h3": host_h3_fields(os.environ["VARIANT"]),
            "physical_tokens_per_gpu": COMPACT_CAPACITY,
            "physical_global_tokens": config.training.num_tokens_per_train_step,
            "workload_equivalent_tokens_per_gpu": ORIGINAL_CAPACITY,
            "parallelism": repr(config.parallelism),
            "dist_moe": repr(config.dist_moe_runtime),
            "precision": repr(config.precision),
            "optimizer": repr(config.optim),
            "loss": repr(config.loss),
            "remat": repr(config.activation_remat),
        }
        (Path(output_dir) / "effective-config.json").write_text(json.dumps(fields, indent=2))
    return config
