"""Validate all 64 rank receipts and report comparable throughput and model FLOPS."""

import argparse
import json
from math import isfinite
from pathlib import Path
from statistics import mean

from torchtitan_recipes.models._muse_spark_indexed_replay import LANES, ORIGINAL_CAPACITY


WINDOW = slice(70, 90)
STEPS = 100
STEP_TIME = "time_metrics/end_to_end(s)"
METRICS = ("loss_metrics/global_avg_loss", "grad_norm", "throughput(tps)", "tflops", STEP_TIME)


def rank_summary(root, rank):
    receipt = json.loads((root / f"completion-rank{rank}.json").read_text())
    audit = json.loads((root / f"dtype-audit-rank{rank}.json").read_text())
    if (
        receipt["rank"] != rank
        or receipt["trainer_close"] != "success"
        or (receipt["completed_steps"], receipt["metric_rows"]) != (STEPS, STEPS)
        or audit["rank"] != rank
        or audit["status"] != "pass"
        or audit["violations"]
    ):
        raise ValueError(f"Incomplete training or dtype checks at rank {rank}")

    rows = [json.loads(line) for line in (root / f"metrics-rank{rank}.jsonl").read_text().splitlines()]
    if [row["step"] for row in rows] != list(range(1, STEPS + 1)) or any(
        row["rank"] != rank or not all(isfinite(row[key]) for key in METRICS) or row[STEP_TIME] <= 0
        for row in rows
    ):
        raise ValueError(f"Missing or invalid metrics at rank {rank}")

    window = rows[WINDOW]
    return {
        "rank": rank,
        "device": receipt["device"],
        "nominal_tokens_per_second_per_gpu": mean(ORIGINAL_CAPACITY / row[STEP_TIME] for row in window),
        "native_tokens_per_second_per_gpu": mean(row["throughput(tps)"] for row in window),
        "model_tflops_per_gpu": mean(row["tflops"] for row in window),
        "seconds_per_update": mean(row[STEP_TIME] for row in window),
        "peak_active_gib": max(row["memory/max_active(GiB)"] for row in rows),
        "peak_reserved_gib": max(row["memory/max_reserved(GiB)"] for row in rows),
        "peak_device_used_at_log_gib": max(row["memory/device_used_at_log(GiB)"] for row in rows),
    }


def summarize(root):
    ranks = [rank_summary(root, rank) for rank in range(LANES)]
    result = {"ranks_completed": LANES, "timed_updates": [71, 90], "rank0": ranks[0], "ranks": ranks}
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "ranks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
