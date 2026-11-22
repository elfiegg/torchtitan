# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize a bench job: throughput per variant plus GPU clock/power spread.

Usage: python3 summarize.py <job-dir> [first_step]
Tokens/s/GPU is the mean over ranks of each rank's logged throughput(tps), then
mean and median over steps first_step..last. Step seconds = 32768 tokens/GPU / tps.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

TOKENS_PER_GPU_STEP = 2097152 // 64


def load_metrics(variant_dir):
    per_step = {}
    for path in variant_dir.glob("metrics-rank*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            per_step.setdefault(row["step"], []).append(
                (
                    row["rank"],
                    float(row["throughput(tps)"]),
                    row.get("memory/device_used_at_log(GiB)"),
                )
            )
    return per_step


def gpu_stats(variant_dir):
    clocks, power, throttled, total = [], [], 0, 0
    per_gpu = {}
    for path in variant_dir.glob("node-*-gpu.csv"):
        node = path.stem.split("-")[1]
        with path.open() as f:
            for row in csv.reader(f):
                try:
                    idx, sm, pw, reasons = (
                        row[1].strip(),
                        float(row[3]),
                        float(row[5]),
                        int(row[7].strip(), 16),
                    )
                except (ValueError, IndexError):
                    continue
                if pw < 800:  # idle / startup / barrier-only samples
                    continue
                total += 1
                clocks.append(sm)
                power.append(pw)
                # NVML clock event reasons: 0x1 idle, 0x2 app clocks setting, 0x4 SW power cap,
                # 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 HW power brake.
                if reasons & ~0x3:
                    throttled += 1
                per_gpu.setdefault(f"n{node}g{idx}", []).append(sm)
    if not total:
        return None
    gpu_means = {k: statistics.median(v) for k, v in per_gpu.items()}
    lo = min(gpu_means, key=gpu_means.get)
    hi = max(gpu_means, key=gpu_means.get)
    return {
        "busy_samples": total,
        "sm_mhz_mean": round(statistics.mean(clocks)),
        "sm_mhz_p5": round(sorted(clocks)[len(clocks) // 20]),
        "power_w_mean": round(statistics.mean(power)),
        "power_w_max": round(max(power)),
        "power_or_thermal_capped_frac": round(throttled / total, 3),
        "slowest_gpu": [lo, round(gpu_means[lo])],
        "fastest_gpu": [hi, round(gpu_means[hi])],
    }


def main():
    job = Path(sys.argv[1])
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    rows = []
    for variant_dir in sorted(p for p in job.iterdir() if p.is_dir()):
        status_path = variant_dir / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        per_step = load_metrics(variant_dir)
        steps = sorted(s for s in per_step if s >= first)
        entry = {
            "variant": variant_dir.name,
            "rc": status.get("rc"),
            "steps": len(steps),
        }
        failures = list(variant_dir.glob("failure-rank*.json"))
        if failures:
            entry["failure"] = json.loads(failures[0].read_text()).get("exception")
        if steps:
            step_tps = [statistics.mean(t for _, t, _ in per_step[s]) for s in steps]
            entry.update(
                tps_mean=round(statistics.mean(step_tps), 1),
                tps_median=round(statistics.median(step_tps), 1),
                step_s_mean=round(TOKENS_PER_GPU_STEP / statistics.mean(step_tps), 3),
                step_s_min=round(TOKENS_PER_GPU_STEP / max(step_tps), 3),
                step_s_max=round(TOKENS_PER_GPU_STEP / min(step_tps), 3),
                max_device_used_gib=round(
                    max((m or 0) for s in steps for _, _, m in per_step[s]), 1
                ),
            )
        entry["gpu"] = gpu_stats(variant_dir)
        rows.append(entry)
    base = next((r for r in rows if r["variant"] == "base" and "tps_mean" in r), None)
    for r in rows:
        if base and "tps_mean" in r:
            r["vs_base_pct"] = round(100 * (r["tps_mean"] / base["tps_mean"] - 1), 2)
    print(json.dumps(rows, indent=2))
    (job / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
