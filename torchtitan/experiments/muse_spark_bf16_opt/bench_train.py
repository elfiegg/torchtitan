# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``torchtitan.train`` for benchmarking: stop after N steps, keep exact per-rank metrics.

- Imports the env-gated Dist-MoE runtime patches first (no-ops when their
  ``EXP_*`` variables are unset).
- Writes every logged metric row of every rank to ``metrics-rank<r>.jsonl``
  (with device memory in use) and checks the FP32-master/BF16-moment dtypes
  after step 1.
- Stops after ``REPRO_STOP_AFTER`` steps (default 16), fails if a rank did not
  finish them, and exits without NCCL teardown after a clean ``Trainer.close``.

Env: ``RANK``, ``REPRO_OUTPUT`` (shared results dir), ``REPRO_LOCAL_OUTPUT``
(node-local dir), ``REPRO_STOP_AFTER``.
"""

import json
import math
import os
import pathlib
import shutil
import sys

import torch

from torchtitan.experiments.dist_moe_patches.train import import_patches
from torchtitan.experiments.muse_spark_bf16_opt import audit
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

import_patches()

rank = int(os.environ["RANK"])
expected_steps = int(os.environ.get("REPRO_STOP_AFTER", "16"))
output = pathlib.Path(os.environ["REPRO_OUTPUT"])
local_output = pathlib.Path(os.environ["REPRO_LOCAL_OUTPUT"])
local_output.mkdir(parents=True, exist_ok=True)
metrics_path = local_output / f"metrics-rank{rank}.jsonl"


class MetricsRecorder:
    def __init__(self):
        self.file = metrics_path.open("w")
        self.steps = []

    def log(self, metrics, step):
        audit.check(step)
        for key in [
            "loss_metrics/global_avg_loss",
            "grad_norm",
            "throughput(tps)",
            "tflops",
        ]:
            if not math.isfinite(float(metrics[key])):
                raise ValueError(f"Non-finite {key} at step {step}")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        metrics = dict(metrics)
        metrics["memory/device_used_at_log(GiB)"] = (total_bytes - free_bytes) / 2**30
        self.file.write(json.dumps({"rank": rank, "step": step, **metrics}) + "\n")
        self.steps.append(step)

    def close(self):
        self.file.close()
        shutil.copy2(metrics_path, output / metrics_path.name)


original_metrics_init = MetricsProcessor.__init__
original_close = Trainer.close
original_should_continue = Trainer.should_continue_training
recorder = None


def initialize_metrics(self, *args, **kwargs):
    global recorder
    original_metrics_init(self, *args, **kwargs)
    recorder = MetricsRecorder()
    self.loggers.append(recorder)


def should_continue(self):
    return (
        self.engine.num_completed_steps < expected_steps
        and original_should_continue(self)
    )


def close_and_exit(self):
    had_error = sys.exc_info()[0] is not None
    completed_steps = self.engine.num_completed_steps if hasattr(self, "engine") else 0
    if had_error:
        context = {"rank": rank, "completed_steps_before_error": completed_steps}
        if torch.cuda.is_initialized():
            context.update(
                allocated_before_close=torch.cuda.memory_allocated(),
                reserved_before_close=torch.cuda.memory_reserved(),
            )
        (output / f"failure-context-rank{rank}.json").write_text(
            json.dumps(context, indent=2)
        )
    original_close(self)
    if had_error:
        return
    if (
        completed_steps != expected_steps
        or recorder is None
        or recorder.steps != list(range(1, expected_steps + 1))
    ):
        raise RuntimeError(
            f"Incomplete benchmark on rank {rank}: {completed_steps} steps"
        )
    traces = list(
        (output / "trainer" / "profiling" / "traces").glob(
            f"iteration_*/rank{rank}_trace.json.gz"
        )
    )
    receipt = {
        "trace_files": [str(path.relative_to(output)) for path in traces],
        "rank": rank,
        "completed_steps": completed_steps,
        "trainer_close": "success",
        "metric_rows": len(recorder.steps),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": torch.cuda.get_device_properties(torch.cuda.current_device()).name,
        "exit_workaround": "os._exit(0) after successful Trainer.close",
    }
    (output / f"completion-rank{rank}.json").write_text(json.dumps(receipt, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


MetricsProcessor.__init__ = initialize_metrics
Trainer.close = close_and_exit
Trainer.should_continue_training = should_continue

if __name__ == "__main__":
    from torchtitan.train import main

    try:
        main()
    except BaseException as error:
        failure = {
            "rank": rank,
            "exception": type(error).__name__,
            "message": str(error),
        }
        if torch.cuda.is_initialized():
            failure.update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
            )
        (output / f"failure-rank{rank}.json").write_text(json.dumps(failure, indent=2))
        raise
