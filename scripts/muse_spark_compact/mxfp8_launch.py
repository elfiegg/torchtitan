"""Run the pinned TorchTitan MXFP8 recipe and save comparable per-rank metrics."""

import json
import math
import os
from pathlib import Path
import shutil
import sys

import torch
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

import audit
import planner_audit


RANK = int(os.environ["RANK"])
STEPS = 100
OUTPUT = Path(os.environ["REPRO_OUTPUT"])
LOCAL_OUTPUT = Path(os.environ["REPRO_LOCAL_OUTPUT"])
LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
METRICS = LOCAL_OUTPUT / f"metrics-rank{RANK}.jsonl"


class Recorder:
    def __init__(self):
        self.file = METRICS.open("w")
        self.steps = []

    def log(self, metrics, step):
        audit.check(step)
        for key in ("loss_metrics/global_avg_loss", "grad_norm", "throughput(tps)", "tflops"):
            if not math.isfinite(float(metrics[key])):
                raise ValueError(f"Nonfinite metric {key} at update {step}")
        free, total = torch.cuda.mem_get_info()
        metrics = dict(metrics)
        metrics["memory/device_used_at_log(GiB)"] = (total - free) / 2**30
        self.file.write(json.dumps({"rank": RANK, "step": step, **metrics}) + "\n")
        self.steps.append(step)

    def close(self):
        self.file.close()
        shutil.copy2(METRICS, OUTPUT / METRICS.name)


original_metrics = MetricsProcessor.__init__
original_close = Trainer.close
original_continue = Trainer.should_continue_training
recorder = None


def initialize_metrics(self, *args, **kwargs):
    global recorder
    original_metrics(self, *args, **kwargs)
    recorder = Recorder()
    self.loggers.append(recorder)


def should_continue(self):
    return self.engine.num_completed_steps < STEPS and original_continue(self)


def close(self):
    failed = sys.exc_info()[0] is not None
    steps = self.engine.num_completed_steps if hasattr(self, "engine") else 0
    original_close(self)
    if failed:
        return
    if steps != STEPS or recorder is None or recorder.steps != list(range(1, STEPS + 1)):
        raise RuntimeError(f"Incomplete MXFP8 benchmark on rank {RANK} at update {steps}")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    result = {
        "rank": RANK, "completed_steps": steps, "metric_rows": len(recorder.steps),
        "trainer_close": "success", "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version, "device": props.name,
        "device_total_memory_gib": props.total_memory / 2**30,
        "warmup_dist_moe_plan": planner_audit.snapshots,
    }
    (OUTPUT / f"completion-rank{RANK}.json").write_text(json.dumps(result, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    # The pinned benchmark uses the same successful-close workaround.
    os._exit(0)


MetricsProcessor.__init__ = initialize_metrics
Trainer.close = close
Trainer.should_continue_training = should_continue

try:
    if torch.version.git_version != "daf1a40a5de981f7cab47fed4dd9883b1581cd63":
        raise RuntimeError("Expected the pinned benchmark PyTorch build")
    planner_audit.install()
    import norm_fusion  # noqa: F401
    from torchtitan.train import main

    main()
except BaseException as error:
    failure = {"rank": RANK, "exception": type(error).__name__, "message": str(error)}
    if torch.cuda.is_initialized():
        failure["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        failure["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (OUTPUT / f"failure-rank{RANK}.json").write_text(json.dumps(failure, indent=2))
    raise
