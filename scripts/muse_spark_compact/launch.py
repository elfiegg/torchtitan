"""Run the upstream entrypoint, retaining exact metrics and skipping NCCL teardown."""

import json
import math
import os
import pathlib
import shutil
import sys

import torch
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer
import audit
import h3_campaign

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
variant = os.environ['VARIANT']
expected_steps = int(os.environ.get('REPRO_STOP_AFTER', '100'))
output = pathlib.Path(os.environ['REPRO_OUTPUT'])
local_output = pathlib.Path(os.environ['REPRO_LOCAL_OUTPUT'])
local_output.mkdir(parents=True, exist_ok=True)
metrics_path = local_output / f'metrics-rank{rank}.jsonl'
h3_binder = None


class MetricsRecorder:
    def __init__(self):
        self.file = metrics_path.open('w')
        self.steps = []
        self.host_snapshots = []

    def record_host_phase(self, phase, step):
        self.host_snapshots.append(h3_campaign.collect_host_phase(torch, h3_binder, phase, step, rank, local_rank))

    def log(self, metrics, step):
        audit.check(step)
        for key in ['loss_metrics/global_avg_loss', 'grad_norm', 'throughput(tps)', 'tflops']:
            if not math.isfinite(float(metrics[key])):
                raise ValueError(f'Non-finite {key} at step {step}')
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        metrics = dict(metrics)
        metrics["memory/device_used_at_log(GiB)"] = (total_bytes - free_bytes) / 2**30
        self.file.write(json.dumps({'rank': rank, 'step': step, **metrics}) + '\n')
        self.steps.append(step)
        if step == 1:
            self.record_host_phase(h3_campaign.HOST_PHASES[0], step)
        elif step == 3:
            self.record_host_phase(h3_campaign.HOST_PHASES[1], step)

    def close(self):
        self.file.close()
        shutil.copy2(metrics_path, output / metrics_path.name)


original_metrics_init = MetricsProcessor.__init__
recorder = None


def initialize_metrics(self, *args, **kwargs):
    global recorder
    original_metrics_init(self, *args, **kwargs)
    recorder = MetricsRecorder()
    self.loggers.append(recorder)


original_close = Trainer.close
original_should_continue = Trainer.should_continue_training


def should_continue(self):
    return self.engine.num_completed_steps < expected_steps and original_should_continue(self)


def close_and_exit(self):
    had_error = sys.exc_info()[0] is not None
    completed_steps = self.engine.num_completed_steps if hasattr(self, 'engine') else 0
    if had_error:
        original_close(self)
        return
    try:
        if completed_steps != expected_steps or recorder is None or recorder.steps != list(range(1, expected_steps + 1)):
            raise RuntimeError(f'Incomplete benchmark on rank {rank}: {completed_steps} steps')
        # Engine.close tears down the graph and shared Dist-MoE runtime.
        recorder.record_host_phase(h3_campaign.HOST_PHASES[2], completed_steps)
        host_h3 = h3_binder.receipt() if h3_binder is not None else None
        graph_enabled = self.engine._forward_backward_accelerator_graph_enabled
    finally:
        original_close(self)
    receipt = {
        'rank': rank,
        'completed_steps': completed_steps,
        'trainer_close': 'success',
        'metric_rows': len(recorder.steps),
        'torch_version': torch.__version__,
        'torch_git_version': torch.version.git_version,
        'device': torch.cuda.get_device_properties(torch.cuda.current_device()).name,
        'device_total_memory_gib': torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30,
        'exit_workaround': 'os._exit(0) after successful Trainer.close',
        'training_accelerator_graph': graph_enabled,
        'learned_route': learned_route.calls.receipt(),
    }
    if host_h3 is not None:
        receipt["host_h3"] = host_h3
    fields = h3_campaign.host_h3_fields(variant)
    h3_campaign.check_host_snapshots(recorder.host_snapshots, rank, fields)
    h3_campaign.check_completion(receipt, rank, fields)
    (output / f'host-allocator-rank{rank}.json').write_text(json.dumps(recorder.host_snapshots, indent=2))
    (output / f'completion-rank{rank}.json').write_text(json.dumps(receipt, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


MetricsProcessor.__init__ = initialize_metrics
Trainer.close = close_and_exit
Trainer.should_continue_training = should_continue

try:
    h3_campaign.check_environment()
    import norm_fusion  # noqa: F401
    import swiglu_recompute  # noqa: F401
    import learned_route

    learned_route.install()
    h3_binder = h3_campaign.install(variant, torch)

    from torchtitan.train import main

    main()
except BaseException as error:
    failure = {'rank': rank, 'exception': type(error).__name__, 'message': str(error)}
    if torch.cuda.is_initialized():
        failure.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
        )
    (output / f'failure-rank{rank}.json').write_text(json.dumps(failure, indent=2))
    raise
