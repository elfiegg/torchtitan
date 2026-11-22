# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check FP32 masters/gradients and BF16 Adam moments after an update."""

import collections
import json
import os
from pathlib import Path

import torch

from torchtitan.components.optim import AdamW


optimizers = []
original_init = AdamW.__init__


def initialize(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    optimizers.append(self)


AdamW.__init__ = initialize


def local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def check(step):
    if step != 1:
        return
    counts = collections.Counter()
    violations = []
    missing_gradients = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            names = group.get("param_names", ["<unnamed>"] * len(group["params"]))
            for name, parameter in zip(names, group["params"], strict=True):
                state = optimizer.state.get(parameter, {})
                tensors = {
                    "parameter": parameter,
                    "gradient": parameter.grad,
                    "exp_avg": state.get("exp_avg"),
                    "exp_avg_sq": state.get("exp_avg_sq"),
                }
                if parameter.grad is None:
                    missing_gradients.append(name)
                for kind, tensor in tensors.items():
                    if tensor is None:
                        if kind != "gradient":
                            violations.append([name, kind, "missing"])
                        continue
                    value = local(tensor)
                    counts[f"{kind}:{value.dtype}:{value.device.type}"] += value.numel()
                    expected = (
                        torch.float32
                        if kind in {"parameter", "gradient"}
                        else torch.bfloat16
                    )
                    if value.dtype != expected or value.device.type != "cuda":
                        violations.append(
                            [name, kind, str(value.dtype), str(value.device)]
                        )
    if missing_gradients:
        violations.append(["missing gradients", len(missing_gradients)])
    if not optimizers or not any(key.startswith("gradient:") for key in counts):
        violations.append(["optimizer/gradient coverage", "empty"])
    result = {
        "step": step,
        "rank": int(os.environ["RANK"]),
        "optimizer_count": len(optimizers),
        "local_element_counts": dict(counts),
        "parameters_without_gradients": missing_gradients,
        "violations": violations,
        "status": "pass" if not violations else "fail",
    }
    destination = (
        Path(os.environ["REPRO_OUTPUT"]) / f"dtype-audit-rank{os.environ['RANK']}.json"
    )
    destination.write_text(json.dumps(result, indent=2))
    if violations:
        raise RuntimeError(f"FP32-master/BF16-moment audit failed: {violations[:5]}")
