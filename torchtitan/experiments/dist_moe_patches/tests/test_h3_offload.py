# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU parity test for h3_offload (EXP_MOE_H3_OFFLOAD), shrunken Dist-MoE.

Run on one node from the repository root (expects 4 GPUs and dist_moe; this
is a torchrun script, not a pytest module):

    PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
        torchtitan/experiments/dist_moe_patches/tests/test_h3_offload.py
    PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
        torchtitan/experiments/dist_moe_patches/tests/test_h3_offload.py \
        --with-swiglu-recompute

The stacked mode needs the swiglu_recompute module; when it is not in this
checkout, the script prints SKIP and exits 0 without starting a process group.

What it does, per case (activation-slot size x postprocess path):
  1. Reference: unpatched dist_moe, run twice to measure run-to-run noise.
  2. Patched: h3_offload.apply(), same inputs and weights, run once.
  3. Same again with every h3 copy delayed ~10 ms on its copy stream, so a
     missing stream wait would read stale data instead of winning a race.
  Steps 2-3 run with one device staging buffer (default) and with direct D2H.
Every run is L=3 MoE layers sharing one context, M=2 microbatches (forward and
backward of each, gradients accumulated), 2 steps, with a learned router (so
grad_topk_scores, the only consumer of h3, reaches a parameter) and, in the
"eager" path, a Muse-like learned RMSNorm callback on the expert output.

Pass criteria (all ranks):
  * every output and gradient of the patched runs differs from reference run 1
    by at most 2x the reference's own run-to-run difference (bitwise equal when
    the reference is deterministic), and all are finite;
  * the patch was really used: the FC2 combine GEMM runs 0 times in patched
    backward (L*M*steps times in reference backward), and the copy counters are
    offloads = restores = L*M*steps, resident = M*steps (the last layer),
    prefetched = (L-1)*M*steps, synchronous restores = 0.
Slot sizes: minimum (all layers recompute), mixed (layer 0 saved), large (all
saved). Expected runtime: < 2 min after the CuTe kernels are JIT-compiled.
"""

import argparse
import importlib.util
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

T, D, FF, K, LOCAL_EXPERTS = 2048, 1024, 1024, 8, 4  # tokens/rank, hidden, intermediate
L, M, STEPS = 3, 2, 2  # MoE layers, microbatches, steps


class RouteNorm(nn.Module):
    """Muse-like learned RMSNorm applied to each route output (eager callback)."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) + 0.1 * torch.randn(dim))

    def forward(self, h):
        out = F.rms_norm(h.float(), (h.shape[-1],), None, 1e-6) * self.weight
        return out.to(h.dtype).contiguous()

    def to_dist_moe_postprocess(self):
        return self


class Model(nn.Module):
    def __init__(self, num_experts, device):
        super().__init__()
        g = torch.Generator().manual_seed(
            1234 + dist.get_rank()
        )  # expert shards differ per rank
        gs = torch.Generator().manual_seed(99)  # router and norms are replicated

        def rand(shape, std, gen, dtype=torch.bfloat16):
            return nn.Parameter(
                (torch.randn(shape, generator=gen) * std).to(device, dtype)
            )

        self.w13 = nn.ParameterList(
            [rand((LOCAL_EXPERTS, 2 * FF, D), D**-0.5, g) for _ in range(L)]
        )
        self.w2 = nn.ParameterList(
            [rand((LOCAL_EXPERTS, D, FF), FF**-0.5, g) for _ in range(L)]
        )
        self.router = nn.ParameterList(
            [rand((num_experts, D), D**-0.5, gs, torch.float32) for _ in range(L)]
        )
        torch.manual_seed(7)
        self.norms = nn.ModuleList([RouteNorm(D) for _ in range(L)]).to(device)


def run(model, context, path, combine_calls, phase):
    """Two steps of M microbatches; returns every output and gradient."""
    import dist_moe

    rank = dist.get_rank()
    out = {}
    for p in model.parameters():
        p.grad = None
    for step in range(STEPS):
        context.reset()
        for mb in range(M):
            g = torch.Generator(device="cuda").manual_seed(1000 * step + 10 * mb + rank)
            x0 = torch.randn(T, D, generator=g, device="cuda", dtype=torch.bfloat16)
            x0.requires_grad_(True)
            target = torch.randn(T, D, generator=g, device="cuda", dtype=torch.float32)
            x = x0
            phase[0] = "fwd"
            for layer in range(L):
                logits = x.float() @ model.router[layer].t()
                top_vals, ids = logits.topk(K, dim=-1)
                scores = torch.softmax(top_vals, dim=-1)
                options = (
                    None
                    if path == "registered"
                    else dist_moe.ExecutionOptions(
                        experts_output_postprocess=model.norms[layer]
                    )
                )
                y = dist_moe.routed_experts(
                    x.contiguous(),
                    ids.contiguous(),
                    scores.contiguous(),
                    model.w13[layer],
                    model.w2[layer],
                    context,
                    options=options,
                )
                out[f"s{step}.mb{mb}.out{layer}"] = y.detach().clone()
                x = x + y
            loss = (x.float() * target).sum() / (T * D)
            phase[0] = "bwd"
            loss.backward()
            phase[0] = "fwd"
            out[f"s{step}.mb{mb}.grad_x0"] = x0.grad.detach().clone()
    for name, p in model.named_parameters():
        if p.grad is not None:
            out[f"grad.{name}"] = p.grad.detach().clone()
    torch.cuda.synchronize()
    out["_combine_fwd"] = combine_calls["fwd"]
    out["_combine_bwd"] = combine_calls["bwd"]
    combine_calls["fwd"] = combine_calls["bwd"] = 0
    return out


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-swiglu-recompute", action="store_true")
    args = parser.parse_args()
    # This test applies and removes the patch itself.
    os.environ.pop("EXP_MOE_H3_OFFLOAD", None)
    swiglu_module = "torchtitan.experiments.dist_moe_patches.swiglu_recompute"
    if args.with_swiglu_recompute and importlib.util.find_spec(swiglu_module) is None:
        if os.environ.get("RANK", "0") == "0":
            print(
                f"[test] SKIP --with-swiglu-recompute: {swiglu_module} is not in this checkout",
                flush=True,
            )
        sys.exit(0)

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if args.with_swiglu_recompute:
        os.environ["EXP_SWIGLU_RECOMPUTE"] = "1"
        # Patches on import; must precede h3_offload.apply().
        from torchtitan.experiments.dist_moe_patches import (  # noqa: F401
            swiglu_recompute,
        )
    import dist_moe
    from dist_moe.kernels import dist_grouped_gemm as dgg

    from torchtitan.experiments.dist_moe_patches import h3_offload

    # Count FC2 combine GEMM launches by phase. dist_moe imports this name at
    # call time, so patching the module attribute sees every call.
    phase = ["fwd"]
    combine_calls = {"fwd": 0, "bwd": 0}
    real_combine = dgg.dist_grouped_gemm_fprop_combine

    def counting_combine(*a, **k):
        combine_calls[phase[0]] += 1
        return real_combine(*a, **k)

    dgg.dist_grouped_gemm_fprop_combine = counting_combine

    num_experts = LOCAL_EXPERTS * world
    model = Model(num_experts, "cuda")
    x_bytes = T * D * 2
    rows = T * K  # balanced receive rows per rank
    saved_layer = (
        rows * D * 2 + rows * 2 * FF * 2 + T * K * D * 2
    )  # x_gathered + h1 + h3
    slots = {
        "all_recompute": None,
        "mixed": L * x_bytes + int(1.5 * saved_layer),
        "all_saved": 4 * L * saved_layer,
    }
    n_layers = L * M * STEPS
    failures = []
    t_start = time.time()
    for slot_name, slot_bytes in slots.items():
        context = dist_moe.create_context(
            group=dist.group.WORLD,
            config=dist_moe.Config(
                max_num_local_input_tokens=T,
                hidden_dim=D,
                intermediate_dim=FF,
                top_k=K,
                num_experts=num_experts,
                max_moe_layers_per_activation_slot=L,
                device_scratch_capacity_factor=2.0,
                activation_slot_bytes=slot_bytes,
            ),
        )
        for path in ("registered", "eager"):
            case = f"{slot_name}/{path}"
            ref1 = run(model, context, path, combine_calls, phase)
            ref2 = run(model, context, path, combine_calls, phase)
            results = {}
            for stage, delay in ((1, 0), (1, 20_000_000), (0, 0), (0, 20_000_000)):
                variant = f"stage={stage} delay={delay}"
                h3_offload.apply()
                h3_offload.reset_stats()
                h3_offload.STAGE_BUFFERS = stage
                h3_offload.DEBUG_COPY_DELAY_CYCLES = delay
                h3_offload._runtime.debug_need_recompute = []
                results[variant] = run(model, context, path, combine_calls, phase)
                stats = h3_offload.stats()
                pattern = [
                    int(t.item()) for t in h3_offload._runtime.debug_need_recompute[:L]
                ]
                h3_offload._runtime.debug_need_recompute = None
                h3_offload.DEBUG_COPY_DELAY_CYCLES = 0
                h3_offload.STAGE_BUFFERS = 1
                h3_offload.remove()
                expect = dict(
                    offloads=n_layers,
                    restore_resident=M * STEPS,
                    restore_prefetched=(L - 1) * M * STEPS,
                    restore_sync=0,
                )
                for key, want in expect.items():
                    got = getattr(stats, key)
                    if got != want:
                        failures.append(
                            f"{case} {variant}: stats.{key}={got}, want {want}"
                        )
                if results[variant]["_combine_bwd"] != 0:
                    failures.append(
                        f"{case} {variant}: combine GEMM ran "
                        f"{results[variant]['_combine_bwd']}x in patched backward"
                    )
            if ref1["_combine_bwd"] != n_layers or ref1["_combine_fwd"] != n_layers:
                failures.append(
                    f"{case}: reference combine calls fwd={ref1['_combine_fwd']} "
                    f"bwd={ref1['_combine_bwd']}, want {n_layers} each"
                )
            for name, ref in ref1.items():
                if name.startswith("_"):
                    continue
                if not torch.isfinite(ref).all():
                    failures.append(f"{case}: reference {name} not finite")
                    continue
                noise = max_diff(ref2[name], ref)
                tol = 2.0 * noise
                for variant, res in results.items():
                    d = max_diff(res[name], ref)
                    if not torch.isfinite(res[name]).all() or d > tol:
                        failures.append(
                            f"{case} {variant}: {name} max|patched-ref|={d:.3e} > "
                            f"2*ref_noise={tol:.3e}"
                        )
            if rank == 0:
                print(
                    f"[test] {case}: need_recompute(layers 0..{L - 1})={pattern} "
                    f"tensors={sum(not k.startswith('_') for k in ref1)} "
                    f"ref_noise_max={max(max_diff(ref2[k], ref1[k]) for k in ref1 if not k.startswith('_')):.3e} "
                    f"stats={stats} t={time.time() - t_start:.0f}s",
                    flush=True,
                )
        context.close()

    failed = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(failed)
    for line in failures[:20]:
        print(f"[test][rank {rank}] FAIL {line}", flush=True)
    if rank == 0:
        verdict = "PASS" if failed.item() == 0 else f"FAIL ({failed.item()} problems)"
        print(f"[test] {verdict} in {time.time() - t_start:.0f}s", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if failed.item() == 0 else 1)


if __name__ == "__main__":
    main()
