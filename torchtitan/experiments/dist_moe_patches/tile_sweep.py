#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tile-config sweep for the Dist-MoE BF16 fprop/dgrad grouped GEMMs (one node).

Times the exact launchers training calls, with the exact keyword arguments
training passes, at Muse Spark 816B production shapes per rank:
  T = 16384 tokens (32K seq / CP2), D = 4096 (compressed_dim), I = 4096
  (expert_hidden_dim), top-8, 4 local experts, round-robin (forced balanced)
  routing -> 4 x 32768 = 131072 routed rows, clamped SwiGLU (1.702, 7.0).

  fc13_fprop  dist_grouped_gemm_fprop_dispatch + emit SwiGLU  N=8192 K=4096  8.80 TFLOP
  fc2_fprop   dist_grouped_gemm_fprop_combine                 N=4096 K=4096  4.40 TFLOP
  fc2_dgrad   dist_grouped_gemm_dgrad_dispatch                N=4096 K=4096  4.40 TFLOP
  fc13_dgrad  dist_grouped_gemm_dgrad_swiglu_bwd_combine      N=4096 K=8192  8.80 TFLOP

How: an EP=<nproc> Dist-MoE context runs ONE real forward+backward with upstream
configs while the launchers are wrapped to record their kwargs (activation-buffer
offsets, gather/scatter pointer tables, symmetric buffers, SwiGLU flags). Weight
gradients and the conditional backward recompute are stubbed during that pass
(not swept, saves compiles). Each GEMM is then replayed with only config=
changed. Combine GEMMs scatter into private "sink" symmetric buffers (pointer
table remapped peer by peer), so GEMMs running concurrently on different GPUs
never write what another GPU reads; the NVLink traffic pattern is unchanged.

Phase 1 (split): rank r sweeps GEMMs GEMM_ORDER[r::world] independently.
  Per GEMM: upstream config runs once as the numerics reference; every
  candidate is JIT-compiled and checked against it (bitwise, or max bf16 ULP
  when not). Then, after --heat-seconds of unmeasured load, all configs are
  timed back-to-back (upstream first and last, as a drift check), so no timing
  follows an idle compile: each gets --warmup-seconds, then >= --seconds of
  sustained CUDA-graph replays timed with CUDA events, with SM clock, power and
  energy from NVML (pynvml) over the same GPU window.
Phase 2 (lockstep, world > 1): all ranks run the same GEMM and config at the
  same time (as in training): upstream, the top --confirm-top candidates from
  phase 1, upstream again. The winner per GEMM comes from phase 2 when it ran.
--verify-spec (optional, ~1 min): one real Dist-MoE forward+backward with
  moe_tiles.apply(<winning spec>) vs upstream (bitwise or within 2x upstream
  run-to-run noise; every spec GEMM must actually be overridden). The spec file
  is not written if this fails.

Fails fast: any exception (compile/launch error, numerics beyond --ulp-tol,
coverage error) aborts the job (torchrun tears down the other ranks) after
writing the partial per-rank JSON. --keep-going records candidate errors instead.
A wall-clock budget (--budget-s, default 540 s from start) skips the
lowest-priority candidates rather than overrunning.

Outputs (rank 0): --out JSON (all records, best per GEMM) and --spec-out, an
EXP_MOE_TILES spec for moe_tiles.py containing only winners that beat upstream
by >= --min-gain-pct.

  torchrun --standalone --nproc_per_node=4 -m torchtitan.experiments.dist_moe_patches.tile_sweep --out sweep.json
  python -m torchtitan.experiments.dist_moe_patches.tile_sweep --list   # candidates + SMEM/TMEM, no GPU
"""

import argparse
import inspect
import json
import math
import os
import sys
import threading
import time
import traceback

_T_START = time.time()
if os.environ.get("EXP_MOE_TILES", "").strip():
    raise SystemExit(
        "tile_sweep.py: unset EXP_MOE_TILES; the sweep sets configs itself"
    )
# Pure python (no torch import) unless EXP_MOE_TILES is set.
from torchtitan.experiments.dist_moe_patches import moe_tiles as mt  # noqa: E402

# Muse Spark 816B-A42B (torchtitan/models/muse_spark/flavors.py) under
# muse_spark_816b_a42b_dist_moe_bf16_spmd_64_benchmark: seq 32K, CP2, TP1, EP64.
TOKENS = 16384
HIDDEN = 4096  # compressed_dim
INTER = 4096  # expert_hidden_dim
TOP_K = 8
LOCAL_EXPERTS = 4  # 256 experts / EP64
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0

GEMM_ORDER = ("fc2_dgrad", "fc13_fprop", "fc13_dgrad", "fc2_fprop")
LAUNCHER = {
    "fc13_fprop": "dist_grouped_gemm_fprop_dispatch",
    "fc2_fprop": "dist_grouped_gemm_fprop_combine",
    "fc2_dgrad": "dist_grouped_gemm_dgrad_dispatch",
    "fc13_dgrad": "dist_grouped_gemm_dgrad_swiglu_bwd_combine",
}
N_OUT = {
    "fc13_fprop": 2 * INTER,
    "fc2_fprop": HIDDEN,
    "fc2_dgrad": INTER,
    "fc13_dgrad": HIDDEN,
}
EXPECTED_DEFAULT = {
    "fc13_fprop": "training_2cta2mma_bm512_bn256",
    "fc2_fprop": "training_2cta1mma_bm256_bn256",
    "fc2_dgrad": "training_2cta2mma_bm512_bn256",
    "fc13_dgrad": "training_2cta2mma_bm512_bn256",
}


def _k(ctas, mmas, bn, s, t, c, e):
    return mt.make_config(
        ctas=ctas, mmas=mmas, bn=bn, smem=s, tmem=t, c_stages=c, epi=e
    )


# Candidates in priority order (the budget drops from the end). The upstream
# config is always run first as the reference and is de-duplicated here.
# k(NUM_CTAS, NUM_MMAS, BLOCK_N, SMEM stages, TMEM buffers, C stages, EPILOGUE_SUBTILE);
# BLOCK_M = 128 * NUM_CTAS * NUM_MMAS. See MOE_TILES.md for the pruned space.
CANDIDATES = {
    # DISPATCH_SWIGLU_FWD + emit h1: sub-tile must be 64 cols (E = BN/64);
    # NUM_C_STAGES = number of 16 KB h1 TMA-store stages.
    "fc13_fprop": [
        (_k(2, 2, 256, 4, 1, 2, 4), "upstream tile + 2nd C stage (the wgrad fix)"),
        (
            _k(2, 1, 256, 6, 2, 2, 4),
            "bm256 2-TMEM (= training_2cta1mma, FC2 combine tile)",
        ),
        (_k(2, 2, 128, 4, 2, 2, 2), "bm512 2-TMEM via BN=128"),
        (_k(2, 1, 256, 5, 2, 4, 4), "bm256 2-TMEM, 5 A/B + 4 C stages"),
        (_k(2, 2, 128, 5, 2, 1, 2), "bm512 bn128 2-TMEM, deepest A/B ring"),
        (_k(2, 1, 256, 6, 2, 1, 4), "bm256 2-TMEM, 1 C stage (isolates C staging)"),
        (_k(2, 2, 128, 4, 2, 4, 2), "bm512 bn128 2-TMEM, 4 C stages"),
        (_k(2, 2, 256, 3, 1, 4, 4), "upstream tile, 3 A/B + 4 C stages"),
        (_k(2, 2, 128, 5, 1, 1, 2), "diagnostic: bn128 without 2nd TMEM buffer"),
        (_k(1, 2, 256, 3, 1, 2, 4), "1-CTA control (no B multicast)"),
    ],
    # plain DISPATCH, TMA-store epilogue: C stages and sub-tile width are free.
    "fc2_dgrad": [
        (_k(2, 2, 256, 4, 1, 2, 4), "upstream tile + 2nd C stage (the wgrad fix)"),
        (_k(2, 1, 256, 6, 2, 2, 4), "bm256 2-TMEM (= training_2cta1mma)"),
        (_k(2, 2, 128, 4, 2, 2, 2), "bm512 2-TMEM via BN=128"),
        (_k(2, 2, 256, 4, 1, 4, 8), "upstream tile, 32-col sub-tiles, 4 C stages"),
        (_k(2, 1, 256, 6, 2, 4, 8), "bm256 2-TMEM, 32-col sub-tiles, 4 C stages"),
        (_k(2, 2, 128, 5, 2, 3, 4), "bm512 bn128 2-TMEM, 5 A/B, 32-col x3 C"),
        (_k(2, 1, 256, 5, 2, 4, 4), "bm256 2-TMEM, 5 A/B + 4 C stages"),
        (_k(2, 2, 256, 4, 1, 1, 2), "upstream tile, 128-col sub-tiles"),
        (_k(2, 2, 128, 4, 2, 2, 1), "bm512 bn128 2-TMEM, whole-tile sub-tile"),
        (_k(2, 1, 256, 6, 2, 1, 2), "bm256 2-TMEM, 128-col sub-tiles"),
        (_k(2, 2, 256, 3, 1, 4, 4), "upstream tile, 3 A/B + 4 C stages"),
        (_k(1, 2, 256, 3, 1, 2, 4), "1-CTA control (no B multicast)"),
    ],
    # COMBINE_SWIGLU_BWD: one scatter stage; NUM_C_STAGES only sets the row skew.
    "fc13_dgrad": [
        (_k(2, 1, 256, 6, 2, 2, 4), "bm256 2-TMEM (= training_2cta1mma)"),
        (_k(2, 2, 128, 5, 2, 2, 2), "bm512 2-TMEM via BN=128"),
        (_k(2, 2, 256, 4, 1, 1, 8), "upstream tile, 32-col scatter sub-tiles"),
        (_k(2, 1, 256, 6, 2, 2, 8), "bm256 2-TMEM, 32-col sub-tiles"),
        (_k(2, 2, 128, 5, 2, 2, 4), "bm512 bn128 2-TMEM, 32-col sub-tiles"),
        (_k(2, 2, 256, 3, 1, 1, 2), "upstream tile, 3 A/B, 128-col sub-tiles"),
        (_k(2, 1, 256, 5, 2, 2, 2), "bm256 2-TMEM, 128-col sub-tiles"),
        (_k(2, 2, 128, 4, 2, 2, 1), "bm512 bn128 2-TMEM, whole-tile sub-tile"),
        (_k(1, 2, 256, 3, 1, 1, 4), "1-CTA control (no B multicast)"),
    ],
    # COMBINE: same epilogue rules; upstream already uses bm256 2-TMEM here.
    "fc2_fprop": [
        (_k(2, 2, 256, 4, 1, 1, 4), "bm512 split-atom (= training_2cta2mma)"),
        (_k(2, 2, 128, 5, 2, 2, 2), "bm512 2-TMEM via BN=128"),
        (_k(2, 1, 256, 6, 2, 2, 8), "bm256 2-TMEM, 32-col sub-tiles"),
        (_k(2, 1, 256, 5, 2, 2, 2), "bm256 2-TMEM, 128-col sub-tiles"),
        (_k(2, 2, 128, 5, 2, 2, 4), "bm512 bn128 2-TMEM, 32-col sub-tiles"),
        (_k(2, 2, 256, 4, 1, 1, 8), "bm512 split-atom, 32-col sub-tiles"),
        (_k(1, 2, 256, 3, 1, 1, 4), "1-CTA control (no B multicast)"),
    ],
}


def list_candidates():
    bad = 0
    for gemm in GEMM_ORDER:
        mode = mt.EPILOGUE_MODE[gemm]
        print(
            f"\n{gemm} ({LAUNCHER[gemm]}, epilogue={mode}, N={N_OUT[gemm]}, "
            f"upstream={EXPECTED_DEFAULT[gemm]})"
        )
        print(f"  {'label':34s} {'smem':>7s} {'tmem':>5s} {'F/B':>6s}  why")
        for cfg, why in CANDIDATES[gemm]:
            problems = mt.check_config(cfg, gemm, n_out=N_OUT[gemm])
            smem = mt.smem_breakdown(cfg, mode)["total"]
            flag = "" if not problems else "  INVALID: " + "; ".join(problems)
            bad += bool(problems)
            print(
                f"  {mt.config_label(cfg):34s} {smem:7d} {mt.tmem_cols(cfg):5d} "
                f"{mt.smem_flop_per_byte(cfg):6.1f}  {why}{flag}"
            )
        print(f"  ({len(CANDIDATES[gemm])} candidates + upstream)")
    return bad


# --------------------------------------------------------------------------
# GPU part (torch / dist_moe imported lazily so --list runs anywhere).
# --------------------------------------------------------------------------


class Log:
    def __init__(self, rank):
        self.rank = rank

    def __call__(self, msg):
        print(
            f"[tile_sweep r{self.rank} +{time.time() - _T_START:6.1f}s] {msg}",
            flush=True,
        )


class NvmlSampler:
    """Background NVML poller: SM clock, power, energy counter, clock-event reasons."""

    POWER_CAP_BIT = 0x4  # nvmlClocksThrottleReasonSwPowerCap

    def __init__(self, device_index, period_s=0.02):
        self.ok = False
        self.err = None
        self.samples = []
        self.lock = threading.Lock()
        self.period_s = period_s
        self.power_limit_w = None
        self.driver = None
        try:
            import pynvml as nv
            import torch

            nv.nvmlInit()
            handle = None
            try:
                uuid = str(torch.cuda.get_device_properties(device_index).uuid)
                for cand in (uuid, "GPU-" + uuid):
                    try:
                        handle = nv.nvmlDeviceGetHandleByUUID(cand)
                        break
                    except Exception:
                        continue
            except Exception:
                handle = None
            if handle is None:
                self.err = "UUID lookup failed; using NVML index = CUDA index"
                handle = nv.nvmlDeviceGetHandleByIndex(device_index)
            self.nv, self.h = nv, handle
            try:
                self.power_limit_w = nv.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except Exception:
                pass
            try:
                d = nv.nvmlSystemGetDriverVersion()
                self.driver = d.decode() if isinstance(d, bytes) else d
            except Exception:
                pass
            self.reasons_fn = getattr(
                nv, "nvmlDeviceGetCurrentClocksEventReasons", None
            ) or getattr(nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
            self.ok = True
        except Exception as exc:  # pynvml missing or no NVML in the container
            self.err = repr(exc)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self.ok:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _poll(self):
        nv, h = self.nv, self.h
        while not self._stop.is_set():
            t = time.perf_counter()
            row = [t, None, None, None, None]
            try:
                row[1] = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
            except Exception:
                pass
            try:
                row[2] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pass
            try:
                row[3] = nv.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
            except Exception:
                pass
            if self.reasons_fn is not None:
                try:
                    row[4] = int(self.reasons_fn(h))
                except Exception:
                    pass
            with self.lock:
                self.samples.append(row)
            time.sleep(self.period_s)

    def window(self, t0, t1):
        if not self.ok:
            return {"nvml": None, "nvml_error": self.err}
        with self.lock:
            rows = [r for r in self.samples if t0 <= r[0] <= t1]
        out = {"nvml_samples": len(rows)}
        clk = [r[1] for r in rows if r[1] is not None]
        pw = [r[2] for r in rows if r[2] is not None]
        en = [(r[0], r[3]) for r in rows if r[3] is not None]
        rs = [r[4] for r in rows if r[4] is not None]
        if clk:
            out["sm_clock_mhz_mean"] = sum(clk) / len(clk)
            out["sm_clock_mhz_min"] = min(clk)
            out["sm_clock_mhz_max"] = max(clk)
        if pw:
            out["power_w_mean"] = sum(pw) / len(pw)
        if len(en) >= 2 and en[-1][0] > en[0][0]:
            out["power_w_energy"] = (en[-1][1] - en[0][1]) / (en[-1][0] - en[0][0])
        if rs:
            mask = 0
            for r in rs:
                mask |= r
            out["clock_event_reasons_or"] = hex(mask)
            out["power_cap_frac"] = sum(1 for r in rs if r & self.POWER_CAP_BIT) / len(
                rs
            )
        return out


class Case:
    """One captured GEMM call plus its activation-buffer regions and sink."""

    def __init__(self, gemm, fn, kwargs, sink, world):
        import torch

        self.gemm = gemm
        self.fn = fn
        self.kwargs = kwargs
        self.default_name = kwargs["config"]
        self.w = kwargs["w"]
        self.ab = kwargs["activation_buffer"]
        self.gm = int(kwargs["num_tokens_per_local_expert"].sum().item())
        self.flops = 2.0 * self.gm * int(self.w.shape[1]) * int(self.w.shape[2])
        n_w, k_w = int(self.w.shape[1]), int(self.w.shape[2])
        if gemm == "fc13_fprop":
            ins, outs = [], [("y", n_w), ("swiglu_h2", n_w // 2), ("x_gathered", k_w)]
        elif gemm == "fc2_fprop":
            ins, outs = [("x", k_w)], []
        elif gemm == "fc2_dgrad":
            ins, outs = [], [("dx", k_w), ("dy_gathered", n_w)]
        elif gemm == "fc13_dgrad":
            ins, outs = [("grad_h2", n_w // 2), ("h1", n_w)], [("grad_h1", n_w)]
        else:
            raise ValueError(gemm)
        self.inputs = [self._region(k, w) for k, w in ins]
        self.outputs = [self._region(k, w) for k, w in outs]
        spans = sorted(r[1] for r in self.inputs + self.outputs)
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            if b0 < a1:
                raise RuntimeError(f"{gemm}: activation-buffer regions overlap {spans}")
        # Peer-scatter GEMMs: this rank writes only the [T*K, D] sink rows its
        # local experts own, on every rank. Index them from the remapped table.
        self.sink_views, self.sink_rows = [], []
        if sink is not None:
            width = int(sink.shape[-1])
            row_bytes = width * sink.dtype.itemsize
            nbytes = math.prod(sink.shape) * sink.dtype.itemsize
            ptrs = kwargs["scatter_ptrs"][: self.gm]
            bases = [int(b) for b in sink.buffer_ptrs_tensor.tolist()]
            covered = 0
            for r in range(world):
                view = sink.hdl.get_buffer(r, tuple(sink.shape), sink.dtype).view(
                    -1, width
                )
                m = (ptrs >= bases[r]) & (ptrs < bases[r] + nbytes)
                rel = ptrs[m] - bases[r]
                if int((rel % row_bytes != 0).sum().item()):
                    raise RuntimeError(f"{gemm}: scatter pointers not row aligned")
                self.sink_views.append(view)
                self.sink_rows.append((rel // row_bytes).to(torch.long))
                covered += int(m.sum().item())
            if covered != self.gm:
                raise RuntimeError(f"{gemm}: sink rows cover {covered} of {self.gm}")
        self._torch = torch

    def _region(self, key, width):
        off = int(self.kwargs[key].item())
        nbytes = self.gm * width * 2
        if off < 0 or off % 128 or off + nbytes > self.ab.numel():
            raise RuntimeError(
                f"{self.gemm}: bad region {key} off={off} bytes={nbytes}"
            )
        view = self.ab[off : off + nbytes].view(self._dtype()).view(self.gm, width)
        return (key, (off, off + nbytes), view)

    @staticmethod
    def _dtype():
        import torch

        return torch.bfloat16

    def outputs_all(self):
        """Output tensors to check; sink rows are gathered copies."""
        rows = [
            v[idx] for v, idx in zip(self.sink_views, self.sink_rows) if idx.numel()
        ]
        return [v for _, _, v in self.outputs] + rows

    def fill_inputs(self, gen):
        for _, _, v in self.inputs:
            v.normal_(generator=gen)

    def sentinel(self):
        # 0xFFFF is a bf16 NaN, never a GEMM output: catches rows left unwritten.
        for v in [v for _, _, v in self.outputs] + self.sink_views:
            v.view(self._torch.int16).fill_(-1)

    def launch(self, name):
        kwargs = dict(self.kwargs)
        kwargs["config"] = name
        return self.fn(**kwargs)

    def check_coverage(self):
        for i, v in enumerate(self.outputs_all()):
            n = int((v.view(self._torch.int16) == -1).sum().item())
            if n:
                raise RuntimeError(
                    f"{self.gemm}: upstream output #{i} left {n} sentinel elements"
                )

    def snapshot(self):
        return [v.clone() for v in self.outputs_all()]

    def compare(self, ref, ulp_tol):
        torch = self._torch
        max_ulp, ndiff, total = 0, 0, 0
        for r, v in zip(ref, self.outputs_all()):
            a, b = r.view(torch.int16).reshape(-1), v.view(torch.int16).reshape(-1)
            total += a.numel()
            if torch.equal(a, b):
                continue
            step = 1 << 26
            for i in range(0, a.numel(), step):
                ai = a[i : i + step].to(torch.int32)
                bi = b[i : i + step].to(torch.int32)
                ao = torch.where(ai < 0, -(ai & 0x7FFF), ai)
                bo = torch.where(bi < 0, -(bi & 0x7FFF), bi)
                d = (ao - bo).abs()
                max_ulp = max(max_ulp, int(d.max().item()))
                ndiff += int((d != 0).sum().item())
        if ndiff == 0:
            status = "bitwise"
        elif max_ulp <= ulp_tol:
            status = "bf16-rounding"
        else:
            status = "FAIL"
        return {"status": status, "max_ulp": max_ulp, "n_diff": ndiff, "n_total": total}


def _capture(dist_moe, dgg, gg, context, args, log):
    """One real forward+backward; record launcher kwargs; stub wgrad/recompute."""
    import torch
    import torch.distributed as dist

    rank, world = dist.get_rank(), dist.get_world_size()
    calls = {name: [] for name in mt.LAUNCHERS}
    originals = {name: getattr(dgg, name) for name in mt.LAUNCHERS}
    orig_wgrad = gg.grouped_gemm_wgrad
    stubbed = {"recompute": 0, "wgrad": 0}

    def recorder(name):
        orig = originals[name]
        sig = inspect.signature(orig)

        def rec(*a, **kw):
            if kw.get("conditional_execution") is not None:
                # Backward recompute (same GEMM as the forward call): not swept.
                stubbed["recompute"] += 1
                if name == "dist_grouped_gemm_fprop_combine":
                    return kw["symm_mem_buffer"].local()
                return None
            calls[name].append(dict(sig.bind(*a, **kw).arguments))
            return orig(*a, **kw)

        return rec

    def wgrad_stub(*a, **kw):
        stubbed["wgrad"] += 1
        out = kw.get("wgrad")
        if out is None:
            raise RuntimeError("tile_sweep: wgrad stub needs wgrad= (api call changed)")
        return out

    for name in mt.LAUNCHERS:
        setattr(dgg, name, recorder(name))
    gg.grouped_gemm_wgrad = wgrad_stub
    try:
        T, E = args.tokens, LOCAL_EXPERTS * world
        g = torch.Generator(device="cuda").manual_seed(1234 + rank)
        x = torch.randn(T, HIDDEN, generator=g, device="cuda", dtype=torch.bfloat16)
        x.requires_grad_(True)
        # RoundRobinTokenChoiceTopKRouter: arange(T*K).reshape(T, K) % E.
        ids = (
            torch.arange(T * TOP_K, device="cuda").reshape(T, TOP_K) % E
        ).contiguous()
        scores = torch.softmax(
            torch.randn(T, TOP_K, generator=g, device="cuda", dtype=torch.float32), -1
        ).contiguous()
        w13 = torch.nn.Parameter(
            torch.randn(
                LOCAL_EXPERTS,
                2 * INTER,
                HIDDEN,
                generator=g,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * HIDDEN**-0.5
        )
        w2 = torch.nn.Parameter(
            torch.randn(
                LOCAL_EXPERTS,
                HIDDEN,
                INTER,
                generator=g,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * INTER**-0.5
        )
        y = dist_moe.routed_experts(x, ids, scores, w13, w2, context)
        dy = torch.randn(y.shape, generator=g, device="cuda", dtype=torch.float32)
        (y.float() * dy).sum().backward()
        torch.cuda.synchronize()
        dist.barrier()
    finally:
        for name, fn in originals.items():
            setattr(dgg, name, fn)
        gg.grouped_gemm_wgrad = orig_wgrad
    num_calls = {k: len(v) for k, v in calls.items()}
    log(f"capture done: stubbed {stubbed}, calls {num_calls}")
    if calls["dist_grouped_gemm_dgrad_combine"]:
        raise RuntimeError(
            "unfused FC13 dgrad combine ran; production uses the fused path"
        )
    out = {}
    for gemm, name in LAUNCHER.items():
        if len(calls[name]) != 1:
            raise RuntimeError(
                f"{gemm}: expected 1 recorded {name} call, got {len(calls[name])}"
            )
        kw = calls[name][0]
        if kw.get("activation_buffer") is None or not isinstance(kw.get("config"), str):
            raise RuntimeError(f"{gemm}: unexpected call convention {sorted(kw)}")
        out[gemm] = (originals[name], kw)
    if out["fc13_fprop"][1].get("swiglu_h2") is None:
        raise RuntimeError("fc13_fprop: forward did not use the fused SwiGLU emit")
    return out, (x, ids, scores, w13, w2, y)


def _remap_ptrs(ptrs, src, dst, gm):
    """Rewrite peer pointers into src's symmetric buffers to dst's, rank by rank."""
    import torch

    base = src.buffer_ptrs_tensor.to(ptrs.device)
    new = dst.buffer_ptrs_tensor.to(ptrs.device)
    nbytes = math.prod(src.shape) * src.dtype.itemsize
    out = ptrs.clone()
    hit = torch.zeros_like(ptrs, dtype=torch.bool)
    for r in range(base.numel()):
        m = (ptrs >= base[r]) & (ptrs < base[r] + nbytes)
        out = torch.where(m, ptrs - base[r] + new[r], out)
        hit |= m
    n = int(hit[:gm].sum().item())
    if n != gm:
        raise RuntimeError(f"pointer remap covered {n} of {gm} rows")
    return out


def timing(case, name, args, sampler):
    """>= args.seconds of sustained back-to-back launches after a warm-up."""
    import torch

    kwargs = dict(case.kwargs)
    kwargs["config"] = name
    fn = case.fn
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(3):
        fn(**kwargs)
    en.record()
    torch.cuda.synchronize()
    t1 = st.elapsed_time(en) / 3.0  # ms
    per_replay = max(1, min(32, round(25.0 / max(t1, 1e-3))))
    graph = None
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn(**kwargs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            for _ in range(per_replay):
                fn(**kwargs)
        replay = graph.replay
        launch_mode = "cuda_graph"
    except Exception as exc:  # fall back to eager launches
        graph = None
        torch.cuda.synchronize()

        def replay():
            for _ in range(per_replay):
                fn(**kwargs)

        launch_mode = f"eager ({type(exc).__name__})"
    rep_ms = per_replay * t1
    for _ in range(max(1, math.ceil(args.warmup_seconds * 1e3 / rep_ms))):
        replay()
    n = max(2, math.ceil(1.1 * args.seconds * 1e3 / rep_ms))
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
    h0 = time.perf_counter()
    evs[0].record()
    for i in range(n):
        replay()
        evs[i + 1].record()
    h_enq = time.perf_counter() - h0
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    per = sorted(evs[i].elapsed_time(evs[i + 1]) / per_replay for i in range(n))
    total_ms = evs[0].elapsed_time(evs[n])
    mean_ms = total_ms / (n * per_replay)
    out = {
        "launch_mode": launch_mode,
        "launches": n * per_replay,
        "window_s": total_ms / 1e3,
        "mean_ms": mean_ms,
        "median_ms": per[len(per) // 2],
        "p10_ms": per[len(per) // 10],
        "p90_ms": per[(9 * len(per)) // 10],
        "pflops": case.flops / (mean_ms * 1e-3) / 1e15,
        "host_enqueue_frac": h_enq / (total_ms / 1e3),
    }
    out.update(sampler.window(t_end - total_ms / 1e3, t_end))
    if out.get("power_w_energy"):
        out["tflop_per_joule"] = (
            case.flops
            * (n * per_replay)
            / (out["power_w_energy"] * total_ms / 1e3)
            / 1e12
        )
    if launch_mode != "cuda_graph" and out["host_enqueue_frac"] > 0.8:
        out["warning"] = "eager launches may be host-bound"
    del graph
    return out


def _record_base(case, label, name, cfg, why, phase, rank):
    mode = mt.EPILOGUE_MODE[case.gemm]
    return {
        "phase": phase,
        "rank": rank,
        "gemm": case.gemm,
        "label": label,
        "name": name,
        "config": cfg,
        "why": why,
        "smem_est": mt.smem_breakdown(cfg, mode)["total"],
        "tmem_cols": mt.tmem_cols(cfg),
        "flop_per_smem_byte": round(mt.smem_flop_per_byte(cfg), 1),
        "tflop": case.flops / 1e12,
        "rows": case.gm,
    }


def _timing_cost_s(args):
    """Wall seconds one timing() call takes (warm-up + window + calibration)."""
    return args.warmup_seconds + 1.1 * args.seconds + 0.6


def _heat(case, name, seconds):
    """Unmeasured back-to-back launches so the first timed config is not run cold."""
    import torch

    end = time.time() + seconds
    while time.time() < end:
        for _ in range(8):
            case.launch(name)
        torch.cuda.synchronize()


def _log_timing(log, prefix, rec):
    tm = rec["timing"]
    log(
        f"{prefix}: {tm['pflops']:.3f} PF/s {tm['mean_ms']:.3f} ms "
        f"{tm.get('sm_clock_mhz_mean', float('nan')):.0f} MHz "
        f"{tm.get('power_w_mean', float('nan')):.0f} W "
        f"cap={tm.get('power_cap_frac', float('nan')):.2f}"
    )


def phase1(cases, my_gemms, args, sampler, deadline, log, records, dump):
    """Per GEMM: compile + numerics for every candidate, then time them all
    back-to-back (upstream first and last) so every timing sees the same
    sustained thermal/power state instead of following an idle compile."""
    import torch
    from dist_moe.kernels import config as dm_config

    rank = int(os.environ.get("RANK", "0"))
    skip = set(filter(None, (args.skip or "").split(",")))
    compile_cost = [args.est_compile_s]
    t_cost = _timing_cost_s(args)
    remaining = list(my_gemms)
    for gemm in my_gemms:
        remaining.remove(gemm)
        case = cases[gemm]
        gen = torch.Generator(device="cuda").manual_seed(4321 + GEMM_ORDER.index(gemm))
        case.fill_inputs(gen)
        default_cfg = mt.normalized(dm_config.GROUPED_GEMM_CONFIGS[case.default_name])
        default_label = "upstream:" + case.default_name
        case.sentinel()
        case.launch(case.default_name)
        torch.cuda.synchronize()
        case.check_coverage()
        ref = case.snapshot()
        # Time later GEMMs on this rank need at least for their upstream runs.
        reserve_later = len(remaining) * (2 * t_cost + args.heat_seconds + 5)
        ready = []
        for cfg, why in CANDIDATES[gemm]:
            cfg = mt.normalized(cfg)
            if cfg == default_cfg:
                continue
            label = mt.config_label(cfg)
            rec = _record_base(case, label, mt.registered_name(cfg), cfg, why, 1, rank)
            if label in skip:
                rec["skipped"] = "--skip"
                records.append(rec)
                continue
            est_c = sum(compile_cost) / len(compile_cost)
            need = est_c + (len(ready) + 3) * t_cost + args.heat_seconds + reserve_later
            if time.time() + need > deadline:
                rec["skipped"] = f"budget (needs ~{need:.0f}s)"
                records.append(rec)
                log(f"{gemm} {label}: skipped, budget")
                continue
            try:
                name = mt.register_config(cfg, gemm)
                case.sentinel()
                t0 = time.time()
                case.launch(name)
                torch.cuda.synchronize()
                rec["first_call_s"] = time.time() - t0  # JIT compile + one launch
                compile_cost.append(rec["first_call_s"])
                rec["numerics"] = case.compare(ref, args.ulp_tol)
                if rec["numerics"]["status"] == "FAIL":
                    raise RuntimeError(f"{gemm} {label}: numerics {rec['numerics']}")
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                records.append(rec)
                dump()
                if not args.keep_going:
                    raise
                torch.cuda.synchronize()  # sticky CUDA errors re-raise here
                log(f"{gemm} {label}: ERROR {rec['error']} (keep-going)")
                continue
            log(
                f"{gemm} {label}: compiled in {rec['first_call_s']:.1f}s, "
                f"numerics {rec['numerics']['status']} "
                f"(max_ulp={rec['numerics']['max_ulp']}, n_diff={rec['numerics']['n_diff']})"
            )
            ready.append(rec)
        del ref
        torch.cuda.empty_cache()
        # Timing block: upstream, candidates, upstream again (drift check).
        _heat(case, case.default_name, args.heat_seconds)
        up = _record_base(
            case,
            default_label,
            case.default_name,
            default_cfg,
            "upstream (reference)",
            1,
            rank,
        )
        up["numerics"] = {"status": "reference"}
        up["timing"] = timing(case, case.default_name, args, sampler)
        records.append(up)
        _log_timing(log, f"{gemm} {default_label}", up)
        for rec in ready:
            try:
                rec["timing"] = timing(case, rec["name"], args, sampler)
            except Exception as exc:
                rec["error"] = f"timing: {type(exc).__name__}: {exc}"
                records.append(rec)
                dump()
                if not args.keep_going:
                    raise
                torch.cuda.synchronize()
                continue
            records.append(rec)
            dump()
            _log_timing(log, f"{gemm} {rec['label']}", rec)
        rep = _record_base(
            case,
            default_label + ":repeat",
            case.default_name,
            default_cfg,
            "upstream (drift check)",
            1,
            rank,
        )
        rep["timing"] = timing(case, case.default_name, args, sampler)
        records.append(rep)
        dump()
        _log_timing(log, f"{gemm} {default_label}:repeat", rep)


def _pf(rec):
    t = rec.get("timing") or {}
    return t.get("pflops_mean_ranks", t.get("pflops"))


def _phase1_ranking(records, gemm):
    ok = [
        r
        for r in records
        if r["gemm"] == gemm
        and r["phase"] == 1
        and not r["label"].startswith("upstream")
        and r.get("timing")
        and (r.get("numerics") or {}).get("status") in ("bitwise", "bf16-rounding")
    ]
    return sorted(ok, key=_pf, reverse=True)


def phase2(cases, all1, args, sampler, deadline, log, records):
    """Lockstep confirmation: every rank runs the same GEMM/config at once."""
    import torch
    import torch.distributed as dist

    rank, world = dist.get_rank(), dist.get_world_size()
    t_cost = _timing_cost_s(args)
    plan = []
    for gemm in GEMM_ORDER:
        top = _phase1_ranking(all1, gemm)[: args.confirm_top]
        if top:
            plan.append((gemm, top))
    # Compile everything first (each rank lacks the candidates of GEMMs it did
    # not sweep), so the lockstep timings below run back-to-back.
    for gemm, top in plan:
        for r in top:
            mt.register_config(r["config"], gemm)
            cases[gemm].launch(r["name"])
    torch.cuda.synchronize()
    dist.barrier()
    log(f"phase 2: compiled {sum(len(t) for _, t in plan)} configs")
    for gemm, top in plan:
        case = cases[gemm]
        seq = [("upstream:" + case.default_name, case.default_name, None)]
        seq += [(r["label"], r["name"], r["config"]) for r in top]
        seq += [("upstream:" + case.default_name + ":repeat", case.default_name, None)]
        go = [time.time() + len(seq) * t_cost + args.heat_seconds + 5 < deadline]
        dist.broadcast_object_list(go, src=0)
        if not go[0]:
            log(f"phase 2: budget exhausted before {gemm}")
            return
        gen = torch.Generator(device="cuda").manual_seed(
            8765 + GEMM_ORDER.index(gemm) + rank
        )
        case.fill_inputs(gen)
        torch.cuda.synchronize()
        dist.barrier()
        _heat(case, case.default_name, args.heat_seconds)
        for label, name, cfg in seq:
            dist.barrier()
            tm = timing(case, name, args, sampler)
            per_rank = [None] * world
            dist.all_gather_object(per_rank, tm)
            pfs = [t["pflops"] for t in per_rank]
            agg = {
                "per_rank": per_rank,
                "pflops_mean_ranks": sum(pfs) / len(pfs),
                "pflops_min_ranks": min(pfs),
                "pflops_max_ranks": max(pfs),
            }
            clk = [
                t.get("sm_clock_mhz_mean")
                for t in per_rank
                if t.get("sm_clock_mhz_mean")
            ]
            if clk:
                agg["sm_clock_mhz_mean_ranks"] = sum(clk) / len(clk)
            pw = [t.get("power_w_mean") for t in per_rank if t.get("power_w_mean")]
            if pw:
                agg["power_w_mean_ranks"] = sum(pw) / len(pw)
            base_cfg = cfg if cfg is not None else None
            rec = {
                "phase": 2,
                "gemm": gemm,
                "label": label,
                "name": name,
                "config": base_cfg,
                "timing": agg,
            }
            records.append(rec)
            log(
                f"phase 2 {gemm} {label}: mean {agg['pflops_mean_ranks']:.3f} PF/s "
                f"(min {agg['pflops_min_ranks']:.3f}) "
                f"{agg.get('sm_clock_mhz_mean_ranks', float('nan')):.0f} MHz"
            )


def choose_best(records, args):
    best, spec = {}, {}
    for gemm in GEMM_ORDER:
        p2 = [r for r in records if r["phase"] == 2 and r["gemm"] == gemm]
        p2_cands = [r for r in p2 if not r["label"].startswith("upstream")]
        if p2_cands and any(r["label"].startswith("upstream") for r in p2):
            source = "phase2_lockstep"
            ups = [_pf(r) for r in p2 if r["label"].startswith("upstream")]
            cands = sorted(p2_cands, key=_pf, reverse=True)
        else:
            source = "phase1_split"
            ups = [
                _pf(r)
                for r in records
                if r["phase"] == 1
                and r["gemm"] == gemm
                and r["label"].startswith("upstream")
                and r.get("timing")
            ]
            cands = _phase1_ranking(records, gemm)
        if not ups:
            continue
        up_pf = sum(ups) / len(ups)
        entry = {
            "source": source,
            "upstream_pflops": up_pf,
            "upstream_spread_pct": 100 * (max(ups) - min(ups)) / up_pf,
        }
        default_name = next(
            r["name"]
            for r in records
            if r["gemm"] == gemm and r["label"].startswith("upstream")
        )
        entry["upstream"] = default_name
        if cands:
            top = cands[0]
            gain = 100 * (_pf(top) / up_pf - 1)
            num = next(
                (
                    r.get("numerics")
                    for r in records
                    if r["phase"] == 1
                    and r["gemm"] == gemm
                    and r["label"] == top["label"]
                ),
                None,
            )
            entry.update(
                label=top["label"],
                name=top["name"],
                config=top["config"],
                pflops=_pf(top),
                gain_pct=gain,
                numerics=num,
                accepted=gain >= args.min_gain_pct,
            )
            if entry["accepted"]:
                spec[gemm] = {
                    "config": top["config"],
                    "expect": default_name,
                    "label": top["label"],
                    "note": f"+{gain:.1f}% vs {default_name} ({source})",
                }
        else:
            entry["accepted"] = False
        best[gemm] = entry
    return best, spec


def verify_spec(spec, dist_moe, context, keep, log):
    """End-to-end check of moe_tiles.apply(spec): one real Dist-MoE forward +
    backward (incl. recompute and wgrad) vs upstream, bitwise or within 2x the
    upstream run-to-run noise, and every spec GEMM must have been overridden."""
    import torch
    import torch.distributed as dist

    x, ids, scores, w13, w2, _ = keep
    rank = dist.get_rank()
    g = torch.Generator(device="cuda").manual_seed(777 + rank)
    dy = torch.randn(x.shape, generator=g, device="cuda", dtype=torch.float32)

    def step():
        context.reset()
        for p in (x, w13, w2):
            p.grad = None
        y = dist_moe.routed_experts(x, ids, scores, w13, w2, context)
        (y.float() * dy).sum().backward()
        torch.cuda.synchronize()
        return {
            "y": y.detach().clone(),
            "grad_x": x.grad.clone(),
            "grad_w13": w13.grad.clone(),
            "grad_w2": w2.grad.clone(),
        }

    ref1 = step()
    ref2 = step()
    mt.apply(mt.load_spec(json.dumps(spec)))
    try:
        got = step()
        used = sorted({gemm for gemm, _ in mt._state["logged"]})
    finally:
        mt.remove()
    out = {"tensors": {}, "overridden": used}
    fail = sorted(set(spec) - set(used))
    if fail:
        out["not_overridden"] = fail
    for k, ref in ref1.items():
        noise = (ref2[k].float() - ref.float()).abs().max().item()
        diff = (got[k].float() - ref.float()).abs().max().item()
        bitwise = torch.equal(
            got[k].view(torch.int16) if got[k].dtype == torch.bfloat16 else got[k],
            ref.view(torch.int16) if ref.dtype == torch.bfloat16 else ref,
        )
        ok = bitwise or (math.isfinite(diff) and diff <= 2 * noise)
        out["tensors"][k] = {
            "bitwise": bitwise,
            "max_abs_diff": diff,
            "ref_noise": noise,
            "ok": ok,
        }
        if not ok:
            fail.append(k)
    flag = torch.tensor([1 if fail else 0], device="cuda")
    dist.all_reduce(flag)
    out["status"] = "PASS" if flag.item() == 0 else "FAIL"
    diffs = {
        k: ("bitwise" if v["bitwise"] else v["max_abs_diff"])
        for k, v in out["tensors"].items()
    }
    log(f"verify_spec: {out['status']} overridden={used} {diffs}")
    return out


def run(args):
    import torch
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    log = Log(rank)
    records = []
    partial = f"{args.out}.rank{rank}.partial.json"

    def dump():
        with open(partial, "w") as f:
            json.dump(records, f, indent=1, default=str)

    deadline_hard = _T_START + args.budget_s
    try:
        import dist_moe
        from dist_moe._buffers import SymmetricMemoryBuffer
        from dist_moe.kernels import dist_grouped_gemm as dgg, grouped_gemm as gg

        props = torch.cuda.get_device_properties(dev)
        sampler = NvmlSampler(local)
        sampler.start()
        log(
            f"{props.name} SMs={props.multi_processor_count} world={world} "
            f"nvml={'ok' if sampler.ok else sampler.err} power_limit={sampler.power_limit_w} W"
        )
        config = dist_moe.Config(
            max_num_local_input_tokens=args.tokens,
            hidden_dim=HIDDEN,
            intermediate_dim=INTER,
            top_k=TOP_K,
            num_experts=LOCAL_EXPERTS * world,
            max_moe_layers_per_activation_slot=1,
            device_scratch_capacity_factor=2.0,
            activation="swiglu_clamped",
            swiglu_alpha=SWIGLU_ALPHA,
            swiglu_limit=SWIGLU_LIMIT,
        )
        context = dist_moe.create_context(group=dist.group.WORLD, config=config)
        captured, keepalive = _capture(dist_moe, dgg, gg, context, args, log)
        for gemm, (_, kw) in captured.items():
            if kw["config"] != EXPECTED_DEFAULT[gemm]:
                log(
                    f"WARNING {gemm}: upstream picked {kw['config']}, "
                    f"expected {EXPECTED_DEFAULT[gemm]} (thresholds changed?)"
                )

        # Private sinks for the two peer-scatter GEMMs; fill the real dispatch
        # (FC13 fprop A) and combine (FC2 dgrad dy) buffers once with randn.
        sinks = {}
        for gemm in ("fc2_fprop", "fc13_dgrad"):
            src = captured[gemm][1]["symm_mem_buffer"]
            sinks[gemm] = SymmetricMemoryBuffer.create(
                tuple(src.shape), src.dtype, dist.group.WORLD, dev
            )
        gen = torch.Generator(device="cuda").manual_seed(99 + rank)
        captured["fc13_fprop"][1]["symm_mem_buffer"].local().normal_(generator=gen)
        captured["fc2_dgrad"][1]["symm_mem_buffer"].local().normal_(generator=gen)
        torch.cuda.synchronize()
        dist.barrier()
        cases = {}
        for gemm, (fn, kw) in captured.items():
            kw = dict(kw)
            sink = sinks.get(gemm)
            gm = int(kw["num_tokens_per_local_expert"].sum().item())
            if sink is not None:
                kw["scatter_ptrs"] = _remap_ptrs(
                    kw["scatter_ptrs"], kw["symm_mem_buffer"], sink, gm
                )
            cases[gemm] = Case(gemm, fn, kw, sink, world)
            log(
                f"{gemm}: rows={gm} TFLOP={cases[gemm].flops / 1e12:.2f} upstream={kw['config']}"
            )
        torch.cuda.synchronize()
        dist.barrier()

        do_confirm = world > 1 and args.confirm_top > 0
        reserve = args.confirm_reserve_s if do_confirm else 0.0
        my_gemms = [
            g for g in GEMM_ORDER[rank::world] if not args.only or g in args.only
        ]
        log(
            f"phase 1: {my_gemms}, deadline in {deadline_hard - reserve - time.time():.0f}s"
        )
        phase1(
            cases, my_gemms, args, sampler, deadline_hard - reserve, log, records, dump
        )
        torch.cuda.synchronize()
        dist.barrier()
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        all1 = [r for part in gathered for r in part]
        p2 = []
        if do_confirm:
            log("phase 2 (lockstep)")
            phase2(cases, all1, args, sampler, deadline_hard, log, p2)
        allrec = all1 + p2
        best, spec = choose_best(allrec, args) if rank == 0 else (None, None)
        obj = [spec]
        dist.broadcast_object_list(obj, src=0)
        spec = obj[0]
        verify = None
        if args.verify_spec and spec:
            go = [deadline_hard - time.time() > args.verify_min_s]
            dist.broadcast_object_list(go, src=0)
            if go[0]:
                verify = verify_spec(spec, dist_moe, context, keepalive, log)
            else:
                verify = {"status": "skipped (budget)"}
                log("verify_spec: skipped, budget")
        if rank == 0:
            meta = {
                "shapes": {
                    "tokens_per_rank": args.tokens,
                    "hidden": HIDDEN,
                    "inter": INTER,
                    "top_k": TOP_K,
                    "local_experts": LOCAL_EXPERTS,
                    "ep": world,
                    "rows_per_rank": cases["fc13_fprop"].gm,
                },
                "gpu": props.name,
                "sms": props.multi_processor_count,
                "driver": sampler.driver,
                "power_limit_w": sampler.power_limit_w,
                "torch": torch.__version__,
                "dist_moe": os.path.dirname(dist_moe.__file__),
                "args": vars(args),
                "elapsed_s": time.time() - _T_START,
            }
            try:
                import cutlass

                meta["cutlass_dsl"] = getattr(cutlass, "__version__", None)
            except Exception:
                pass
            spec_doc = dict(spec)
            spec_doc["_meta"] = {
                "source": os.path.abspath(args.out),
                "min_gain_pct": args.min_gain_pct,
                "verify_spec": None if verify is None else verify["status"],
            }
            meta["verify_spec"] = verify
            meta["elapsed_s"] = time.time() - _T_START
            with open(args.out, "w") as f:
                json.dump(
                    {
                        "meta": meta,
                        "best": best,
                        "moe_tiles_spec": spec_doc,
                        "records": allrec,
                    },
                    f,
                    indent=1,
                    default=str,
                )
            _print_summary(allrec, best, log)
            if verify is not None and verify["status"] == "FAIL":
                log(f"verify_spec FAILED: spec NOT written; see {args.out}")
            else:
                with open(args.spec_out, "w") as f:
                    json.dump(spec_doc, f, indent=1)
                log(f"wrote {args.spec_out} ({len(spec)} GEMM override(s))")
            log(f"wrote {args.out}; elapsed {time.time() - _T_START:.0f}s")
        if verify is not None and verify["status"] == "FAIL":
            raise RuntimeError("verify_spec failed")
        sampler.stop()
        del keepalive
        torch.cuda.synchronize()
        dist.barrier()
        context.close()
        dist.destroy_process_group()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        try:
            dump()
        except Exception:
            pass
        log(f"FAILED; partial results in {partial}")
        sys.stdout.flush()
        os._exit(1)


def _print_summary(records, best, log):
    for gemm in GEMM_ORDER:
        rows = [r for r in records if r["gemm"] == gemm and r.get("timing")]
        if not rows:
            continue
        log(f"--- {gemm} ---")
        for r in sorted(rows, key=lambda rec: (rec["phase"], -(_pf(rec) or 0))):
            t = r["timing"]
            num = (r.get("numerics") or {}).get("status", "")
            clk = t.get("sm_clock_mhz_mean", t.get("sm_clock_mhz_mean_ranks"))
            pw = t.get("power_w_mean", t.get("power_w_mean_ranks"))
            log(
                f"  p{r['phase']} {r['label']:44s} {_pf(r):6.3f} PF/s "
                f"{clk if clk is None else round(clk)} MHz {pw if pw is None else round(pw)} W {num}"
            )
        b = best.get(gemm, {})
        if b.get("label"):
            log(
                f"  best {b['label']} {b['gain_pct']:+.2f}% vs upstream ({b['source']}) "
                f"-> {'USE' if b['accepted'] else 'keep upstream'}"
            )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--list", action="store_true", help="print candidates and exit (no GPU)"
    )
    p.add_argument("--out", default="tile_sweep.json")
    p.add_argument(
        "--spec-out",
        default=None,
        help="EXP_MOE_TILES spec (default <out>.moe_tiles.json)",
    )
    p.add_argument("--tokens", type=int, default=TOKENS, help="tokens per rank")
    p.add_argument(
        "--seconds", type=float, default=2.0, help="sustained timing window per config"
    )
    p.add_argument("--warmup-seconds", type=float, default=0.5)
    p.add_argument(
        "--budget-s", type=float, default=540.0, help="wall budget from start"
    )
    p.add_argument(
        "--confirm-top", type=int, default=1, help="phase-2 candidates per GEMM"
    )
    p.add_argument(
        "--confirm-reserve-s",
        type=float,
        default=75.0,
        help="budget held back from phase 1 for phase 2",
    )
    p.add_argument(
        "--heat-seconds",
        type=float,
        default=1.5,
        help="unmeasured load before each back-to-back timing block",
    )
    p.add_argument(
        "--est-compile-s",
        type=float,
        default=10.0,
        help="initial guess of one CuTe JIT compile (then measured)",
    )
    p.add_argument(
        "--ulp-tol", type=int, default=2, help="max bf16 ULP for 'bf16-rounding'"
    )
    p.add_argument("--min-gain-pct", type=float, default=1.0)
    p.add_argument("--only", default=None, help="comma list of GEMMs")
    p.add_argument("--skip", default=None, help="comma list of candidate labels")
    p.add_argument(
        "--keep-going", action="store_true", help="record candidate errors, continue"
    )
    p.add_argument(
        "--verify-spec",
        action="store_true",
        help="end-to-end Dist-MoE fwd+bwd with moe_tiles.apply(spec) vs upstream",
    )
    p.add_argument(
        "--verify-min-s",
        type=float,
        default=90.0,
        help="budget left required to run --verify-spec",
    )
    args = p.parse_args()
    if args.list:
        raise SystemExit(1 if list_candidates() else 0)
    if args.only:
        args.only = [g.strip() for g in args.only.split(",") if g.strip()]
        unknown = set(args.only) - set(GEMM_ORDER)
        if unknown:
            raise SystemExit(f"--only: unknown {sorted(unknown)}")
    if args.spec_out is None:
        stem = args.out[:-5] if args.out.endswith(".json") else args.out
        args.spec_out = stem + ".moe_tiles.json"
    bad = sum(
        bool(mt.check_config(c, g, n_out=N_OUT[g]))
        for g in GEMM_ORDER
        for c, _ in CANDIDATES[g]
    )
    if bad:
        raise SystemExit(f"{bad} invalid candidates; run --list")
    run(args)


if __name__ == "__main__":
    main()
