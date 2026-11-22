# h3 host offload: skip the BF16 backward FC2 recompute

| | |
|---|---|
| Enable | `EXP_MOE_H3_OFFLOAD=1` on every rank (or `h3_offload.apply()`) |
| Import order | after `swiglu_recompute` when both are used |
| Device memory | +1 GiB (one staging buffer at the shapes below); `EXP_MOE_H3_STAGE_BUFFERS=0` gives +0 but risks a forward stall |
| Host memory | one microbatch of h3 in pinned memory per GPU (+62 GiB at the shapes below) |
| Numerics | no GEMM changes; bitwise identical when the combine GEMM is deterministic |
| Review dump | `EXP_MOE_H3_OFFLOAD_DUMP=<dir>` writes the two patched functions |

Line numbers refer to dist_moe at commit `b76e0e0`. Sizes and timings are for
Muse Spark 816B BF16 at 64 GPUs (EP64, 16384 tokens per rank, top-8,
D = 4096), with gradient accumulation over 2 microbatches and no PP.

## Problem

For every layer whose activations were not saved (124 per step at the shapes
above), the Dist-MoE BF16 backward reruns two GEMMs:

1. The FC13 dispatch GEMM, to rebuild `x_gathered` and `h1` for the W13 wgrad
   and the SwiGLU backward.
2. The FC2 combine GEMM, to rebuild the route output `h3` (api.py:2428):
   about 141 ms per step in the trace.

`h3` has one consumer: the output postprocess backward (api.py:2458), which
computes `grad_topk_scores` and replays the expert-output norm callback
(`_resolve_callback.backward`, _postprocess.py:1091). It feeds no expert GEMM.
The second GEMM exists only to rebuild 1 GiB that the forward already computed.

## Mechanism

`h3` is a view of the rank's own combine symmetric buffer, at a fixed address
and size (`[T*K, D]` = 16384 * 8 * 4096 * 2 B = 1 GiB). The upstream recompute
writes `h3` back into that same view. The patch saves the forward `h3` to pinned
host memory with copy engines and copies it back into the same view in backward,
instead of rerunning the GEMM. FC13 recompute, the activation planner and the
activation slot are unchanged.

```
forward,  layer i:  combine GEMM -> h3 (combine buffer) -> postprocess
                                     |  D2D (copy engine), frees the combine buffer
                                     v
                                  staging (1 GiB HBM)  --D2H over a whole layer-->  pinned host[i]

backward, layer i:  pinned host[i] --H2D, queued at the end of layer i+1's backward--> combine buffer view
                    (no combine GEMM) -> postprocess backward -> dgrad / wgrad as before
```

The patch inserts 5 lines (exact-match anchors, each tagged `# CHANGED`) into
`_Bf16Autograd.forward` (2) and `_Bf16Autograd._backward_impl` (3):

| Hook | Where | What it does |
|---|---|---|
| `before_call` | start of forward and backward | compute stream waits for the last copy that reads the combine buffer; counts calls |
| `offload` | forward, after the postprocess | D2D combine buffer -> staging, then D2H staging -> pinned host; pushes (key, host buffer) |
| `restore` | backward, replaces the combine GEMM | pops the entry and checks its key; waits for the H2D prefetched into the combine view; returns that view |
| `after_backward` | backward, after the final dispatch barrier | starts the H2D of the next entry into the combine view |

The upstream combine GEMM still runs when the patch is inactive, and only then.

## Race freedom

**Forward.** Peers write our combine buffer only in their combine kernel of our
next Dist-MoE call. That kernel runs after the call's first EP barrier, which
needs our arrival, and our compute stream waits for the D2D at the start of that
call (`before_call`).

**Backward.** No rank runs the combine GEMM any more, so no peer writes our
combine buffer in backward. Peers read it (for `grad_h3`) in their dgrad
dispatch, which finishes before they arrive at the final
`_ep_barrier(buffers.dispatch)` (api.py:2627). The prefetch H2D for the next
layer is queued after that barrier. Our own gradient publish happens after
`restore` has waited.

**Special cases.**

- **Last layer before the loss.** No Dist-MoE call ran between its forward and
  its backward, so `h3` is still in the combine buffer and is used in place.
- **Saved layers** (`need_recompute = 0`). They are offloaded too, because the
  predicate lives on the device and the host cannot see it. The existing
  conditional copy then overwrites the restored `h3` with the identical saved
  one. Harmless; it wastes a few GiB of copies per microbatch.
- **Pairing.** Forwards and backwards are matched as a LIFO stack per context,
  keyed by `need_recompute.data_ptr()` (a tensor autograd keeps alive until
  backward). A mismatch, for example under PP 1F1B, raises. All ranks must use
  the same setting: a rank that fell back to the GEMM would scatter into its
  peers' restored `h3`.
- **Fake process groups** keep the upstream recompute: FakePG zero-fills the
  local combine buffer at every recompute.
- **CUDA graphs.** Every side stream is joined back into the compute stream
  within the microbatch (D2H at `after_backward`, H2D at `restore`, D2D through
  D2H). Pinned buffers are reused through event waits, with the same
  eager-versus-captured event rule as torchtitan's `_PinnedBufferPool`. The
  hooks only run Python at capture time, so the schedule is fixed.

## Copy deadlines

**Forward D2H.** The activation offloader already uses the D2H link. Between the
end of `combine_fprop` and the next layer's dispatch there are 5.9 ms (median),
4.98 ms at minimum, with the link already 66-72% busy. A direct 1 GiB D2H would
finish about 0.9 ms late, about 0.11 s/step of stall, which would cancel the
gain. Staging fixes it: a copy-engine D2D into one staging buffer frees the
combine buffer in well under 1 ms, and the D2H then has a whole layer (11.3 ms)
to finish; the link work per layer is about 7.6 ms. If traces show forward
stalls, set `EXP_MOE_H3_STAGE_BUFFERS=2` (+1 GiB).

**Backward H2D.** About 3.4 ms per layer against a window of 19.7 ms (median),
15.2 ms at minimum, from the end of the previous Dist-MoE backward to the point
of use.

Forward is D2H-only and backward is H2D-only (gradient accumulation, no PP), so
the two directions never share the link.

## Memory

| | Added |
|---|---|
| Device | +1 GiB staging per staging buffer (persistent). Combine buffer and activation slot unchanged |
| Host, per GPU | one microbatch of h3 (62 layers x 1 GiB = 62 GiB); buffers are reused across microbatches and steps |

Check pinnable host memory (`/proc/meminfo`, `ulimit -l` inside the container)
before enabling, and device headroom for the staging buffer. With little HBM
headroom, `EXP_MOE_H3_STAGE_BUFFERS=0` removes the staging buffer at the cost of
the forward stall risk above.

## Numerics

No GEMM, dtype or reduction-order change. The restored `h3` is exactly the bytes
the forward produced and the forward postprocess consumed, including padded rows
the callback path had already zeroed in place. With a deterministic combine GEMM
(the upstream recompute is then bitwise equal to the forward), loss, grad_norm
and all gradients are bitwise identical to `EXP_MOE_H3_OFFLOAD=0`; with a
non-deterministic one, the patch is closer to exact, since it reuses the
forward's values.

## Expected and measured savings

Expected: the backward `combine_fprop` removal saves 0.141 s/step on rank 0
(0.166 s scaled to the slowest rank), with no new SM work (copy engines only)
and up to 0.02 s of background HBM traffic: about 0.14 s/step net
(0.12-0.16). Adds to `swiglu_recompute`; the two touch independent kernels.

Measured: this patch, `swiglu_recompute` and two norm fusions together gave
+5.0% end-to-end throughput on the 64-GPU benchmark recipe, with loss and
grad_norm within run-to-run noise. The offload engaged with 0 synchronous
restores.

## Diagnostics and failure modes

- Rank 0 prints `[h3_offload] first microbatch drained: Stats(...)`. For one
  microbatch at the shapes above expect `offloads=62, restore_resident=1,
  restore_prefetched=61, restore_sync=0`. `restore_sync > 0` means an exposed
  H2D; `combine_fprop` kernels in the backward mean the patch did not engage.
- Pinned allocation failure or host OOM at the first microbatch; HBM OOM from
  the staging buffer.
- Loud errors by design: anchor drift, wrong import order relative to
  `swiglu_recompute`, non-LIFO forward/backward (PP), more than
  `EXP_MOE_H3_MAX_LIVE` (default 512) live forwards.
- Silent corruption if ranks disagree on `EXP_MOE_H3_OFFLOAD`: the launcher must
  set it for every rank.

## Tests

CPU anchor check (skips without dist_moe):

```bash
pytest torchtitan/experiments/dist_moe_patches/tests/test_h3_offload_anchors.py
```

GPU parity, one node with 4 GPUs, from the repository root:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
    torchtitan/experiments/dist_moe_patches/tests/test_h3_offload.py
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
    torchtitan/experiments/dist_moe_patches/tests/test_h3_offload.py --with-swiglu-recompute
```

The test runs a shrunken EP4 Dist-MoE (3 layers sharing one context, 2
microbatches, 2 steps, learned router and a learned RMSNorm callback) for all
three activation-slot regimes (all recompute, mixed, all saved) and both
postprocess paths, with staged and direct copies, and with about 10 ms of
injected copy delay to expose any missing stream wait. Every output and gradient
must match unpatched dist_moe within 2x its own run-to-run difference (bitwise
when deterministic), the combine GEMM must run 0 times in the patched backward,
and the copy counters must show 0 synchronous restores. The stacked mode
applies `swiglu_recompute` first and prints SKIP when that module is not in the
checkout.
