# SwiGLU fused into the BF16 backward recompute

Enable with `EXP_SWIGLU_RECOMPUTE=1` (or call `swiglu_recompute.apply()`).
When unset, importing `swiglu_recompute` does nothing. To dump the patched
function for review, set `EXP_SWIGLU_RECOMPUTE_DUMP=/path/out.py`.

Line numbers refer to dist_moe at commit `b76e0e0`.

## Problem

For every layer whose activations were not saved, the Dist-MoE BF16 backward
recomputes FC13 to rebuild H1, then rebuilds H2 = SwiGLU(H1) with a standalone
Triton `swiglu_fwd`. The forward already produces H1 and H2 from one kernel: its
FC13 dispatch runs the native emit-SwiGLU epilogue (`swiglu_h2=` in
`_Bf16Autograd.forward`). The backward recompute does not use it, so it pays
for one extra elementwise pass over H1 and H2 per recomputed layer.

## What is patched

One function: `dist_moe.api._Bf16Autograd._backward_impl` (api.py:2242-2656).
This is the BF16 autograd only. MXFP8 and NVFP4 use
`_BlockScaledAutograd._backward_impl`, which is untouched.

The module does not carry a copy of the function body. It reads the upstream
source with `inspect.getsource`, applies 4 exact-text edits, compiles the result
against `dist_moe.api.__dict__`, and reassigns the staticmethod. Each anchor must
match exactly once, otherwise the module raises and patches nothing. All callers
go through the class attribute, so both the autograd backward (api.py:2240) and
the custom-op backward (api.py:3396) pick up the patched version.

The edits only add lines (12 in total, each tagged `# CHANGED`):

1. Import `_native_swiglu_fwd_config_is_compatible`, the predicate the forward
   uses (api.py:1977).
2. `fused_swiglu_recompute = activation in (swiglu, swiglu_clamped) and
   _native_swiglu_fwd_config_is_compatible(fprop_config, F)`. This is the
   forward's condition (api.py:2086-2091) without the clip-stats term; the
   backward never collects clip counters. `fprop_config` is built with the same
   arguments as in the forward (api.py:2363 vs 2077).
3. The recompute `dist_grouped_gemm_fprop_dispatch` (api.py:2396-2411) gets
   `swiglu_h2=backward_plan.h2_offset` plus the clamped/alpha/limit arguments,
   exactly as the forward passes them (api.py:2124-2127). It is still gated on
   `conditional_execution=need_recompute`.
4. The standalone `swiglu_fwd` (api.py:2414) gets
   `conditional_execution=torch.logical_not(need_recompute)`.

```
                     need_recompute = 1 (recomputed layer)   need_recompute = 0 (saved layer)
before:  FC13 dispatch   writes H1                            skipped on device
         swiglu_fwd      H1 -> H2                             H1 (saved) -> H2
after:   FC13 dispatch   writes H1 and H2 (emit epilogue)     skipped on device
         swiglu_fwd      skipped on device                    H1 (saved) -> H2
```

### Why `swiglu_fwd` stays

The recompute dispatch is skipped on the device when `need_recompute == 0`, for
the layers whose H1 was saved. H2 is always transient scratch, so for those
layers it must still be rebuilt from the saved H1. With the patch, exactly one
of the two kernels does real work on any layer, and both are still launched, so
the CUDA-graph launch topology is unchanged.

### The backward still has H1

The SwiGLU backward reads H1 (`dist_grouped_gemm_dgrad_swiglu_bwd_combine(h1=...)`
at api.py:2540-2555, and the fallback `swiglu_bwd` at api.py:2557). The emit
kernel writes both outputs: H1 through the normal C TMA store to `y`
(`backward_plan.h1_offset`) and H2 to `swiglu_h2`. The forward relies on the
same behavior (api.py:2129-2131).

## Race freedom and buffers

H1 and H2 use the backward-plan offsets the recompute already uses, so no new
buffer space is needed. In recompute mode the planner places H2 at
`scratch_end - h2_size` and H1 and x_gathered below the gradient scratch; these
regions are disjoint (_activation_buffer_planner_kernel.py:1389-1406), so one
kernel writing H1, H2 and x_gathered is safe. All offsets are 128-byte aligned;
the emit kernel needs 16-byte alignment. The patch adds no streams and no
cross-rank traffic: the two kernels run in stream order, as before.

## Numerics

Bitwise identical to the unpatched backward. The emit epilogue is documented as
bitwise identical to plain H1 + `swiglu_fwd` (swiglu_fragment.py:16-20),
`fast_math` is False on both paths, and H1 comes from the same GEMM. The
recomputed H2 now matches the forward's H2 by construction. Verified bitwise as
part of the 64-GPU run below.

## Memory

No new persistent memory. One 1-element bool per backward from
`torch.logical_not`, with no host sync; under capture it comes from the graph
pool. The existing launcher already makes the same kind of allocation for this
call (`need_recompute.to(torch.int32)`).

## Expected and measured savings

From a 64-GPU Muse Spark 816B BF16 trace: the patch removes about 120 of the
124 `_triton_swiglu_fwd` launches per step, about 58 of 60 ms per step. The
fused dispatch costs the same as the unfused one (forward: 304 vs 305 ms per
step). The remaining saved-H1 layers still run `swiglu_fwd`, and each backward
adds a `logical_not` launch of about 2 us.

Measured: this patch, `h3_offload` and two norm fusions together gave +5.0%
end-to-end throughput on the 64-GPU benchmark recipe, with loss and grad_norm
within run-to-run noise.

## Risks

- **New JIT variants.** The emit-SwiGLU dispatch with `conditional_execution`
  and `swiglu_fwd` with a condition are new compile keys. They compile on the
  first backward, so warmup must happen before graph capture, as it already does
  for the existing conditional variants.
- **Upstream drift.** If `_backward_impl` changes upstream, the anchors fail at
  import. The patch never applies to drifted code.
- **Stale upstream comment.** "Rebuild it unconditionally" (api.py:2412-2413) no
  longer describes the patched behavior.

## Tests

`tests/test_swiglu_recompute_anchors.py` checks the anchors and that the patched
source compiles, on CPU. On GPU, the stacked mode of the `h3_offload` test
(`--with-swiglu-recompute`) runs this patch under `h3_offload` and checks that
the two compose (its reference also has this patch applied). Parity of this
patch against unpatched dist_moe was verified as part of the 64-GPU run.
