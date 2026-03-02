# TE Grouped GEMM for MoE — Design Document

## Overview

The implementation routes MoE expert computations (`torch._grouped_mm`) through
TransformerEngine's `general_grouped_gemm` CUDA kernel, supporting both MXFP8
quantized and BF16 modes. It is entirely transparent to the model code — the
existing `GroupedExperts` module and `_run_experts_grouped_mm` function work
unchanged.

## Architecture

```
CLI flags                          Model code (unchanged)
──────────                         ──────────────────────
--model.converters=                GroupedExperts.forward()
    quantize.grouped_mm.te            └─ _run_experts_grouped_mm()
--quantize.grouped_mm.te.               └─ torch._grouped_mm(x, w1, offs=...)
    mode=mxfp8                                     │
--quantize.grouped_mm.te.                          ▼
    fqns=experts              TEGroupedMMTensor.__torch_function__
                                           │
                                           ▼
                                  _TEGroupedGEMM.forward()
                                           │
                                  ┌────────┴────────┐
                                  │  Custom Ops      │
                                  │  (torch.library) │
                                  ├──────────────────┤
                                  │ te_moe::gemm_fwd │──► general_grouped_gemm
                                  │ te_moe::gemm_dgrad│──► general_grouped_gemm
                                  │ te_moe::gemm_wgrad│──► general_grouped_gemm
                                  └──────────────────┘
```

## Files

| File | Role |
|---|---|
| `components/quantization/te.py` | Converter registration. Parses config, calls `swap_params_to_te_grouped_mm` at model init time. Registered as `"quantize.grouped_mm.te"`. |
| `models/moe/te_backend.py` | Core implementation. Contains the tensor subclass, autograd function, custom ops, MXFP8 quantization helpers, and the parameter-swapping utility. |
| `models/moe/moe.py` | Unmodified. `GroupedExperts` calls `torch._grouped_mm` as usual; the intercept is invisible to it. |

## Key Components (all in `te_backend.py`)

### 1. `TEGroupedMMTensor` (tensor subclass)

A `torch.Tensor` subclass wrapping MoE weight parameters (3D `[E, out_features, in_features]`).
Its only job is interception:

- **`__torch_function__`**: When `torch._grouped_mm(A, B, offs=...)` is called and `B` is a
  `TEGroupedMMTensor`, unwraps `B` to plain tensor and dispatches to
  `_TEGroupedGEMM.apply(A, B_plain, offs, ...)`.
- **`__torch_dispatch__`**: For all other ATen ops, unwraps to plain tensor, runs the op, and
  re-wraps the result for a whitelist of shape-preserving ops (`slice`, `view`, `clone`,
  `transpose`, etc.).
- **`__tensor_flatten__` / `__tensor_unflatten__`**: Required for `torch.compile` to
  serialize/deserialize the subclass through FX graph boundaries.
- **`fsdp_pre_all_gather` / `fsdp_post_all_gather`**: FSDP2 hooks. Pre-all-gather casts to
  `param_dtype` and unwraps; post-all-gather re-wraps the gathered data into
  `TEGroupedMMTensor`.

### 2. `_TEGroupedGEMM` (autograd.Function)

Standard `torch.autograd.Function` with `setup_context` (required for `torch.compile`
compatibility). Delegates to three custom ops:

| Pass | Custom Op | GEMM Layout | Computes |
|---|---|---|---|
| Forward | `te_moe::gemm_fwd` | TN | `out[i] = X[i] @ W[i]^T` |
| DGRAD | `te_moe::gemm_dgrad` | TN | `dX[i] = dOut[i] @ W[i]` |
| WGRAD | `te_moe::gemm_wgrad` | NT | `dW[i] = dOut[i]^T @ X[i]` |

`setup_context` saves `A`, `B_t`, and `offs` for backward. The backward returns gradients
for `A` (dgrad) and `B_t` (wgrad); `offs`, `out_dtype`, and `use_fp8` get `None`.

### 3. Custom Ops (`torch.library`)

Each op is registered with `@torch.library.custom_op` and has a `register_fake` (meta)
implementation. This is the mechanism that makes `torch.compile` work:

- **Tracing time**: Dynamo calls the fake implementation which returns an empty tensor of the
  correct shape/dtype. No TE C++ code runs.
- **Execution time**: The real implementation calls TE's `general_grouped_gemm`.

### 4. MXFP8 Quantization Helpers

When `use_fp8=True`, inputs/weights/gradients must be quantized to MXFP8 (E8M0 block-scaled
FP8) before the GEMM. MXFP8 requires tensor dimensions divisible by 32 (block size).

| Helper | Purpose |
|---|---|
| `_pad_for_mxfp8` | Pads each expert's token chunk to a multiple of 32 rows. Single buffer allocation + per-expert slice copies. |
| `_unpad_mxfp8_output` | Extracts real rows from padded GEMM output back into the caller's buffer. |
| `_mxfp8_quantize_inputs` | Forward pass: quantizes activations rowwise-only (`rowwise=True, columnwise=False`). |
| `_mxfp8_quantize_weights` | Forward/DGRAD: quantizes weights rowwise-only per expert via `MXFP8Quantizer.quantize_impl()`. Optionally transposes `[K,N]→[N,K]`. |
| `_mxfp8_quantize_wgrad` | WGRAD: quantizes both inputs and gradients with `rowwise=True, columnwise=True`, plus `internal=True` and `optimize_for_gemm=True`. Then calls `update_usage(rowwise_usage=False, columnwise_usage=True)` on the input tensors to present the columnwise view required by NT-layout GEMM. |

The WGRAD quantization is the subtlest part — the NT-layout GEMM (`dW = dOut^T @ X`) requires
both operands to have column-wise data available. The quantizers produce both row+column
scaling, then `update_usage` masks off the rowwise view so TE's kernel sees the columnwise
layout it expects. This was the root cause of the 671B NaN bug: the original code only
produced 1D scaling, which TE silently consumed in release builds.

### 5. `swap_params_to_te_grouped_mm`

Called by the converter at model init time. Walks the model's named modules, matches against
`fqn_filter` (e.g., `["experts"]`), and replaces each `nn.Parameter`'s data with
`TEGroupedMMTensor(param.data, use_fp8=...)`. The model is otherwise untouched.

## Data Flow (MXFP8 forward, single GEMM)

```
_run_experts_grouped_mm:
    torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2,-1), offs=offsets)
                              │
    __torch_function__ intercepts (w1 is TEGroupedMMTensor)
                              │
    _TEGroupedGEMM.apply(A=x, B_t=w1_plain, offs=offsets)
                              │
    te_moe::gemm_fwd(A, B_t, offs, out_dtype=bf16, use_fp8=True)
        │
        ├─ _offs_to_m_splits(offs)     → [n1, n2, ..., nE]   (GPU→CPU sync)
        ├─ A_used = A[:total_tokens]   → slice off HybridEP padding
        ├─ _pad_for_mxfp8(A_used)      → padded_A, padded_splits
        ├─ _mxfp8_quantize_inputs()    → inputmats_fp8  (split_quantize)
        ├─ _mxfp8_quantize_weights()   → weights_fp8    (per-expert quantize_impl)
        ├─ general_grouped_gemm(weights_fp8, inputmats_fp8, ...)
        └─ _unpad_mxfp8_output()       → copy real rows to output
```

## GEMM Layout Details

All GEMMs use TE's `general_grouped_gemm`. The layout string refers to the operand storage:

- **TN (forward, dgrad)**: A is transposed (weight `[N,K]`), B is normal (input `[m,K]`).
  Computes `C = B @ A^T`, i.e., `[m,K] @ [K,N] = [m,N]`.
- **NT (wgrad)**: Computes `C = A^T @ B`, i.e., `[N,m] @ [m,K] = [N,K]`. Requires both
  operands to have column-wise MXFP8 data.

The `grad=True` parameter is passed for DGRAD and WGRAD to signal to TE that these are
backward-pass GEMMs (affects internal scale handling).

## Parallelism Compatibility

The tensor subclass preserves shape, dtype, stride, and storage semantics, so it is fully
transparent to:

- **FSDP2**: Via `fsdp_pre_all_gather` / `fsdp_post_all_gather` hooks. Parameters are
  unwrapped before all-gather and re-wrapped after.
- **Expert Parallel (EP)**: Weights are 3D `[E, out, in]`; EP shards on dim 0.
  `GroupedExperts.forward` calls `self.w1.to_local()` for EP, which passes through
  `__torch_dispatch__` and preserves the subclass.
- **Tensor Parallel (TP)**: w1/w3 column-sharded (Shard(1)), w2 row-sharded (Shard(2)).
  Works because the subclass doesn't alter shapes.
- **Pipeline Parallel (PP)**: No special handling needed; the subclass is just a weight wrapper.
- **HybridEP/DeepEP**: Dispatched buffers may be larger than `sum(tokens_per_expert)`. The
  GEMM ops slice `A[:total_tokens]` to avoid passing padding to TE, and the output is sized
  to match `A.shape[0]` so `combine_tokens` gets the expected row count.

## torch.compile Compatibility

The three-layer design (tensor subclass → autograd.Function → custom ops) is specifically
structured for `torch.compile`:

1. `TEGroupedMMTensor.__tensor_flatten__`/`__tensor_unflatten__` let Dynamo serialize the
   subclass through FX graph captures.
2. `_TEGroupedGEMM` uses `setup_context` (not `ctx` in `forward`) as required by
   `torch.compile`.
3. The custom ops (`te_moe::gemm_fwd/dgrad/wgrad`) have `register_fake` implementations that
   return correctly-shaped empty tensors, so tracing never touches TE's C++ code.

## Known Limitations

1. **GPU→CPU sync per GEMM**: `_offs_to_m_splits` calls `offs.tolist()`, triggering a D2H
   sync. This happens 3x per MoE layer (once per GEMM: w1, w3, w2). It's unavoidable because
   TE's `general_grouped_gemm` requires `m_splits` as `List[int]`.

2. **MXFP8 padding overhead**: When expert token counts aren't multiples of 32, zero-padding
   is applied. This increases the effective GEMM size but is necessary for MXFP8 block scaling.
   HybridEP can be configured to pad token counts at dispatch time
   (`TOKEN_GROUP_ALIGN_SIZE_M=32`) to avoid this.

3. **No weight quantization caching**: Unlike the deleted `moe_te.py` (which cached quantized
   weights across microbatches), the converter path re-quantizes weights every GEMM call. This
   is because the tensor subclass intercept doesn't have module-level state to manage a cache.

## Configuration

```bash
# MXFP8 mode (default, recommended)
--model.converters="quantize.grouped_mm.te"
--quantize.grouped_mm.te.fqns="experts"
--quantize.grouped_mm.te.mode="mxfp8"

# BF16 mode (no quantization, uses TE's GEMM kernel directly)
--model.converters="quantize.grouped_mm.te"
--quantize.grouped_mm.te.fqns="experts"
--quantize.grouped_mm.te.mode="bf16"
```

Can be combined with other converters (e.g., `quantize.linear.mx` for attention layers):

```bash
--model.converters="quantize.linear.mx,quantize.grouped_mm.te"
```
