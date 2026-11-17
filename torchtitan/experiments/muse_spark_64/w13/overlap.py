# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Owned stream fork/join. Qualification-only; Python guards do not run on graph replay."""
import threading
import weakref

from .build_variant import require


class OwnedOverlap:
    def __init__(self, torch, api, context, callbacks, barrier):
        self.torch = torch
        self.api = api
        self.context = context
        self.callbacks = tuple(callbacks)
        self.barrier = barrier
        self.device = context.buffers.dispatch.local().device
        self.check_static()
        require(
            not torch.cuda.is_current_stream_capturing(),
            "Create persistent resources outside capture",
        )
        self.side = torch.cuda.Stream(device=self.device)
        self.produced = torch.cuda.Event(enable_timing=False)
        self.completed = torch.cuda.Event(enable_timing=False)
        # CUDA events are lazy: create both actual handles eagerly, then wait before installation.
        self.produced.record(self.side)
        self.completed.record(self.side)
        self.side.synchronize()
        self.caller = None
        self.active = False
        self.forked = False
        self.poisoned = False
        self.graph_refs = []
        self.calls = 0
        self.joined = False
        self.lock = threading.Lock()

    def check_static(self):
        c = self.context.config
        p = self.context.memory_plan
        t = self.torch
        require(not self.context._closed, "Live original context")
        require(
            type(c) is self.api.Config and type(p) is self.api.MemoryPlan,
            "Native config and MemoryPlan",
        )
        expected = {
            "max_num_local_input_tokens": 16384,
            "hidden_dim": 4096,
            "intermediate_dim": 4096,
            "top_k": 8,
            "num_experts": 256,
            "device_scratch_capacity_factor": 10.0,
            "activation_slot_capacity_factor": 0.0,
            "activation_slot_bytes": None,
            "vmm": None,
            "block_scaled": None,
            "activation": "swiglu_clamped",
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "wgrad_dtype": None,
            "inference": False,
            "num_activation_slots": 1,
            "num_sms": None,
            "bf16_grouped_gemm_preset": None,
        }
        require(
            all(getattr(c, k) == v for k, v in expected.items()),
            "Only exact DP32 scratch10/save0 BF16 training policy",
        )
        require(
            c.max_moe_layers_per_activation_slot >= 2,
            "Shared-context multi-layer scope",
        )
        require(
            p.total_scratch_capacity_rows == 1310720
            and p.host_scratch_bytes == 0
            and p.activation_slot_bytes == p.minimum_activation_slot_bytes
            and p.activation_slot_capacity_factor == 0.0,
            "Original save0 mandatory-input-only arena and hard capacity",
        )

    def begin(self, ctx, context, grad, provided, dtype):
        require(
            not self.poisoned and not self.active and context is self.context,
            "One non-reentrant owned native context",
        )
        self.check_static()
        o = ctx.options
        t = self.torch
        require(
            type(o) is self.api.ExecutionOptions
            and o.inplace_wgrad_accum is True
            and o.requires_eager,
            "Original learned callback and in-place gradient path",
        )
        require(
            provided is None and dtype is None,
            "Only native callback backward entry, not registered overrides",
        )
        require(
            all(
                getattr(o, n) is None
                for n in (
                    "weights_preprocess_fn",
                    "wgrad_postprocess_fn",
                    "wgrad_destination_fn",
                    "swiglu_clip_stats_out_3",
                )
            ),
            "No external schedule callbacks or clip observer",
        )
        require(
            any(o.experts_output_postprocess is m for m in self.callbacks),
            "Exact admitted original learned-norm module identity",
        )
        require(
            grad.device == self.device and grad.dtype == t.bfloat16,
            "Native BF16 gradient device",
        )
        require(
            not t.compiler.is_compiling(),
            "Python eager orchestration, native CUDA graph capture allowed",
        )
        self.caller = t.cuda.current_stream(self.device)
        require(self.caller != self.side, "Distinct caller and barrier streams")
        self.active = True

    def resolved(self, destinations, dtype, w13, w2):
        t = self.torch
        require(
            self.active
            and dtype in (t.bfloat16, t.float32)
            and type(destinations) is tuple
            and len(destinations) == 2,
            "Native resolver selected two supported destinations",
        )
        for d, w, shape in zip(
            destinations, (w13, w2), ((4, 8192, 4096), (4, 4096, 4096))
        ):
            require(
                type(d) is self.api._WgradDestination
                and d.parameter is not None
                and d.dtype == dtype,
                "Original parameter-owned native dtype destination",
            )
            declared = d.parameter.grad_dtype
            grad = d.parameter.grad
            require(
                dtype
                == (
                    declared
                    if declared is not None
                    else grad.dtype
                    if grad is not None
                    else d.parameter.dtype
                ),
                "Native owner declaration/storage/fallback dtype unchanged",
            )
            require(
                tuple(w.shape) == shape
                and w.dtype == t.bfloat16
                and w.device == self.device,
                "Original BF16 expert compute geometry",
            )
            if d.output is None:
                require(
                    d.accumulate is False and d.parameter.grad is None,
                    "Original first-write destination",
                )
            else:
                require(
                    d.accumulate is True
                    and d.parameter.grad is not None
                    and tuple(d.output.shape) == shape
                    and d.output.dtype == dtype
                    and d.output.device == self.device
                    and d.output.is_contiguous(),
                    "Original resolved in-place native dtype gradient storage",
                )

    def fork(self, buffer):
        require(
            self.active and not self.forked and buffer is self.context.buffers.dispatch,
            "One final barrier fork",
        )
        require(
            self.torch.cuda.current_stream(self.device) == self.caller,
            "Original DGrad caller stream",
        )
        self.produced.record(self.caller)
        with self.torch.cuda.stream(self.side):
            self.side.wait_event(self.produced)
            self.barrier(
                buffer
            )  # Original _ep_barrier, channel0, no predicate, no changed kernel.
            self.completed.record(self.side)
        self.forked = True

    def join(self, buffer):
        require(
            self.active and self.forked and buffer is self.context.buffers.dispatch,
            "Matching final barrier join",
        )
        require(
            self.torch.cuda.current_stream(self.device) == self.caller,
            "W13 and top-k remain on caller",
        )
        self.caller.wait_event(self.completed)
        self.joined = True

    def complete(self):
        require(
            self.active and self.forked and self.joined and not self.poisoned,
            "Complete original backward after joined top-k/return",
        )
        self.active = False
        self.forked = False
        self.joined = False
        self.caller = None
        self.calls += 1

    def fail(self):
        self.poisoned = True

    def register_graph(self, graph):
        require(
            not self.torch.cuda.is_current_stream_capturing(),
            "Register completed graph outside capture",
        )
        require(
            isinstance(graph, self.torch.cuda.CUDAGraph),
            "Actual native CUDA graph owner",
        )
        self.graph_refs.append(weakref.ref(graph))

    def teardown_ready(self):
        require(
            not self.poisoned and not self.active and not self.forked,
            "Failed native call is terminal; no recovery teardown",
        )
        require(
            not self.torch.cuda.is_current_stream_capturing(), "Restore outside capture"
        )
        require(
            self.context._closed, "Original native context close must complete first"
        )
        require(
            (bool(self.graph_refs) or self.calls == 0)
            and all(r() is None for r in self.graph_refs),
            "Caller must register all graphs, destroy them before resource release",
        )
        self.side.synchronize()
