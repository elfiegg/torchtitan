# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Skip the Dist-MoE FC2 (combine) recompute in BF16 backward by keeping h3 on the host.

The BF16 backward reruns the FC2 combine GEMM (``dist_grouped_gemm_fprop_combine``)
for every layer whose activations were not saved, only to rebuild the route
output h3 ``[T*K, D]`` (1 GiB per layer-microbatch at Muse Spark 816B). h3 feeds
only the output postprocess backward (the router-score gradient and the
expert-output norm callback). This patch copies the forward h3 to pinned host
memory with a copy engine and copies it back in backward instead of rerunning
the GEMM. FC13 (dispatch) recompute, the planner, and the slot are unchanged.

Where h3 lives. The forward h3 is a view of this rank's own Dist-MoE combine
symmetric buffer (fixed address, fixed size). In backward the original code
recomputes h3 into that same view; the patch copies it back into that same
view. Forward copies go combine buffer -> device staging (copy-engine D2D,
default 1 buffer = 1 GiB HBM) -> pinned host, so the combine buffer is free
long before the next layer and the D2H has a whole layer to finish.
Ordering argument (see H3_OFFLOAD.md):
  * forward: peers write our combine buffer only in their combine kernel of
    our next Dist-MoE call, which runs after that call's first EP barrier. The
    compute stream waits for the copy that reads the combine buffer at the
    start of every patched call.
  * backward: with the patch no rank runs the combine GEMM, so no peer writes
    our combine buffer in backward. Peers read it (grad_h3) in their dgrad
    dispatch, before they arrive at the final dispatch barrier of that
    backward. The H2D for the next layer is queued after that barrier.

Forward/backward pairing is a per-context LIFO stack checked by key (the data
pointer of the forward's ``need_recompute`` tensor, which autograd keeps alive
until backward). A mismatch raises: the patch is all-or-nothing on every rank,
because a rank that fell back to the combine GEMM would scatter into peers'
combine buffers.

Patched functions (exact-match anchors, each must match once or the patch
refuses to apply, every changed line carries ``# CHANGED``):
  * ``dist_moe.api._Bf16Autograd.forward``         (2 insertions)
  * ``dist_moe.api._Bf16Autograd._backward_impl``  (3 insertions)
Composes with swiglu_recompute.py when imported AFTER it (this patch reads the
already patched source via linecache; swiglu_recompute's anchors only match
the original layout, so the other order fails loudly at import).

Applied on import when EXP_MOE_H3_OFFLOAD=1; otherwise a no-op. ``apply()``
applies it explicitly. Every rank must use the same setting.
  EXP_MOE_H3_OFFLOAD_DUMP=<dir>   write both patched sources for review
  EXP_MOE_H3_MAX_LIVE=<int>       max offloaded layers alive at once (default 512)
  EXP_MOE_H3_STAGE_BUFFERS=<int>  device staging buffers for the forward copy
                                  (default 1 = +1 GiB HBM; 0 = direct D2H)

TODO: this belongs in meta-pytorch/dist_moe (``dist_moe/api.py``
``_Bf16Autograd``); delete this module once it lands there.
"""

import dataclasses
import inspect
import linecache
import os
import textwrap

import torch

ENABLED = os.environ.get("EXP_MOE_H3_OFFLOAD", "0") == "1"
MAX_LIVE = int(os.environ.get("EXP_MOE_H3_MAX_LIVE", "512"))
# Device staging buffers for the forward copy (1 GiB each at Muse Spark 816B).
# >0: copy engine D2D combine buffer -> staging (fast, frees the combine buffer
#     well before the next layer), then D2H staging -> host over a whole layer.
#  0: D2H straight from the combine buffer; no extra HBM, but the copy must
#     finish within the ~5.9 ms before the next layer's Dist-MoE, where the
#     ActivationOffloader already keeps the D2H link ~66% busy (stall risk).
STAGE_BUFFERS = int(os.environ.get("EXP_MOE_H3_STAGE_BUFFERS", "1"))
_RANK0 = os.environ.get("RANK", "0") == "0"
# Test only: spin this many GPU cycles on the copy stream before every h3 copy,
# so a missing stream wait shows up as wrong gradients instead of a lucky race.
DEBUG_COPY_DELAY_CYCLES = 0

# (old, new) exact-text edits, written against the dedented method source
# (method body at 4 spaces). Insertions only; nothing upstream is removed.
_FORWARD_EDITS = (
    # F1. Order this call after the previous h3 D2H (and after any prefetched
    #     H2D) before anything can let a peer write our combine buffer.
    (
        "    context = _get_context(context_id)\n"
        "    options = _resolve_execution_options(options)\n",
        "    context = _get_context(context_id)\n"
        "    options = _resolve_execution_options(options)\n"
        "    _moe_h3_offload.before_call(context_id, forward=True)  # CHANGED: wait for h3 copies\n",
    ),
    # F2. After postprocess consumed h3 (and zeroed padded rows in place),
    #     queue its D2H and push the forward on the per-context stack.
    (
        "    if not save_for_backward:\n"
        "        ctx.mark_non_differentiable(output_TD)\n"
        "        return output_TD\n",
        "    if save_for_backward:  # CHANGED: keep h3 in pinned host memory\n"
        "        _moe_h3_offload.offload(context_id, forward_plan.need_recompute, h3_MD)  # CHANGED\n"
        "    if not save_for_backward:\n"
        "        ctx.mark_non_differentiable(output_TD)\n"
        "        return output_TD\n",
    ),
)

_BACKWARD_EDITS = (
    # B1. Count this call; order it after the last h3 D2H.
    (
        "    ) = ctx.saved_tensors\n",
        "    ) = ctx.saved_tensors\n"
        "    _moe_h3_offload.before_call(ctx.context_id, forward=False)  # CHANGED\n",
    ),
    # B2. h3 comes back from the host into the same combine-buffer view; the
    #     FC2 combine GEMM runs only when the patch is inactive (returns None).
    (
        "        h3_saved_MD = dist_grouped_gemm_fprop_combine(\n",
        "        _h3_restored = _moe_h3_offload.restore(ctx.context_id, need_recompute)  # CHANGED\n"
        "        h3_saved_MD = _h3_restored if _h3_restored is not None else dist_grouped_gemm_fprop_combine(  # CHANGED\n",
    ),
    # B3. Every peer has passed this barrier only after its dgrad dispatch
    #     finished reading our grad_h3, so the combine buffer is free: prefetch
    #     the next backward's h3 into it.
    (
        "    _ep_barrier(buffers.dispatch)\n" "    grad_x_TD = reduce_from_topk(\n",
        "    _ep_barrier(buffers.dispatch)\n"
        "    _moe_h3_offload.after_backward(ctx.context_id)  # CHANGED: prefetch next h3\n"
        "    grad_x_TD = reduce_from_topk(\n",
    ),
)


def patch_source(src: str, edits, what: str) -> str:
    """Dedent ``src`` and apply ``edits``; raise if any anchor drifted."""
    src = textwrap.dedent(src)
    for i, (old, new) in enumerate(edits, 1):
        n = src.count(old)
        if n != 1:
            raise RuntimeError(
                f"h3_offload: {what} edit {i} anchor matched {n} times; "
                "dist_moe.api changed upstream, refusing to patch"
            )
        src = src.replace(old, new)
    return src


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------


def _capturing() -> bool:
    return torch.cuda.is_current_stream_capturing()


@dataclasses.dataclass
class _Mark:
    """An event plus the capture mode it was recorded in."""

    event: torch.cuda.Event
    captured: bool


def _record(stream: torch.cuda.Stream) -> _Mark:
    event = torch.cuda.Event()
    event.record(stream)
    return _Mark(event, _capturing())


def _wait(stream: torch.cuda.Stream, mark: "_Mark | None") -> None:
    # Same rule as torchtitan's _PinnedBufferPool: an event recorded in the
    # other mode (eager vs capture) cannot be waited on, and its work is
    # ordered anyway (torch.cuda.graph synchronizes before capture; a replay
    # runs on the compute stream, which every side-stream copy waits on).
    if mark is not None and mark.captured == _capturing():
        stream.wait_event(mark.event)


class _HostPool:
    """Pinned buffers reused in GPU stream order (no host sync, capture-safe)."""

    def __init__(self) -> None:
        self._free: dict[int, list[tuple[torch.Tensor, _Mark]]] = {}
        self.allocated_bytes = 0

    def acquire(self, nbytes: int, copy_stream: torch.cuda.Stream) -> torch.Tensor:
        free = self._free.get(nbytes)
        if free:
            buffer, last_use = free.pop(0)
            _wait(copy_stream, last_use)
            return buffer
        self.allocated_bytes += nbytes
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

    def release(self, buffer: torch.Tensor, last_use: _Mark) -> None:
        self._free.setdefault(buffer.numel(), []).append((buffer, last_use))


@dataclasses.dataclass
class _Entry:
    key: int  # need_recompute.data_ptr() of the forward
    call_index: int  # value of _State.calls at that forward
    h3: torch.Tensor  # [T*K, D] view of this rank's combine symmetric buffer
    host: torch.Tensor  # pinned uint8 copy of h3
    d2h: _Mark  # host copy complete


@dataclasses.dataclass
class _Pending:
    key: int
    h2d: _Mark


class _State:
    def __init__(self, device: torch.device, fake: bool) -> None:
        self.device = device
        self.fake = fake
        self.d2d_stream = torch.cuda.Stream(device)
        self.d2h_stream = torch.cuda.Stream(device)
        self.h2d_stream = torch.cuda.Stream(device)
        self.calls = 0
        self.stack: list[_Entry] = []
        # Last forward copy that reads the combine buffer (the D2D when
        # staged, else the D2H); the combine buffer is free after it.
        self.combine_free: _Mark | None = None
        self.last_d2h: _Mark | None = None
        self.pending: _Pending | None = None
        self.pool = _HostPool()
        self.staging: list[torch.Tensor] = []
        self.staging_free: list[_Mark | None] = []
        self.next_staging = 0


@dataclasses.dataclass
class Stats:
    offloads: int = 0
    restore_resident: int = 0  # h3 still in the combine buffer (last layer)
    restore_prefetched: int = 0  # H2D queued at the previous backward's end
    restore_sync: int = 0  # H2D queued at use (exposed); expected 0
    prefetch_invalidated: int = 0
    bytes_d2h: int = 0
    bytes_h2d: int = 0


def _delay() -> None:
    if DEBUG_COPY_DELAY_CYCLES:
        torch.cuda._sleep(DEBUG_COPY_DELAY_CYCLES)


class _Runtime:
    def __init__(self) -> None:
        self.active = False
        self.states: dict[str, _State] = {}
        self.stats = Stats()
        self.debug_need_recompute: list[torch.Tensor] | None = None
        self._reported = False

    # -- helpers ------------------------------------------------------------
    def _state(self, context_id: str) -> _State:
        st = self.states.get(context_id)
        if st is None:
            from dist_moe import api
            from dist_moe._buffers import is_fake_symmetric_memory

            context = api._get_context(context_id)
            combine = context.buffers.combine
            # FakePG zero-fills the local combine buffer at every recompute
            # (_initialize_fake_peer_scatter_output), which would wipe a
            # restored h3. Fake runs keep the upstream combine recompute.
            st = _State(combine.local().device, is_fake_symmetric_memory(combine))
            self.states[context_id] = st
        return st

    def _staging(self, st: _State, nbytes: int) -> tuple[int, torch.Tensor]:
        if len(st.staging) != STAGE_BUFFERS or (
            st.staging and st.staging[0].numel() != nbytes
        ):
            if st.stack:
                raise RuntimeError(
                    "h3_offload: staging changed with forwards in flight"
                )
            st.staging = [
                torch.empty(nbytes, dtype=torch.uint8, device=st.device)
                for _ in range(STAGE_BUFFERS)
            ]
            st.staging_free = [None] * STAGE_BUFFERS
            st.next_staging = 0
        i = st.next_staging
        st.next_staging = (i + 1) % STAGE_BUFFERS
        return i, st.staging[i]

    def _issue_h2d(self, st: _State, entry: _Entry) -> _Mark:
        cur = torch.cuda.current_stream()
        # cur is past the point where the combine buffer is free (see callers).
        st.h2d_stream.wait_stream(cur)
        _wait(st.h2d_stream, entry.d2h)  # the host copy must be complete
        with torch.cuda.stream(st.h2d_stream):
            _delay()
            entry.h3.view(-1).view(torch.uint8).copy_(entry.host, non_blocking=True)
            mark = _record(st.h2d_stream)
        self.stats.bytes_h2d += entry.host.numel()
        return mark

    # -- hooks called from the patched dist_moe code -------------------------
    def before_call(self, context_id: str, *, forward: bool) -> None:
        if not self.active:
            return
        st = self._state(context_id)
        if st.fake:
            return
        cur = torch.cuda.current_stream()
        # The forward copy that reads the combine buffer must finish before
        # this call can overwrite it (our grad publish) or let peers write it.
        _wait(cur, st.combine_free)
        if forward and st.pending is not None:
            # A forward between two backwards (not expected without PP) would
            # clobber the prefetched h3: let the H2D land, then forget it so
            # the matching backward copies again.
            _wait(cur, st.pending.h2d)
            st.pending = None
            self.stats.prefetch_invalidated += 1
        st.calls += 1

    def offload(
        self, context_id: str, need_recompute: torch.Tensor, h3: torch.Tensor
    ) -> None:
        if not self.active:
            return
        st = self._state(context_id)
        if st.fake:
            return
        if len(st.stack) >= MAX_LIVE:
            raise RuntimeError(
                f"h3_offload: {len(st.stack)} forwards without backward on one "
                "Dist-MoE context (EXP_MOE_H3_MAX_LIVE); a backward never ran?"
            )
        if not h3.is_contiguous():
            raise RuntimeError("h3_offload: forward h3 view is not contiguous")
        cur = torch.cuda.current_stream()
        src = h3.view(-1).view(torch.uint8)
        nbytes = src.numel()
        host = st.pool.acquire(nbytes, st.d2h_stream)
        if STAGE_BUFFERS > 0:
            i, staging = self._staging(st, nbytes)
            st.d2d_stream.wait_stream(cur)  # h3 is complete
            _wait(st.d2d_stream, st.staging_free[i])  # previous D2H out of it done
            with torch.cuda.stream(st.d2d_stream):
                _delay()
                staging.copy_(src, non_blocking=True)
                combine_free = _record(st.d2d_stream)
            _wait(st.d2h_stream, combine_free)
            with torch.cuda.stream(st.d2h_stream):
                _delay()
                host.copy_(staging, non_blocking=True)
                d2h = _record(st.d2h_stream)
            st.staging_free[i] = d2h
        else:
            st.d2h_stream.wait_stream(cur)
            with torch.cuda.stream(st.d2h_stream):
                _delay()
                host.copy_(src, non_blocking=True)
                d2h = _record(st.d2h_stream)
            combine_free = d2h
        st.combine_free = combine_free
        st.last_d2h = d2h
        st.stack.append(
            _Entry(
                key=need_recompute.data_ptr(),
                call_index=st.calls,
                h3=h3,
                host=host,
                d2h=d2h,
            )
        )
        self.stats.offloads += 1
        self.stats.bytes_d2h += nbytes
        if self.debug_need_recompute is not None:
            self.debug_need_recompute.append(need_recompute.detach().clone())

    def restore(
        self, context_id: str, need_recompute: torch.Tensor
    ) -> "torch.Tensor | None":
        """Return h3 for this backward in the combine-buffer view, or None (inactive)."""
        if not self.active:
            return None
        st = self._state(context_id)
        if st.fake:
            return None
        if not st.stack:
            raise RuntimeError(
                "h3_offload: Dist-MoE backward with no offloaded h3 on this "
                "context (was EXP_MOE_H3_OFFLOAD toggled between forward and backward?)"
            )
        entry = st.stack.pop()
        if entry.key != need_recompute.data_ptr():
            raise RuntimeError(
                "h3_offload: Dist-MoE backward is not the most recent unmatched "
                "forward on this context (non-LIFO schedule, e.g. PP 1F1B); "
                "this patch only supports gradient accumulation without PP"
            )
        cur = torch.cuda.current_stream()
        if entry.call_index == st.calls - 1:
            # No Dist-MoE call since this forward except this backward, so the
            # combine buffer still holds h3 (the last layer before the loss).
            # before_call already ordered us after the copy that reads it.
            self.stats.restore_resident += 1
            last_use = entry.d2h
        else:
            if st.pending is not None and st.pending.key == entry.key:
                self.stats.restore_prefetched += 1
                last_use = st.pending.h2d
            else:
                # Nothing prefetched (first backward after an interleaved
                # forward): copy now. Exposed, but ordered: cur waited for
                # combine_free in before_call.
                self.stats.restore_sync += 1
                last_use = self._issue_h2d(st, entry)
            _wait(cur, last_use)
        st.pending = None
        st.pool.release(entry.host, last_use)
        return entry.h3

    def after_backward(self, context_id: str) -> None:
        if not self.active:
            return
        st = self._state(context_id)
        if st.fake:
            return
        cur = torch.cuda.current_stream()
        # Join the forward copy streams back into the compute stream (CUDA
        # graph capture needs every forked stream joined). By now even the
        # last layer's D2H, queued before the loss, has long finished.
        _wait(cur, st.last_d2h)
        if st.stack:
            # Every peer arrived at the barrier just passed only after its
            # dgrad dispatch finished reading our grad_h3, and no peer writes
            # our combine buffer in a patched backward: prefetch into it.
            nxt = st.stack[-1]
            st.pending = _Pending(key=nxt.key, h2d=self._issue_h2d(st, nxt))
        elif _RANK0 and not self._reported:
            self._reported = True
            print(f"[h3_offload] first microbatch drained: {self.stats}", flush=True)


_runtime = _Runtime()


# --------------------------------------------------------------------------
# Patch application
# --------------------------------------------------------------------------

_ORIGINALS: dict[str, staticmethod] = {}


def _compile_staticmethod(cls, name: str, edits, filename: str):
    from dist_moe import api

    orig = cls.__dict__[name]
    if not isinstance(orig, staticmethod):
        raise RuntimeError(f"h3_offload: {name} is not a staticmethod")
    func = orig.__func__
    src = inspect.getsource(func)
    if not src.lstrip().startswith("@staticmethod"):
        raise RuntimeError(f"h3_offload: unexpected {name} source layout")
    new_src = patch_source(src, edits, name)
    dump = os.environ.get("EXP_MOE_H3_OFFLOAD_DUMP")
    if dump:
        os.makedirs(dump, exist_ok=True)
        with open(os.path.join(dump, f"{name}.py"), "w") as f:
            f.write(new_src)
    linecache.cache[filename] = (len(new_src), None, new_src.splitlines(True), filename)
    namespace: dict = {}
    # Globals are dist_moe.api's own module dict, so every upstream name
    # resolves to the upstream object; _moe_h3_offload is injected below.
    exec(compile(new_src, filename, "exec"), api.__dict__, namespace)
    patched = namespace[name]
    if not isinstance(patched, staticmethod):
        raise RuntimeError(f"h3_offload: patched {name} lost @staticmethod")
    patched.__func__.__doc__ = func.__doc__
    patched.__func__.__qualname__ = func.__qualname__
    patched.__func__.__module__ = func.__module__
    return orig, patched


def apply() -> None:
    """Patch dist_moe's BF16 autograd and activate the runtime (idempotent)."""
    from dist_moe import api

    cls = api._Bf16Autograd
    if getattr(cls, "_exp_moe_h3_offload_patched", False):
        _runtime.active = True
        return
    api.__dict__["_moe_h3_offload"] = _runtime
    # Compile both before installing either, so a drifted anchor leaves
    # dist_moe untouched.
    fwd = _compile_staticmethod(
        cls,
        "forward",
        _FORWARD_EDITS,
        "<h3_offload: dist_moe.api._Bf16Autograd.forward>",
    )
    bwd = _compile_staticmethod(
        cls,
        "_backward_impl",
        _BACKWARD_EDITS,
        "<h3_offload: dist_moe.api._Bf16Autograd._backward_impl>",
    )
    _ORIGINALS["forward"], cls.forward = fwd
    _ORIGINALS["_backward_impl"], cls._backward_impl = bwd
    cls._exp_moe_h3_offload_patched = True
    _runtime.active = True
    if _RANK0:
        print(
            "[h3_offload] patched dist_moe BF16 autograd: h3 kept in pinned host "
            "memory, FC2 combine recompute skipped",
            flush=True,
        )


def remove() -> None:
    """Restore the functions that apply() replaced (tests only)."""
    from dist_moe import api

    cls = api._Bf16Autograd
    if not getattr(cls, "_exp_moe_h3_offload_patched", False):
        return
    for st in _runtime.states.values():
        if st.stack:
            raise RuntimeError("h3_offload.remove() with forwards awaiting backward")
    cls.forward = _ORIGINALS.pop("forward")
    cls._backward_impl = _ORIGINALS.pop("_backward_impl")
    cls._exp_moe_h3_offload_patched = False
    _runtime.active = False


def stats() -> Stats:
    return _runtime.stats


def reset_stats() -> None:
    _runtime.stats = Stats()


if ENABLED:
    apply()
