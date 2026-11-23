"""Bind the host H3 controller to the exact compact BF16 activation layout."""

import linecache

from borrowed_h3 import BorrowedH3
from replay_parity import compare_valid_words
from rewrite_dist_moe import compile_methods


LAYERS = 62
TOKENS = 29_184
WIDTH = 4_096
TOPK = 8
RANKS = 64
H3_BYTES = TOKENS * WIDTH * TOPK * 2
SLOT_BYTES = 49 * 2**30


def stage_spans(slot_bytes, slots, scratch_bytes):
    if (slot_bytes, slots, scratch_bytes) != (SLOT_BYTES, 1, 6 * H3_BYTES):
        raise ValueError("Host H3 was validated for the fixed one-slot compact BF16 planner")
    return [(SLOT_BYTES + i * H3_BYTES, H3_BYTES) for i in range(2)]


class TorchCopyIO:
    def __init__(self, torch, device):
        self.torch = torch
        self.device = device
        self.streams = {name: torch.cuda.Stream(device=device) for name in ("device_stage", "host", "restore")}

    def compute_done(self):
        # The caller is on the forward producer or final EP release stream.
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream(device=self.device))
        return event

    def wait_compute(self, event):
        self.torch.cuda.current_stream(device=self.device).wait_event(event)

    def copy(self, name, dst, src, *predecessors):
        torch = self.torch
        stream = self.streams[name]
        for event in predecessors:
            if event is not None:
                stream.wait_event(event)
        with torch.cuda.stream(stream):
            dst_bytes = dst.view(-1).view(torch.uint8)
            src_bytes = src.view(-1).view(torch.uint8)
            if dst_bytes.numel() != H3_BYTES or src_bytes.numel() != H3_BYTES:
                raise ValueError("Host H3 copy requires the full compact route extent")
            dst_bytes.copy_(src_bytes, non_blocking=True)
            copied = torch.cuda.Event()
            copied.record(stream)
        return copied

    def check_replay(self, snapshot, replayed, expert_ids, copied):
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Host H3 replay parity must run in eager warmup")
        copied.synchronize()
        return compare_valid_words(self.torch, snapshot, replayed, expert_ids, WIDTH)


class BindHostH3:
    def __init__(self, torch, postprocess_type, selected_layers, is_fake_combine, *, validate_replay=False):
        self.torch = torch
        self.postprocess_type = postprocess_type
        self.is_fake_combine = is_fake_combine
        self.selected = tuple(sorted(selected_layers))
        if (not self.selected or len(set(self.selected)) != len(self.selected)
                or any(layer < 5 or layer >= LAYERS - 1 for layer in self.selected)):
            raise ValueError("Host H3 requires unique replay layers in the range 5–60")
        self.states = {}
        self.probe_layer = self.selected[0] if validate_replay else None

    def receipt(self):
        if len(self.states) != 1:
            raise RuntimeError("Host H3 receipt requires one bound Dist-MoE context")
        (_, _, schedule), = self.states.values()
        return {
            **schedule.receipt(),
            "snapshot_bytes": H3_BYTES,
            "reserved_host_bytes": len(schedule.host_buffers) * H3_BYTES,
        }

    def __call__(self, context_id, context, activation, slot, depth, postprocess, x, save):
        torch = self.torch
        if self.is_fake_combine(context.buffers.combine):
            return None
        if not save:
            if context_id in self.states and self.states[context_id][2].stack:
                raise RuntimeError("A scratch-only forward would reuse the live host H3 storage")
            return None
        config, plan = context.config, context.memory_plan
        if (
            type(postprocess) is not self.postprocess_type or depth != LAYERS
            or x.dtype != torch.bfloat16 or tuple(x.shape) != (TOKENS, WIDTH)
            or context.top_k != TOPK or torch.distributed.get_world_size(context.group) != RANKS
            or activation.inference_mode or activation.device_scratch_size is not None
            or activation.host_scratch_size is not None or activation.buffer.dtype != torch.uint8
            or not activation.buffer.is_cuda or not activation.buffer.is_contiguous()
        ):
            raise RuntimeError("Selective host H3 requires the typed EP64 BF16 compact training recipe")
        # The pinned routing kernel traps before any forward scratch write
        # when receive rows exceed the capacity passed from this plan.
        if (config.hidden_dim, config.intermediate_dim) != (WIDTH, WIDTH) or (
            plan is None or not 0 < plan.total_scratch_capacity_rows <= TOKENS * TOPK
        ):
            raise RuntimeError("Selective host H3 routing capacity would overlap its borrowed banks")
        if slot.data_ptr() != activation.activation_slot_ids_S.data_ptr():
            raise RuntimeError("Selective host H3 requires the first and only activation slot")
        if context_id not in self.states:
            if self.states:
                raise RuntimeError("Selective host H3 uses one Dist-MoE context per worker")
            device = activation.buffer.device
            with torch.cuda.device(device):
                capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                raise RuntimeError("Host H3 buffers must be allocated in warmup before CUDA graph capture")
            intervals = stage_spans(
                activation.activation_slot_bytes,
                activation.num_activation_slots,
                activation.scratch_region_size,
            )
            stages = [activation.buffer.narrow(0, start, size) for start, size in intervals]
            hosts = {layer: torch.empty(H3_BYTES, dtype=torch.uint8, device="cpu", pin_memory=True)
                     for layer in self.selected}
            schedule = BorrowedH3(TorchCopyIO(torch, device), stages, hosts, LAYERS, probe_layer=self.probe_layer)
            self.states[context_id] = (activation, activation.buffer.data_ptr(), schedule)
        cached, raw_address, schedule = self.states[context_id]
        if activation is not cached or activation.buffer.data_ptr() != raw_address:
            raise RuntimeError("Selective host H3 activation allocation changed")
        return schedule


def install(api, torch, selected_layers, is_fake_combine, *, validate_replay=False):
    """Install after the typed route. Kept outside the submitted verifier."""
    cls = api._Bf16Autograd
    if getattr(cls, "_borrowed_h3_installed", False):
        raise RuntimeError("The selective H3 plan is already installed")
    binder = BindHostH3(torch, api._LearnedRoute, selected_layers, is_fake_combine, validate_replay=validate_replay)
    methods = compile_methods(api)
    api._select_borrowed_h3 = binder
    for name, (method, source) in methods.items():
        physical = getattr(cls, name)
        method.__func__.__qualname__ = physical.__qualname__
        filename = method.__func__.__code__.co_filename
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        setattr(cls, name, method)
    cls._borrowed_h3_installed = True
    return binder


def install_with_exact_host_pool(api, torch, offload_module, selected_layers, is_fake_combine, *, validate_replay=False):
    """Prepare the capacity and copy changes together before model warmup."""
    from exact_host_pool import exact_acquire

    if getattr(api._Bf16Autograd, "_borrowed_h3_installed", False):
        raise RuntimeError("The selective H3 plan is already installed")
    pool = offload_module._PinnedBufferPool
    acquire = exact_acquire(pool.acquire)
    binder = install(api, torch, selected_layers, is_fake_combine, validate_replay=validate_replay)
    pool.acquire = acquire
    return binder
