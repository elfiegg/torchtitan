"""Compose selective host H3 with the pinned typed route and fused SwiGLU."""

import inspect
import textwrap


def replace(source, boundary, changed, label):
    if source.count(boundary) != 1:
        raise RuntimeError(f"Pinned Dist-MoE source changed at {label}")
    return source.replace(boundary, changed)


def forward_source(source):
    source = textwrap.dedent(source)
    before_input = "    # 2. Publish routing and activation payloads before deriving peer\n"
    prepare = """    host_h3 = _select_borrowed_h3(
        context_id, context, activation_buffer, activation_slot_id_1,
        num_moe_layers_in_slot, postprocess, x_TD, save_for_backward,
    )
    if host_h3 is not None:
        host_h3.before_forward()

"""
    source = replace(source, before_input, prepare + before_input, "forward input publication")
    native_dispatch = "    # 3. Retain the local input only when backward may need device-selected\n"
    require_fused = """    if host_h3 is not None and not fused_swiglu_forward:
        raise RuntimeError("Selective host H3 requires native fused SwiGLU in forward")

"""
    source = replace(source, native_dispatch, require_fused + native_dispatch, "native forward activation")
    before_save = "    # 6. Retain only graph-stable routing, planner, policy, and activation\n"
    publish = """    if host_h3 is not None:
        if host_h3.saves_next_forward():
            torch._assert_async(
                forward_plan.need_recompute,
                "Host H3 layer unexpectedly retained in the activation slot",
            )
        ctx._borrowed_h3 = host_h3
        ctx._host_h3_frame = host_h3.after_forward(
            forward_plan.need_recompute.data_ptr(), h3_MD,
        )

"""
    return replace(source, before_save, publish + before_save, "forward H3 snapshot")


def backward_source(source):
    source = textwrap.dedent(source)
    before_plan = "    # 2. Rebuild typed plan views from saved scalar offsets, then let the\n"
    drain = """    host_h3 = getattr(ctx, "_borrowed_h3", None)
    if host_h3 is not None:
        host_h3.before_backward(ctx._host_h3_frame, need_recompute.data_ptr())

"""
    source = replace(source, before_plan, drain + before_plan, "backward scratch ownership")
    replay_begin = "        h3_saved_MD = dist_grouped_gemm_fprop_combine(\n"
    replay_end = "        _ep_barrier(buffers.combine, conditional_execution=need_recompute)\n"
    if source.count(replay_begin) != 1 or source.count(replay_end) != 1:
        raise RuntimeError("Pinned Dist-MoE W2 replay block changed")
    start = source.index(replay_begin)
    end = source.index(replay_end) + len(replay_end)
    if start >= end:
        raise RuntimeError("W2 replay and combine barrier order changed")
    stock = textwrap.indent(source[start:end], "    ")
    replay = """        host_h3_selected = host_h3 is not None and ctx._host_h3_frame.host is not None
        verify_h3_replay = host_h3_selected and host_h3.probes(ctx._host_h3_frame)
        if host_h3_selected and not verify_h3_replay:
            if not fused_swiglu_recompute:
                raise RuntimeError("Selective host H3 requires native fused SwiGLU in replay")
            h3_saved_MD = host_h3.restore(ctx._host_h3_frame)
        else:
""" + stock + """            if verify_h3_replay:
                if not fused_swiglu_recompute:
                    raise RuntimeError("Host H3 parity requires native fused SwiGLU replay")
                host_h3.check_replay(ctx._host_h3_frame, h3_saved_MD, topk_expert_ids_TK)
"""
    source = source[:start] + replay + source[end:]
    release = "    _ep_barrier(buffers.dispatch)\n    grad_x_TD = reduce_from_topk(\n"
    prefetch = """    _ep_barrier(buffers.dispatch)
    if host_h3 is not None:
        host_h3.after_backward_barrier(ctx._host_h3_frame)
    grad_x_TD = reduce_from_topk(
"""
    return replace(source, release, prefetch, "backward EP release")


def compile_methods(api):
    cls = api._Bf16Autograd
    if not getattr(cls, "_exp_learned_route_patched", False):
        raise RuntimeError("Install typed learned route before selective host H3")
    if not getattr(cls, "_exp_swiglu_recompute_patched", False):
        raise RuntimeError("Install fused SwiGLU replay before selective host H3")
    methods = {}
    for name, transform in (("forward", forward_source), ("_backward_impl", backward_source)):
        descriptor = cls.__dict__[name]
        if not isinstance(descriptor, staticmethod):
            raise TypeError(f"Dist-MoE {name} changed from a static method")
        source = transform(inspect.getsource(descriptor.__func__))
        filename = f"<selective_host_h3:{name}>"
        namespace = {}
        exec(compile(source, filename, "exec"), api.__dict__, namespace)
        if not isinstance(namespace[name], staticmethod):
            raise TypeError(f"Selective H3 {name} must stay a static method")
        methods[name] = (namespace[name], source)
    return methods
