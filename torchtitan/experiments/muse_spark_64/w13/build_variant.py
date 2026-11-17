# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Four explicit AST edits to the pinned callback-compatible native backward."""
import ast
import copy
import hashlib

SHA = "e8def4834e7cc57c2ff54f052dcc58b14e04dc8ae5b9b74281860a1f04ec3497"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def original(source):
    require(
        hashlib.sha256(source.read_bytes()).hexdigest() == SHA,
        "Exact native API source",
    )
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "_Bf16Autograd"
    )
    return copy.deepcopy(
        next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_backward_impl"
        )
    )


def variant(source):
    fn = original(source)
    i = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "buffers"
    )
    fn.body.insert(
        i + 1,
        ast.parse(
            "_overlap.begin(ctx, context, grad_output_TD, wgrad_destinations, wgrad_output_dtype)"
        ).body[0],
    )
    i = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign)
        and ast.unparse(n.targets[0]) == "wgrad_output_dtype"
    )
    fn.body.insert(
        i + 1,
        ast.parse(
            "_overlap.resolved(wgrad_destinations, wgrad_output_dtype, w13_EFD, w2_EDF)"
        ).body[0],
    )
    blocks = [
        n
        for n in fn.body
        if isinstance(n, ast.With)
        and isinstance(n.items[0].context_expr, ast.Call)
        and n.items[0].context_expr.args[0].value == "dist_moe_backward"
    ]
    require(len(blocks) == 1, "One original backward record scope")
    block = blocks[0]
    i = next(
        i
        for i, n in enumerate(block.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "grad_x_TKD"
    )
    block.body.insert(i + 1, ast.parse("_overlap.fork(buffers.dispatch)").body[0])
    indices = [
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Expr)
        and ast.unparse(n.value) == "_ep_barrier(buffers.dispatch)"
    ]
    require(len(indices) == 1, "One final unconditional dispatch barrier")
    fn.body[indices[0]] = ast.parse("_overlap.join(buffers.dispatch)").body[0]
    return ast.fix_missing_locations(fn)


def text(source):
    fn = variant(source)
    fn.decorator_list = []
    return ast.unparse(fn) + "\n"
