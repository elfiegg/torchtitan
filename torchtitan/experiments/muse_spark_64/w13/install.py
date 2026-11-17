# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""No import-time mutation. Explicit native operation qualification only."""
import ast
import copy
import hashlib
import importlib
import inspect
import json
import marshal
import types
from pathlib import Path

from .build_variant import original, require, SHA, text
from .overlap import OwnedOverlap

HERE = Path(__file__).resolve().parent
TORCH_GIT = "bf270c3251d8e84723cedbf5b60eaa60cacaa3f2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(code):
    return code.replace(
        co_consts=tuple(
            normalized(x) if isinstance(x, types.CodeType) else x
            for x in code.co_consts
        ),
        co_filename="<bound-native>",
        co_firstlineno=0,
        co_linetable=b"",
        co_qualname=code.co_name,
    )


def code_sha(fn):
    return hashlib.sha256(marshal.dumps(normalized(fn.__code__))).hexdigest()


def source_callable(node, namespace):
    node = copy.deepcopy(node)
    node.decorator_list = []
    ns = dict(namespace)
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(node),
            "<source-identity>",
            "exec",
        ),
        ns,
    )
    return ns[node.name]


def whole_function_code(node, namespace):
    path = Path(namespace["__file__"])
    pins = json.loads((HERE / "RUNTIME_PINS.json").read_text())
    require(
        namespace["__name__"] in pins and digest(path) == pins[namespace["__name__"]],
        "Exact whole native module source",
    )
    raw = path.read_bytes()
    tree = ast.parse(raw)
    declarations = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and ast.dump(n, include_attributes=True)
        == ast.dump(node, include_attributes=True)
    ]
    require(len(declarations) == 1, "Unique exact native function declaration")
    line = min([node.lineno, *[n.lineno for n in node.decorator_list]])
    root = compile(raw, str(path), "exec", dont_inherit=True, optimize=0)
    found = []

    def walk(code):
        for child in code.co_consts:
            if isinstance(child, types.CodeType):
                if child.co_name == node.name and child.co_firstlineno == line:
                    found.append(child)
                walk(child)

    walk(root)
    require(len(found) == 1, "Unique whole-module compiled native function")
    return found[0]


def verify_function(active, node, namespace):
    require(
        type(active) is types.FunctionType and active.__globals__ is namespace,
        "Original native function global namespace",
    )
    code = whole_function_code(node, namespace)
    expected = source_callable(
        node, namespace
    )  # Defaults only; never expected body code.
    expected_sha = hashlib.sha256(marshal.dumps(normalized(code))).hexdigest()
    require(
        code_sha(active) == expected_sha
        and active.__defaults__ == expected.__defaults__
        and active.__kwdefaults__ == expected.__kwdefaults__,
        "Active function body/defaults match pinned source",
    )


def guarded(fn, owner):
    def run(*args, **kwargs):
        if not owner.lock.acquire(blocking=False):
            owner.fail()
            raise RuntimeError("Concurrent Python backward forbidden")
        try:
            value = fn(*args, **kwargs)
            owner.complete()
            return value
        except BaseException:
            owner.fail()
            raise
        finally:
            owner.lock.release()

    return run


class Installation:
    def __init__(self, api, descriptor, replacement, owner, barrier, proof):
        self.api = api
        self.original = descriptor
        self.replacement = replacement
        self.owner = owner
        self.barrier = barrier
        self.proof = proof
        self.restored = False

    def attest(self):
        require(
            not self.restored
            and inspect.getattr_static(self.api._Bf16Autograd, "_backward_impl")
            is self.replacement,
            "Exact installed static descriptor",
        )
        require(
            self.api._ep_barrier is self.barrier,
            "Original final barrier function identity unchanged",
        )
        require(
            code_sha(self.replacement.__func__) == self.proof["guard_code_sha256"],
            "Installed wrapper code unchanged",
        )
        if "variant_code_sha256" in self.proof:
            cells = dict(
                zip(
                    self.replacement.__func__.__code__.co_freevars,
                    (
                        c.cell_contents
                        for c in self.replacement.__func__.__closure__ or ()
                    ),
                )
            )
            require(
                cells.get("owner") is self.owner
                and code_sha(cells["fn"]) == self.proof["variant_code_sha256"],
                "Exact derivative and owned-resource closure",
            )
            require(
                cells["fn"].__globals__.get("_overlap") is self.owner,
                "Derivative uses owned barrier resources",
            )
        return {
            **self.proof,
            "python_backward_calls": self.owner.calls,
            "failed_terminal": self.owner.poisoned,
        }

    def register_graph(self, graph):
        self.attest()
        self.owner.register_graph(graph)

    def restore_after_native_teardown(self):
        proof = self.attest()
        self.owner.teardown_ready()
        self.api._Bf16Autograd._backward_impl = self.original
        require(
            inspect.getattr_static(self.api._Bf16Autograd, "_backward_impl")
            is self.original,
            "Exact original descriptor restored",
        )
        self.restored = True
        # Release only after native context/registered graph destruction and side completion.
        self.owner.callbacks = ()
        self.owner.context = None
        self.owner.side = None
        self.owner.produced = None
        self.owner.completed = None
        return {
            **proof,
            "original_descriptor_restored": True,
            "resources_released_after_declared_graph_inventory_teardown": True,
        }


def install(context, callbacks, *, manifest_sha256, screen_only=False):
    require(__debug__, "Native installation disallows optimized Python")
    require(
        screen_only is True,
        "Explicit combined unqualified screen only; no numerical or performance acceptance",
    )
    require(
        digest(HERE / "SOURCE_MANIFEST.json") == manifest_sha256,
        "Exact prototype manifest",
    )
    for name, pin in json.loads((HERE / "SOURCE_MANIFEST.json").read_text())[
        "files"
    ].items():
        require(digest(HERE / name) == pin, "Exact prototype payload " + name)
    import torch

    require(
        torch.version.git_version == TORCH_GIT
        and not torch.cuda.is_current_stream_capturing(),
        "Exact runtime; installation outside capture",
    )
    pins = json.loads((HERE / "RUNTIME_PINS.json").read_text())
    modules = {name: importlib.import_module(name) for name in pins}
    for name, module in modules.items():
        require(digest(module.__file__) == pins[name], "Exact native source " + name)
    api = modules["dist_moe.api"]
    buffers = modules["dist_moe._buffers"]
    model = modules["torchtitan.models.muse_spark.model"]
    require(
        type(context) is api.Context
        and api._get_context(context.context_id) is context,
        "Original registered native context",
    )
    require(
        torch.distributed.get_backend(context.group) == "nccl"
        and context.group.size() == 64,
        "Native EP64 NCCL context",
    )
    require(
        context.buffers.dispatch.group is context.group
        and context.buffers.combine.group is context.group,
        "Same original group-wide barrier channel",
    )
    workspace = buffers.get_multimem_barrier_workspace(context.group)
    require(
        workspace is not None and workspace.multicast_ptr != 0,
        "Only initialized native NVLS barrier backend",
    )
    require(
        not buffers.is_fake_symmetric_memory(context.buffers.dispatch),
        "No fake backend",
    )
    callbacks = tuple(callbacks)
    require(
        callbacks and len({id(m) for m in callbacks}) == len(callbacks),
        "Explicit unique original learned-norm owners",
    )
    norm = model.MuseSparkRMSNormWithInputScale
    require(
        all(type(m) is norm and m.to_dist_moe_postprocess() is m for m in callbacks),
        "Exact original learned RMSNorm callback class/identity",
    )
    cls = next(
        n
        for n in ast.parse(Path(model.__file__).read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == norm.__name__
    )
    for name in ("forward", "to_dist_moe_postprocess"):
        node = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        verify_function(inspect.unwrap(getattr(norm, name)), node, model.__dict__)
    descriptor = inspect.getattr_static(api._Bf16Autograd, "_backward_impl")
    require(
        type(descriptor) is staticmethod, "Exact original static backward descriptor"
    )
    verify_function(descriptor.__func__, original(Path(api.__file__)), api.__dict__)
    barrier = api._ep_barrier
    node = next(
        n
        for n in ast.parse(Path(api.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_ep_barrier"
    )
    verify_function(barrier, node, api.__dict__)
    owner = OwnedOverlap(torch, api, context, callbacks, barrier)
    owner.check_static()
    ns = dict(api.__dict__)
    ns["_overlap"] = owner
    variant_text = text(Path(api.__file__))
    exec(
        compile(
            "from __future__ import annotations\n" + variant_text,
            str(HERE / "<explicit-native-variant>"),
            "exec",
        ),
        ns,
    )
    variant = ns["_backward_impl"]
    run = guarded(variant, owner)
    replacement = staticmethod(run)
    proof = {
        "schema": "w13-final-barrier-only-overlap-prototype-v1",
        "source_api_sha256": SHA,
        "original_code_sha256": code_sha(descriptor.__func__),
        "variant_ast_source_sha256": hashlib.sha256(variant_text.encode()).hexdigest(),
        "variant_code_sha256": code_sha(variant),
        "guard_code_sha256": code_sha(run),
        "native_math_changed": False,
        "native_source_files_modified": False,
        "active_function_is_original": False,
        "qualification_only": False,
        "unqualified_screen_only": True,
        "native_lifetime_qualified": False,
        "native_capture_replay_qualified": False,
        "performance_qualified": False,
        "scope": (
            "Combined unqualified screen: EP64 BF16 clamped SwiGLU scratch10/save0 "
            "original learned callback; final NVLS channel0 barrier only"
        ),
    }
    handle = Installation(api, descriptor, replacement, owner, barrier, proof)
    api._Bf16Autograd._backward_impl = replacement
    return handle
