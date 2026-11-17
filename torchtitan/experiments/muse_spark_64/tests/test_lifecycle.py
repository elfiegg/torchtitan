# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for native-math preservation and graph/resource ownership."""

import ast
import gc
import hashlib
import tempfile
import unittest
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from torchtitan.experiments.muse_spark_64.w13 import build_variant
from torchtitan.experiments.muse_spark_64.w13.overlap import OwnedOverlap

ROOT = Path(__file__).resolve().parents[1]

BACKWARD = """
class _Bf16Autograd:
    @staticmethod
    def _backward_impl(ctx, grad_output_TD, *, wgrad_destinations=None,
                       wgrad_output_dtype=None):
        context = ctx.context
        buffers = context.buffers
        wgrad_output_dtype = resolve_dtype()
        with record_function("dist_moe_backward"):
            prefix()
            grad_x_TKD = dgrad_and_scatter()
            wgrad()
        _ep_barrier(buffers.dispatch)
        return reduce_from_topk(grad_x_TKD)
"""


class BackwardTransformation(unittest.TestCase):
    def test_only_scheduling_statements_change(self):
        # Guards against moving the barrier across a producer or changing math.
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "api.py"
            source.write_text(BACKWARD)
            with patch.object(
                build_variant, "SHA", hashlib.sha256(source.read_bytes()).hexdigest()
            ):
                original = build_variant.original(source)
                changed = build_variant.variant(source)
        block = next(node for node in changed.body if isinstance(node, ast.With))
        statements = [ast.unparse(node) for node in block.body]
        self.assertLess(
            statements.index("grad_x_TKD = dgrad_and_scatter()"),
            statements.index("_overlap.fork(buffers.dispatch)"),
        )
        self.assertLess(
            statements.index("_overlap.fork(buffers.dispatch)"),
            statements.index("wgrad()"),
        )
        changed.body = [
            node
            for node in changed.body
            if not (
                isinstance(node, ast.Expr)
                and ast.unparse(node.value).startswith(
                    ("_overlap.begin(", "_overlap.resolved(")
                )
            )
        ]
        block.body = [
            node
            for node in block.body
            if ast.unparse(node) != "_overlap.fork(buffers.dispatch)"
        ]
        join = next(
            i
            for i, node in enumerate(changed.body)
            if ast.unparse(node) == "_overlap.join(buffers.dispatch)"
        )
        changed.body[join] = ast.parse("_ep_barrier(buffers.dispatch)").body[0]
        self.assertEqual(ast.dump(original), ast.dump(changed))

    def test_changed_dependency_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "api.py"
            source.write_text(BACKWARD)
            with self.assertRaisesRegex(RuntimeError, "Exact native API source"):
                build_variant.variant(source)


class Graph:
    pass


class InputWork(unittest.TestCase):
    def loader_type(self):
        tree = ast.parse((ROOT / "data.py").read_text())
        loader = next(node for node in tree.body if isinstance(node, ast.ClassDef))

        @dataclass(kw_only=True, slots=True)
        class NativeConfig:
            pass

        class NativeLoader:
            Config = NativeConfig

            def __init__(self, config, *, dp_world_size, dp_rank, **kwargs):
                self.source_world = dp_world_size
                self.source_rank = dp_rank
                self.iterator = iter((dp_rank, index) for index in range(32))

            def __iter__(self):
                return self.iterator

        namespace = {"GrainDataLoader": NativeLoader, "dataclass": dataclass}
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                loader,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        exec(compile(module, "<input-work>", "exec"), namespace)
        return namespace["OriginalWorkDataLoader"]

    def test_each_update_preserves_every_original_global_microbatch(self):
        loader_type = self.loader_type()
        loaders = [
            loader_type(loader_type.Config(), dp_world_size=32, dp_rank=rank)
            for rank in range(32)
        ]
        iterators = [iter(loader) for loader in loaders]
        for update in range(3):
            actual = [next(iterator) for iterator in iterators for _ in range(2)]
            expected = [
                (rank, index)
                for rank in range(8)
                for index in range(update * 8, (update + 1) * 8)
            ]
            self.assertCountEqual(actual, expected)
        self.assertTrue(all(loader.source_world == 8 for loader in loaders))

    def test_new_iterator_keeps_subsequence_position(self):
        loader_type = self.loader_type()
        loader = loader_type(loader_type.Config(), dp_world_size=32, dp_rank=17)
        self.assertEqual(next(iter(loader)), (1, 2))
        self.assertEqual(next(iter(loader)), (1, 6))

    def test_unsupported_data_geometry_is_rejected(self):
        loader_type = self.loader_type()
        with self.assertRaisesRegex(ValueError, "32 data-parallel"):
            loader_type(loader_type.Config(), dp_world_size=16, dp_rank=0)


class Teardown(unittest.TestCase):
    def owner(self):
        owner = OwnedOverlap.__new__(OwnedOverlap)
        owner.poisoned = owner.active = owner.forked = False
        owner.context = SimpleNamespace(_closed=True)
        owner.torch = SimpleNamespace(
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: False)
        )
        owner.side = SimpleNamespace(synchronize=lambda: None)
        owner.graph_refs = []
        owner.calls = 0
        return owner

    def test_live_graph_blocks_stream_release(self):
        owner = self.owner()
        graph = Graph()
        owner.graph_refs = [weakref.ref(graph)]
        owner.calls = 1
        with self.assertRaisesRegex(RuntimeError, "destroy them"):
            owner.teardown_ready()
        del graph
        gc.collect()
        owner.teardown_ready()

    def test_failed_backward_is_terminal(self):
        owner = self.owner()
        owner.poisoned = True
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            owner.teardown_ready()

    def test_unused_installation_can_close_without_captured_graphs(self):
        self.owner().teardown_ready()

    def test_engine_drops_graphs_before_releasing_overlap_resources(self):
        # Exercise the actual engine methods without importing GPU dependencies.
        tree = ast.parse((ROOT / "trainer.py").read_text())
        engine = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ExperimentalTrainingEngine"
        )
        engine.body = [
            node
            for node in engine.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("close", "_register_overlap_graphs")
        ]
        manager = SimpleNamespace(
            _dummy_graph=Graph(), _wrappers=[SimpleNamespace(_graph=Graph())]
        )
        context = SimpleNamespace(_closed=False)
        owner = self.owner()
        owner.context = context
        owner.calls = 1

        class NativeEngine:
            def close(self):
                context._closed = True
                manager._dummy_graph = None
                manager._wrappers.clear()

        namespace = {
            "TrainingEngine": NativeEngine,
            "accelerator_graph_manager": manager,
        }
        exec(
            compile(
                ast.Module(body=[engine], type_ignores=[]), "<engine-lifecycle>", "exec"
            ),
            namespace,
        )
        instance = namespace["ExperimentalTrainingEngine"]()
        restored = []

        def restore():
            owner.teardown_ready()
            restored.append(True)

        instance._w13_overlap = SimpleNamespace(
            owner=owner,
            register_graph=lambda graph: owner.graph_refs.append(weakref.ref(graph)),
            restore_after_native_teardown=restore,
        )
        instance.close()
        self.assertEqual(restored, [True])
        self.assertIsNone(instance._w13_overlap)
        instance.close()


if __name__ == "__main__":
    unittest.main()
