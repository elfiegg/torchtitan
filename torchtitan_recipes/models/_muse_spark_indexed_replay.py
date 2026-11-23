# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Memory-mapped C4 replay for the compact Muse Spark benchmark."""

from __future__ import annotations

import json

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torchtitan.components.data.dataset import GrainDataset
    from torchtitan.components.data.types import (
        DatasetBuildContext,
        DatasetIterationPolicy,
    )


SOURCE_FORMAT = "muse-c4-indexed-source-v1"
PLAN_FORMAT = "muse-c4-indexed-plan-v1"
LANES = 64
SOURCE_SEGMENTS_PER_ROW = 64
ORIGINAL_CAPACITY = 32_768
COMPACT_CAPACITY = 29_184
REPLAY_UPDATES = 100
SOURCE_COMMIT = "814df1c9e892f0e45bb2af5523c1968c52e7262b"
INPUT_PLAN_SHA256 = "48edfcbac44fbf3ffd4a68285f0a99029ce7275c5cbece738c37abe50b5aa945"
ORDINAL_PERIOD = 32
IGNORE_INDEX = -100
SEGMENT_DTYPE = np.dtype(
    [("offset", "<u4"), ("length", "<u2"), ("ordinal_mod32", "u1")]
)


@dataclass(frozen=True, slots=True)
class PackedRow:
    input_ids: np.ndarray
    labels: np.ndarray
    positions: np.ndarray
    padding_mask: np.ndarray
    router_ordinal_mod32: np.ndarray | None


class RankRows:
    """Index one rank in O(1); share file-backed source and plan arrays across ranks."""

    def __init__(self, source_dir, plan_dir, rank):
        source_dir, plan_dir = Path(source_dir), Path(plan_dir)
        source_manifest = (source_dir / "manifest.json").read_bytes()
        source = json.loads(source_manifest)
        plan = json.loads((plan_dir / "manifest.json").read_text())
        if source["format"] != SOURCE_FORMAT or plan["format"] != PLAN_FORMAT:
            raise ValueError("Unsupported C4 indexed source or replay plan format.")
        if plan["source_manifest_sha256"] != sha256(source_manifest).hexdigest():
            raise ValueError("The replay plan references a different source bundle.")
        if not 0 <= rank < LANES:
            raise ValueError(f"DP rank must be in [0, {LANES}), got {rank}.")
        self.capacity = plan["capacity"]
        self.phase_preserved = plan["phase_preserved"]
        self.rank = rank
        self.steps = plan["updates"]
        self.tokens = np.load(
            source_dir / "tokens.npy", mmap_mode="r", allow_pickle=False
        )
        self.offsets = np.load(
            plan_dir / "lane_offsets.npy", mmap_mode="r", allow_pickle=False
        )
        self.segments = np.load(
            plan_dir / "lane_segments.npy", mmap_mode="r", allow_pickle=False
        )
        if (
            self.tokens.dtype != np.dtype("<u4")
            or self.segments.dtype != SEGMENT_DTYPE
            or self.offsets.dtype != np.dtype("<u4")
            or len(self.offsets) != LANES * self.steps + 1
            or self.offsets[-1] != len(self.segments)
        ):
            raise ValueError("C4 bundle arrays disagree with the index format.")
        self._ordinal = np.arange(self.capacity, dtype="<i8")

    def __len__(self):
        return self.steps

    def load(self, update, *, with_router_residues=False):
        if not 1 <= update <= self.steps:
            raise IndexError(update)
        lane = (update - 1) * LANES + self.rank
        begin, end = self.offsets[lane : lane + 2]
        segments = self.segments[begin:end]
        live = int(segments["length"].sum())
        input_ids = np.zeros(self.capacity, dtype="<i8")
        labels = np.full(self.capacity, IGNORE_INDEX, dtype="<i8")
        positions = np.empty(self.capacity, dtype="<i8")
        positions[live:] = self._ordinal[: self.capacity - live]
        padding_mask = self._ordinal >= live
        residues = np.zeros(self.capacity, dtype="u1") if with_router_residues else None

        out = 0
        for offset, length, ordinal in segments.tolist():
            destination = slice(out, out + length)
            input_ids[destination] = self.tokens[offset : offset + length]
            labels[destination] = self.tokens[offset + 1 : offset + length + 1]
            positions[destination] = self._ordinal[:length]
            if residues is not None:
                residues[destination] = (
                    self._ordinal[:length] + ordinal
                ) % ORDINAL_PERIOD
            out += length
        return PackedRow(input_ids, labels, positions, padding_mask, residues)


@dataclass(frozen=True, kw_only=True, slots=True)
class IndexedPackedReplay:
    """TorchTitan DatasetConfig for a phase-preserving balanced replay."""

    source_dir: str
    plan_dir: str

    def validate_compact_plan(self) -> None:
        """Check the saved workload geometry and the matching source manifest."""
        source_bytes = (Path(self.source_dir) / "manifest.json").read_bytes()
        source = json.loads(source_bytes)
        plan = json.loads((Path(self.plan_dir) / "manifest.json").read_text())
        if (
            source.get("format") != SOURCE_FORMAT
            or source.get("source_commit") != SOURCE_COMMIT
            or plan.get("source_manifest_sha256") != sha256(source_bytes).hexdigest()
            or plan.get("format") != PLAN_FORMAT
            or plan.get("phase_preserved") is not True
            or plan.get("capacity") != COMPACT_CAPACITY
            or plan.get("updates") != REPLAY_UPDATES
            or plan.get("input_plan_sha256") != INPUT_PLAN_SHA256
        ):
            raise ValueError(
                f"Expected the {REPLAY_UPDATES}-update phase-preserving "
                f"C4 replay with {COMPACT_CAPACITY:,} slots per GPU."
            )

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> GrainDataset:
        import grain.python as grain
        from torchtitan.components.data.dataset import TextSequence

        if dataset_iteration_policy.dp_world_size != LANES:
            raise ValueError(
                f"Indexed C4 replay requires DP{LANES} and one local row per update."
            )
        rows = RankRows(
            self.source_dir, self.plan_dir, dataset_iteration_policy.dp_rank
        )
        if not rows.phase_preserved:
            raise ValueError(
                "The plan changes router phases and requires a router-ordinal sidecar."
            )
        if (
            context.num_tokens_per_microbatch != rows.capacity
            or context.max_context_length != rows.capacity
            or context.max_num_documents != SOURCE_SEGMENTS_PER_ROW
            or context.per_sample_pad_multiple != 1
        ):
            raise ValueError("The indexed C4 plan and local microbatch shape disagree.")

        def load(index):
            row = rows.load(index + 1)
            return TextSequence(
                input_ids=row.input_ids,
                labels=row.labels,
                positions=row.positions,
                padding_mask=row.padding_mask,
            )

        # The stock trainer provides the step limit. Finite plans must not wrap at resume.
        return grain.MapDataset.range(len(rows)).map(load)
