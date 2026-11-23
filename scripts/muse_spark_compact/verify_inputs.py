"""Check staged benchmark arrays and the expected forced-router phase."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path

import numpy as np

from indexed_packed_replay import LANES, ORDINAL_PERIOD, ORIGINAL_CAPACITY, RankRows


def verify_inputs(source_dir, plan_dir):
    source_manifest = (source_dir / "manifest.json").read_bytes()
    source = json.loads(source_manifest)
    plan = json.loads((plan_dir / "manifest.json").read_bytes())
    if (
        source["source_commit"] != "814df1c9e892f0e45bb2af5523c1968c52e7262b"
        or plan["source_manifest_sha256"] != sha256(source_manifest).hexdigest()
        or plan["input_plan_sha256"] != "48edfcbac44fbf3ffd4a68285f0a99029ce7275c5cbece738c37abe50b5aa945"
        or not plan["phase_preserved"]
        or (plan["updates"], plan["capacity"]) != (100, 29_184)
    ):
        raise ValueError("Expected the pinned 100-update compact replay")
    for directory, checksums in (
        (source_dir, source["array_sha256"]),
        (plan_dir, {f"{name}.npy": plan[f"{name}_sha256"] for name in ("lane_offsets", "lane_segments")}),
    ):
        for name, expected in checksums.items():
            with (directory / name).open("rb") as file:
                if file_digest(file, "sha256").hexdigest() != expected:
                    raise ValueError(f"Checksum mismatch: {directory / name}")

    ranks = [RankRows(source_dir, plan_dir, rank) for rank in range(LANES)]
    ordinal = np.arange(plan["capacity"]) % ORDINAL_PERIOD
    for update in range(1, plan["updates"] + 1):
        valid_labels = 0
        for rank in ranks:
            row = rank.load(update, with_router_residues=True)
            live = int(np.count_nonzero(~row.padding_mask))
            valid_labels += np.count_nonzero(row.labels != -100)
            if not np.array_equal(row.router_ordinal_mod32[:live], ordinal[:live]):
                raise ValueError(f"Route phase changed at update {update}, rank {rank.rank}")
        if valid_labels != source["global_valid_labels"][update - 1]:
            raise ValueError(f"Valid label count changed at update {update}")
    print(f"Verified 100 updates across {LANES} ranks ({ORIGINAL_CAPACITY:,} nominal slots per GPU).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    verify_inputs(args.source, args.plan)
