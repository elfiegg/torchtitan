# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Redistribute the measured DP8 input streams over DP32 without repacking."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.types import TrainingMicrobatch


class OriginalWorkDataLoader(GrainDataLoader):
    """Take one of four complete-microbatch subsequences from each DP8 stream."""

    @dataclass(kw_only=True, slots=True)
    class Config(GrainDataLoader.Config):
        pass

    def __init__(
        self, config: Config, *, dp_world_size: int, dp_rank: int, **kwargs: Any
    ) -> None:
        if dp_world_size != 32 or not 0 <= dp_rank < 32:
            raise ValueError("This input mapping requires 32 data-parallel ranks.")
        self._target_rank = dp_rank
        self._source_index = 0
        super().__init__(config, dp_world_size=8, dp_rank=dp_rank % 8, **kwargs)

    def __iter__(self) -> Iterator[TrainingMicrobatch]:
        for microbatch in super().__iter__():
            index = self._source_index
            self._source_index += 1
            if index % 4 == self._target_rank // 8:
                yield microbatch

    def state_dict(self) -> dict[str, Any]:
        return {
            "target_rank": self._target_rank,
            "source_index": self._source_index,
            "native": super().state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict:
            raise ValueError("This experimental input mapping does not support resume.")
