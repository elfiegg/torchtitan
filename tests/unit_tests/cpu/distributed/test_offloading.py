# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import Mock, patch

import torch
from torch.numa.binding import AffinityMode

from torchtitan.distributed import maybe_apply_numa_binding
from torchtitan.distributed.activation_offload import _PinnedBufferPool


def test_numa_binding_skips_non_cuda_devices() -> None:
    with patch(
        "torch.numa.binding._maybe_apply_numa_binding_to_current_process"
    ) as bind:
        maybe_apply_numa_binding(0, "cpu")

    bind.assert_not_called()


def test_numa_binding_uses_cuda_device_node() -> None:
    with patch(
        "torch.numa.binding._maybe_apply_numa_binding_to_current_process"
    ) as bind:
        maybe_apply_numa_binding(3, "cuda")

    assert bind.call_args.kwargs["device_index"] == 3
    numa_options = bind.call_args.kwargs["numa_options"]
    assert numa_options.affinity_mode == AffinityMode.NODE
    assert numa_options.should_fall_back_if_binding_fails


def test_pinned_pool_reuses_exact_and_rounded_buckets() -> None:
    allocate = torch.empty
    pool = _PinnedBufferPool(exact_sizes=(6,))
    stream = Mock()
    last_read = Mock()
    with (
        patch.object(
            torch, "empty", side_effect=lambda n, **_: allocate(n, dtype=torch.uint8)
        ),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
    ):
        exact = pool.acquire(6, stream)
        rounded = pool.acquire(5, stream)
        assert exact._base.nbytes == 6
        assert rounded._base.nbytes == 8

        pool.release(exact, last_read)
        pool.release(rounded, last_read)
        assert pool.acquire(6, stream).data_ptr() == exact.data_ptr()
        assert pool.acquire(7, stream).data_ptr() == rounded.data_ptr()
    assert stream.wait_event.call_count == 2
