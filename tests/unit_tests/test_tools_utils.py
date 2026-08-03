# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import torchtitan.tools.utils as tools_utils
from torchtitan.tools.utils import (
    get_cuda_flash_attention_impl,
    maybe_activate_cuda_flash_attention_impl,
)


@pytest.fixture(autouse=True)
def _unset_flash_attention_impl_env(monkeypatch):
    """Keep an operator's override out of the architecture-default tests."""
    monkeypatch.delenv(tools_utils.FLASH_ATTENTION_IMPL_ENV, raising=False)


@pytest.mark.parametrize(
    ("capability", "expected_impl"),
    [
        ((8, 0), None),
        ((9, 0), "FA3"),
        ((9, 1), "FA3"),
        ((10, 0), "FA4"),
        ((10, 3), "FA4"),
        # SM 11.0+ falls through to the newest known impl (FA4).
        ((11, 0), "FA4"),
    ],
)
def test_get_cuda_flash_attention_impl(monkeypatch, capability, expected_impl):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.version, "hip", None)

    assert get_cuda_flash_attention_impl() == expected_impl


def test_get_cuda_flash_attention_impl_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert get_cuda_flash_attention_impl() is None


def test_get_cuda_flash_attention_impl_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.0")

    assert get_cuda_flash_attention_impl() is None


@pytest.mark.parametrize(
    ("override", "capability", "expected_impl"),
    [
        # FA2 keeps PyTorch's built-in kernels on any architecture.
        ("FA2", (10, 0), None),
        ("FA2", (9, 0), None),
        # An explicit impl is honored regardless of the architecture default.
        ("FA4", (9, 0), "FA4"),
        ("FA3", (10, 0), "FA3"),
        # "auto" is the documented default value, same as an unset variable.
        ("auto", (10, 0), "FA4"),
    ],
)
def test_get_cuda_flash_attention_impl_env_override(
    monkeypatch, override, capability, expected_impl
):
    monkeypatch.setenv(tools_utils.FLASH_ATTENTION_IMPL_ENV, override)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.version, "hip", None)

    assert get_cuda_flash_attention_impl() == expected_impl


def test_get_cuda_flash_attention_impl_env_override_invalid(monkeypatch):
    monkeypatch.setenv(tools_utils.FLASH_ATTENTION_IMPL_ENV, "fa2")

    with pytest.raises(ValueError, match="TORCHTITAN_FLASH_ATTENTION_IMPL"):
        get_cuda_flash_attention_impl()


def test_maybe_activate_flash_attention_impl_env_override_fa2(monkeypatch):
    """FA2 must be restored when another impl is already registered over aten."""
    monkeypatch.setenv(tools_utils.FLASH_ATTENTION_IMPL_ENV, "FA2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(tools_utils, "current_flash_attention_impl", lambda: "FA4")

    restored = []
    monkeypatch.setattr(
        tools_utils, "restore_flash_attention_impl", lambda: restored.append(True)
    )

    assert maybe_activate_cuda_flash_attention_impl() is None
    assert restored == [True]


def _patch_capability(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(tools_utils, "current_flash_attention_impl", lambda: None)


def test_maybe_activate_flash_attention_impl(monkeypatch):
    activated = []
    _patch_capability(monkeypatch, (10, 0))
    monkeypatch.setattr(tools_utils, "activate_flash_attention_impl", activated.append)

    assert maybe_activate_cuda_flash_attention_impl() == "FA4"
    assert activated == ["FA4"]


def test_maybe_activate_flash_attention_impl_already_active(monkeypatch):
    _patch_capability(monkeypatch, (9, 0))
    monkeypatch.setattr(tools_utils, "current_flash_attention_impl", lambda: "FA3")

    def fail(impl):
        raise AssertionError("activate_flash_attention_impl called twice")

    monkeypatch.setattr(tools_utils, "activate_flash_attention_impl", fail)

    assert maybe_activate_cuda_flash_attention_impl() == "FA3"


@pytest.mark.parametrize(
    "error",
    [
        # The wheel is not installed in this environment (NVBug 6433694).
        ModuleNotFoundError("No module named 'flash_attn_interface'"),
        # The module is importable but does not expose the expected kernels.
        RuntimeError("Module does not expose FA4 kernels"),
    ],
)
def test_maybe_activate_flash_attention_impl_falls_back_to_fa2(monkeypatch, error):
    _patch_capability(monkeypatch, (10, 0))

    def raise_error(impl):
        raise error

    monkeypatch.setattr(tools_utils, "activate_flash_attention_impl", raise_error)

    assert maybe_activate_cuda_flash_attention_impl() is None


def test_maybe_activate_flash_attention_impl_pre_hopper(monkeypatch):
    _patch_capability(monkeypatch, (8, 0))

    def fail(impl):
        raise AssertionError("no FA3/FA4 impl exists for SM 8.0")

    monkeypatch.setattr(tools_utils, "activate_flash_attention_impl", fail)

    assert maybe_activate_cuda_flash_attention_impl() is None
