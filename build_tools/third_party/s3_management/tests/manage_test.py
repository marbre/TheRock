# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from manage import (
    S3Index,
    S3Object,
    StructuredS3Index,
    _make_list_prefix,
    package_name_from_distribution_filename,
    pep503_normalize_package_name,
    update_pep503_index,
)


# ---------------------------------------------------------------------------
# _make_list_prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix, package_name, expected",
    [
        ("v4/whl", "torch", "v4/whl/torch-"),
        ("v4/whl", "amd_torch_device_gfx942", "v4/whl/amd_torch_device_gfx942-"),
        ("v4/whl", "torchaudio", "v4/whl/torchaudio-"),
        ("v4/whl", None, "v4/whl"),
        ("v2/gfx94X-dcgpu", None, "v2/gfx94X-dcgpu"),
        # Empty string must not narrow the prefix; treated as full sweep.
        ("v4/whl", "", "v4/whl"),
    ],
)
def test_make_list_prefix(prefix: str, package_name: str | None, expected: str) -> None:
    assert _make_list_prefix(prefix, package_name) == expected


def test_update_pep503_index_rejects_package_name_with_update_root_index() -> None:
    with pytest.raises(
        ValueError,
        match="package_name and update_root_index=True cannot be used together",
    ):
        update_pep503_index(
            prefix="v4/whl", package_name="torch", update_root_index=True
        )


def test_make_list_prefix_torch_does_not_match_torchaudio() -> None:
    # The hyphen delimiter ensures torch- cannot match torchaudio- wheels.
    prefix = _make_list_prefix("v4/whl", "torch")
    assert not "torchaudio-2.10.0.whl".startswith(prefix.split("/")[-1] + "audio")
    assert "torch-2.10.0.whl".startswith(prefix.split("/")[-1])
    assert not "torchaudio-2.10.0.whl".startswith(prefix.split("/")[-1])


# ---------------------------------------------------------------------------
# S3Index.obj_to_package_name
# ---------------------------------------------------------------------------


def _make_obj(key: str) -> S3Object:
    return S3Object(key=key, orig_key=key, checksum=None, size=None, pep658=None)


@pytest.mark.parametrize(
    "key, expected",
    [
        (
            "v4/whl/torch-2.10.0%2Brocm7.14.0a20260617-cp310-cp310-linux_x86_64.whl",
            "torch",
        ),
        (
            "v4/whl/amd_torch_device_gfx942-2.10.0%2Brocm7.14.0a20260617-py3-none-linux_x86_64.whl",
            "amd_torch_device_gfx942",
        ),
        (
            "v4/whl/torchaudio-2.10.0%2Brocm7.14.0a20260617-cp310-cp310-linux_x86_64.whl",
            "torchaudio",
        ),
        (
            "v4/whl/torchvision-0.21.0%2Brocm7.14.0a20260617-cp312-cp312-linux_x86_64.whl",
            "torchvision",
        ),
        (
            "v2/gfx94X-dcgpu/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
            "torch",
        ),
    ],
)
def test_obj_to_package_name(key: str, expected: str) -> None:
    idx = S3Index([], prefix="v4/whl")
    assert idx.obj_to_package_name(_make_obj(key)) == expected


# ---------------------------------------------------------------------------
# PEP 503 helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ROCm.SDK_Core", "rocm-sdk-core"),
        ("amd_torch_device_gfx942", "amd-torch-device-gfx942"),
        ("llnl.hatchet", "llnl-hatchet"),
        ("a---b___c...d", "a-b-c-d"),
    ],
)
def test_pep503_normalize_package_name(name: str, expected: str) -> None:
    assert pep503_normalize_package_name(name) == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "torch-2.10.0+rocm7.14.0a20260617-cp310-cp310-linux_x86_64.whl",
            "torch",
        ),
        (
            "amd_torch_device_gfx942-2.10.0+rocm7.14.0a20260617-py3-none-linux_x86_64.whl",
            "amd-torch-device-gfx942",
        ),
        ("rocm_sdk_core-7.13.0.tar.gz", "rocm-sdk-core"),
        ("llnl-hatchet-2024.1.0.tar.gz", "llnl-hatchet"),
    ],
)
def test_package_name_from_distribution_filename(filename: str, expected: str) -> None:
    assert package_name_from_distribution_filename(filename) == expected


def test_package_name_from_distribution_filename_rejects_unknown_extension() -> None:
    with pytest.raises(ValueError, match="unsupported Python distribution file"):
        package_name_from_distribution_filename("torch-2.10.0.txt")


# ---------------------------------------------------------------------------
# Structured product-local layout
# ---------------------------------------------------------------------------


def test_structured_index_discovers_package_directories() -> None:
    torch_obj = _make_obj(
        "rocm/pytorch/whl/torch/torch-2.10.0%2Brocm7.14.0-cp310-cp310-linux_x86_64.whl"
    )
    device_obj = _make_obj(
        "rocm/pytorch/whl/amd-torch-device-gfx942/"
        "amd_torch_device_gfx942-2.10.0%2Brocm7.14.0-py3-none-linux_x86_64.whl"
    )
    idx = StructuredS3Index([torch_obj, device_obj], prefix="rocm/pytorch/whl")

    assert idx.subdirs == {"rocm/pytorch/whl"}
    assert idx.get_package_names() == ["amd-torch-device-gfx942", "torch"]
    assert list(idx.gen_file_list(package_name="torch")) == [torch_obj]


def test_structured_package_html_uses_same_directory_links() -> None:
    obj = S3Object(
        key="rocm/pytorch/whl/torch/torch-2.10.0%2Brocm7.14.0-cp310-cp310-linux_x86_64.whl",
        orig_key="rocm/pytorch/whl/torch/torch-2.10.0+rocm7.14.0-cp310-cp310-linux_x86_64.whl",
        checksum="abc123",
        size=1,
        pep658="def456",
    )
    idx = StructuredS3Index([obj], prefix="rocm/pytorch/whl")

    html = idx.to_simple_package_html(subdir=None, package_name="torch")

    assert (
        'href="torch-2.10.0%2Brocm7.14.0-cp310-cp310-linux_x86_64.whl#sha256=abc123"'
        in html
    )
    assert 'href="../' not in html
    assert 'data-dist-info-metadata="sha256=def456"' in html


def test_structured_root_html_lists_local_package_directories() -> None:
    idx = StructuredS3Index(
        [
            _make_obj(
                "rocm/pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl"
            ),
            _make_obj(
                "rocm/pytorch/whl/torchaudio/torchaudio-2.10.0-cp310-cp310-linux_x86_64.whl"
            ),
        ],
        prefix="rocm/pytorch/whl",
    )

    html = idx.to_simple_packages_html(subdir=None)

    assert '<a href="torch/">torch</a>' in html
    assert '<a href="torchaudio/">torchaudio</a>' in html


def test_legacy_package_html_keeps_parent_directory_links() -> None:
    idx = S3Index(
        [_make_obj("v4/whl/torch-2.10.0%2Brocm7.14.0-cp310-cp310-linux_x86_64.whl")],
        prefix="v4/whl",
    )

    html = idx.to_simple_package_html(subdir="v4/whl", package_name="torch")

    assert 'href="../torch-2.10.0%2Brocm7.14.0-cp310-cp310-linux_x86_64.whl"' in html
