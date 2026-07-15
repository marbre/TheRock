# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.python_indexes import (
    PythonIndexOwner,
    build_python_index_manifest,
    product_index_relative_path,
    render_python_index_manifest_yaml,
    validate_product_local_index_tree,
)


def test_product_index_relative_path_uses_normalized_package_directory() -> None:
    assert (
        product_index_relative_path(
            product="pytorch",
            index_name="whl-next",
            filename="amd_torch_device_gfx942-2.10.0+rocm7.14.0-py3-none-linux_x86_64.whl",
        )
        == "rocm/pytorch/whl-next/amd-torch-device-gfx942/"
        "amd_torch_device_gfx942-2.10.0+rocm7.14.0-py3-none-linux_x86_64.whl"
    )


def test_product_index_relative_path_accepts_nested_product_path() -> None:
    assert (
        product_index_relative_path(
            product="extras/rocoptiq",
            index_name="whl",
            filename="rocoptiq-1.0.0-py3-none-any.whl",
        )
        == "rocm/extras/rocoptiq/whl/rocoptiq/rocoptiq-1.0.0-py3-none-any.whl"
    )


def test_build_python_index_manifest_normalizes_and_sorts_packages() -> None:
    manifest = build_python_index_manifest(
        [
            PythonIndexOwner(
                public_base="/rocm/whl",
                owner_path="core/whl",
                packages=frozenset({"rocm_sdk_core", "ROCm.SDK"}),
            ),
            PythonIndexOwner(
                public_base="/rocm/whl-next",
                owner_path="pytorch/whl-next",
                packages=frozenset({"torch"}),
            ),
        ]
    )

    assert manifest == {
        "python_indexes": [
            {
                "public_base": "/rocm/whl",
                "packages": {
                    "rocm-sdk": {"owner_path": "core/whl"},
                    "rocm-sdk-core": {"owner_path": "core/whl"},
                },
            },
            {
                "public_base": "/rocm/whl-next",
                "packages": {
                    "torch": {"owner_path": "pytorch/whl-next"},
                },
            },
        ]
    }


def test_render_python_index_manifest_yaml_is_infra_loadable_shape() -> None:
    manifest = build_python_index_manifest(
        [
            PythonIndexOwner(
                public_base="/rocm/whl",
                owner_path="core/whl",
                packages=frozenset({"rocm-sdk-core"}),
            )
        ]
    )

    text = render_python_index_manifest_yaml(manifest)

    assert yaml.safe_load(text) == manifest
    assert "python_indexes:" in text
    assert "owner_path: core/whl" in text


def test_build_python_index_manifest_rejects_duplicate_owners() -> None:
    with pytest.raises(
        ValueError, match="assigned to both 'core/whl' and 'pytorch/whl'"
    ):
        build_python_index_manifest(
            [
                PythonIndexOwner(
                    public_base="/rocm/whl",
                    owner_path="core/whl",
                    packages=frozenset({"torch"}),
                ),
                PythonIndexOwner(
                    public_base="/rocm/whl",
                    owner_path="pytorch/whl",
                    packages=frozenset({"torch"}),
                ),
            ]
        )


def test_validate_product_local_index_tree_rejects_undeclared_package(
    tmp_path: Path,
) -> None:
    _write_package_index(tmp_path, "core/whl", "rocm-sdk-core")
    _write_package_index(tmp_path, "core/whl", "rocm-sdk-devel")
    manifest = build_python_index_manifest(
        [
            PythonIndexOwner(
                public_base="/rocm/whl",
                owner_path="core/whl",
                packages=frozenset({"rocm-sdk-core"}),
            )
        ]
    )

    with pytest.raises(ValueError, match="undeclared product-local package"):
        validate_product_local_index_tree(tmp_path, manifest)


def test_validate_product_local_index_tree_rejects_missing_package_index(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "rocm" / "core" / "whl" / "rocm-sdk-core"
    package_dir.mkdir(parents=True)
    manifest = build_python_index_manifest(
        [
            PythonIndexOwner(
                public_base="/rocm/whl",
                owner_path="core/whl",
                packages=frozenset({"rocm-sdk-core"}),
            )
        ]
    )

    with pytest.raises(ValueError, match="missing product-local package index"):
        validate_product_local_index_tree(tmp_path, manifest)


def test_validate_product_local_index_tree_rejects_non_normalized_package_dir(
    tmp_path: Path,
) -> None:
    _write_package_index(tmp_path, "core/whl", "rocm_sdk_core")
    manifest = build_python_index_manifest(
        [
            PythonIndexOwner(
                public_base="/rocm/whl",
                owner_path="core/whl",
                packages=frozenset({"rocm-sdk-core"}),
            )
        ]
    )

    with pytest.raises(ValueError, match="package directory is not normalized"):
        validate_product_local_index_tree(tmp_path, manifest)


def test_validate_product_local_index_tree_rejects_escaping_package_links(
    tmp_path: Path,
) -> None:
    _write_package_index(
        tmp_path,
        "pytorch/whl",
        "torch",
        '<a href="../torch-2.10.0.whl">torch</a>',
    )
    manifest = build_python_index_manifest(
        [
            PythonIndexOwner(
                public_base="/rocm/whl",
                owner_path="pytorch/whl",
                packages=frozenset({"torch"}),
            )
        ]
    )

    with pytest.raises(ValueError, match="escapes package directory"):
        validate_product_local_index_tree(tmp_path, manifest)


def test_validate_product_local_index_tree_accepts_declared_packages(
    tmp_path: Path,
) -> None:
    _write_package_index(tmp_path, "pytorch/whl", "torch")
    manifest = build_python_index_manifest(
        [
            PythonIndexOwner(
                public_base="/rocm/whl",
                owner_path="pytorch/whl",
                packages=frozenset({"torch"}),
            )
        ]
    )

    validate_product_local_index_tree(tmp_path, manifest)


def _write_package_index(
    root: Path,
    owner_path: str,
    package: str,
    html: str | None = None,
) -> None:
    package_dir = root / "rocm" / owner_path / package
    package_dir.mkdir(parents=True)
    if html is None:
        html = f'<a href="{package}-1.0.0-py3-none-any.whl">{package}</a>'
    (package_dir / "index.html").write_text(html, encoding="utf-8")
