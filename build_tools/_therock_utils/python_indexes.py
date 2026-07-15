# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Helpers for ROCm stream-subdomain Python package indexes."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Mapping, Sequence

from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    parse_sdist_filename,
    parse_wheel_filename,
)
import yaml

from _therock_utils.storage_location import StorageLocation

ACCEPTED_PYTHON_DISTRIBUTION_SUFFIXES = (".whl", ".tar.gz", ".zip")
ALLOWED_PUBLIC_BASES = frozenset({"/rocm/whl", "/rocm/whl-next"})
STREAM_REPO_HOSTS = {
    "dev": "dev.repo.amd.com",
    "nightly": "nightly.repo.amd.com",
    "prerelease": "rc.repo.amd.com",
}

_NORMALIZE_RE = re.compile(r"[-_.]+")
_NORMALIZED_PACKAGE_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class _PackageIndexLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {name.lower(): value for name, value in attrs}
        href = attr_map.get("href")
        if href:
            self.hrefs.append(href)


@dataclass(frozen=True)
class PythonIndexOwner:
    public_base: str
    owner_path: str
    packages: frozenset[str]


def pep503_normalize_package_name(name: str) -> str:
    """Return the PEP 503-normalized package name."""
    return _NORMALIZE_RE.sub("-", name).lower()


def package_name_from_distribution_filename(filename: str) -> str:
    """Extract the normalized package name from a wheel or sdist filename."""
    basename = Path(filename).name
    if basename.endswith(".whl"):
        try:
            package_name, _version, _build, _tags = parse_wheel_filename(basename)
        except InvalidWheelFilename as exc:
            raise ValueError(f"invalid wheel filename: {filename}") from exc
        return pep503_normalize_package_name(str(package_name))

    if basename.endswith((".tar.gz", ".zip")):
        try:
            package_name, _version = parse_sdist_filename(basename)
        except InvalidSdistFilename as exc:
            raise ValueError(f"invalid sdist filename: {filename}") from exc
        return pep503_normalize_package_name(str(package_name))

    raise ValueError(f"unsupported Python distribution file: {filename}")


def is_python_distribution_filename(filename: str) -> bool:
    return Path(filename).name.endswith(ACCEPTED_PYTHON_DISTRIBUTION_SUFFIXES)


def product_index_relative_path(
    *,
    product: str,
    index_name: str,
    filename: str,
) -> str:
    """Return the repo-bucket object key for a product-local package file."""
    product = _validate_relative_path(product, "product")
    if index_name not in {"whl", "whl-next"}:
        raise ValueError(f"unsupported Python index: {index_name}")
    basename = Path(filename).name
    package = package_name_from_distribution_filename(basename)
    return f"rocm/{product}/{index_name}/{package}/{basename}"


def python_index_public_base(index_name: str) -> str:
    if index_name not in {"whl", "whl-next"}:
        raise ValueError(f"unsupported Python index: {index_name}")
    return f"/rocm/{index_name}"


def python_index_public_url(release_type: str, index_name: str) -> str:
    host = STREAM_REPO_HOSTS.get(release_type)
    if host is None:
        allowed = ", ".join(sorted(STREAM_REPO_HOSTS))
        raise ValueError(
            f"release_type={release_type!r} is invalid, expected one of {allowed}"
        )
    return f"https://{host}{python_index_public_base(index_name)}/"


def iter_python_distribution_files(source_dir: Path) -> list[Path]:
    """Return Python distribution files under source_dir in stable order."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    files: list[Path] = []
    for suffix in ACCEPTED_PYTHON_DISTRIBUTION_SUFFIXES:
        files.extend(source_dir.rglob(f"*{suffix}"))
    return sorted(f for f in files if f.is_file() and not f.is_symlink())


def build_product_index_uploads(
    *,
    source_dir: Path,
    dest_bucket: str,
    product: str,
    index_name: str,
) -> tuple[list[tuple[Path, StorageLocation]], frozenset[str]]:
    pairs: list[tuple[Path, StorageLocation]] = []
    packages: set[str] = set()
    for source in iter_python_distribution_files(source_dir):
        package = package_name_from_distribution_filename(source.name)
        packages.add(package)
        pairs.append(
            (
                source,
                StorageLocation(
                    dest_bucket,
                    product_index_relative_path(
                        product=product,
                        index_name=index_name,
                        filename=source.name,
                    ),
                ),
            )
        )
    return pairs, frozenset(packages)


def build_product_index_copies(
    *,
    source_files: Sequence[StorageLocation],
    dest_bucket: str,
    product: str,
    index_name: str,
) -> tuple[list[tuple[StorageLocation, StorageLocation]], frozenset[str]]:
    pairs: list[tuple[StorageLocation, StorageLocation]] = []
    packages: set[str] = set()
    for source in sorted(source_files, key=lambda loc: loc.relative_path):
        filename = Path(source.relative_path).name
        package = package_name_from_distribution_filename(filename)
        packages.add(package)
        pairs.append(
            (
                source,
                StorageLocation(
                    dest_bucket,
                    product_index_relative_path(
                        product=product,
                        index_name=index_name,
                        filename=filename,
                    ),
                ),
            )
        )
    return pairs, frozenset(packages)


def build_python_index_manifest(
    owners: Sequence[PythonIndexOwner],
) -> dict[str, list[dict[str, object]]]:
    """Build a concrete Python index ownership manifest."""
    by_public_base: dict[str, dict[str, dict[str, str]]] = {}
    for owner in owners:
        public_base = _validate_public_base(owner.public_base)
        owner_path = _validate_owner_path(owner.owner_path, public_base)
        package_map = by_public_base.setdefault(public_base, {})
        for raw_package in owner.packages:
            package = _validate_normalized_package(
                pep503_normalize_package_name(raw_package)
            )
            previous_owner = package_map.get(package, {}).get("owner_path")
            if previous_owner is not None and previous_owner != owner_path:
                raise ValueError(
                    f"package '{package}' is assigned to both "
                    f"'{previous_owner}' and '{owner_path}' in {public_base}"
                )
            package_map[package] = {"owner_path": owner_path}

    indexes: list[dict[str, object]] = []
    for public_base in sorted(by_public_base):
        packages = by_public_base[public_base]
        indexes.append(
            {
                "public_base": public_base,
                "packages": {
                    package: packages[package] for package in sorted(packages)
                },
            }
        )
    return {"python_indexes": indexes}


def render_python_index_manifest_yaml(manifest: Mapping[str, object]) -> str:
    return yaml.safe_dump(dict(manifest), sort_keys=False)


def write_python_index_manifest(
    output_path: Path,
    owners: Sequence[PythonIndexOwner],
) -> None:
    manifest = build_python_index_manifest(owners)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_python_index_manifest_yaml(manifest), encoding="utf-8"
    )


def validate_product_local_index_tree(
    content_root: Path,
    manifest: Mapping[str, object],
) -> None:
    """Validate product-local package directories against a manifest."""
    declared = _declared_owner_packages(manifest)
    errors: list[str] = []
    for owner_path, packages in declared.items():
        owner_root = content_root / "rocm" / owner_path
        if owner_root.is_dir():
            for package_dir in sorted(p for p in owner_root.iterdir() if p.is_dir()):
                expected = pep503_normalize_package_name(package_dir.name)
                if package_dir.name != expected:
                    errors.append(
                        f"package directory is not normalized: {package_dir}; "
                        f"expected {expected}"
                    )
                    continue
                if package_dir.name not in packages:
                    errors.append(f"undeclared product-local package: {package_dir}")

        for package in sorted(packages):
            package_dir = owner_root / package
            index_path = package_dir / "index.html"
            if not index_path.is_file():
                errors.append(f"missing product-local package index: {index_path}")
                continue
            if index_path.stat().st_size == 0:
                errors.append(f"empty product-local package index: {index_path}")
                continue
            errors.extend(_validate_package_index_links(index_path))

    if errors:
        raise ValueError("\n".join(errors))


def _validate_relative_path(value: str, where: str) -> str:
    if value.startswith("/") or value.endswith("/"):
        raise ValueError(f"invalid {where}: {value}")
    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(f"invalid {where}: {value}")
    return value


def _validate_public_base(public_base: str) -> str:
    public_base = public_base.rstrip("/")
    if public_base not in ALLOWED_PUBLIC_BASES:
        allowed = ", ".join(sorted(ALLOWED_PUBLIC_BASES))
        raise ValueError(f"unsupported public_base '{public_base}', expected {allowed}")
    return public_base


def _validate_owner_path(owner_path: str, public_base: str) -> str:
    if owner_path.startswith("/") or owner_path.endswith("/"):
        raise ValueError(f"owner_path '{owner_path}' must be relative")
    segments = owner_path.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(f"owner_path '{owner_path}' contains unsafe segments")
    index_name = public_base.rsplit("/", 1)[-1]
    if segments[-1] != index_name:
        raise ValueError(
            f"owner_path '{owner_path}' must end with '{index_name}' "
            f"for {public_base}"
        )
    return owner_path


def _validate_normalized_package(package: str) -> str:
    if package != pep503_normalize_package_name(package):
        raise ValueError(f"package '{package}' is not normalized")
    if not _NORMALIZED_PACKAGE_RE.fullmatch(package):
        raise ValueError(f"package '{package}' is not a valid normalized name")
    return package


def _declared_owner_packages(manifest: Mapping[str, object]) -> dict[str, set[str]]:
    raw_indexes = manifest.get("python_indexes")
    if not isinstance(raw_indexes, list):
        raise ValueError("python_indexes must be a list")
    declared: dict[str, set[str]] = {}
    for raw_index in raw_indexes:
        if not isinstance(raw_index, Mapping):
            raise ValueError("python index entry must be a mapping")
        public_base = _validate_public_base(str(raw_index.get("public_base", "")))
        raw_packages = raw_index.get("packages")
        if not isinstance(raw_packages, Mapping):
            raise ValueError("packages must be a mapping")
        for raw_package, raw_route in raw_packages.items():
            package = _validate_normalized_package(str(raw_package))
            if not isinstance(raw_route, Mapping):
                raise ValueError(f"route for package '{package}' must be a mapping")
            owner_path = _validate_owner_path(
                str(raw_route.get("owner_path", "")),
                public_base,
            )
            declared.setdefault(owner_path, set()).add(package)
    return declared


def _validate_package_index_links(index_path: Path) -> list[str]:
    parser = _PackageIndexLinkParser()
    parser.feed(index_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for href in parser.hrefs:
        link_path = href.split("#", 1)[0].split("?", 1)[0]
        if not link_path:
            continue
        segments = link_path.split("/")
        if link_path.startswith("/") or "/" in link_path or ".." in segments:
            errors.append(
                f"package index link escapes package directory: {index_path}: {href}"
            )
    return errors
