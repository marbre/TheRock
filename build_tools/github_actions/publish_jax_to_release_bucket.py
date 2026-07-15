#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Upload multi-arch JAX wheels from a local directory to a release bucket."""

import argparse
import logging
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.python_indexes import (
    PythonIndexOwner,
    build_product_index_uploads,
    python_index_public_base,
    python_index_public_url,
    write_python_index_manifest,
)
from _therock_utils.s3_buckets import get_release_bucket_config, get_repo_bucket_config
from _therock_utils.storage_backend import create_storage_backend
from _therock_utils.storage_location import StorageLocation
from github_actions.github_actions_api import gha_set_output

logger = logging.getLogger(__name__)

MULTI_ARCH_INDEX_URLS = {
    "dev": "https://rocm.devreleases.amd.com/whl-multi-arch/",
    "nightly": "https://rocm.nightlies.amd.com/whl-multi-arch/",
    "prerelease": "https://rocm.prereleases.amd.com/whl-multi-arch/",
}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Upload multi-arch JAX wheels to a release bucket"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Local directory containing the wheels to upload",
    )
    parser.add_argument(
        "--release-type",
        required=True,
        choices=["dev", "nightly", "prerelease"],
        help="Release type (selects therock-{release_type}-python bucket)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without uploading"
    )
    parser.add_argument(
        "--python-publish-target",
        choices=["legacy", "rfc0012"],
        default="legacy",
        help=(
            "Destination for Python wheels. 'legacy' keeps the existing "
            "therock-{release_type}-python/v4/whl layout. 'rfc0012' publishes "
            "the RFC0012 stream-subdomain product-local layout."
        ),
    )
    parser.add_argument(
        "--python-index",
        choices=["whl", "whl-next"],
        default="whl",
        help="Stream-subdomain Python index name when --python-publish-target=rfc0012.",
    )
    parser.add_argument(
        "--python-index-manifest-output",
        type=Path,
        help="Optional path to write a concrete Python index ownership manifest.",
    )
    args = parser.parse_args(argv)

    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    backend = create_storage_backend(dry_run=args.dry_run)

    if args.python_publish_target == "legacy":
        bucket = get_release_bucket_config(args.release_type, "python")
        dest = StorageLocation(bucket.name, "v4/whl")

        logger.info("JAX wheels: %s -> %s", args.source_dir, dest.s3_uri)
        count = backend.upload_directory(args.source_dir, dest, include=["*.whl"])
        logger.info("Uploaded %d wheel files", count)
        if count == 0:
            raise FileNotFoundError(f"No wheels found at {args.source_dir}")

        gha_set_output({"package_index_url": MULTI_ARCH_INDEX_URLS[args.release_type]})
        return

    bucket = get_repo_bucket_config(args.release_type)
    uploads, packages = build_product_index_uploads(
        source_dir=args.source_dir,
        dest_bucket=bucket.name,
        product="jax",
        index_name=args.python_index,
    )
    if not uploads:
        raise FileNotFoundError(
            f"No Python distribution files found at {args.source_dir}"
        )

    logger.info(
        "JAX stream index wheels: %s -> s3://%s/rocm/jax/%s/",
        args.source_dir,
        bucket.name,
        args.python_index,
    )
    count = backend.upload_files(uploads)
    logger.info("Uploaded %d stream-index wheel files", count)
    if count == 0:
        raise FileNotFoundError(f"No wheels found at {args.source_dir}")

    owner = PythonIndexOwner(
        public_base=python_index_public_base(args.python_index),
        owner_path=f"jax/{args.python_index}",
        packages=packages,
    )
    if args.python_index_manifest_output:
        write_python_index_manifest(args.python_index_manifest_output, [owner])
    gha_set_output(
        {
            "package_index_url": python_index_public_url(
                args.release_type, args.python_index
            )
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
