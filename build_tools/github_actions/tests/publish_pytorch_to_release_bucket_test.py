#!/usr/bin/env python
"""Unit tests for publish_pytorch_to_release_bucket.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

from github_actions.publish_pytorch_to_release_bucket import main


class TestPublishPytorchToReleaseBucket(unittest.TestCase):
    """Tests for the main() CLI entry point."""

    def setUp(self):
        # Real directory so the script's existence check passes; the
        # upload itself is mocked so no S3 contact happens.
        self._tmp = tempfile.TemporaryDirectory()
        self.source_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    @mock.patch("github_actions.publish_pytorch_to_release_bucket.gha_set_output")
    def test_dev_uploads_to_v4_whl_in_dev_python(self, mock_set_output, mock_upload):
        mock_upload.return_value = 3
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "dev",
                "--dry-run",
            ]
        )

        self.assertEqual(mock_upload.call_count, 1)
        call_args = mock_upload.call_args
        source, dest = call_args.args
        self.assertEqual(source, self.source_dir)
        self.assertEqual(dest.bucket, "therock-dev-python")
        self.assertEqual(dest.relative_path, "v4/whl")
        self.assertEqual(call_args.kwargs.get("include"), ["*.whl"])
        mock_set_output.assert_called_once_with(
            {"package_index_url": "https://rocm.devreleases.amd.com/whl-multi-arch/"}
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_nightly_selects_nightly_bucket(self, mock_upload):
        mock_upload.return_value = 2
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "nightly",
                "--dry-run",
            ]
        )

        _source, dest = mock_upload.call_args.args
        self.assertEqual(dest.bucket, "therock-nightly-python")
        self.assertEqual(dest.relative_path, "v4/whl")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_prerelease_selects_prerelease_bucket(self, mock_upload):
        mock_upload.return_value = 2
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "prerelease",
                "--dry-run",
            ]
        )

        _source, dest = mock_upload.call_args.args
        self.assertEqual(dest.bucket, "therock-prerelease-python")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_files")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    @mock.patch("github_actions.publish_pytorch_to_release_bucket.gha_set_output")
    def test_can_publish_pytorch_stream_index(
        self, mock_set_output, mock_upload_directory, mock_upload_files
    ):
        (
            self.source_dir / "torch-2.10.0+rocm7.14.0-cp312-cp312-linux_x86_64.whl"
        ).write_text(
            "wheel",
            encoding="utf-8",
        )
        (
            self.source_dir
            / "amd_torch_device_gfx942-2.10.0+rocm7.14.0-py3-none-linux_x86_64.whl"
        ).write_text("wheel", encoding="utf-8")
        mock_upload_directory.return_value = 2
        mock_upload_files.return_value = 2
        manifest_path = self.source_dir / "python-indexes.yaml"

        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "nightly",
                "--python-publish-target",
                "rfc0012",
                "--python-index",
                "whl-next",
                "--python-index-manifest-output",
                os.fspath(manifest_path),
                "--dry-run",
            ]
        )

        mock_upload_directory.assert_not_called()

        uploaded_pairs = mock_upload_files.call_args.args[0]
        self.assertEqual(
            [(src.name, dst.bucket, dst.relative_path) for src, dst in uploaded_pairs],
            [
                (
                    "amd_torch_device_gfx942-2.10.0+rocm7.14.0-py3-none-linux_x86_64.whl",
                    "therock-repo-amd-nightly",
                    "rocm/pytorch/whl-next/amd-torch-device-gfx942/"
                    "amd_torch_device_gfx942-2.10.0+rocm7.14.0-py3-none-linux_x86_64.whl",
                ),
                (
                    "torch-2.10.0+rocm7.14.0-cp312-cp312-linux_x86_64.whl",
                    "therock-repo-amd-nightly",
                    "rocm/pytorch/whl-next/torch/"
                    "torch-2.10.0+rocm7.14.0-cp312-cp312-linux_x86_64.whl",
                ),
            ],
        )
        self.assertEqual(
            mock_set_output.call_args.args[0]["package_index_url"],
            "https://nightly.repo.amd.com/rocm/whl-next/",
        )

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["python_indexes"][0]["packages"],
            {
                "amd-torch-device-gfx942": {"owner_path": "pytorch/whl-next"},
                "torch": {"owner_path": "pytorch/whl-next"},
            },
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_raises_when_no_wheels_uploaded(self, mock_upload):
        mock_upload.return_value = 0
        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--source-dir",
                    os.fspath(self.source_dir),
                    "--release-type",
                    "dev",
                    "--dry-run",
                ]
            )

    def test_raises_when_source_dir_missing(self):
        missing = self.source_dir / "does-not-exist"
        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--source-dir",
                    os.fspath(missing),
                    "--release-type",
                    "dev",
                    "--dry-run",
                ]
            )

    def test_invalid_release_type_rejected(self):
        with self.assertRaises(SystemExit):
            main(
                [
                    "--source-dir",
                    os.fspath(self.source_dir),
                    "--release-type",
                    "release",
                    "--dry-run",
                ]
            )


if __name__ == "__main__":
    unittest.main()
