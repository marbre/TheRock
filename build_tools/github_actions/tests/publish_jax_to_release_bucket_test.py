#!/usr/bin/env python
"""Unit tests for publish_jax_to_release_bucket.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

from github_actions.publish_jax_to_release_bucket import main


class TestPublishJaxToReleaseBucket(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.source_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    @mock.patch("github_actions.publish_jax_to_release_bucket.gha_set_output")
    def test_dev_uploads_to_v4_whl_in_dev_python(self, mock_set_output, mock_upload):
        mock_upload.return_value = 2
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "dev",
                "--dry-run",
            ]
        )

        source, dest = mock_upload.call_args.args
        self.assertEqual(source, self.source_dir)
        self.assertEqual(dest.bucket, "therock-dev-python")
        self.assertEqual(dest.relative_path, "v4/whl")
        mock_set_output.assert_called_once_with(
            {"package_index_url": "https://rocm.devreleases.amd.com/whl-multi-arch/"}
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_files")
    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    @mock.patch("github_actions.publish_jax_to_release_bucket.gha_set_output")
    def test_can_publish_jax_stream_index(
        self, mock_set_output, mock_upload_directory, mock_upload_files
    ):
        (self.source_dir / "jax-0.6.0-py3-none-any.whl").write_text(
            "wheel",
            encoding="utf-8",
        )
        (
            self.source_dir / "jax_rocm7_plugin-0.6.0-cp312-cp312-linux_x86_64.whl"
        ).write_text(
            "wheel",
            encoding="utf-8",
        )
        mock_upload_directory.return_value = 2
        mock_upload_files.return_value = 2
        manifest_path = self.source_dir / "python-indexes.yaml"

        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "dev",
                "--python-publish-target",
                "rfc0012",
                "--python-index",
                "whl",
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
                    "jax-0.6.0-py3-none-any.whl",
                    "therock-repo-amd-dev",
                    "rocm/jax/whl/jax/jax-0.6.0-py3-none-any.whl",
                ),
                (
                    "jax_rocm7_plugin-0.6.0-cp312-cp312-linux_x86_64.whl",
                    "therock-repo-amd-dev",
                    "rocm/jax/whl/jax-rocm7-plugin/"
                    "jax_rocm7_plugin-0.6.0-cp312-cp312-linux_x86_64.whl",
                ),
            ],
        )
        self.assertEqual(
            mock_set_output.call_args.args[0]["package_index_url"],
            "https://dev.repo.amd.com/rocm/whl/",
        )

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["python_indexes"][0]["packages"],
            {
                "jax": {"owner_path": "jax/whl"},
                "jax-rocm7-plugin": {"owner_path": "jax/whl"},
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


if __name__ == "__main__":
    unittest.main()
