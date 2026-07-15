#!/usr/bin/env python
"""Unit tests for write_artifacts_bucket_info.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import write_artifacts_bucket_info


class TestWriteArtifactsBucketInfo(unittest.TestCase):
    @mock.patch("write_artifacts_bucket_info.gha_set_output")
    def test_default_scope_keeps_release_artifacts_role(self, mock_set_output):
        write_artifacts_bucket_info.main(["--release-type", "dev"])

        mock_set_output.assert_called_once_with(
            {
                "bucket": "therock-dev-artifacts",
                "iam_role": "arn:aws:iam::692859939525:role/therock-dev",
                "aws_region": "us-east-2",
            }
        )

    @mock.patch("write_artifacts_bucket_info.gha_set_output")
    def test_rfc0012_scope_uses_stream_repo_role(self, mock_set_output):
        write_artifacts_bucket_info.main(
            ["--release-type", "nightly", "--bucket-scope", "rfc0012"]
        )

        mock_set_output.assert_called_once_with(
            {
                "bucket": "therock-repo-amd-nightly",
                "iam_role": "arn:aws:iam::692859939525:role/therock-repo-nightly",
                "aws_region": "us-east-2",
            }
        )


if __name__ == "__main__":
    unittest.main()
