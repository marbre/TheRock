#!/usr/bin/env python
"""Regression tests for RFC0012 Python index workflow wiring."""

from pathlib import Path


THEROCK_DIR = Path(__file__).resolve().parents[3]
WORKFLOWS_DIR = THEROCK_DIR / ".github" / "workflows"
ACTION_PATH = (
    THEROCK_DIR
    / ".github"
    / "actions"
    / "configure_aws_artifacts_credentials"
    / "action.yml"
)


def test_configure_aws_action_accepts_rfc0012_scope() -> None:
    text = ACTION_PATH.read_text(encoding="utf-8")

    assert "outputs:" in text
    assert "bucket:" in text
    assert "${{ steps.iam.outputs.bucket }}" in text
    assert "bucket_scope:" in text
    assert '--bucket-scope "${{ inputs.bucket_scope }}"' in text


def test_rocm_release_workflows_publish_python_only_to_rfc0012() -> None:
    top_level_text = (WORKFLOWS_DIR / "multi_arch_release.yml").read_text(
        encoding="utf-8"
    )
    assert "rfc0012_python_index:" in top_level_text
    assert (
        "rfc0012_python_index: ${{ inputs.rfc0012_python_index || 'whl' }}"
        in top_level_text
    )

    for workflow_name in (
        "multi_arch_release_linux.yml",
        "multi_arch_release_windows.yml",
    ):
        text = (WORKFLOWS_DIR / workflow_name).read_text(encoding="utf-8")

        assert "publish_rfc0012_python_indexes:" not in text
        assert "--skip-python-packages" in text
        assert "bucket_scope: rfc0012" in text
        assert "id: rfc0012_aws" in text
        assert '--python-publish-target="rfc0012"' in text
        assert "--python-index-manifest-output=" in text
        assert "actions/upload-artifact@" in text
        assert "build_tools/third_party/s3_management/manage.py" in text
        assert '"rocm/core/${{ inputs.rfc0012_python_index }}"' in text
        assert '--bucket="${{ steps.rfc0012_aws.outputs.bucket }}"' in text
        assert "--structured-layout" in text


def test_pytorch_and_jax_workflows_publish_rfc0012_python() -> None:
    for workflow_name in (
        "multi_arch_build_portable_linux_pytorch_wheels.yml",
        "multi_arch_build_windows_pytorch_wheels.yml",
        "multi_arch_build_linux_jax_wheels.yml",
    ):
        text = (WORKFLOWS_DIR / workflow_name).read_text(encoding="utf-8")

        assert "python_index:" in text
        assert "bucket_scope: rfc0012" in text
        assert "id: rfc0012_aws" in text
        assert '--python-publish-target="rfc0012"' in text
        assert '--python-index="${{ inputs.python_index }}"' in text
        assert "--python-index-manifest-output=" in text
        assert "actions/upload-artifact@" in text
        assert "build_tools/third_party/s3_management/manage.py" in text
        assert '--bucket="${{ steps.rfc0012_aws.outputs.bucket }}"' in text
        assert "--structured-layout" in text
