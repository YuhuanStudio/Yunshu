"""No launchd job may be killed merely because it has the product label."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "deploy_service",
    Path(__file__).resolve().parents[2] / "scripts/research/deploy27b_service.py",
)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def test_owned_pid_requires_our_executable():
    assert (
        service.require_owned(
            "program = /Volumes/P5Plus/test/bin/python\n pid = 4321\n",
            Path("/Volumes/P5Plus/test/bin/python"),
        )
        == 4321
    )
    with pytest.raises(RuntimeError):
        service.require_owned(
            "program = /usr/bin/python\n pid = 4321\n",
            Path("/Volumes/P5Plus/test/bin/python"),
        )
    with pytest.raises(RuntimeError):
        service.require_owned(
            "program = /Volumes/P5Plus/test/bin/python\n",
            Path("/Volumes/P5Plus/test/bin/python"),
        )


def test_deployment_stage_is_opt_in_and_rejects_false_complete(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify.stages import _deployment_valid
    from verify.suites import LADDER, STAGES

    assert "deployment" in STAGES and "service_deploy" in STAGES
    assert "deployment" not in LADDER and "service_deploy" not in LADDER
    path = tmp_path / "result.jsonl"
    path.write_text('{"complete":true,"passed":false}\n')
    assert _deployment_valid(path)[0] is False
    path.write_text('{"complete":true,"passed":true}\n')
    assert _deployment_valid(path)[0] is True
