"""The m5_gpu marker is registered, and the gate decides M5 from the GPU architecture string alone."""

import pytest
from m5_gpu_gate import m5_gpu_allowed


def test_the_m5_gpu_marker_is_registered(pytestconfig):
    markers = pytestconfig.getini("markers")
    assert any(line.startswith("m5_gpu:") for line in markers)


@pytest.mark.parametrize(
    "architecture", ["applegpu_g15p", "applegpu_g16g", "applegpu_g14s", "", "garbage"]
)
def test_non_m5_architectures_do_not_run_m5_tests(architecture):
    assert m5_gpu_allowed(architecture) is False


@pytest.mark.parametrize(
    "architecture", ["applegpu_g17p", "applegpu_g17g", "applegpu_g18s"]
)
def test_m5_architectures_run_m5_tests(architecture):
    assert m5_gpu_allowed(architecture) is True


def test_the_env_override_wins_over_the_architecture():
    assert m5_gpu_allowed("applegpu_g15p", override="1") is True
    assert m5_gpu_allowed("applegpu_g17p", override="0") is False
