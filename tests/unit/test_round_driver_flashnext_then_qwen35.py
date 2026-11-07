"""Switching models in one process: a Flash-Next DFlash setup made while the default device is the CPU must
not decide the GPU kernels of the model loaded after it.

Cause (found with a before / after snapshot of every module-level value): ``lane_qmm._resolve_variant`` and
``lane_qmm.ready`` cache "which lane variant / is the lane certified" on first call from
``mx.device_info()`` and a self-test kernel run, both of which read the *default device*. The first call of a
process came from the Flash-Next DFlash batch (drafter projections) under a CPU default device, so the
process cached ``portable`` / ``not ready`` and the Qwen3.5 lane, APC hit == cold and DFlash tests that ran
next used another arithmetic. Fix: kernels/gpu_probe.py (every such probe runs under the GPU stream)."""

# ruff: noqa: F811  (imported fixtures are used as test arguments)

import contextlib
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.models.qwen4_exp.language")

from tests.unit.test_round_driver_apc import (  # noqa: E402,F401
    test_a_prompt_without_media_tokens_skips_the_media_scans as _apc_hit_equals_cold,
)
from tests.unit.test_round_driver_apc import tiny  # noqa: E402,F401
from tests.unit.test_round_driver_flashnext import CHUNK, PROMPTS, make_tiny  # noqa: E402
from tests.unit.test_round_driver_flashnext_dflash import make_dflash  # noqa: E402
from yunshu_engine.kernels import gpu_probe  # noqa: E402
from yunshu_engine.kernels.tensorfold import lane_qmm  # noqa: E402
from yunshu_engine.round_driver.driver import Request, RoundDriver  # noqa: E402


@contextlib.contextmanager
def cpu_default():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _gpu_variant() -> str:
    arch = str(gpu_probe.device_info().get("architecture", ""))
    generation = lane_qmm._generation(arch)
    return "m5" if generation in (17, 18) else "portable"


def test_a_flash_next_dflash_setup_on_the_cpu_leaves_the_gpu_probes_alone():
    # a fresh process: nothing probed yet
    lane_qmm._VARIANT = None
    lane_qmm._READY.clear()
    restore = None
    try:
        with cpu_default():
            lm, _ = make_tiny()
            draft, restore = make_dflash(lm)
            driver = RoundDriver(
                SimpleNamespace(language_model=lm),
                dflash_drafter=draft,
                stop_tokens=set(),
                chunk=CHUNK,
            )
            driver.add(Request(PROMPTS[0], 6, handle=0))
            for _ in range(100):
                if not driver.busy():
                    break
                driver.step()
            # whatever probed during that setup, the cached answers are the GPU's
            variant = lane_qmm._resolve_variant()
            ready = lane_qmm.ready()
        assert variant == _gpu_variant()
        if _gpu_variant() == "m5":
            assert ready is True
    finally:
        if restore is not None:
            restore()


def test_probes_ignore_the_default_device():
    """The cached answers equal the GPU's even when asked first, and only, from a CPU default device."""
    lane_qmm._VARIANT = None
    lane_qmm._READY.clear()
    with cpu_default():
        assert lane_qmm._resolve_variant() == _gpu_variant()
        first = lane_qmm.ready()
    lane_qmm._VARIANT = None
    lane_qmm._READY.clear()
    assert lane_qmm.ready() is first  # the same answer asked from the GPU default
    assert lane_qmm._resolve_variant() == _gpu_variant()


def test_c_qwen35_apc_hit_equals_cold_after_a_flash_next_dflash_setup(
    tiny, monkeypatch
):
    """The Qwen3.5 test that used to fail here (token 11 differed between the hit and the cold run)."""
    _apc_hit_equals_cold(tiny, monkeypatch)
