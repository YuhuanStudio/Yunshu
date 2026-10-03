"""The direct-capacity research arm preserves prefix bytes and cache ownership."""

import importlib.util
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_vlm.apc_adapters import clone_cache_entry  # noqa: E402
from mlx_vlm.models.cache import KVCache  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "restore_probe",
    Path(__file__).resolve().parents[2] / "scripts/research/probe_prefill_restore.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def cpu_arrays():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


@pytest.mark.parametrize("prefix,capacity", [(32, 33), (256, 257), (257, 1025)])
def test_direct_capacity_is_detached_and_matches_adapter(prefix, capacity):
    source = KVCache()
    source.update_and_fetch(
        mx.arange(prefix * 4).reshape(1, 1, prefix, 4).astype(mx.bfloat16),
        mx.ones((1, 1, prefix, 2), mx.bfloat16) * 3,
    )
    targets = []

    def no_exact_clone(*args, **kwargs):
        raise AssertionError("direct capacity restore must bypass exact cloning")

    actual = probe.direct_clone(
        source,
        min_capacity_tokens=capacity,
        eval_targets=targets,
        clone=clone_cache_entry,
    )
    expected = clone_cache_entry(
        source, min_capacity_tokens=capacity, eval_targets=targets
    )
    source.keys[..., :prefix, :] = -1
    mx.eval(targets)
    assert actual.offset == expected.offset == prefix
    assert actual.meta_state == expected.meta_state
    assert actual.keys.shape == expected.keys.shape
    assert mx.array_equal(actual.keys, expected.keys).item()
    assert mx.array_equal(actual.values, expected.values).item()
    actual.values[..., :prefix, :] = 9
    assert mx.all(source.values[..., :prefix, :] == 3).item()


def test_custom_contract_stays_on_adapter():
    source = object()
    calls = []

    def adapter(c, **kwargs):
        calls.append((c, kwargs))
        return "fallback"

    targets = []
    assert (
        probe.direct_clone(
            source, min_capacity_tokens=1024, eval_targets=targets, clone=adapter
        )
        == "fallback"
    )
    assert calls == [(source, dict(min_capacity_tokens=1024, eval_targets=targets))]
