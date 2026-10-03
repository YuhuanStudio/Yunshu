"""Native restore bypasses a detached intermediate copy; stores stay upstream."""

import pytest

mx = pytest.importorskip("mlx.core")


def test_native_restore_bypasses_snapshot_clone_and_store_uses_it(monkeypatch):
    from mlx_vlm import apc_adapters
    from mlx_vlm.models.cache import KVCache

    from yunshu_engine.kernels import cache_restore

    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    original = apc_adapters.clone_cache_entry
    calls = []

    def adapter(cache, **kwargs):
        calls.append(kwargs["min_capacity_tokens"])
        return original(cache, **kwargs)

    monkeypatch.setattr(apc_adapters, "clone_cache_entry", adapter)
    try:
        cache_restore.install()
        installed = apc_adapters.clone_cache_entry
        cache_restore.install()
        assert apc_adapters.clone_cache_entry is installed
        cache = KVCache()
        cache.update_and_fetch(mx.ones((1, 1, 8, 4)), mx.ones((1, 1, 8, 4)))
        targets = []
        restored = installed(cache, min_capacity_tokens=80, eval_targets=targets)
        mx.eval(targets)
        assert calls == []
        assert restored.offset == 8
        assert restored.keys.shape[2] == 256
        restored.keys[..., :8, :] = 9
        assert mx.all(cache.keys[..., :8, :] == 1).item()
        installed(cache, min_capacity_tokens=None, eval_targets=[])
        assert calls == [None]
    finally:
        mx.set_default_device(device)


def test_custom_cache_restore_retains_adapter_contract(monkeypatch):
    from mlx_vlm import apc_adapters

    from yunshu_engine.kernels import cache_restore

    calls = []
    cache = object()
    targets = []

    def adapter(value, **kwargs):
        calls.append((value, kwargs))
        return "custom"

    monkeypatch.setattr(apc_adapters, "clone_cache_entry", adapter)
    cache_restore.install()
    assert (
        apc_adapters.clone_cache_entry(
            cache, min_capacity_tokens=80, eval_targets=targets
        )
        == "custom"
    )
    assert calls == [(cache, {"min_capacity_tokens": 80, "eval_targets": targets})]
