"""The idle trim returns the prompt-end replay store's memory."""

import pytest

pytest.importorskip("mlx.core")

from yunshu_engine import idle_memory  # noqa: E402
from yunshu_engine.round_driver import replay as driver_replay  # noqa: E402


def test_idle_trim_clears_driver_replay(monkeypatch):
    cleared = []
    monkeypatch.setattr(driver_replay, "resident_bytes", lambda: 1234)
    monkeypatch.setattr(driver_replay, "clear", lambda: cleared.append(True))
    monkeypatch.setattr(
        "yunshu_engine.kernels.buffer_cache.clear_restore_buffers", lambda: None
    )
    monkeypatch.setattr("yunshu_engine.mlx_executor.sync_and_clear_cache", lambda: None)
    monkeypatch.setattr(idle_memory, "_trim_host_allocator", lambda: 0)
    monkeypatch.setattr(idle_memory, "_gib", lambda: (0.0, 0.0))
    monkeypatch.setattr("yunshu_engine.ane_ffn_runtime.release_idle", lambda: False)
    idle_memory._trim()
    assert cleared == [True]


def test_idle_trim_drops_replay_blocks_before_clearing_the_pool(monkeypatch):
    """Dropped replay blocks land in MLX's freed-buffer pool: clearing the pool first left them there
    (release gate 0.1.6 soak: 502 MiB of replay blocks released after the clear, idle cache 505 MiB)."""
    calls = []
    monkeypatch.setattr(driver_replay, "resident_bytes", lambda: 1234)
    monkeypatch.setattr(driver_replay, "clear", lambda: calls.append("replay"))
    monkeypatch.setattr(
        "yunshu_engine.kernels.buffer_cache.clear_restore_buffers",
        lambda: calls.append("spares"),
    )
    monkeypatch.setattr(
        "yunshu_engine.mlx_executor.sync_and_clear_cache", lambda: calls.append("mlx")
    )
    monkeypatch.setattr(idle_memory, "_trim_host_allocator", lambda: 0)
    monkeypatch.setattr(idle_memory, "_gib", lambda: (0.0, 0.0))
    monkeypatch.setattr(
        "yunshu_engine.ane_ffn_runtime.release_idle", lambda: calls.append("ane")
    )
    idle_memory._trim()
    assert calls.index("replay") < calls.index("mlx")
    assert calls.index("ane") < calls.index("mlx")
