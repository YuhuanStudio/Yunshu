"""The idle trim returns the prompt-end replay store's memory."""

import pytest

pytest.importorskip("mlx.core")

from yunshu_engine import idle_memory, prompt_end_replay  # noqa: E402


def test_idle_trim_clears_prompt_end_replay(monkeypatch):
    cleared = []
    monkeypatch.setattr(prompt_end_replay, "resident_bytes", lambda: 1234)
    monkeypatch.setattr(prompt_end_replay, "clear", lambda: cleared.append(True))
    monkeypatch.setattr(
        "yunshu_engine.kernels.buffer_cache.clear_restore_buffers", lambda: None
    )
    monkeypatch.setattr("yunshu_engine.mlx_executor.sync_and_clear_cache", lambda: None)
    monkeypatch.setattr(idle_memory, "_trim_host_allocator", lambda: 0)
    monkeypatch.setattr(idle_memory, "_gib", lambda: (0.0, 0.0))
    monkeypatch.setattr("yunshu_engine.ane_ffn_runtime.release_idle", lambda: False)
    idle_memory._trim()
    assert cleared == [True]
