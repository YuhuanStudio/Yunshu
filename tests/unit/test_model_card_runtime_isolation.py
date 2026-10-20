"""Runtime facts applied to a loaded card must not leak into the cached static card."""

from __future__ import annotations

import json
from types import SimpleNamespace

from yunshu_engine.model_card import build_model_card


def test_loaded_runtime_facts_do_not_leak_into_later_cards(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
    )
    engine = SimpleNamespace(
        get_stats=lambda: {"apc": {"memory_max_bytes": 32 * 1024**3}}
    )
    loaded = build_model_card(tmp_path, loaded=True, engine=engine)
    assert loaded.prefix_cache.get("memory_max_bytes") == 32 * 1024**3
    unloaded = build_model_card(tmp_path)
    assert unloaded.state["loaded"] is False
    assert unloaded.prefix_cache.get("memory_max_bytes") != 32 * 1024**3
    assert not unloaded.prefix_cache.get("active")
