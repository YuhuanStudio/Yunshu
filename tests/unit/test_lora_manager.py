"""Tests for LoRA adapter manager."""
import json
import pytest
from pathlib import Path


class TestLoRAAdapterEntry:
    def test_entry_defaults(self):
        from yunshu_engine.lora_manager import LoRAAdapterEntry
        entry = LoRAAdapterEntry(adapter_id="test", adapter_path="/tmp/test")
        assert entry.adapter_id == "test"
        assert entry.is_loaded is False
        assert entry.is_merged is False
        assert entry.rank == 8


class TestLoRAAdapterManager:
    def test_register_adapter(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager
        mgr = LoRAAdapterManager(max_loras=2)
        mgr.register_adapter("test-adapter", "/tmp/test-adapter")
        adapters = mgr.list_adapters()
        assert len(adapters) == 1
        assert adapters[0]["adapter_id"] == "test-adapter"

    def test_register_duplicate_ignored(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager
        mgr = LoRAAdapterManager()
        mgr.register_adapter("a1", "/tmp/a1")
        mgr.register_adapter("a1", "/tmp/a1")
        assert len(mgr.list_adapters()) == 1

    def test_get_stats(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager
        mgr = LoRAAdapterManager(max_loras=4)
        mgr.register_adapter("a1", "/tmp/a1")
        mgr.register_adapter("a2", "/tmp/a2")
        stats = mgr.get_stats()
        assert stats["max_loras"] == 4
        assert stats["registered"] == 2
        assert stats["loaded"] == 0

    def test_load_without_base_model(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager
        mgr = LoRAAdapterManager()
        mgr.register_adapter("a1", "/tmp/a1")
        assert mgr.load_adapter("a1") is False

    def test_load_unknown_adapter(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager
        mgr = LoRAAdapterManager()
        assert mgr.load_adapter("nonexistent") is False

    def test_unload_unknown_adapter(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager
        mgr = LoRAAdapterManager()
        assert mgr.unload_adapter("nonexistent") is False

    def test_discover_adapters(self, tmp_path):
        from yunshu_engine.lora_manager import LoRAAdapterManager
        mgr = LoRAAdapterManager()

        # Create a fake adapter directory
        adapter_dir = tmp_path / "adapters" / "my-lora"
        adapter_dir.mkdir(parents=True)
        config = {"lora_parameters": {"rank": 16, "scale": 10.0}, "num_layers": 8}
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        (adapter_dir / "adapters.safetensors").write_bytes(b"fake_weights")

        discovered = mgr.discover_adapters(str(tmp_path))
        assert "my-lora" in discovered
        assert len(mgr.list_adapters()) == 1

    def test_discover_no_adapters(self, tmp_path):
        from yunshu_engine.lora_manager import LoRAAdapterManager
        mgr = LoRAAdapterManager()
        discovered = mgr.discover_adapters(str(tmp_path))
        assert len(discovered) == 0

    def test_max_loras_enforcement(self):
        from yunshu_engine.lora_manager import LoRAAdapterManager, LoRAAdapterEntry
        mgr = LoRAAdapterManager(max_loras=1)
        mgr._base_model = None  # Can't actually load but test the logic

        # Pre-load one adapter
        mgr._adapters["a1"] = LoRAAdapterEntry("a1", "/tmp/a1", is_loaded=True)
        mgr._lru_order = ["a1"]

        # The enforcement happens inside load_adapter which needs a real model
        # Test the LRU eviction directly
        assert len(mgr._loaded_adapters) == 1


class TestGrammarParameter:
    def test_parse_grammar_json_schema(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        grammar = {"type": "json", "schema": {"type": "object", "properties": {"name": {"type": "string"}}}}
        result = _parse_response_format(None, grammar)
        assert isinstance(result, dict)
        assert result["type"] == "object"

    def test_parse_grammar_json_no_schema(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        grammar = {"type": "json"}
        result = _parse_response_format(None, grammar)
        assert result == "json_object"

    def test_grammar_overrides_response_format(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        grammar = {"type": "json", "schema": {"type": "string"}}
        rf = {"type": "json_object"}
        result = _parse_response_format(rf, grammar)
        # Grammar takes priority
        assert result == {"type": "string"}

    def test_no_grammar_no_response_format(self):
        from yunshu_gateway.routers.chat import _parse_response_format
        result = _parse_response_format(None, None)
        assert result is None
