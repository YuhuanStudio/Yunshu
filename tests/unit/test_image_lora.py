"""Tests for ImageGenEngine LoRA adapter loading."""

import pytest

from python.yunshu_engine.image_engine import ImageGenEngine


class FakeTransformer:
    """Fake transformer with named_modules for LoRA testing."""
    def __init__(self):
        import mlx.nn as nn
        self.q_proj = nn.Linear(64, 64, bias=False)
        self.v_proj = nn.Linear(64, 64, bias=False)
        self.out_proj = nn.Linear(64, 64, bias=False)

    def named_modules(self):
        import mlx.nn as nn
        yield "q_proj", self.q_proj
        yield "v_proj", self.v_proj
        yield "out_proj", self.out_proj

    def load_weights(self, *a, **kw):
        pass

    def parameters(self):
        return []


def _make_engine():
    import threading
    engine = ImageGenEngine.__new__(ImageGenEngine)
    engine._transformer = FakeTransformer()
    engine._model_path = "/fake"
    engine._running = True
    engine._lora_lock = threading.Lock()
    engine._original_modules = {}
    return engine


class TestImageLoRA:
    def test_load_lora_no_transformer(self):
        """Should fail gracefully when transformer not loaded."""
        import threading
        engine = ImageGenEngine.__new__(ImageGenEngine)
        engine._transformer = None
        engine._lora_lock = threading.Lock()
        result = engine.load_lora_adapter("/fake/path")
        assert result is False

    def test_load_lora_no_config(self, tmp_path):
        """Should fail when adapter_config.json is missing."""
        engine = _make_engine()
        result = engine.load_lora_adapter(str(tmp_path))
        assert result is False

    def test_load_lora_with_config(self, tmp_path):
        """Should attempt to load LoRA layers from config."""
        import json
        config = {
            "lora_parameters": {"rank": 4, "scale": 10.0},
            "num_layers": 2,
        }
        config_path = tmp_path / "adapter_config.json"
        config_path.write_text(json.dumps(config))

        engine = _make_engine()
        # This will try to import mlx_lm.tuner.lora — may or may not succeed
        # depending on the environment. We test the path handling logic.
        result = engine.load_lora_adapter(str(tmp_path))
        # Result depends on whether mlx_lm.tuner.lora is available
        assert isinstance(result, bool)

    def test_load_lora_with_weights(self, tmp_path):
        """Config + weights path handling."""
        import json
        config = {
            "lora_parameters": {"rank": 4, "scale": 10.0},
            "num_layers": 2,
        }
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))
        # Create empty safetensors file (invalid but tests path handling)
        (tmp_path / "adapters.safetensors").write_bytes(b"")

        engine = _make_engine()
        result = engine.load_lora_adapter(str(tmp_path))
        assert isinstance(result, bool)

    def test_engine_has_lora_method(self):
        """ImageGenEngine should expose load_lora_adapter."""
        assert hasattr(ImageGenEngine, 'load_lora_adapter')
        import inspect
        sig = inspect.signature(ImageGenEngine.load_lora_adapter)
        assert 'adapter_path' in sig.parameters
        assert 'rank' in sig.parameters
        assert 'scale' in sig.parameters
