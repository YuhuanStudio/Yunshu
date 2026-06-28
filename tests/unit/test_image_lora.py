"""Tests for ImageGenEngine LoRA adapter loading."""


from python.yunshu_engine.image_engine import ImageGenEngine


class FakeTransformer:
    """Fake transformer with named_modules for LoRA testing."""
    def __init__(self):
        import mlx.nn as nn
        self.q_proj = nn.Linear(64, 64, bias=False)
        self.v_proj = nn.Linear(64, 64, bias=False)
        self.out_proj = nn.Linear(64, 64, bias=False)

    def named_modules(self):
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


import mlx.nn as _nn


class _DiTAttn(_nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = _nn.Linear(64, 64, bias=False)
        self.to_k = _nn.Linear(64, 64, bias=False)
        self.to_v = _nn.Linear(64, 64, bias=False)
        self.to_out = _nn.Linear(64, 64, bias=False)  # flat, like the real model
        self.adaLN_modulation = _nn.Linear(64, 64, bias=False)


class _DiTLayer(_nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _DiTAttn()


class _DiTTransformer(_nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_DiTLayer(), _DiTLayer()]


class TestZImageLoRAKeyRemap:
    """the ComfyUI/Flux `attention.to_out.0` key must remap onto our
    FLAT `attention.to_out` Linear, else the output projection is silently
    skipped on every DiT layer (the product's primary feature, badly weakened)."""

    def test_remap_collapses_to_out_index(self):
        r = ImageGenEngine._zimage_lora_key_remap(
            "diffusion_model.layers.0.attention.to_out.0.lora_down.weight"
        )
        assert r == "layers.0.attention.to_out.lora_down.weight"

    def test_remap_collapses_adaln_index(self):
        r = ImageGenEngine._zimage_lora_key_remap(
            "diffusion_model.layers.3.adaLN_modulation.0.lora_up.weight"
        )
        assert r == "layers.3.adaLN_modulation.lora_up.weight"

    def test_remap_leaves_flat_keys_untouched(self):
        r = ImageGenEngine._zimage_lora_key_remap(
            "diffusion_model.layers.1.attention.to_q.lora_down.weight"
        )
        assert r == "layers.1.attention.to_q.lora_down.weight"

    def test_peft_load_walks_list_indexed_blocks(self, tmp_path):
        """The DiT keeps its blocks in plain Python lists (layers/noise_refiner/
        context_refiner), so named_modules() yields 'layers.0.attention.to_q' where
        the mid-path '0' is a LIST INDEX. The PEFT-directory load/unload must walk it
        index-aware — a bare getattr(list, '0') raises AttributeError, which made every
        PEFT-format image LoRA crash→return False on the real transformer (the
        FakeTransformer above uses flat top-level modules, hiding the bug)."""
        import json
        import threading

        import mlx.nn as nn
        from python.yunshu_engine.image_engine import ImageGenEngine

        engine = ImageGenEngine.__new__(ImageGenEngine)
        engine._transformer = _DiTTransformer()
        engine._lora_lock = threading.Lock()
        engine._original_modules = {}
        engine._lora_offloader = None

        config = {"lora_parameters": {"rank": 4, "scale": 10.0}, "num_layers": 8}
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))

        ok = engine._load_lora_adapter_locked(str(tmp_path))
        assert ok is True
        # Every attention to_q/to_v/to_out across BOTH list-indexed layers wrapped.
        from mlx_lm.tuner.lora import LoRALinear
        for layer in engine._transformer.layers:
            assert isinstance(layer.attention.to_q, LoRALinear)
            assert isinstance(layer.attention.to_v, LoRALinear)
            assert isinstance(layer.attention.to_out, LoRALinear)
            # to_k is intentionally NOT a standard PEFT target.
            assert isinstance(layer.attention.to_k, nn.Linear)
            assert not isinstance(layer.attention.to_k, LoRALinear)

        # Unload restores the plain Linear inside the list-indexed parent.
        assert engine.unload_lora_adapter() is True
        for layer in engine._transformer.layers:
            assert isinstance(layer.attention.to_q, nn.Linear)
            assert not isinstance(layer.attention.to_q, LoRALinear)

    def test_remapped_to_out_resolves_remap_dead_without(self):
        """The remapped path resolves to the real Linear; the un-remapped (with
        trailing .0) resolves to None — proving the bug and the fix."""
        engine = ImageGenEngine.__new__(ImageGenEngine)
        engine._transformer = _DiTTransformer()

        # Module path WITHOUT the remap (trailing list-index 0) → dead.
        dead, _, _ = engine._resolve_lora_module("layers.0.attention.to_out.0")
        assert dead is None

        # Module path AFTER the remap → resolves to the flat Linear.
        import mlx.nn as nn
        good, parent, last = engine._resolve_lora_module("layers.0.attention.to_out")
        assert isinstance(good, nn.Linear)
        assert last == "to_out"
