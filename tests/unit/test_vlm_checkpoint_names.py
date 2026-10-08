"""Mixed precision must follow sanitized checkpoint names, before filtering."""

from types import SimpleNamespace

from yunshu_engine.vlm_engine import _prepare_vlm_weights, _vlm_module_quantization


def test_sanitizes_before_quantization_keys_are_used():
    class Model:
        def sanitize(self, weights):
            return {
                f"language_model.{k}": v
                for k, v in weights.items()
                if not k.startswith("mtp.")
            }

    weights = _prepare_vlm_weights(
        Model(),
        SimpleNamespace(),
        SimpleNamespace(),
        {"model.layers.0.ffn.up_proj.scales": "scales", "mtp.0.weight": "draft"},
    )
    assert weights == {"language_model.model.layers.0.ffn.up_proj.scales": "scales"}
    q = {"bits": 2, "model.layers.0.ffn.up_proj": {"bits": 8, "group_size": 64}}
    assert _vlm_module_quantization(
        Model(), q, "language_model.model.layers.0.ffn.up_proj"
    ) == {"bits": 8, "group_size": 64}


def test_exact_name_wins_and_false_override_is_preserved():
    q = {"language_model.lm_head": False, "lm_head": {"bits": 8}}
    assert _vlm_module_quantization(object(), q, "language_model.lm_head") is False
    assert _vlm_module_quantization(object(), q, "language_model.missing") is None


def test_model_supplied_legacy_alias():
    model = SimpleNamespace(quantization_path_aliases=lambda p: ("head",))
    assert _vlm_module_quantization(
        model, {"head": {"bits": 8}}, "language_model.lm_head"
    ) == {"bits": 8}


def test_vlm_only_language_model_routes_to_shared_runner(tmp_path, monkeypatch):
    import json

    from yunshu_engine import model_manager

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "new_text_moe"}))

    def imports(name):
        if name == "mlx_lm.models.new_text_moe":
            raise ImportError(name)
        return object()

    monkeypatch.setattr(model_manager.importlib, "import_module", imports)
    assert (
        model_manager._detect_model_type(str(tmp_path)) == model_manager.ModelType.VLM
    )
    monkeypatch.setattr(model_manager.importlib, "import_module", lambda _: object())
    assert (
        model_manager._detect_model_type(str(tmp_path)) == model_manager.ModelType.LLM
    )
