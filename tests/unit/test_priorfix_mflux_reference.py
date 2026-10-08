"""CPU reproduction of the published packed Qwen projection loader failure."""

import importlib.util
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def bridge():
    path = Path(__file__).parents[2] / "scripts/research/priorfix_mflux_reference.py"
    spec = importlib.util.spec_from_file_location("mflux_reference_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packed_projection_reproduces_and_decodes_on_cpu():
    import mlx.core as mx

    with mx.stream(mx.cpu):
        weight = mx.array(np.linspace(-1, 1, 128 * 64).reshape(128, 64), mx.float32)
        packed, scales, biases = mx.quantize(weight, group_size=64, bits=4)
        x = mx.ones((1, 4, 64))
        # Same failure as (1,512,2560) @ (320,4096): packed words are not features.
        with pytest.raises(ValueError, match="Last dimension"):
            mx.eval(x @ packed.T)
        raw = {
            "q.weight": packed,
            "q.scales": scales,
            "q.biases": biases,
            "q.bias": mx.ones(128),
        }
        decoded = bridge().decode_affine_weights(raw)
        assert decoded["q.weight"].shape == (128, 64)
        assert decoded["q.bias"] is raw["q.bias"]
        assert "q.scales" not in decoded and "q.biases" not in decoded
        expected = mx.dequantize(packed, scales, biases, group_size=64, bits=4)
        actual = x @ decoded["q.weight"].T
        mx.eval(actual, expected)
        assert actual.shape == (1, 4, 128)
        assert mx.array_equal(actual, x @ expected.T).item()
        assert "q.scales" in raw  # caller input remains intact
        token = bridge().decode_affine_weights(
            {
                "cap_pad_token": packed[:1],
                "cap_pad_token.scales": scales[:1],
                "cap_pad_token.biases": biases[:1],
            }
        )
        assert token["cap_pad_token"].shape == (1, 64)
        assert mx.array_equal(token["cap_pad_token"], expected[:1]).item()
        with pytest.raises(ValueError, match="missing affine"):
            bridge().decode_affine_weights({"q.weight": packed})
        with pytest.raises(ValueError, match="invalid packed shape"):
            bridge().decode_affine_weights({**raw, "q.scales": scales[:, :0]})


def write_header(path, tensors):
    header = json.dumps(tensors).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def test_header_contract_rejects_wrong_hidden_size_and_missing_triplet(tmp_path):
    module = bridge()
    quant = {"bits": 4, "group_size": 64}
    (tmp_path / "quantize_config.json").write_text(json.dumps({"quantization": quant}))
    for name in ("text_encoder", "transformer", "vae"):
        (tmp_path / name).mkdir()
        write_header(tmp_path / name / "model.safetensors", {})
    config = {
        "quantization": quant,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "head_dim": 32,
        "hidden_size": 64,
    }
    config_path = tmp_path / "text_encoder/config.json"
    config_path.write_text(json.dumps(config))
    key = "model.layers.0.self_attn.q_proj"
    tensors = {
        key + ".weight": {"dtype": "U32", "shape": [128, 8]},
        key + ".scales": {"dtype": "F32", "shape": [128, 1]},
        key + ".biases": {"dtype": "F32", "shape": [128, 1]},
    }
    header_path = tmp_path / "text_encoder/model.safetensors"
    write_header(header_path, tensors)
    assert module.checkpoint_contract(tmp_path)["packed_layers"]["text_encoder"] == 1
    config_path.write_text(json.dumps({**config, "hidden_size": 128}))
    with pytest.raises(ValueError, match="projection contract"):
        module.checkpoint_contract(tmp_path)
    config_path.write_text(json.dumps(config))
    tensors.pop(key + ".biases")
    write_header(header_path, tensors)
    with pytest.raises(ValueError, match="affine triplet"):
        module.checkpoint_contract(tmp_path)


def test_reference_bridge_loader_orchestration_without_real_model(
    monkeypatch, tmp_path
):
    module = bridge()
    monkeypatch.setattr(
        module, "checkpoint_contract", lambda path: {"bits": 4, "group_size": 64}
    )
    monkeypatch.setattr(
        module, "decode_affine_weights", lambda weights, **kwargs: {"decoded": weights}
    )
    component = SimpleNamespace(
        name="text_encoder",
        hf_subdir="text_encoder",
        loading_mode="mlx_native",
        weight_files=None,
    )

    @dataclass
    class MetaData:
        quantization_level: int | None = None

    @dataclass
    class LoadedWeights:
        components: dict
        meta_data: MetaData

    class WeightLoader:
        @staticmethod
        def _load_safetensors(path, mode, weight_files):
            assert path == tmp_path / "text_encoder"
            assert mode == "mlx_native" and weight_files is None
            return {"packed": True}

        @staticmethod
        def _load_component(root, comp, raw_weights_cache):
            assert root == tmp_path and comp is component
            key = (str(root / "text_encoder"), "mlx_native", ())
            assert raw_weights_cache[key] == {"decoded": {"packed": True}}
            return {"q_proj": "floating"}, None, None

    model = SimpleNamespace(bits=4)

    class Initializer:
        @staticmethod
        def _apply_weights(obj, weights, quantize):
            assert obj is model and quantize is None
            assert weights.components == {"text_encoder": {"q_proj": "floating"}}
            assert weights.meta_data.quantization_level is None
            obj.bits = None

    modules = {
        "mflux.models.common.weights.loading.loaded_weights": {
            "LoadedWeights": LoadedWeights,
            "MetaData": MetaData,
        },
        "mflux.models.common.weights.loading.weight_loader": {
            "WeightLoader": WeightLoader
        },
        "mflux.models.z_image.variants.z_image": {"ZImage": lambda **kwargs: model},
        "mflux.models.z_image.weights.z_image_weight_definition": {
            "ZImageWeightDefinition": SimpleNamespace(
                get_components=lambda: [component]
            )
        },
        "mflux.models.z_image.z_image_initializer": {"ZImageInitializer": Initializer},
    }
    for name, attrs in modules.items():
        fake = ModuleType(name)
        fake.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, fake)
    got, evidence = module.load_reference(tmp_path)
    assert got is model and got.bits is None
    assert "floating reference" in evidence["format_adapter"]


def test_real_mflux_mapper_packed_projection_on_cpu(tmp_path):
    pytest.importorskip("mflux")
    import mlx.core as mx
    from mflux.models.common.weights.loading.weight_definition import (
        ComponentDefinition,
    )
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.common.weights.mapping.weight_mapping import WeightTarget

    component_path = tmp_path / "text_encoder"
    component_path.mkdir()
    component = ComponentDefinition(
        name="text_encoder",
        hf_subdir="text_encoder",
        precision=mx.float16,
        mapping_getter=lambda: [
            WeightTarget(
                to_pattern="q_proj.weight", from_pattern=["model.q_proj.weight"]
            )
        ],
    )
    with mx.stream(mx.cpu):
        weight = mx.ones((128, 64))
        packed, scales, biases = mx.quantize(weight, group_size=64, bits=4)
        raw = {
            "model.q_proj.weight": packed,
            "model.q_proj.scales": scales,
            "model.q_proj.biases": biases,
        }
        mx.save_safetensors(str(component_path / "model.safetensors"), raw)
        native, _, _ = WeightLoader._load_component(tmp_path, component)
        assert native["q_proj"]["weight"].shape == (128, 8)
        with pytest.raises(ValueError, match="Last dimension"):
            mx.eval(mx.ones((1, 4, 64)) @ native["q_proj"]["weight"].T)
        key = (str(component_path), component.loading_mode, ())
        decoded = bridge().decode_affine_weights(raw)
        fixed, quantization, _ = WeightLoader._load_component(
            tmp_path, component, raw_weights_cache={key: decoded}
        )
        assert quantization is None
        output = mx.ones((1, 4, 64), dtype=mx.float16) @ fixed["q_proj"]["weight"].T
        mx.eval(output)
        assert output.shape == (1, 4, 128)
        assert mx.array_equal(
            output, mx.full(output.shape, 64, dtype=mx.float16)
        ).item()
