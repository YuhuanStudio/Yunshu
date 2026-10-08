"""Explicit diffusers affine checkpoint bridge for the independent mflux probe.

This is not native mflux packed-format support: decode stored affine weights before
its HF key mapping, which otherwise drops scales and casts uint32 words to floats.
No upstream source or checkpoint is modified. Serving never imports this module.
"""

# Upstream (inspired): mflux-community/mflux 465df30a, MIT; unchanged HF mapper/applier APIs.
from __future__ import annotations

import json
import struct
from pathlib import Path


def checkpoint_contract(path):
    """Header-only admission: reject the known projection mismatch before GPU load."""
    root = Path(path)
    quant = json.loads((root / "quantize_config.json").read_text())["quantization"]
    bits, group = quant["bits"], quant["group_size"]
    if (bits, group) != (4, 64):
        raise ValueError("this bounded reference bridge requires affine4/group64")
    config = json.loads((root / "text_encoder/config.json").read_text())
    if config.get("quantization") != {"bits": bits, "group_size": group}:
        raise ValueError("text encoder and checkpoint quantization disagree")
    counts, headers = {}, {}
    for component in ("text_encoder", "transformer", "vae"):
        header = {}
        files = sorted((root / component).glob("*.safetensors"))
        if not files:
            raise ValueError(f"missing {component} weights")
        for file in files:
            with file.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                shard = json.loads(stream.read(size))
            for key, value in shard.items():
                if key == "__metadata__":
                    continue
                if key in header:
                    raise ValueError(f"duplicate tensor {component}/{key}")
                header[key] = value
        packed = 0
        for key, value in header.items():
            if value["dtype"] != "U32":
                continue
            packed += 1
            shape = value["shape"]
            prefix = key.removesuffix(".weight")
            scales = header.get(prefix + ".scales", {}).get("shape")
            biases = header.get(prefix + ".biases", {}).get("shape")
            if (
                (
                    not key.endswith(".weight")
                    and key not in ("x_pad_token", "cap_pad_token")
                )
                or len(shape) != 2
                or not scales
                or len(scales) != 2
                or scales != biases
                or shape[0] != scales[0]
                or shape[1] * 32 != scales[1] * group * bits
            ):
                raise ValueError(f"invalid packed affine triplet {component}/{key}")
        counts[component] = packed
        headers[component] = header
    for layer in range(config["num_hidden_layers"]):
        key = f"model.layers.{layer}.self_attn.q_proj.weight"
        q = headers["text_encoder"].get(key, {})
        expected = [
            config["num_attention_heads"] * config["head_dim"],
            config["hidden_size"] * bits // 32,
        ]
        if q.get("dtype") != "U32" or q.get("shape") != expected:
            raise ValueError(f"text encoder projection contract failed: {key}")
    return {"bits": bits, "group_size": group, "packed_layers": counts}


def decode_affine_weights(weights, *, bits=4, group_size=64):
    """Decode before mapping; keep floating tensors and genuine linear bias intact."""
    import mlx.core as mx

    result = dict(weights)
    for key, weight in weights.items():
        if weight.dtype != mx.uint32:
            continue
        if not key.endswith(".weight") and key not in ("x_pad_token", "cap_pad_token"):
            raise ValueError(f"unexpected packed tensor {key}")
        prefix = key.removesuffix(".weight")
        scales, biases = (
            weights.get(prefix + ".scales"),
            weights.get(prefix + ".biases"),
        )
        if scales is None or biases is None:
            raise ValueError(f"missing affine triplet {key}")
        if (
            weight.ndim != 2
            or scales.ndim != 2
            or scales.shape != biases.shape
            or weight.shape[0] != scales.shape[0]
            or weight.shape[1] * 32 != scales.shape[1] * group_size * bits
        ):
            raise ValueError(f"invalid packed shape {key}")
        result[key] = mx.dequantize(weight, scales, biases, group_size, bits)
        del result[prefix + ".scales"]
        del result[prefix + ".biases"]
    return result


def load_reference(path):
    contract = checkpoint_contract(path)
    from mflux.models.common.weights.loading.loaded_weights import (
        LoadedWeights,
        MetaData,
    )
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.z_image.variants.z_image import ZImage
    from mflux.models.z_image.weights.z_image_weight_definition import (
        ZImageWeightDefinition,
    )
    from mflux.models.z_image.z_image_initializer import ZImageInitializer

    # Construct the upstream model without evaluating the initially mismapped weights.
    # Then use its unchanged mapper/applier on explicitly decoded HF inputs. The
    # upstream loader's raw cache is an argument, so no global monkeypatch is needed.
    root = Path(path)
    model = ZImage(model_path=str(root))
    components = {}
    for component in ZImageWeightDefinition.get_components():
        component_path = root / component.hf_subdir
        raw = decode_affine_weights(
            WeightLoader._load_safetensors(
                component_path, component.loading_mode, component.weight_files
            ),
            bits=contract["bits"],
            group_size=contract["group_size"],
        )
        cache_key = (
            str(component_path),
            component.loading_mode,
            tuple(component.weight_files or []),
        )
        mapped, quantization, _ = WeightLoader._load_component(
            root, component, raw_weights_cache={cache_key: raw}
        )
        if quantization is not None:
            raise ValueError("expected diffusers source, got native mflux metadata")
        components[component.name] = mapped
    ZImageInitializer._apply_weights(
        model, LoadedWeights(components=components, meta_data=MetaData()), quantize=None
    )
    return model, {
        "format_adapter": "diffusers affine4 decoded before mflux HF mapping; floating reference arithmetic",
        **contract,
    }
