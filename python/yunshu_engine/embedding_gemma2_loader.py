# Upstream (derived): Blaizzy/mlx-vlm (MIT), encoder_loader.py at 3d87e884
"""Prefer mlx-vlm's maintained loader; bridge its unreleased Gemma 2 port.

Fallback derived from mlx-vlm encoder_loader.py at 3d87e884 (MIT). No global
registry patch: other encoder loaders keep their normal model resolution.
"""

from importlib import import_module
from pathlib import Path


def load_published_model(model_path: Path, *, lazy: bool = False):
    try:
        import_module("mlx_vlm.models.embedding_gemma2")
    except ModuleNotFoundError as exc:
        if exc.name != "mlx_vlm.models.embedding_gemma2":
            raise
    else:
        from mlx_vlm.embedding_loader import load_embedding_model

        return load_embedding_model(model_path, strict=True, lazy=lazy)
    return _load_derived(model_path, lazy=lazy)


def _load_derived(model_path: Path, *, lazy: bool = False):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.encoder_loader import _weight_files
    from mlx_vlm.utils import (
        _load_safetensors,
        _quantization_for_module_path,
        load_config,
    )

    from ._vendor.embedding_gemma2.config import ModelConfig
    from ._vendor.embedding_gemma2.model import Model

    config = load_config(model_path)
    files = _weight_files(model_path)
    if not files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights = {}
    for file in files:
        weights.update(_load_safetensors(file))
    model = Model(ModelConfig.from_dict(config))
    weights = model.sanitize(weights)
    quantization = config.get("quantization") or config.get("quantization_config")
    if quantization is not None:

        def predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            per_module = _quantization_for_module_path(quantization, path, model)
            if per_module is not None:
                return per_module
            if hasattr(module, "weight") and module.weight.size % 64 != 0:
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=predicate,
        )
    model.load_weights(list(weights.items()), strict=True)
    if not lazy:
        mx.eval(model.parameters())
    model.model_path = model_path
    model.eval()
    return model
