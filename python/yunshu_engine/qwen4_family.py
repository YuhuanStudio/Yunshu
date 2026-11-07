"""One place that knows which ``model_type`` values are the Qwen3.8-Flash-Next (Qwen4) architecture.

The preview checkpoints ship as ``qwen4_exp`` with the upstream module of the same name. The official
release may rename both. To support a new name, append it to ``QWEN4_MODEL_TYPES`` below (one line);
every gate (MTP/spec, PLE, fused kernels) and every patch target follows. The upstream module is
expected to be named after the model type; a type whose module does not import is skipped.

Class names inside the module (``Qwen4Exp*``) are still upstream's and are looked up by name.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any

# Add the official release's model_type here (e.g. "qwen4", "qwen4_moe").
QWEN4_MODEL_TYPES: tuple[str, ...] = ("qwen4_exp",)

_PREFIX = "mlx_vlm.models."


def is_qwen4_type(model_type: Any) -> bool:
    """A Qwen4 model_type, or its text sub-config's (mlx-vlm names the language model's config
    ``<type>_text``: the Flash-Next checkpoint's text_config.model_type is ``qwen4_exp_text``)."""
    if not isinstance(model_type, str):
        return False
    base = model_type[: -len("_text")] if model_type.endswith("_text") else model_type
    return base in QWEN4_MODEL_TYPES


def is_qwen4_family(config: Mapping[str, Any] | None) -> bool:
    """True when ``config`` (or its ``text_config``) names a Qwen4 model_type."""
    if not config:
        return False
    if is_qwen4_type(config.get("model_type")):
        return True
    text = config.get("text_config")
    return isinstance(text, Mapping) and is_qwen4_type(text.get("model_type"))


def qwen4_module_names(sub: str = "language") -> tuple[str, ...]:
    """Dotted module names (``sub`` = submodule, "" for the package) for every type."""
    suffix = f".{sub}" if sub else ""
    return tuple(f"{_PREFIX}{t}{suffix}" for t in QWEN4_MODEL_TYPES)


def is_qwen4_module_name(name: str, sub: str = "language") -> bool:
    return name in qwen4_module_names(sub)


def qwen4_model_modules(sub: str = "language") -> list[ModuleType]:
    """Importable modules for every type in the family; missing ones are skipped."""
    out: list[ModuleType] = []
    for name in qwen4_module_names(sub):
        try:
            out.append(importlib.import_module(name))
        except ImportError:
            continue
    return out


def qwen4_first_module(sub: str = "language") -> ModuleType | None:
    mods = qwen4_model_modules(sub)
    return mods[0] if mods else None


def loaded_qwen4_modules(sub: str = "language") -> list[ModuleType]:
    """Family modules already in ``sys.modules`` (never imports)."""
    return [m for n in qwen4_module_names(sub) if (m := sys.modules.get(n)) is not None]
