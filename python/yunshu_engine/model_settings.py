"""Per-Model Settings — configurable runtime parameters per model.

Inspired by oMLX's per-model config system. Each registered model can
have its own settings that override global defaults. Settings are:

1. Set via API (admin endpoint)
2. Loaded from model directory (model_settings.json)
3. Overridden by environment variables (YUNSHU_MODEL_{ID}_*)

Settings are applied when loading the engine and can be hot-reloaded.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelSettings:
    """Per-model runtime configuration.

    All fields have sensible defaults. Override via:
    - admin API: PUT /admin/models/{id}/settings
    - model directory: model_settings.json
    - env var: YUNSHU_MODEL_{MODEL_ID}_{FIELD}
    """

    # ── Generation defaults ──
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0  # 0 = disabled
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # ── Stop sequences ──
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)

    # ── Context window ──
    max_context_tokens: int = 0  # 0 = use model's max_position_embeddings

    # ── KV cache ──
    kv_cache_quant_bits: int | None = None  # None = no quantization
    kv_cache_quant_group_size: int = 64
    kv_cache_quant_start_layer: int = 0

    # ── Prefix cache ──
    prefix_cache_enabled: bool = True
    prefix_cache_max_entries: int = 64
    prefix_cache_min_prefix: int = 32

    # ── Speculative decoding ──
    spec_decode_enabled: bool = False
    ngram_spec_enabled: bool = False
    ngram_spec_max_n: int = 5
    ngram_spec_k: int = 5

    # ── SpecPrefill ──
    spec_prefill_enabled: bool = False
    spec_prefill_threshold: int = 8192
    spec_prefill_keep_rate: float = 0.20

    # ── Thinking / reasoning ──
    thinking_budget: int | None = None  # None = model default
    enable_thinking: bool | None = None  # None = model default

    # ── Memory management ──
    memory_pressure_threshold: float = 85.0
    max_kv_cache_memory: int = 0  # 0 = auto

    # ── Performance ──
    batch_size: int = 1
    prefill_chunk_size: int = 0  # 0 = auto
    use_fast_path: bool = True

    # ── Engine behavior ──
    seed: int | None = None  # None = random
    logprobs: bool = False

    # ── Streaming ──
    stream_keepalive_interval: float = 15.0
    stream_chunk_timeout: float = 30.0

    # ── MoE optimization ──
    moe_top_k: int = 0  # 0 = use model default, >0 = override expert count

    # ── LoRA ──
    lora_enabled: bool = False
    max_loras: int = 4

    # ── SSD cache ──
    ssd_cache_enabled: bool = False
    ssd_cache_dir: str = "~/.cache/yunshu/kv-ssd"
    ssd_cache_max_gb: int = 10

    def to_dict(self) -> dict[str, Any]:
        """Export settings as a flat dict."""
        result = {}
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if val is not None and not (isinstance(val, list) and not val):
                result[f] = val
        return result

    def apply_overrides(self, overrides: dict[str, Any]) -> list[str]:
        """Apply override dict to settings. Returns list of changed field names."""
        changed = []
        for key, value in overrides.items():
            if key in self.__dataclass_fields__ and value is not None:
                current = getattr(self, key)
                if current != value:
                    try:
                        field_type = self.__dataclass_fields__[key].type
                        if field_type in ("int", int) and isinstance(value, (int, float)):
                            value = int(value)
                        elif field_type in ("float", float) and isinstance(value, (int, float)):
                            value = float(value)
                        elif field_type in ("bool", bool):
                            value = bool(value)
                    except Exception:
                        pass
                    setattr(self, key, value)
                    changed.append(key)
        return changed


def load_model_settings(model_path: str, model_id: str) -> ModelSettings:
    """Load per-model settings from model directory or env vars.

    Priority (highest to lowest):
    1. Environment variable overrides
    2. model_settings.json in model directory
    3. Defaults
    """
    settings = ModelSettings()

    # 2. Load from model_settings.json
    settings_path = Path(model_path) / "model_settings.json"
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                overrides = json.load(f)
            settings.apply_overrides(overrides)
            logger.info(f"Loaded model settings from {settings_path}: {len(overrides)} fields")
        except Exception:
            logger.debug(f"Failed to load model settings from {settings_path}", exc_info=True)

    # 1. Override from env vars: YUNSHU_MODEL_{SANITIZED_ID}_{FIELD}
    env_prefix = f"YUNSHU_MODEL_{model_id.upper().replace('-', '_').replace('.', '_').replace('/', '_')}_"
    env_overrides: dict[str, Any] = {}
    for env_key, env_val in os.environ.items():
        if env_key.startswith(env_prefix):
            field_name = env_key[len(env_prefix):].lower()
            if field_name in settings.__dataclass_fields__:
                try:
                    # Parse booleans and ints
                    val: Any = env_val
                    if env_val.lower() in ("true", "1", "yes"):
                        val = True
                    elif env_val.lower() in ("false", "0", "no"):
                        val = False
                    elif env_val.isdigit():
                        val = int(env_val)
                    else:
                        try:
                            val = float(env_val)
                        except ValueError:
                            pass
                    env_overrides[field_name] = val
                except Exception:
                    pass

    if env_overrides:
        changed = settings.apply_overrides(env_overrides)
        if changed:
            logger.info(f"Model {model_id} env overrides: {changed}")

    return settings
