"""Model discovery — auto-discovers models from disk with modality detection.

oMLX pattern: scans model directories, detects modality from config.json,
estimates memory usage, and returns structured DiscoveredModel entries.

Uses dynamic detection: delegates to model_manager._detect_model_type()
which uses mlx-lm importlib probing for 0-day support.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ModelType = Literal["llm", "vlm", "audio_tts", "audio_stt", "image_gen"]
EngineType = Literal["batched", "vlm", "audio", "image"]

IMAGE_GEN_MODEL_TYPES = {"flux", "sd3", "sdxl", "z_image"}


@dataclass
class DiscoveredModel:
    model_id: str
    model_path: str
    model_type: ModelType
    engine_type: EngineType
    estimated_size: int
    config_model_type: str = ""


def detect_model_type(model_path: Path) -> ModelType:
    """Detect model type using model_manager's dynamic detection."""
    from .model_manager import _detect_model_type, ModelType as MT

    detected = _detect_model_type(str(model_path))

    # Map internal ModelType enum to string literal
    return {
        MT.LLM: "llm",
        MT.VLM: "vlm",
        MT.TTS: "audio_tts",
        MT.ASR: "audio_stt",
        MT.IMAGE_GEN: "image_gen",
    }[detected]


def estimate_model_size(model_path: Path) -> int:
    total = 0
    for f in model_path.glob("*.safetensors"):
        total += f.stat().st_size
    if total == 0:
        for f in model_path.glob("*.bin"):
            total += f.stat().st_size
    if total == 0:
        for f in model_path.glob("**/*.safetensors"):
            total += f.stat().st_size
    return int(total * 1.05)


def _is_model_dir(path: Path) -> bool:
    return (path / "config.json").exists() or any(path.glob("*.safetensors"))


def _engine_for_type(mt: ModelType) -> EngineType:
    return {
        "llm": "batched",
        "vlm": "vlm",
        "audio_tts": "audio",
        "audio_stt": "audio",
        "image_gen": "image",
    }[mt]


def discover_models(model_dir: Path) -> dict[str, DiscoveredModel]:
    """Scan directory for models, auto-detecting modality type."""
    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")

    models: dict[str, DiscoveredModel] = {}

    for subdir in sorted(model_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue

        if _is_model_dir(subdir):
            _register(models, subdir)
        else:
            for child in sorted(subdir.iterdir()):
                if child.is_dir() and not child.name.startswith(".") and _is_model_dir(child):
                    _register(models, child)

    if not models and _is_model_dir(model_dir):
        _register(models, model_dir)

    return models


def discover_models_from_dirs(model_dirs: list[Path]) -> dict[str, DiscoveredModel]:
    """Scan multiple directories and merge results (first wins on conflicts)."""
    merged: dict[str, DiscoveredModel] = {}
    for d in model_dirs:
        if not d.exists() or not d.is_dir():
            continue
        try:
            for mid, info in discover_models(d).items():
                if mid not in merged:
                    merged[mid] = info
        except ValueError:
            continue
    return merged


def _register(models: dict[str, DiscoveredModel], model_dir: Path) -> None:
    try:
        mt = detect_model_type(model_dir)
        size = estimate_model_size(model_dir)

        config_model_type = ""
        try:
            with open(model_dir / "config.json") as f:
                config_model_type = json.load(f).get("model_type", "")
        except Exception:
            logger.debug("model config.json read failed", exc_info=True)

        models[model_dir.name] = DiscoveredModel(
            model_id=model_dir.name,
            model_path=str(model_dir),
            model_type=mt,
            engine_type=_engine_for_type(mt),
            estimated_size=size,
            config_model_type=config_model_type,
        )
        logger.info(
            "Discovered: %s (type=%s, engine=%s, size=%.2fGB)",
            model_dir.name, mt, _engine_for_type(mt), size / 1024 ** 3,
        )
    except Exception as e:
        logger.error("Failed to discover model %s: %s", model_dir.name, e)
