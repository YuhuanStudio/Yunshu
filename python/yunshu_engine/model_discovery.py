"""Model discovery — auto-discovers models from disk with modality detection.

scans model directories, detects modality from config.json,
estimates memory usage, and returns structured DiscoveredModel entries.

Uses dynamic detection: delegates to model_manager._detect_model_type()
which uses mlx-lm importlib probing for 0-day support.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ModelType = Literal[
    "llm", "vlm", "audio_tts", "audio_stt", "image_gen", "ocr", "sts", "video"
]
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
    from .model_manager import ModelType as MT
    from .model_manager import _detect_model_type

    detected = _detect_model_type(str(model_path))

    # Map internal ModelType enum to string literal
    type_map = {
        MT.LLM: "llm",
        MT.VLM: "vlm",
        MT.TTS: "audio_tts",
        MT.ASR: "audio_stt",
        MT.IMAGE_GEN: "image_gen",
        MT.OCR: "ocr",
        MT.STS: "sts",
        MT.VIDEO: "video",
    }
    return type_map.get(detected, "llm")


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
    # 1.10x covers MLX module overhead + activation scratch at moderate seq
    # lengths. KV-cache headroom is added separately by ModelManager via
    # kv_reserve_ratio. The old 1.8x double-counted KV and rejected loads
    # that actually fit (e.g. 20GB 4-bit model needs ~22GB, not 36GB).
    return int(total * 1.10)


def _is_model_dir(path: Path) -> bool:
    return (
        (path / "config.json").exists()
        or any(path.glob("*.safetensors"))
        or (path / "model_index.json").exists()
    )


def _engine_for_type(mt: ModelType) -> EngineType:
    return {
        "llm": "batched",
        "vlm": "vlm",
        "audio_tts": "audio",
        "audio_stt": "audio",
        "image_gen": "image",
        "ocr": "vlm",
        "sts": "audio",
        "video": "vlm",
    }.get(mt, "batched")


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
        elif not (subdir / "model_index.json").exists():
            # Only recurse into non-diffusion directories
            for child in sorted(subdir.iterdir()):
                if (
                    child.is_dir()
                    and not child.name.startswith(".")
                    and _is_model_dir(child)
                ):
                    _register(models, child)

    if not models and _is_model_dir(model_dir):
        _register(models, model_dir)

    return models


def hf_cache_snapshots() -> list[tuple[str, Path, int]]:
    """Model repos in the Hugging Face cache that can load without a download:
    ``(repo_id, snapshot_path, size_on_disk)``, newest complete revision per
    repo (a ``config.json`` and ``*.safetensors`` weights)."""
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
    except Exception:  # noqa: BLE001 - no cache yet, or an unreadable one
        logger.debug("Hugging Face cache scan failed", exc_info=True)
        return []
    out = []
    for repo in sorted(cache.repos, key=lambda r: r.repo_id):
        if repo.repo_type != "model":
            continue
        for rev in sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True):
            names = {f.file_name for f in rev.files}
            if "config.json" in names and any(
                n.endswith(".safetensors") for n in names
            ):
                out.append((repo.repo_id, Path(rev.snapshot_path), rev.size_on_disk))
                break
    return out


def hf_repo_id_for(path: str | Path | None) -> str | None:
    """``org/name`` for a snapshot directory inside the Hugging Face cache
    (``.../models--org--name/snapshots/<revision>``), else None."""
    if not path:
        return None
    parts = Path(path).parts
    for i, part in enumerate(parts[:-2]):
        if part.startswith("models--") and parts[i + 1] == "snapshots":
            org, sep, name = part[len("models--") :].partition("--")
            return f"{org}/{name}" if sep and org and name else None
    return None


def discover_hf_cache_models(
    models: dict[str, DiscoveredModel],
) -> dict[str, DiscoveredModel]:
    """Add Hugging Face cache models to ``models`` (already-discovered entries
    win; a clashing short name is registered as ``org/name``)."""
    taken = {m.model_path for m in models.values()}
    for repo_id, snapshot, _size in hf_cache_snapshots():
        if str(snapshot) in taken:
            continue
        org, _, name = repo_id.rpartition("/")
        _register(models, snapshot, name=name, org=org or None)
    return models


def resolve_model_ref(ref: str | None, models_dir: Path | None = None) -> str | None:
    """A local directory for a model reference, when one is on disk.

    ``ref`` may be a path, a name under the models directory (``name`` or
    ``org/name``, the layout ``yunshu pull`` writes), or a Hugging Face repo id
    already in the Hugging Face cache. Returns that directory, or ``ref``
    unchanged when nothing local matches (the loader then downloads it).
    """
    if not ref:
        return ref
    p = Path(ref).expanduser()
    if p.exists():
        return str(p)
    if ref.startswith(("/", ".", "~")):
        return ref
    if models_dir is None:
        from .paths import models_dir as _models_dir

        models_dir = _models_dir()
    for cand in (models_dir / ref, models_dir / ref.rsplit("/", 1)[-1]):
        if cand.is_dir() and _is_model_dir(cand):
            return str(cand)
    if ref.count("/") == 1:
        try:
            from huggingface_hub import try_to_load_from_cache

            config = try_to_load_from_cache(ref, "config.json")
        except Exception:  # noqa: BLE001 - not a valid repo id / no cache
            config = None
        if isinstance(config, str):
            snapshot = Path(config).parent
            if any(snapshot.glob("*.safetensors")):
                return str(snapshot)
    return ref


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


def _register(
    models: dict[str, DiscoveredModel],
    model_dir: Path,
    name: str | None = None,
    org: str | None = None,
) -> None:
    try:
        mt = detect_model_type(model_dir)
        size = estimate_model_size(model_dir)

        config_model_type = ""
        try:
            with open(model_dir / "config.json") as f:
                config_model_type = json.load(f).get("model_type", "")
        except Exception:
            logger.debug("model config.json read failed", exc_info=True)

        # Use model_dir.name as the key, but disambiguate if a different
        # model with the same name was already registered (e.g. same model
        # name under different org directories like mlx-community/Qwen2.5
        # vs custom-org/Qwen2.5).  The existing entry wins (first-found).
        base_name = name or model_dir.name
        key = base_name
        if key in models:
            existing_path = models[key].model_path
            if existing_path != str(model_dir):
                # Collision: disambiguate with parent directory prefix
                parent_name = org or model_dir.parent.name
                key = f"{parent_name}/{base_name}"
                logger.warning(
                    "Model name collision: '%s' from %s shadows %s, "
                    "using disambiguated key '%s'",
                    model_dir.name,
                    existing_path,
                    model_dir,
                    key,
                )

        models[key] = DiscoveredModel(
            model_id=base_name,
            model_path=str(model_dir),
            model_type=mt,
            engine_type=_engine_for_type(mt),
            estimated_size=size,
            config_model_type=config_model_type,
        )
        logger.info(
            "Discovered: %s (type=%s, engine=%s, size=%.2fGB)",
            key,
            mt,
            _engine_for_type(mt),
            size / 1024**3,
        )
    except Exception as e:
        logger.error(
            "Failed to discover model %s: %s", model_dir.name, e, exc_info=True
        )
