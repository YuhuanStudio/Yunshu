"""Which speculative path a Qwen3.5-family model will use.

Order: ``YUNSHU_VLM_DRAFT`` (a drafter directory, or ``mtp`` to force the
checkpoint's MTP head, or ``off``), else a DFlash2 drafter found on disk that
matches the target's shape (models dir or Hugging Face cache; measured faster
than MTP on Qwen3.8-27B and lossless), else the checkpoint's MTP head when it
has one (``YUNSHU_MTP``), else plain decode.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from . import settings

logger = logging.getLogger(__name__)

FORCE_MTP = ("mtp", "force-mtp")
FORCE_OFF = ("off", "none")


@dataclass(frozen=True)
class SpecChoice:
    kind: str  # "dflash" | "mtp" | "none"
    drafter: str | None  # drafter directory for dflash
    reason: str
    automatic: bool = False


def _text_config(config: dict) -> dict:
    return config.get("text_config") or config


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def drafter_matches(drafter_dir: Path, target_config: dict) -> bool:
    """A DFlash drafter is trained for one target: same hidden size and depth."""
    cfg = _read_json(drafter_dir / "config.json")
    if not cfg or "dflash_config" not in cfg:
        return False
    if not any(drafter_dir.glob("*.safetensors")):
        return False
    text = _text_config(target_config)
    return cfg.get("num_target_layers") == text.get("num_hidden_layers") and cfg.get(
        "hidden_size"
    ) == text.get("hidden_size")


def forced_mismatch(drafter_dir: Path, target_config: dict) -> str | None:
    """Why a forced local drafter cannot serve this target, before any weight loads (None: fine or not local)."""
    cfg = _read_json(drafter_dir / "config.json")
    if not cfg or "dflash_config" not in cfg:
        return None
    text = _text_config(target_config)
    for key, want in (
        ("hidden_size", "hidden_size"),
        ("num_target_layers", "num_hidden_layers"),
    ):
        have, need = cfg.get(key), text.get(want)
        if have is not None and need is not None and have != need:
            return f"does not match this model: drafter {key} {have} != target {want} {need}"
    return None


def is_27b_class(target_config: dict) -> bool:
    text = _text_config(target_config)
    return text.get("num_hidden_layers") == 64 and text.get("hidden_size") == 5120


def _candidate_dirs(models_dir: Path | None) -> list[Path]:
    from .model_discovery import hf_cache_snapshots
    from .paths import models_dir as _models_dir

    base = models_dir or _models_dir()
    found: list[Path] = []
    if base.is_dir():
        # ``org/name`` layout (yunshu pull) and flat layout.
        for pattern in ("*DFlash*", "*/*DFlash*"):
            found += sorted(p for p in base.glob(pattern) if p.is_dir())
    found += [snap for repo, snap, _ in hf_cache_snapshots() if "DFlash" in repo]
    return found


def find_dflash_drafter(
    target_config: dict, models_dir: Path | None = None
) -> str | None:
    for cand in _candidate_dirs(models_dir):
        if drafter_matches(cand, target_config):
            return str(cand)
    return None


def choose(
    target_config: dict,
    *,
    spec_family: bool,
    mtp_capable: bool,
    models_dir: Path | None = None,
    ignore_override: bool = False,
) -> SpecChoice:
    if not spec_family:
        return SpecChoice("none", None, "model family has no speculative path")
    override = None if ignore_override else settings.get("YUNSHU_VLM_DRAFT")
    if override:
        low = str(override).lower()
        if low in FORCE_OFF:
            return SpecChoice("none", None, "YUNSHU_VLM_DRAFT=off")
        if low in FORCE_MTP:
            if mtp_capable and settings.get_bool("YUNSHU_MTP"):
                return SpecChoice("mtp", None, "YUNSHU_VLM_DRAFT=mtp")
            return SpecChoice("none", None, "MTP forced but the checkpoint has no head")
        mismatch = forced_mismatch(Path(str(override)).expanduser(), target_config)
        if mismatch:
            raise ValueError(f"YUNSHU_VLM_DRAFT={override} {mismatch}")
        return SpecChoice("dflash", str(override), "YUNSHU_VLM_DRAFT")
    found = find_dflash_drafter(target_config, models_dir)
    if found:
        return SpecChoice("dflash", found, "DFlash2 drafter found on disk", True)
    if mtp_capable and settings.get_bool("YUNSHU_MTP"):
        return SpecChoice("mtp", None, "checkpoint MTP head")
    return SpecChoice("none", None, "no drafter and no MTP head")


def plain_rows_use_dflash_batch(
    target_config: dict,
    *,
    spec_family: bool,
    mtp_capable: bool,
    models_dir: Path | None = None,
) -> bool:
    """Which batch decodes the rows of a request that does not draft (``YUNSHU_VLM_DRAFT=off``, or a
    model with no drafter): the batch of the drafter the model would otherwise use, minus drafting.
    A model whose drafter is the checkpoint MTP head decodes plain rows in the MTP batch (one token
    per round), so spec on == spec off is one forward; a DFlash model (or no drafter at all) keeps
    the DFlash batch's plain verify. One function decides every on/off pair."""
    natural = choose(
        target_config,
        spec_family=spec_family,
        mtp_capable=mtp_capable,
        models_dir=models_dir,
        ignore_override=True,
    )
    return natural.kind != "mtp"
