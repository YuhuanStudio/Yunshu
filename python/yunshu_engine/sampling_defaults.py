"""Sampling defaults for requests that leave a sampling parameter out.

Precedence per field, highest first:

1. the value the request sent (an explicit value always wins, including one equal to an old default);
2. the model's published recommendation for the mode THIS request runs in (``thinking`` or
   ``non_thinking``; ``data/sampling_profiles.json``, copied from model cards, matched by model name);
3. the served model's ``generation_config.json`` (read once per model and file version);
4. the API's official default (``temperature`` 1.0, ``top_p`` 1.0; ``top_k`` / ``min_p`` / penalties off).

``YUNSHU_SAMPLING_DEFAULTS=api`` skips 2 and 3. Unknown models, a missing or malformed profile file and a
profile without the field all fall through to the next level.
The request models declare these fields as ``None`` by default; ``resolve_sampling`` fills them in at
the top of each generation route so everything downstream sees concrete numbers, and records where
each value came from (reported as ``x_yunshu.sampling``).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# request field -> generation_config key
_GC_KEYS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}
# The official defaults (OpenAI chat / completions / Responses; Anthropic Messages). Anthropic leaves
# top_p / top_k unset, which the engine spells top_p 1.0 / top_k 0.
API_DEFAULTS: dict[str, float | int] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}
_RANGES = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (0, None),
    "min_p": (0.0, 1.0),
    "repetition_penalty": (0.0, 2.0),
    "presence_penalty": (-2.0, 2.0),
    "frequency_penalty": (-2.0, 2.0),
}

_CACHE: dict[str, tuple[tuple[float, int], dict[str, float | int]]] = {}
_LOCK = threading.Lock()


def _valid(field: str, v: Any) -> float | int | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    lo, hi = _RANGES[field]
    if v != v or (lo is not None and v < lo) or (hi is not None and v > hi):
        return None
    if field == "top_k":
        return int(v) if float(v).is_integer() else None
    return float(v)


def read_generation_config(model_path: str | Path | None) -> dict[str, float | int]:
    """The sampling keys ``generation_config.json`` defines (valid values only), cached per file version."""
    if not model_path:
        return {}
    fp = Path(str(model_path)) / "generation_config.json"
    try:
        st = fp.stat()
    except OSError:
        return {}
    ver = (st.st_mtime, st.st_size)
    key = str(fp)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == ver:
            return dict(hit[1])
    out: dict[str, float | int] = {}
    try:
        g = json.loads(fp.read_text())
        if isinstance(g, dict):
            for field, gk in _GC_KEYS.items():
                if gk in g:
                    v = _valid(field, g[gk])
                    if v is not None:
                        out[field] = v
                    else:
                        logger.warning(
                            "ignoring %s=%r in %s (out of range)", gk, g[gk], fp
                        )
    except Exception:
        logger.warning("unreadable generation_config.json at %s", fp, exc_info=True)
    with _LOCK:
        _CACHE[key] = (ver, out)
    return dict(out)


# ── published per-mode recommendations ───────────────────────────────────────

PROFILES_PATH = Path(__file__).parent / "data" / "sampling_profiles.json"
_QUANT_SUFFIX = re.compile(
    r"[-_](mlx|bf16|fp16|f16|[2-8]bit|mxfp[48]|nvfp4|oq\d\w*|mtp|dflash\w*|q\d\w*|gguf)$"
)
_PROFILES: tuple[tuple[float, int], list[dict]] | None = None


def normalize_name(name: str) -> str:
    """Model directory / repo name -> lowercase, quantization and packaging suffixes stripped."""
    n = name.rstrip("/").rsplit("/", 1)[-1].lower()
    while True:
        m = _QUANT_SUFFIX.sub("", n)
        if m == n:
            return n
        n = m


def load_profiles(path: Path | None = None) -> list[dict]:
    """The valid entries of the profile file; a malformed file or entry is skipped with a warning."""
    global _PROFILES
    fp = path or PROFILES_PATH
    try:
        st = fp.stat()
    except OSError:
        return []
    ver = (st.st_mtime, st.st_size)
    if path is None and _PROFILES is not None and _PROFILES[0] == ver:
        return _PROFILES[1]
    out: list[dict] = []
    try:
        doc = json.loads(fp.read_text())
        for e in doc.get("profiles", []) if isinstance(doc, dict) else []:
            names = e.get("names") if isinstance(e, dict) else None
            if not (
                isinstance(names, list)
                and names
                and all(isinstance(n, str) for n in names)
            ):
                logger.warning("skipping malformed sampling profile %r", e)
                continue
            ent: dict[str, Any] = {
                "names": [n.lower() for n in names],
                "source_url": str(e.get("source_url", "")),
                "read": str(e.get("read", "")),
            }
            for mode in ("thinking", "non_thinking"):
                raw = e.get(mode)
                vals: dict[str, Any] = {}
                if isinstance(raw, dict):
                    for f, v in raw.items():
                        ok = _valid(f, v) if f in _RANGES else None
                        if ok is not None:
                            vals[f] = ok
                        else:
                            logger.warning(
                                "ignoring %s.%s=%r in sampling profile", mode, f, v
                            )
                ent[mode] = vals
            out.append(ent)
    except Exception:
        logger.warning("unreadable sampling profile file %s", fp, exc_info=True)
        out = []
    if path is None:
        _PROFILES = (ver, out)
    return out


def profile_for(model_path: str | Path | None, path: Path | None = None) -> dict | None:
    """The published-recommendation entry for a model directory, matched by normalized name."""
    if not model_path:
        return None
    n = normalize_name(str(model_path))
    for e in load_profiles(path):
        if n in e["names"]:
            return e
    return None


_MODE_CACHE: dict[str, tuple[tuple[float, int], str]] = {}


def default_mode(model_path: str | Path | None) -> str:
    """Mode a request with no thinking switch runs in: what the model's chat template does by default."""
    if not model_path:
        return "non_thinking"
    d = Path(str(model_path))
    probe = d / "chat_template.jinja"
    try:
        st = (probe if probe.exists() else d / "tokenizer_config.json").stat()
    except OSError:
        return "non_thinking"
    ver = (st.st_mtime, st.st_size)
    with _LOCK:
        hit = _MODE_CACHE.get(str(d))
        if hit is not None and hit[0] == ver:
            return hit[1]
    mode = "non_thinking"
    try:
        from .model_card import _reasoning, chat_template_text

        r = _reasoning(chat_template_text(d), {}, {})
        if r.get("supported") and r.get("default_enabled"):
            mode = "thinking"
    except Exception:
        logger.debug("template mode probe failed", exc_info=True)
    with _LOCK:
        _MODE_CACHE[str(d)] = (ver, mode)
    return mode


def _policy() -> str:
    try:
        from . import settings

        return str(settings.get("YUNSHU_SAMPLING_DEFAULTS") or "model")
    except Exception:
        return "model"


def effective_defaults(
    model_path: str | Path | None, mode: str | None = None
) -> dict[str, dict[str, Any]]:
    """Per field: ``{"value", "source"}``; source is model_card, generation_config or api_default.

    ``mode`` is ``thinking`` / ``non_thinking`` (None: the model's template default)."""
    model_policy = _policy() != "api"
    gc = read_generation_config(model_path) if model_policy else {}
    prof = profile_for(model_path) if model_policy else None
    m = mode or default_mode(model_path)
    rec = (prof or {}).get(m, {}) if prof else {}
    out: dict[str, dict[str, Any]] = {}
    for f in _GC_KEYS:
        if f in rec:
            out[f] = {"value": rec[f], "source": "model_card"}
        elif f in gc:
            out[f] = {"value": gc[f], "source": "generation_config"}
        else:
            out[f] = {"value": API_DEFAULTS[f], "source": "api_default"}
    return out


def defaults_by_mode(model_path: str | Path | None) -> dict[str, Any]:
    """Both modes' defaults, the model's default mode and the matched profile (for surfaces)."""
    prof = profile_for(model_path) if _policy() != "api" else None
    return {
        "policy": _policy(),
        "default_mode": default_mode(model_path),
        "profile": {k: prof[k] for k in ("names", "source_url", "read")}
        if prof
        else None,
        "thinking": effective_defaults(model_path, "thinking"),
        "non_thinking": effective_defaults(model_path, "non_thinking"),
    }


def fill_sampling(
    values: dict[str, Any], model_path: str | Path | None, mode: str | None = None
) -> dict[str, dict[str, Any]]:
    """Fill the None entries of ``values`` (field -> request value); returns the full per-field report."""
    eff = effective_defaults(model_path, mode)
    if values.get("temperature") == 0:
        # Greedy is not sampling, so a card's sampling recipe does not apply to its penalties (user 2026-10-25): a
        # greedy request keeps the API's neutral penalties instead of the non-thinking presence_penalty 1.5, which
        # changed greedy text and kept the request off the fast tree (release gate serve-27b: server/in-process 0.68).
        eff = {
            f: {"value": API_DEFAULTS[f], "source": "api_default"}
            if f.endswith("_penalty") and f in API_DEFAULTS
            else e
            for f, e in eff.items()
        }
    report: dict[str, dict[str, Any]] = {}
    for f, v in values.items():
        if v is None:
            values[f] = eff[f]["value"]
            report[f] = dict(eff[f])
        else:
            report[f] = {"value": v, "source": "request"}
    return report


def log_line(model_path: str | Path | None) -> str | None:
    """One startup line: the defaults a request without sampling parameters gets, per mode."""
    if not model_path:
        return None
    d = defaults_by_mode(model_path)

    def fmt(m: str) -> str:
        e = d[m]
        src = sorted({v["source"] for v in e.values()} - {"api_default"}) or [
            "api_default"
        ]
        return (
            f"{m} T={e['temperature']['value']} top_p={e['top_p']['value']} "
            f"top_k={e['top_k']['value']} presence={e['presence_penalty']['value']} [{','.join(src)}]"
        )

    return (
        f"Sampling defaults (policy {d['policy']}, default mode {d['default_mode']}): "
        f"{fmt('thinking')}; {fmt('non_thinking')}"
    )
