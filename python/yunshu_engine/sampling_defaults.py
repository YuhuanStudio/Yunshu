"""Sampling defaults for requests that leave a sampling parameter out.

Precedence, highest first:

1. the value the request sent (an explicit value always wins, including one equal to an old default);
2. the served model's ``generation_config.json`` (read once per model and file version);
3. the API's official default (``temperature`` 1.0, ``top_p`` 1.0; ``top_k`` / ``min_p`` / penalties off).

There is no operator setting for default sampling, so nothing sits between 1 and 2.
The request models declare these fields as ``None`` by default; ``resolve_sampling`` fills them in at
the top of each generation route so everything downstream sees concrete numbers, and records where
each value came from (reported as ``x_yunshu.sampling``).
"""

from __future__ import annotations

import json
import logging
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


def effective_defaults(model_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Per field: ``{"value": ..., "source": "generation_config" | "api_default"}``."""
    gc = read_generation_config(model_path)
    return {
        f: (
            {"value": gc[f], "source": "generation_config"}
            if f in gc
            else {"value": API_DEFAULTS[f], "source": "api_default"}
        )
        for f in _GC_KEYS
    }


def fill_sampling(
    values: dict[str, Any], model_path: str | Path | None
) -> dict[str, dict[str, Any]]:
    """Fill the None entries of ``values`` (field -> request value); returns the full per-field report."""
    eff = effective_defaults(model_path)
    report: dict[str, dict[str, Any]] = {}
    for f, v in values.items():
        if v is None:
            values[f] = eff[f]["value"]
            report[f] = dict(eff[f])
        else:
            report[f] = {"value": v, "source": "request"}
    return report
