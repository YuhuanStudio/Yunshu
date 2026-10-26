"""``suppress_tokens`` from a checkpoint's generation_config.json: the default token mask.

The model's author bans these ids from the output (audio/image placeholder tokens and the like).
A request's own ``suppress_tokens`` wins; when it sets none, the checkpoint's list applies.
Shared by the text path (``BatchedEngine.generate`` / ``stream_generate``) and meant for the VLM
path's identical rule, so a unified path keeps one reader.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
# path -> ((mtime, size), raw int list as written)
_CACHE: dict[str, tuple[tuple[float, int], list[int]]] = {}


_DIRS: dict[str, Path | None] = {}


def _model_dir(model_path: str | Path | None) -> Path | None:
    """The checkpoint directory for a local path, or for a hub id its already-downloaded snapshot
    (offline lookup in the Hugging Face cache, the directory mlx-lm loaded the weights from)."""
    if not model_path:
        return None
    key = str(model_path)
    with _LOCK:
        if key in _DIRS:
            return _DIRS[key]
    path: Path | None = Path(key)
    if not path.is_dir():
        try:
            from mlx_lm.utils import hf_repo_to_path

            path = hf_repo_to_path(key)
        except Exception:  # not a cached hub id (or no mlx-lm): nothing to read
            path = None
    with _LOCK:
        _DIRS[key] = path
    return path


def _raw_ids(model_path: str | Path | None) -> list[int]:
    root = _model_dir(model_path)
    if root is None:
        return []
    fp = root / "generation_config.json"
    try:
        st = fp.stat()
    except OSError:
        return []
    ver = (st.st_mtime, st.st_size)
    key = str(fp)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == ver:
            return hit[1]
    ids: list[int] = []
    try:
        raw = json.loads(fp.read_text()).get("suppress_tokens")
        if isinstance(raw, list):
            ids = [i for i in raw if isinstance(i, int) and not isinstance(i, bool)]
    except Exception:
        logger.warning("unreadable generation_config.json at %s", fp, exc_info=True)
    with _LOCK:
        _CACHE[key] = (ver, ids)
    return ids


def default_suppress_ids(model_path: str | Path | None, tokenizer=None) -> list[int]:
    """The checkpoint's ``suppress_tokens``, sorted and de-duplicated, ids outside the
    vocabulary dropped (the mask indexes the logits row). Empty when there is no such list."""
    ids = _raw_ids(model_path)
    if not ids:
        return []
    if tokenizer is not None:
        try:
            vocab = len(getattr(tokenizer, "_tokenizer", tokenizer))
            ids = [i for i in ids if 0 <= i < vocab]
        except Exception:
            ids = [i for i in ids if i >= 0]
    else:
        ids = [i for i in ids if i >= 0]
    return sorted(set(ids))


def resolve_suppress_tokens(
    requested: list[int] | None, model_path: str | Path | None, tokenizer=None
) -> list[int] | None:
    """The request's own list when it set one, else the checkpoint default (None when neither)."""
    if requested:
        return requested
    return default_suppress_ids(model_path, tokenizer) or None
