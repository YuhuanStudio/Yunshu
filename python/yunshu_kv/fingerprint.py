"""Identity of a checkpoint for persisted KV / recurrent states.

A cached state is only valid for the exact model that produced it. Naming the
cache directory by model name or path is not enough: weights replaced in place
(re-quantized, fine-tuned, a new revision downloaded over the old one), an edited
``config.json``, a different tokenizer / chat template, a LoRA adapter, or a change
of the cache's own layout all yield different states under the same name.

``checkpoint_fingerprint`` folds all of those into one short hex digest used as (part
of) the on-disk namespace, so any change simply selects a new, empty namespace.
Nothing here reads weight tensors: files are identified by name, size and mtime
(``st_mtime_ns``); the small JSON / template files that define behaviour are hashed
by content.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Bump when the layout or meaning of persisted states changes: every older namespace
# stops matching and is never read back.
FORMAT_VERSION = 1

# Small files whose content (not mtime) defines behaviour.
_CONTENT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "added_tokens.json",
    "model.safetensors.index.json",
)
# Larger files identified by name, size and mtime.
_STAT_GLOBS = (
    "*.safetensors",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)
_CONTENT_MAX_BYTES = 4 << 20


def _update_file(h: Any, tag: str, path: Path, *, content: bool) -> None:
    try:
        st = path.stat()
    except OSError:
        h.update(f"|{tag}:missing".encode())
        return
    if content and st.st_size <= _CONTENT_MAX_BYTES:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        except OSError:
            digest = "unreadable"
        h.update(f"|{tag}:{st.st_size}:{digest}".encode())
    else:
        h.update(f"|{tag}:{st.st_size}:{st.st_mtime_ns}".encode())


def _update_tree(h: Any, label: str, root: Path) -> None:
    """Identity of a directory (an adapter): every regular file's name, size, mtime."""
    h.update(f"|{label}".encode())
    if root.is_file():
        _update_file(h, root.name, root, content=root.suffix == ".json")
        return
    try:
        files = sorted(p for p in root.rglob("*") if p.is_file())
    except OSError:
        files = []
    for p in files:
        _update_file(
            h, str(p.relative_to(root)), p, content=p.suffix in (".json", ".jinja")
        )


def checkpoint_fingerprint(
    model_path: str | Path,
    *,
    adapter_paths: tuple[str | Path, ...] = (),
    extra: Mapping[str, Any] | None = None,
    digest_size: int = 8,
) -> str:
    """Hex digest identifying the checkpoint at ``model_path`` and its cache layout.

    Covers: path, weight files (name, size, mtime), config / tokenizer / template /
    preprocessor contents, adapter files, ``extra`` (KV precision, cache layout tags,
    anything the caller's persisted format depends on) and ``FORMAT_VERSION``.
    A path that is not a local directory (a hub id) is identified by the string alone
    plus ``extra``.
    """
    h = hashlib.sha256(f"yunshu-kv-fingerprint:{FORMAT_VERSION}".encode())
    h.update(f"|path:{model_path}".encode())
    root = Path(str(model_path)).expanduser()
    if root.is_dir():
        seen: set[str] = set()
        for pattern in _STAT_GLOBS:
            for p in sorted(root.glob(pattern)):
                if p.name not in seen:
                    seen.add(p.name)
                    _update_file(h, p.name, p, content=False)
        for name in _CONTENT_FILES:
            if name not in seen and (root / name).exists():
                seen.add(name)
                _update_file(h, name, root / name, content=True)
    for adapter in adapter_paths:
        _update_tree(h, "adapter", Path(str(adapter)).expanduser())
    if extra:
        h.update(b"|extra:" + json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()[: 2 * digest_size]
