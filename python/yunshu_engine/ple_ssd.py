"""Serve a Qwen3.8-Flash-Next (qwen4_exp) n-gram embedding (PLE) table from the checkpoint files.

The PLE table is ~30 GB (29.8 GiB on Jundot oQ2.5e), and a token touches only a few dozen of its
rows. Loaded like any weight, it sits in memory where macOS compresses the untouched pages (that
was the 30 GB "swap" every Flash-Next job showed), pushing the oQ3e pack against Metal's working set.
mlx-vlm can read the table row by row from the safetensors files instead
(``text_config.ple_storage``, ``QuantizedMMapNGramEmbedding``); until now that needed a hand-built
model view (``scripts/research/flashnext80_prepare_view.py``).

``external_ple`` does the same at load time without touching the model directory: it builds the row
manifest once under ``~/.yunshu/cache/ple/<key>/`` (byte offsets into the pack's own shards; no
payload is copied), points ``ple_storage`` at it and drops the table from the quantization map; the
loader then skips the table's tensors. The rows read are byte-identical, so output is unchanged.
A pack that already declares ``ple_storage`` (a prepared view) is used as it is.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from . import paths, settings

logger = logging.getLogger(__name__)

# mlx_vlm.models.qwen4_exp.ple_storage.PLE_MARKER: the n-gram table shards
PLE_MARKER = ".ple.ple_embedding.ngram_embedding.shards."


def _cache_key(model_path: Path, index_path: Path) -> str:
    st = index_path.stat()
    raw = f"{model_path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def manifest_for(model_path: Path) -> Path:
    """Build (once) and return the row manifest for ``model_path``'s PLE table."""
    index_path = model_path / "model.safetensors.index.json"
    target = (
        paths.home()
        / "cache"
        / "ple"
        / _cache_key(model_path, index_path)
        / "ple-store.json"
    )
    if target.exists():
        return target
    from mlx_vlm.models.qwen4_exp.ple_storage import build_quantized_ple_manifest

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"ple-store.{os.getpid()}.partial.json")
    build_quantized_ple_manifest(model_path, partial)
    os.replace(partial, target)  # an interrupted build never leaves a manifest behind
    return target


def external_ple(config: dict, model_path) -> bool:
    """Switch ``config`` to row-wise PLE reads from the pack's files when that applies.

    Returns True when the loader must skip the table's tensors (``is_ple_tensor``)."""
    if config.get("model_type") != "qwen4_exp":
        return False
    text = config.get("text_config")
    if not isinstance(text, dict) or text.get("ple_storage"):
        return False
    if settings.get("YUNSHU_PLE_STORAGE") != "ssd":
        return False
    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return False
    weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
    if not any(PLE_MARKER in key for key in weight_map):
        return False
    manifest = manifest_for(model_path)
    text["ple_storage"] = {"manifest": str(manifest), "cache_rows": 0}
    quantization = config.get("quantization")
    if isinstance(quantization, dict):
        config["quantization"] = {
            k: v for k, v in quantization.items() if PLE_MARKER not in k
        }
    logger.info(
        "Qwen4 PLE: n-gram table read from the checkpoint files (manifest %s); "
        "YUNSHU_PLE_STORAGE=memory loads it instead",
        manifest,
    )
    return True


def is_ple_tensor(key: str) -> bool:
    return PLE_MARKER in key


def bytes_read_from_files(model_path) -> int:
    """Bytes of the PLE table ``external_ple`` will leave on disk for this pack (0 when it does not
    apply), from the safetensors headers alone, so the load-time memory plan does not count them."""
    model_path = Path(model_path)
    try:
        config = json.loads((model_path / "config.json").read_text())
        if (
            config.get("model_type") != "qwen4_exp"
            or (config.get("text_config") or {}).get("ple_storage")
            or settings.get("YUNSHU_PLE_STORAGE") != "ssd"
        ):
            return 0
        weight_map = json.loads(
            (model_path / "model.safetensors.index.json").read_text()
        )["weight_map"]
    except (OSError, ValueError, KeyError):
        return 0
    total = 0
    for file_name in sorted({f for k, f in weight_map.items() if PLE_MARKER in k}):
        with (model_path / file_name).open("rb") as fh:
            header = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
        for key, entry in header.items():
            if key != "__metadata__" and PLE_MARKER in key:
                start, end = entry["data_offsets"]
                total += end - start
    return total
