# Patches upstream (MIT): mlx-vlm qwen4_exp PLE lookups gain the checkpoint's per-table scale
# (tracked in vendor.json; Blaizzy/mlx-vlm).
"""Per-table scale of the Qwen4-Exp n-gram (PLE) embedding.

Some packs (Jundot's oQ4e) store the table at a different magnitude and ship the factor as
``...ple_embedding.ngram_embedding.weight_scale`` (0.0002 on Flash-Next oQ4e; absent on mlx-community
conversions).  mlx-vlm 0.7.6 discards that tensor, so every looked-up row is ~5000x too large and the
model emits noise.  Looked-up bf16 rows are multiplied by the scale in float32 and rounded once to bf16,
exactly as the checkpoint's reference runtime does (identity for scale 1, so packs without the tensor are
untouched).
"""

from __future__ import annotations

import re
from typing import Any

_LAYER = re.compile(
    r"(?:^|\.)layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
)


def collect_table_scales(weights: dict) -> dict[int, float]:
    """{decoder layer index: scale} from tensors named ``...layers.N.ple.ple_embedding.ngram_embedding.weight_scale``."""
    import mlx.core as mx

    out: dict[int, float] = {}
    for key, value in weights.items():
        match = _LAYER.search(key)
        if not match:
            continue
        if value.size != 1:
            raise ValueError(
                f"{key}: expected one n-gram table scale, got shape {tuple(value.shape)}"
            )
        out[int(match.group(1))] = float(value.astype(mx.float32).reshape(-1)[0].item())
    return out


def scaled_rows(rows: Any, scale: float) -> Any:
    """Rows times the table scale, rounded once to the rows' dtype (identity for scale 1)."""
    if scale == 1.0:
        return rows
    import mlx.core as mx

    return (rows.astype(mx.float32) * scale).astype(rows.dtype)


def apply_table_scales(language_model: Any, scales: dict[int, float]) -> int:
    """Make each PLE table's lookup apply its scale.  Returns how many tables were scaled (scale != 1)."""
    applied = 0
    layers = language_model.model.layers
    for index, scale in scales.items():
        if scale == 1.0:
            continue
        table = layers[index].ple.ple_embedding.ngram_embedding
        base = type(table)
        if not getattr(base, "_yunshu_scaled", False):

            def __call__(self, indices, _base=base):
                return scaled_rows(_base.__call__(self, indices), self.table_scale)

            table.__class__ = type(
                "Scaled" + base.__name__,
                (base,),
                {"__call__": __call__, "lookup": __call__, "_yunshu_scaled": True},
            )
        table.table_scale = scale
        applied += 1
    return applied
