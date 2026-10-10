# Upstream (patches): Blaizzy/mlx-vlm (MIT) mlx_vlm/generate/ar.py PromptProcessingBatch @ v0.7.3
"""Give logits processors the full prompt as their token context on an APC hit.

mlx-vlm builds a prefill batch's ``_token_context`` from the tokens that batch prefills.
After an APC hit that is only the uncached suffix, so a repetition / presence / frequency
penalty (which counts the context) answered differently on a hit than on a miss. The
context is the whole prompt either way: hit == miss.
"""

from __future__ import annotations

_STATE = {"installed": False}


def full_context(batch) -> None:
    """Replace each row's suffix context with ``full_input_ids`` from its APC metadata."""
    contexts = getattr(batch, "_token_context", None)
    metas = getattr(batch, "_apc_meta", None) or []
    if not contexts:
        return
    for i, meta in enumerate(metas):
        if meta is None or i >= len(contexts):
            continue
        full = meta.get("full_input_ids")
        if full is not None and int(meta.get("prefix_len") or 0) > 0:
            contexts[i] = [int(t) for t in full]


def install(cls=None) -> bool:
    if cls is None:
        if _STATE["installed"]:
            return True
        from mlx_vlm.generate import ar

        cls = ar.PromptProcessingBatch
    original = cls.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        full_context(self)

    cls.__init__ = __init__
    if cls.__module__.startswith("mlx_vlm"):
        _STATE["installed"] = True
    return True
