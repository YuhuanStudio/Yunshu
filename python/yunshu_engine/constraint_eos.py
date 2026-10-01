"""Shared EOS-id normalisation for every constrained-decoding path."""

from __future__ import annotations

from typing import Any


def normalize_eos_ids(tokenizer: Any) -> list[int]:
    """Return the tokenizer's EOS ids as a deduplicated ``list[int]``.

    Handles ``int``, any iterable, ``None`` and empty collections for both
    ``eos_token_ids`` and ``eos_token_id`` (mlx-lm / HF tokenizers disagree on
    the shape).  Never raises on odd shapes.
    """
    out: list[int] = []
    for name in ("eos_token_ids", "eos_token_id"):
        val = getattr(tokenizer, name, None)
        if val is None:
            continue
        if isinstance(val, bool):
            continue
        if isinstance(val, int):
            items: Any = [val]
        else:
            try:
                items = list(val)
            except TypeError:
                continue
        for t in items:
            if isinstance(t, int) and not isinstance(t, bool) and t not in out:
                out.append(int(t))
        if out:
            break
    return out


class ConstrainedDecodingError(ValueError):
    """The constraint reached a dead end: no token can continue a valid output.

    An empty allowed set is never answered with the unconstrained argmax (that
    would emit exactly the token the constraint forbade).  At an accepting
    state the allowed set already contains the normalised EOS ids; reaching
    this error means no EOS is legal either, so the request fails closed.
    """
