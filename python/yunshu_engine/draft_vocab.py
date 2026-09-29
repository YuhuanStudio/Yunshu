"""A reduced-vocabulary greedy readout for the MTP head's drafts.

Every MTP draft is an argmax over the whole vocabulary: on Qwen3.8-27B the
quantized LM head is 0.72 GB, so each of the ~5 drafts in a cycle streams it once
(about 1.4 ms at the machine's bandwidth) — more than the head's own decoder
layer. A draft only proposes; the target model verifies every token it commits, so
the drafter may search a smaller vocabulary without changing any output: a draft
outside the searched set is simply a miss the target corrects.

``DraftVocab`` keeps the rows of the quantized head for the ``keep`` first ids
(byte-pair ids follow merge order, so the low ids are the frequent tokens) plus the
token ids each request's prompt contains. The readout is one ``quantized_matmul``
over that subset and an argmax, mapped back to full-vocabulary ids.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

DRAFT_VOCAB = 65536  # ids searched besides the prompt's (27B: 95.5 vs 86.5 tok/s, code)


class DraftVocab:
    def __init__(self, head: Any, keep: int):
        if not (
            isinstance(head, nn.QuantizedEmbedding | nn.QuantizedLinear)
            or hasattr(head, "quantized_rows")
        ):
            raise TypeError("draft vocabulary needs a quantized head")
        self.head = head
        self.bits = int(head.bits)
        self.group_size = int(head.group_size)
        self.vocab = int(
            head.output_dims if hasattr(head, "output_dims") else head.weight.shape[0]
        )
        self.keep = min(int(keep), self.vocab)
        self._base_ids = mx.arange(self.keep, dtype=mx.uint32)
        self._base = self._rows(self._base_ids)
        mx.eval(self._base, self._base_ids)
        self.ids, self.rows = self._base_ids, self._base

    def _rows(self, ids: mx.array) -> tuple:
        if hasattr(self.head, "quantized_rows"):  # packed layouts
            return tuple(self.head.quantized_rows(ids))
        h = self.head
        return (
            mx.take(h.weight, ids, axis=0),
            mx.take(h.scales, ids, axis=0),
            mx.take(h.biases, ids, axis=0),
        )

    def set_context(self, token_ids: list[int]) -> int:
        """Add the ids a request's prompt uses that the base set lacks; returns
        how many. Call before the request's first draft (host thread)."""
        seen = sorted({int(t) for t in token_ids if self.keep <= int(t) < self.vocab})
        if not seen:
            self.ids, self.rows = self._base_ids, self._base
            return 0
        extra_ids = mx.array(seen, dtype=mx.uint32)
        extra = self._rows(extra_ids)
        self.ids = mx.concatenate([self._base_ids, extra_ids])
        self.rows = tuple(
            mx.concatenate([b, x]) for b, x in zip(self._base, extra, strict=True)
        )
        mx.eval(self.ids, *self.rows)
        return len(seen)

    def argmax(self, hidden: mx.array) -> mx.array:
        """Greedy token ids for ``hidden`` [..., D], shaped like an argmax over
        full-vocabulary logits [..., V]."""
        lead = hidden.shape[:-1]
        w, s, b = self.rows
        logits = mx.quantized_matmul(
            hidden.reshape(-1, hidden.shape[-1]),
            w,
            s,
            b,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        pick = mx.argmax(logits, axis=-1)
        return mx.take(self.ids, pick).astype(mx.int32).reshape(lead)


def install(drafter: Any, target_language_model: Any, keep: int) -> DraftVocab | None:
    """Route ``drafter``'s greedy readout through a ``DraftVocab`` of the target's
    quantized head (its ``lm_head``, or the embedding table when tied); None when
    the head is not affine 4-bit quantized."""
    lm = target_language_model
    head = getattr(lm, "lm_head", None)
    if head is None:
        head = lm.model.embed_tokens
    try:
        vocab = DraftVocab(head, keep)
    except TypeError:
        return None
    if vocab.bits != 4 or getattr(head, "mode", "affine") != "affine":
        return None

    def _greedy_token(hidden: mx.array) -> mx.array:
        return vocab.argmax(hidden)

    object.__setattr__(drafter, "_greedy_token", _greedy_token)
    object.__setattr__(drafter, "_draft_vocab", vocab)
    return vocab


__all__ = ["DraftVocab", "install"]
