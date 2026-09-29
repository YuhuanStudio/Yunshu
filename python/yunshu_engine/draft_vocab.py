# Upstream (inspired): ashhart/TensorFold (MIT) src/tensorfold/drafters/dflash_drafter.py @ 34bae79a
"""A reduced-vocabulary greedy readout for the MTP head's drafts.

Every MTP draft is an argmax over the whole vocabulary: on Qwen3.8-27B the
quantized LM head is 0.72 GB, so each of the ~5 drafts in a cycle streams it once
(about 1.4 ms at the machine's bandwidth) — more than the head's own decoder
layer. A draft only proposes; the target model verifies every token it commits, so
the drafter may search a smaller vocabulary without changing any output: a draft
outside the searched set is simply a miss the target corrects.

``DraftVocab`` keeps the rows of the quantized head for the ``keep`` first ids
(byte-pair ids follow merge order, so the low ids are the frequent tokens) plus a
small per-request extension: the ids the request's prompt contains and every id
the target has committed so far. The extension is what keeps other scripts fast —
Chinese and Japanese text lives mostly above id 65536, so a fixed prefix drafted
it poorly (zh 1K: 38.5 vs 53.6 tok/s on the full vocabulary); once the target has
committed a few of a script's tokens the drafts find them (zh 1K: 49.7). The
readout is a ``quantized_matmul`` over the base rows and one over the extension,
an argmax of each, and the better of the two mapped back to full-vocabulary ids.

When half or more of a request's prompt (16+ ids) or of its committed tokens
(24+) lie above the base set, the request is in such a script and the extension
would grow without bound: the readout goes back to the full vocabulary for the
rest of the request (zh 1K: 53.5, same as no reduction).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

PROMPT_MIN = 16  # prompt ids before its script is judged
COMMITTED_MIN = 24  # committed tokens before the output's script is judged
DRAFT_VOCAB = (
    65536  # ids searched besides the request's own (27B: 95.5 vs 86.5 tok/s, code)
)


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
        self.extra_ids: mx.array | None = None
        self.extra_rows: tuple | None = None
        self._seen: set[int] = set()
        self.fallback: Any = None  # the drafter's own full-vocabulary readout
        self.full = False
        self._high = 0
        self._total = 0

    def _rows(self, ids: mx.array) -> tuple:
        if hasattr(self.head, "quantized_rows"):  # packed layouts
            return tuple(self.head.quantized_rows(ids))
        h = self.head
        return (
            mx.take(h.weight, ids, axis=0),
            mx.take(h.scales, ids, axis=0),
            mx.take(h.biases, ids, axis=0),
        )

    def _add(self, wanted: list[int]) -> int:
        new = sorted(t for t in set(wanted) if self.keep <= t < self.vocab)
        new = [t for t in new if t not in self._seen]
        if not new:
            return 0
        ids = mx.array(new, dtype=mx.uint32)
        rows = self._rows(ids)
        if self.extra_ids is None:
            self.extra_ids, self.extra_rows = ids, rows
        else:
            self.extra_ids = mx.concatenate([self.extra_ids, ids])
            self.extra_rows = tuple(
                mx.concatenate([a, b])
                for a, b in zip(self.extra_rows, rows, strict=True)
            )
        self._seen.update(new)
        return len(new)

    def set_context(self, token_ids: list[int]) -> int:
        """Start a request: the extension holds the ids its prompt uses that the
        base set lacks; returns how many. Call before the request's first draft
        (host thread)."""
        self.extra_ids = None
        self.extra_rows = None
        self._seen = set()
        self.full = False
        self._high = self._total = 0
        ids = [int(t) for t in token_ids]
        if len(ids) >= PROMPT_MIN and self._mostly_high(ids):
            self.full = self.fallback is not None
        if self.full:
            return 0
        count = self._add(ids)
        if count:
            mx.eval(self.extra_ids, *self.extra_rows)
        return count

    def _mostly_high(self, ids: list[int]) -> bool:
        high = sum(1 for t in ids if self.keep <= t < self.vocab)
        return high * 2 >= len(ids)

    def learn(self, token_ids: list[int]) -> int:
        """Add ids the target committed (lazy: built with the next draft's graph)."""
        if self.full:
            return 0
        ids = [int(t) for t in token_ids]
        self._total += len(ids)
        self._high += sum(1 for t in ids if self.keep <= t < self.vocab)
        if (
            self.fallback is not None
            and self._total >= COMMITTED_MIN
            and self._high * 2 >= self._total
        ):
            self.full = True  # a non-Latin script: search everything
            self.extra_ids = self.extra_rows = None
            return 0
        return self._add(ids)

    def _matmul(self, x: mx.array, rows: tuple) -> mx.array:
        w, s, b = rows
        return mx.quantized_matmul(
            x,
            w,
            s,
            b,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

    def argmax(self, hidden: mx.array) -> mx.array:
        """Greedy token ids for ``hidden`` [..., D], shaped like an argmax over
        full-vocabulary logits [..., V]."""
        if self.full:
            return self.fallback(hidden)
        lead = hidden.shape[:-1]
        x = hidden.reshape(-1, hidden.shape[-1])
        logits = self._matmul(x, self.rows)
        pick = mx.argmax(logits, axis=-1)
        best = mx.take(self.ids, pick).astype(mx.int32)
        if self.extra_ids is not None:
            top = mx.max(logits, axis=-1)
            x_logits = self._matmul(x, self.extra_rows)
            x_top = mx.max(x_logits, axis=-1)
            x_best = mx.take(self.extra_ids, mx.argmax(x_logits, axis=-1)).astype(
                mx.int32
            )
            best = mx.where(x_top > top, x_best, best)
        return best.reshape(lead)


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

    vocab.fallback = drafter._greedy_token

    def _greedy_token(hidden: mx.array) -> mx.array:
        return vocab.argmax(hidden)

    object.__setattr__(drafter, "_greedy_token", _greedy_token)
    object.__setattr__(drafter, "_draft_vocab", vocab)
    return vocab


__all__ = ["DraftVocab", "install"]
