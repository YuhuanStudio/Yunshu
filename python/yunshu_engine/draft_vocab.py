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
would grow without bound. If those ids are mostly Chinese (``script_ids``: every
vocabulary id whose text holds a Han character, CJK punctuation or a full-width form;
55.5k of Qwen's 248k ids, all but 61 above the base set), the readout adds that
script set to the base and keeps the extension for the rest: about half the full
head per draft instead of all of it. Any other script goes back to the full
vocabulary for the rest of the request (zh 1K: 53.5, same as no reduction).
"""

from __future__ import annotations

import re
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .prompt_ids import ids_array

_CJK = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]"
)
SCRIPT_SHARE = 0.8  # of a request's high ids that must be in the script set


def cjk_ids(tokenizer: Any, vocab: int) -> np.ndarray | None:
    """Sorted ids whose decoded text holds a Han character, CJK punctuation or a
    full-width form (Simplified and Traditional alike); None without a tokenizer.
    One decode per id: 0.26 s for Qwen's 248k vocabulary."""
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return None
    n = min(int(vocab), len(tokenizer)) if hasattr(tokenizer, "__len__") else int(vocab)
    out = [i for i in range(n) if _CJK.search(decode([i]))]
    return np.asarray(out, dtype=np.int64) if out else None


def _as_ids(token_ids):
    """The prompt's int32 array (shared with the APC lookup's), else a list of ints."""
    arr = ids_array(token_ids)
    return arr if arr is not None else [int(t) for t in token_ids]


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
        # Media placeholder ids (image/video/audio pads, vision start/end): not text, so they
        # neither judge the prompt's script nor join the searched set. Without this an image
        # prompt (thousands of image-pad ids above the base set) read the full vocabulary.
        self.skip_ids: frozenset[int] = frozenset()
        # Chinese script set (see the module doc): ids above the base set, rows built on the
        # first Chinese request and kept (Qwen 27B: 55.5k rows, ~160 MB).
        self.script_ids: np.ndarray | None = None
        self._script: tuple | None = None  # (mx ids, rows)
        self.script = False
        self.full = False
        self._high_ids: list[int] = []
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

    def _text(self, token_ids):
        ids = _as_ids(token_ids)
        if not self.skip_ids:
            return ids
        if isinstance(ids, np.ndarray):
            return ids[~np.isin(ids, np.fromiter(self.skip_ids, dtype=ids.dtype))]
        return [t for t in ids if t not in self.skip_ids]

    def set_script_ids(self, ids) -> None:
        """Install the script set (ids at or below the base set are dropped)."""
        if ids is None:
            self.script_ids = None
            return
        arr = np.unique(np.asarray(ids, dtype=np.int64))
        arr = arr[(arr >= self.keep) & (arr < self.vocab)]
        self.script_ids = arr if arr.size else None
        self._script = None

    def _script_share(self, high: np.ndarray) -> float:
        if self.script_ids is None or high.size == 0:
            return 0.0
        return float(np.count_nonzero(np.isin(high, self.script_ids))) / high.size

    def _escalate(self, high: np.ndarray) -> None:
        """The request is mostly above the base set: the script set when those ids are
        mostly in it, else the full vocabulary."""
        if self.script_ids is not None and self._script_share(high) >= SCRIPT_SHARE:
            self.script = True
            if self._script is None:
                ids = mx.array(self.script_ids.astype(np.uint32))
                rows = self._rows(ids)
                mx.eval(ids, *rows)
                self._script = (ids, rows)
            if self.extra_ids is not None and self.extra_rows is not None:
                keep = ~np.isin(np.array(self.extra_ids), self.script_ids)
                if not keep.all():
                    idx = mx.array(np.nonzero(keep)[0].astype(np.uint32))
                    if keep.any():
                        self.extra_ids = mx.take(self.extra_ids, idx)
                        self.extra_rows = tuple(
                            mx.take(a, idx, axis=0) for a in self.extra_rows
                        )
                    else:
                        self.extra_ids = self.extra_rows = None
            return
        self.full = True  # a non-Latin script: search everything
        self.extra_ids = self.extra_rows = None

    def _add(self, wanted: list[int]) -> int:
        if isinstance(wanted, np.ndarray):
            # unique ids in [keep, vocab): the sort and range filter run in numpy, not per id
            sel = wanted[(wanted >= self.keep) & (wanted < self.vocab)]
            if self.script and self.script_ids is not None:
                sel = sel[~np.isin(sel, self.script_ids)]
            new = [t for t in np.unique(sel).tolist() if t not in self._seen]
        else:
            new = sorted(t for t in set(wanted) if self.keep <= t < self.vocab)
            if self.script and self.script_ids is not None:
                new = [t for t in new if not np.isin(t, self.script_ids)]
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
        self.script = False
        self._high = self._total = 0
        self._high_ids = []
        ids = self._text(token_ids)
        if (
            len(ids) >= PROMPT_MIN
            and self._mostly_high(ids)
            and self.fallback is not None
        ):
            self._escalate(self._high_of(ids))
        if self.full:
            return 0
        count = self._add(ids)
        if count:
            mx.eval(self.extra_ids, *self.extra_rows)
        return count

    def add_context(self, token_ids: list[int]) -> int:
        """Another request joins while others run: its prompt's ids join the
        extension (``set_context`` would drop the running requests')."""
        ids = self._text(token_ids)
        if self.full:
            return 0
        if (
            not self.script
            and len(ids) >= PROMPT_MIN
            and self._mostly_high(ids)
            and self.fallback
        ):
            self._escalate(self._high_of(ids))
            if self.full:
                return 0
        count = self._add(ids)
        if count and self.extra_ids is not None and self.extra_rows is not None:
            mx.eval(self.extra_ids, *self.extra_rows)
        return count

    def _high_of(self, ids) -> np.ndarray:
        arr = np.asarray(ids, dtype=np.int64)
        return arr[(arr >= self.keep) & (arr < self.vocab)]

    def _mostly_high(self, ids: list[int]) -> bool:
        if isinstance(ids, np.ndarray):
            high = int(np.count_nonzero((ids >= self.keep) & (ids < self.vocab)))
        else:
            high = sum(1 for t in ids if self.keep <= t < self.vocab)
        return high * 2 >= len(ids)

    def learn(self, token_ids: list[int]) -> int:
        """Add ids the target committed (lazy: built with the next draft's graph)."""
        if self.full:
            return 0
        ids = [int(t) for t in token_ids]
        if self.script:
            return self._add(ids)
        self._total += len(ids)
        high = [t for t in ids if self.keep <= t < self.vocab]
        self._high += len(high)
        self._high_ids.extend(high)
        if (
            self.fallback is not None
            and self._total >= COMMITTED_MIN
            and self._high * 2 >= self._total
        ):
            self._escalate(np.asarray(self._high_ids, dtype=np.int64))
            if self.full:
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

    def logits_subset(self, hidden: mx.array) -> tuple[mx.array, mx.array] | None:
        """Logits over the searched ids, ``([rows, n], ids [n])`` for ``hidden`` ``[..., D]``;
        None once the request is in a script the base set does not cover (the caller then
        reads the full vocabulary). Shared by the MTP chain / tree and the DFlash lattice:
        a draft only proposes, the target verifies it, so a reduced candidate set cannot
        change an output token."""
        if self.full:
            return None
        x = hidden.reshape(-1, hidden.shape[-1])
        logits = self._matmul(x, self.rows)
        ids = self.ids
        if self.script and self._script is not None:
            logits = mx.concatenate([logits, self._matmul(x, self._script[1])], axis=-1)
            ids = mx.concatenate([ids, self._script[0]])
        if self.extra_ids is not None and self.extra_rows is not None:
            logits = mx.concatenate([logits, self._matmul(x, self.extra_rows)], axis=-1)
            ids = mx.concatenate([ids, self.extra_ids])
        return logits, ids

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
        chunks = []
        if self.script and self._script is not None:
            chunks.append(self._script)
        if self.extra_ids is not None and self.extra_rows is not None:
            chunks.append((self.extra_ids, self.extra_rows))
        if chunks:
            top = mx.max(logits, axis=-1)
            for c_ids, c_rows in chunks:
                c_logits = self._matmul(x, c_rows)
                c_top = mx.max(c_logits, axis=-1)
                c_best = mx.take(c_ids, mx.argmax(c_logits, axis=-1)).astype(mx.int32)
                best = mx.where(c_top > top, c_best, best)
                top = mx.maximum(top, c_top)
        return best.reshape(lead)


def install(
    drafter: Any, target_language_model: Any, keep: int, tokenizer: Any = None
) -> DraftVocab | None:
    """Route ``drafter``'s greedy readout through a ``DraftVocab`` of the target's
    quantized head (its ``lm_head``, or the embedding table when tied); None when
    the head is not affine 4-bit quantized."""
    lm = target_language_model
    head: Any = getattr(lm, "lm_head", None)
    tied = head is None
    if tied:
        head = lm.model.embed_tokens
    try:
        vocab = DraftVocab(head, keep)
    except TypeError:
        return None
    if vocab.bits != 4 or getattr(head, "mode", "affine") != "affine":
        return None

    greedy = getattr(drafter, "_greedy_token", None)
    # a DFlash drafter has no greedy readout: its lattice reads logits_subset instead, and the
    # fallback only marks that a full-vocabulary readout exists
    if greedy is not None:
        # The full-vocabulary readout reads the target head directly. Upstream's own
        # _greedy_token needs drafter.bind(target), which only upstream's generate loop
        # calls; on the round driver it was None, so a mostly non-Latin prompt (or a media
        # prompt full of image-pad ids) crashed the MTP draft on its first full readout.
        head_fn = head.as_linear if tied else head

        def _full_greedy(hidden: mx.array) -> mx.array:
            return mx.argmax(head_fn(hidden), axis=-1)

        vocab.fallback = _full_greedy
    else:
        vocab.fallback = drafter._logits

    if greedy is not None:

        def _greedy_token(hidden: mx.array) -> mx.array:
            return vocab.argmax(hidden)

        object.__setattr__(drafter, "_greedy_token", _greedy_token)
    if tokenizer is not None:
        try:
            vocab.set_script_ids(
                cjk_ids(getattr(tokenizer, "_tokenizer", tokenizer), vocab.vocab)
            )
        except (
            Exception
        ):  # a tokenizer without per-id decode: other scripts read the full head
            vocab.set_script_ids(None)
    object.__setattr__(drafter, "_draft_vocab", vocab)
    return vocab


__all__ = ["DraftVocab", "install"]
