"""Per-request logits processors for the text fast path (mlx-lm ``generate_step``).

frequency/presence penalty, ``logit_bias``, ``suppress_tokens`` and ``min_tokens`` used to
walk their token ids and issue 2-3 tiny MLX ops per id per decoded token
(``logits[..., tid] = logits[..., tid] - x``), so a presence penalty over a 300-token
answer launched ~900 extra kernels per step and grew with the generation. These versions
keep a dense (vocab,) vector, patched with ONE scatter per newly generated token, and
apply it in one or two elementwise ops. The vectors are cast to the logits dtype before
the op, which is what the per-id scalar arithmetic did, so the result is bit-identical
(tests/unit/test_text_logit_processors.py checks float32 and bfloat16 against the old
loops).

Signature is mlx-lm's ``(tokens, logits) -> logits``. With processors set, mlx-lm's
``generate_step`` concatenates EVERY chunk it prefills (the whole uncached prompt suffix, last
token included) into ``tokens`` before the generated ids, so ``tokens`` is
``[prompt suffix..., gen1, gen2, ...]``. Processors that count generated tokens take
``n_prompt`` = the length of the ids handed to ``generate_step`` (``set_prompt_len``, once that is
known); ``tokens`` is not just ``[last_prompt_token, ...]`` (that held on older mlx-lm).
"""

from __future__ import annotations

from typing import cast

import mlx.core as mx


def _ids(tokens) -> list[int]:
    return tokens.tolist() if hasattr(tokens, "tolist") else [int(t) for t in tokens]


class FrequencyPresencePenalty:
    """``logits[t] -= fp * count[t]`` then ``-= pp`` for every generated token ``t``.

    OpenAI permits both in [-2, 2]; a negative value boosts the token, so neither is
    gated on its sign. Counts only cover generated tokens (``tokens[n_prompt:]``) and are
    rebuilt when ``tokens`` shrinks (speculative rollback).
    """

    def __init__(self, frequency_penalty: float, presence_penalty: float, n_prompt=1):
        self.fp = frequency_penalty
        self.pp = presence_penalty
        self.n_prompt = n_prompt
        self._counts: dict[int, int] = {}
        self._last_len = -1
        self._vocab = -1
        self._fpc: mx.array | None = None  # fp * count[t]
        self._pres: mx.array | None = None  # pp where count[t] > 0

    def set_prompt_len(self, n: int) -> None:
        """Length of the ids handed to ``generate_step`` (what ``tokens`` holds before gen1)."""
        self.n_prompt = int(n)

    def _dense(self, ids, vocab: int) -> None:
        """Write the entries for ``ids`` (all of them when ``vocab`` changed/reset)."""
        ids = [t for t in ids if 0 <= t < vocab]
        if self._vocab != vocab:
            self._vocab = vocab
            self._fpc = mx.zeros((vocab,), dtype=mx.float32)
            self._pres = mx.zeros((vocab,), dtype=mx.float32)
        if not ids:
            return
        idx = mx.array(ids)
        if self.fp != 0.0:
            cast(mx.array, self._fpc)[idx] = mx.array(
                [self.fp * self._counts[t] for t in ids]
            )
        if self.pp != 0.0:
            cast(mx.array, self._pres)[idx] = mx.array([self.pp] * len(ids))

    def __call__(self, tokens, logits):
        cur_len = len(tokens)
        if self.n_prompt is None:  # the first call precedes the first generated token
            self.n_prompt = cur_len
        if cur_len <= self.n_prompt:
            self._last_len = cur_len
            return logits
        vocab = logits.shape[-1]
        rebuild = cur_len < self._last_len or self._last_len < self.n_prompt
        if rebuild:
            self._counts = {}
            self._vocab = -1  # zero the dense vectors
            new = _ids(tokens[self.n_prompt :])
        else:
            new = _ids(tokens[max(self._last_len, self.n_prompt) :])
        touched: dict[int, None] = {}
        for t in new:
            self._counts[t] = self._counts.get(t, 0) + 1
            touched[t] = None
        self._last_len = cur_len
        if self._vocab != vocab:  # first call, rebuild, or a different logits width
            self._dense(list(self._counts), vocab)
        else:
            self._dense(list(touched), vocab)
        dt = logits.dtype
        if self.fp != 0.0:
            logits = logits - self._fpc.astype(dt)
        if self.pp != 0.0:
            logits = logits - self._pres.astype(dt)
        return logits


class GeneratedTokenPenalty:
    """OpenAI frequency / presence penalty for the VLM paths: every GENERATED token counts
    (``logits[t] -= fp * count[t]`` then ``-= pp`` once ``count[t] > 0``), not mlx-lm's window of
    the last 20 prompt + generated tokens. Same semantics as :class:`FrequencyPresencePenalty`
    (the text path); this one is driven by the VLM batch generator, which calls
    ``process_last_token(token, logits)`` with each token the step consumed (the first
    generated token is the first one fed, so nothing generated is counted before it).

    A speculative window is pure: ``window_row(fed, logits)`` applies the committed counts plus
    the tokens ``fed`` (the bonus and the drafts before this row), changing nothing, and
    ``commit_tokens(fed)`` counts what the window accepted. Both paths run the same arithmetic
    on the same ``[1, V]`` row, so speculation on equals speculation off bit for bit.
    """

    def __init__(self, frequency_penalty: float, presence_penalty: float):
        self.fp = float(frequency_penalty or 0.0)
        self.pp = float(presence_penalty or 0.0)
        self._cnt: mx.array | None = None  # (vocab,) float32 counts of the fed tokens
        self._vocab = -1

    def _counts(self, vocab: int) -> mx.array:
        if self._cnt is None or self._vocab != vocab:
            self._cnt = mx.zeros((vocab,), dtype=mx.float32)
            self._vocab = vocab
        return self._cnt

    @staticmethod
    def _add(cnt: mx.array, ids) -> mx.array:
        ids = _in_range(ids, cnt.shape[0])
        if not ids:
            return cnt
        return cnt.at[mx.array(ids, dtype=mx.int32)].add(1.0)

    def _apply(self, cnt: mx.array, logits: mx.array) -> mx.array:
        dt = logits.dtype
        if self.fp != 0.0:
            logits = logits - (cnt * self.fp).astype(dt)
        if self.pp != 0.0:
            logits = logits - mx.where(cnt > 0, self.pp, 0.0).astype(dt)
        return logits

    def __call__(self, tokens, logits):
        return logits  # the prompt pass: nothing generated yet

    def process_last_token(self, token: int, logits: mx.array) -> mx.array:
        self._cnt = self._add(self._counts(logits.shape[-1]), [int(token)])
        return self._apply(self._cnt, logits)

    def window_row(self, fed, logits: mx.array) -> mx.array:
        cnt = self._add(self._counts(logits.shape[-1]), [int(t) for t in fed])
        return self._apply(cnt, logits)

    def commit_tokens(self, fed) -> None:
        # a window always ran ``window_row`` first, so the vector exists
        if self._cnt is not None and len(fed):
            self._cnt = self._add(self._cnt, [int(t) for t in fed])


def _in_range(ids, vocab: int) -> list[int]:
    return [t for t in ids if 0 <= t < vocab]


class LogitBias:
    """``logits[t] += bias[t]``; out-of-range ids are ignored (OpenAI ignores them)."""

    def __init__(self, biases: dict[int, float]):
        self._biases = dict(biases)
        self._vocab = -1
        self._vec: mx.array | None = None

    def __call__(self, _tokens, logits):
        vocab = logits.shape[-1]
        if self._vocab != vocab:
            self._vocab = vocab
            ids = _in_range(self._biases, vocab)
            vec = mx.zeros((vocab,), dtype=mx.float32)
            if ids:
                vec[mx.array(ids)] = mx.array([self._biases[t] for t in ids])
            self._vec = vec
        return logits + self._vec.astype(logits.dtype)


class TokenBan:
    """Set the logits of ``ids`` to -inf, optionally only while fewer than ``floor``
    tokens were generated (``min_tokens``: ``len(tokens) - n_prompt`` counts generated ones;
    ``n_prompt`` is the length of the ids given to ``generate_step`` on the mlx-lm text path
    (``set_prompt_len``), the prompt length in the engine loop; both see the whole prompt plus
    the generated ids in ``tokens``)."""

    def __init__(self, ids, floor: int | None = None, n_prompt: int | None = 1):
        self._ids = [int(t) for t in ids]
        self._floor = floor
        self._n_prompt = n_prompt
        self._vocab = -1
        self._mask: mx.array | None = None

    def set_prompt_len(self, n: int) -> None:
        self._n_prompt = int(n)

    def __call__(self, tokens, logits):
        if self._n_prompt is None:  # the first call precedes the first generated token
            self._n_prompt = len(tokens)
        if self._floor is not None and (len(tokens) - self._n_prompt) >= self._floor:
            return logits
        vocab = logits.shape[-1]
        if self._vocab != vocab:
            self._vocab = vocab
            ids = _in_range(self._ids, vocab)
            mask = mx.zeros((vocab,), dtype=mx.bool_)
            if ids:
                mask[mx.array(ids)] = True
            self._mask = mask
        return mx.where(self._mask, mx.array(-float("inf"), logits.dtype), logits)
