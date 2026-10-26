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

Signature is mlx-lm's ``(tokens, logits) -> logits``. What ``tokens`` holds depends on the
mlx-lm release: older ones passed ``[last_prompt_token, gen1, ...]`` (the rest of the prompt was
prefilled outside ``_step``), current ones (0.32) append every prefill chunk too, so ``tokens`` is
``[prompt..., gen1, ...]`` and its first length is the prompt length that THIS request prefilled
(1 after a prefix-cache hit, the whole prompt on a miss). ``n_prompt=None`` (what the mlx-lm
text paths pass) reads the offset off the first call, which is made before any token is generated,
so both shapes work; an int is the explicit offset (the default 1 is the old shape; the engine loop
passes the prompt length of its prompt + generated ``tokens``).
"""

from __future__ import annotations

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
            self._fpc[idx] = mx.array([self.fp * self._counts[t] for t in ids])
        if self.pp != 0.0:
            self._pres[idx] = mx.array([self.pp] * len(ids))

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
    ``n_prompt`` is read off the first call on the mlx-lm text path, and is the prompt length
    in the engine loop where ``tokens`` is the whole prompt plus the generated ids)."""

    def __init__(self, ids, floor: int | None = None, n_prompt: int | None = 1):
        self._ids = [int(t) for t in ids]
        self._floor = floor
        self._n_prompt = n_prompt
        self._vocab = -1
        self._mask: mx.array | None = None

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
