"""Per-row-length batch KV cache for the VLM runner's shared decode batch.

Replaces upstream mlx-lm ``BatchKVCache`` (one left-padded ``[B, H, L_max, D]``
array) for full-attention layers once rows are decoding:

- each row's keys sit at ``0 .. lengths[b] - 1`` of a ``[B, HKV, CAP, D]``
  buffer (no left padding); attention runs through
  ``ragged_attention.ragged_decode_attention`` and reads only the row's own
  keys;
- ``update_and_fetch`` returns the whole buffer (never a strided slice), so no
  per-layer ``mx.contiguous`` copy of the padded KV;
- ``extend`` takes a stock ``BatchKVCache`` (a freshly prefilled row) or
  another ``RaggedKVCache``; ``filter`` drops finished rows and releases
  capacity when the longest row leaves.

It deliberately has no ``left_padding`` attribute: mlx-vlm's qwen3_5 model keys
its left-padded decode and single-row shortcuts off that attribute, and neither
applies here. Prefill (one request at a time in the runner) stays on the stock
cache; the runner converts a generation batch's caches after each step.

Opt-in (``YUNSHU_RAGGED_KV=1``) until measured end to end.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .ragged_attention import ragged_decode_attention

STEP = 256


def _round_up(n: int) -> int:
    return -(-max(int(n), 1) // STEP) * STEP


class RaggedKVCache:
    """Batch KV with per-row lengths (see module docstring)."""

    def __init__(
        self, keys: mx.array | None, values: mx.array | None, lengths: list[int]
    ):
        self.keys = keys
        self.values = values
        self.lengths = [int(n) for n in lengths]
        self._sync()

    # ── bookkeeping ──────────────────────────────────────────────────────
    def _sync(self) -> None:
        # ``offset``: per-row positions of the next token (the model builds
        # rotary positions from it before update_and_fetch). ``_idx``: the
        # longest row, read by the model where it expects a scalar offset.
        self.offset = mx.array(self.lengths, dtype=mx.int32)
        self._idx = max(self.lengths) if self.lengths else 0
        self.metadata_revision = getattr(self, "metadata_revision", 0) + 1

    @property
    def capacity(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])

    @property
    def state(self):
        return [a for a in (self.keys, self.values) if a is not None]

    def empty(self) -> bool:
        return self.keys is None

    def size(self) -> int:
        return self._idx

    def make_mask(self, n: int, **_kw):
        return None  # masking is by per-row length inside the kernel

    # ── construction ─────────────────────────────────────────────────────
    @classmethod
    def from_batch_kv(cls, cache: Any) -> RaggedKVCache:
        """Convert a left-padded mlx-lm ``BatchKVCache`` (valid keys of row b
        at ``left_padding[b] .. _idx - 1``) into right-aligned rows."""
        if cache.keys is None:
            return cls(None, None, [0] * int(cache.offset.shape[0]))
        idx = int(cache._idx)
        pads = [int(p) for p in cache.left_padding.tolist()]
        lengths = [idx - p for p in pads]
        B, H, _, D = cache.keys.shape
        Dv = cache.values.shape[3]
        cap = _round_up(max(lengths) + 1)
        ks, vs = [], []
        for b, p in enumerate(pads):
            n = lengths[b]
            k = cache.keys[b, :, p:idx, :]
            v = cache.values[b, :, p:idx, :]
            ks.append(mx.concatenate([k, mx.zeros((H, cap - n, D), k.dtype)], axis=1))
            vs.append(mx.concatenate([v, mx.zeros((H, cap - n, Dv), v.dtype)], axis=1))
        return cls(mx.stack(ks), mx.stack(vs), lengths)

    def _grow(self, needed: int) -> None:
        cap = _round_up(needed)
        if cap <= self.capacity:
            return
        B, H, old, D = self.keys.shape
        Dv = self.values.shape[3]
        self.keys = mx.concatenate(
            [self.keys, mx.zeros((B, H, cap - old, D), self.keys.dtype)], axis=2
        )
        self.values = mx.concatenate(
            [self.values, mx.zeros((B, H, cap - old, Dv), self.values.dtype)], axis=2
        )

    # ── model interface ──────────────────────────────────────────────────
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, T, D = keys.shape
        if self.keys is None:
            cap = _round_up(max(self.lengths) + T + 1)
            self.keys = mx.zeros((B, H, cap, D), keys.dtype)
            self.values = mx.zeros((B, H, cap, values.shape[3]), values.dtype)
        self._grow(max(self.lengths) + T + 1)
        if len(set(self.lengths)) == 1:
            n = self.lengths[0]
            self.keys[:, :, n : n + T, :] = keys
            self.values[:, :, n : n + T, :] = values
        else:
            for b, n in enumerate(self.lengths):
                self.keys[b, :, n : n + T, :] = keys[b]
                self.values[b, :, n : n + T, :] = values[b]
        self.lengths = [n + T for n in self.lengths]
        self._sync()
        return self.keys, self.values

    def attend(self, queries: mx.array, scale: float) -> mx.array:
        """Attention of this step's queries over each row's keys (after
        ``update_and_fetch`` stored them)."""
        return ragged_decode_attention(
            queries, self.keys, self.values, self.offset, scale, max_length=self._idx
        )

    # ── batch ops used by upstream GenerationBatch ───────────────────────
    def filter(self, batch_indices) -> None:
        idx = [
            int(i)
            for i in (
                batch_indices.tolist()
                if hasattr(batch_indices, "tolist")
                else batch_indices
            )
        ]
        self.lengths = [self.lengths[i] for i in idx]
        if self.keys is not None:
            sel = mx.array(idx, dtype=mx.int32)
            self.keys = self.keys[sel]
            self.values = self.values[sel]
            need = _round_up(max(self.lengths, default=0) + 1)
            if self.lengths and need * 2 <= self.capacity:
                # the longest row left: give its capacity back
                self.keys = mx.contiguous(self.keys[:, :, :need, :])
                self.values = mx.contiguous(self.values[:, :, :need, :])
        self._sync()

    def extend(self, other: Any) -> None:
        if not isinstance(other, RaggedKVCache):
            other = RaggedKVCache.from_batch_kv(other)
        if other.keys is None:
            self.lengths += other.lengths
            self._sync()
            return
        if self.keys is None:
            self.keys, self.values = other.keys, other.values
            self.lengths += other.lengths
            self._sync()
            return
        cap = max(self.capacity, other.capacity)
        for c in (self, other):
            if c.capacity < cap:
                c._grow(cap)
        self.keys = mx.concatenate([self.keys, other.keys], axis=0)
        self.values = mx.concatenate([self.values, other.values], axis=0)
        self.lengths += other.lengths
        self._sync()

    def trim(self, n: int) -> int:
        """Speculative rollback: drop the last ``n`` keys of every row."""
        n = min(int(n), min(self.lengths, default=0))
        self.lengths = [x - n for x in self.lengths]
        self._sync()
        return n


_STATE = {"installed": False}


def install() -> bool:
    """Route qwen3_5 attention through the ragged kernel when its cache is a
    ``RaggedKVCache`` (any other cache keeps the stock path)."""
    if _STATE["installed"]:
        return True
    from mlx_vlm.models.qwen3_5 import language as lang

    cls = lang.Qwen3_5Attention
    orig = cls.__call__

    def __call__(
        self, x, mask=None, cache=None, position_ids=None, position_embeddings=None
    ):
        if not isinstance(cache, RaggedKVCache):
            return orig(self, x, mask, cache, position_ids, position_embeddings)
        B, L, _ = x.shape
        queries, _keys, _values, gate, _ = self._prepare_projected_qkv(
            self.q_proj(x),
            self.k_proj(x),
            self.v_proj(x),
            cache,
            position_ids,
            position_embeddings,
            None,
        )
        out = cache.attend(queries, self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out * mx.sigmoid(gate))

    cls.__call__ = __call__
    _STATE["installed"] = True
    return True


def convert_batch(prompt_cache: list) -> int:
    """Swap stock ``BatchKVCache`` entries of a generation batch for ragged
    ones; returns how many were converted (0 when already ragged)."""
    from mlx_lm.models.cache import BatchKVCache

    n = 0
    for i, c in enumerate(prompt_cache):
        if type(c) is BatchKVCache and c.keys is not None:
            prompt_cache[i] = RaggedKVCache.from_batch_kv(c)
            n += 1
    return n


__all__ = ["RaggedKVCache", "convert_batch", "install"]
