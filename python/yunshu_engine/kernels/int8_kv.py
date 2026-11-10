"""int8 K/V row cache for the round driver's DFlash / plain-verify batch (row 17, stage 2).

``Int8KVCache`` is a single-row attention cache whose ``keys`` / ``values`` are int8 codes
``[1, HKV, CAP, D]`` and whose ``k_scales`` / ``v_scales`` are fp16 scales per (KV head, token,
32-dim group) ``[1, HKV, CAP, D / 32]`` (``ragged_attention.quantize_kv_pair``: symmetric, ``max|x| / 127``,
codes use the fp16-rounded scale). Rows are quantized once on write; a prefill cache (bf16, restored from
the APC or computed) is quantized when the row joins the decode batch (:meth:`Int8KVCache.from_kv`), so a
cache hit and a miss quantize the same bf16 values and decode on identical int8 state.

Layout contract for kernels reading it (the tree attention):

* codes and scales are contiguous with ``CAP`` keys per head, ``CAP % 64 == 0`` (every growth is a
  multiple of ``step`` = 256 and ``from_kv`` pads to 64);
* the stored keys are ``0 .. offset - 1``; rows past ``offset`` are zero codes / zero scales;
* dequantization is defined as ``bf16(float(code) * float(scale))`` per element (:func:`dequantize_kv`), so a
  kernel that dequantizes tiles to bf16 and runs the bf16 tile arithmetic is bit-identical to the bf16
  kernel over ``dequantize_kv(...)`` of the cache (the reference ``tree_attention_int8_reference``).
"""

from __future__ import annotations

import mlx.core as mx
from mlx_vlm.models.cache import KVCache

from .ragged_attention import GROUP, quantize_kv_pair


def dequantize_kv(codes: mx.array, scales: mx.array) -> mx.array:
    """int8 codes [..., D] and fp16 scales [..., D / 32] -> bf16 [..., D] (``bf16(code * scale)``)."""
    *lead, d = codes.shape
    x = codes.astype(mx.float32).reshape(*lead, d // GROUP, GROUP)
    out = x * scales.astype(mx.float32)[..., None]
    return out.reshape(*lead, d).astype(mx.bfloat16)


def _pad_tokens(a: mx.array, n: int) -> mx.array:
    if n <= 0:
        return a
    return mx.concatenate(
        [a, mx.zeros((*a.shape[:2], n, a.shape[3]), dtype=a.dtype)], axis=2
    )


class Int8KVCache(KVCache):
    """Single-row KV cache stored as int8 codes + fp16 group scales (see the module docstring)."""

    offset: int
    keys: mx.array | None
    values: mx.array | None

    def __init__(self) -> None:
        super().__init__()
        self.k_scales: mx.array | None = None
        self.v_scales: mx.array | None = None

    # ── construction from a bf16 prefill cache ───────────────────────────
    @classmethod
    def from_kv(cls, cache: KVCache) -> Int8KVCache:
        """Quantize a bf16 ``KVCache`` (its first ``offset`` keys) once; capacity rounds up to 64."""
        out = cls()
        if cache.keys is None or int(cache.offset) == 0:
            out.offset = int(cache.offset)
            return out
        n = int(cache.offset)
        kq, ks, vq, vs = quantize_kv_pair(
            cache.keys[..., :n, :], cache.values[..., :n, :]
        )
        cap = -(-max(int(cache.keys.shape[2]), n) // 64) * 64
        out.keys, out.values = _pad_tokens(kq, cap - n), _pad_tokens(vq, cap - n)
        out.k_scales, out.v_scales = _pad_tokens(ks, cap - n), _pad_tokens(vs, cap - n)
        out.offset = n
        return out

    # ── writes ───────────────────────────────────────────────────────────
    def _reserve(self, need: int, like: mx.array) -> None:
        if self.keys is not None and need <= self.keys.shape[2]:
            return
        step = max(int(self.step), 64)
        cap = -(-need // step) * step
        if self.keys is None:
            b, h, _, d = like.shape
            self.keys = mx.zeros((b, h, cap, d), mx.int8)
            self.values = mx.zeros((b, h, cap, d), mx.int8)
            self.k_scales = mx.zeros((b, h, cap, d // GROUP), mx.float16)
            self.v_scales = mx.zeros((b, h, cap, d // GROUP), mx.float16)
            return
        grow = cap - int(self.keys.shape[2])
        self.keys = _pad_tokens(self.keys, grow)
        assert self.values is not None
        self.values = _pad_tokens(self.values, grow)
        assert self.k_scales is not None and self.v_scales is not None
        self.k_scales = _pad_tokens(self.k_scales, grow)
        self.v_scales = _pad_tokens(self.v_scales, grow)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Append bf16 K/V ``[1, HKV, T, D]``: quantized on write. Returns the stored codes (the
        tree attention reads ``keys`` / ``values`` / scales off the cache, not the return value)."""
        prev, t = self.offset, int(keys.shape[2])
        self._reserve(prev + t, keys)
        assert self.keys is not None and self.values is not None
        assert self.k_scales is not None and self.v_scales is not None
        kq, ks, vq, vs = quantize_kv_pair(keys, values)
        end = prev + t
        self.keys[..., prev:end, :] = kq
        self.values[..., prev:end, :] = vq
        self.k_scales[..., prev:end, :] = ks
        self.v_scales[..., prev:end, :] = vs
        self.offset = end
        return self.keys[..., :end, :], self.values[..., :end, :]

    def prefix_cache_reserve(self, min_capacity_tokens):
        if self.keys is None:
            return ()
        need = max(self.offset, int(min_capacity_tokens))
        step = max(int(self.step), 64)
        cap = -(-need // step) * step
        if cap > self.keys.shape[2]:
            self._reserve(cap, self.keys)
        return (self.keys, self.values)

    def extract(self, idx: int) -> Int8KVCache:
        """Extract native codes and scales for APC without re-quantizing them.

        Upstream KVCache.extract assumes a two-tensor state; this cache has four.
        Keep the logical prefix, as state does, and detach writes from the source.
        """
        batch_size = 1 if self.keys is None else int(self.keys.shape[0])
        if idx < 0:
            idx += batch_size
        if not 0 <= idx < batch_size:
            raise IndexError("Int8KVCache row index out of range")
        out = type(self)()
        out.state = tuple(
            None if a is None else mx.contiguous(a[idx : idx + 1]) for a in self.state
        )
        out.offset = self.offset
        return out

    # ── bookkeeping ──────────────────────────────────────────────────────
    @property
    def state(self):
        if self.keys is None:
            return None, None, None, None
        n = self.offset
        return (
            self.keys[..., :n, :],
            self.values[..., :n, :],
            self.k_scales[..., :n, :],
            self.v_scales[..., :n, :],
        )

    @state.setter
    def state(self, v):
        self.keys, self.values, self.k_scales, self.v_scales = v
        self.offset = 0 if self.keys is None else int(self.keys.shape[2])

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        assert (
            self.values is not None
            and self.k_scales is not None
            and self.v_scales is not None
        )
        return sum(
            int(a.nbytes)
            for a in (self.keys, self.values, self.k_scales, self.v_scales)
        )

    def dequantized(self, n: int | None = None) -> tuple[mx.array, mx.array]:
        """bf16 K, V of the first ``n`` (default: all ``CAP``) positions: the reference's input."""
        assert self.keys is not None and self.values is not None
        assert self.k_scales is not None and self.v_scales is not None
        end = int(self.keys.shape[2]) if n is None else int(n)
        return (
            dequantize_kv(self.keys[..., :end, :], self.k_scales[..., :end, :]),
            dequantize_kv(self.values[..., :end, :], self.v_scales[..., :end, :]),
        )


def to_bf16_cache(cache):
    """A bf16 ``KVCache`` holding ``dequantize_kv`` of an int8 row cache's stored keys.

    A restored int8 cache (a standing reply handed to its follow-up) must not enter a prefill as it
    is: the model's attention uses what ``update_and_fetch`` returns, which for an int8 cache is the
    raw codes. Converted here, the suffix prefills on the same values decode reads, and the row
    joins the batch through ``from_kv`` again (quantizing dequantized codes returns the same codes
    and scales), so the follow-up decodes on the state the reply left."""
    out = KVCache()
    n = int(cache.offset)
    if cache.keys is None or n == 0:
        out.offset = n
        return out
    out.keys, out.values = cache.dequantized(n)
    out.offset = n
    return out


def is_int8_cache(cache) -> bool:
    return getattr(cache, "k_scales", None) is not None


__all__ = ["Int8KVCache", "dequantize_kv", "is_int8_cache", "to_bf16_cache"]
