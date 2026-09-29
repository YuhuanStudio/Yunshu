# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""Per-row-length batch KV cache for the VLM runner's shared decode batch.

Replaces upstream ``BatchKVCache`` (one left-padded ``[B, H, L_max, D]`` array)
for full-attention layers once two or more rows decode together:

- rows live in *slots* of one ``[S, HKV, CAP, D]`` buffer (``S >= B`` rows,
  ``CAP`` keys); batch row ``b`` keeps its ``lengths[b]`` keys at positions
  ``0 .. lengths[b] - 1`` of slot ``slots[b]``. Attention runs through
  ``ragged_attention.ragged_decode_attention`` and reads only the row's own
  keys;
- a decode step writes every row's new token(s) with one scatter per buffer
  (in place), not one slice update per row;
- ``extend`` copies only the joining rows' keys into free slots; ``filter``
  only drops slot ids. The buffer is reallocated (one copy of the live rows)
  when rows or keys outgrow it — rows grow to the next power of two, keys by
  ``STEP`` — and when the longest row leaves and half the keys are unused;
- ``update_and_fetch`` returns the whole buffer (never a strided slice), so no
  per-layer ``mx.contiguous`` copy.

``kv_format="int8"`` stores K/V as int8 codes plus fp16 scales per (slot, KV
head, token, 32-dim group), quantized on write, and the kernel reads them
directly (half the K/V bytes per decode step; see ``ragged_attention``).

It deliberately has no ``left_padding`` attribute: mlx-vlm's qwen3_5 model keys
its left-padded decode and single-row shortcuts off that attribute, and neither
applies here. Prefill (one request at a time in the runner) stays on the stock
cache; ``enable`` makes the generation batch's merge (upstream
``_extend_cache``) build ragged caches directly, so a single request never
converts and a join copies only the joining row.

The single-row speculative lane (``set_dense_lane``) uses the same kernels
without a ragged cache: its stock ``KVCache`` (decode) / one-row
``BatchKVCache`` (verify) buffer already has the ragged layout (one row, keys at
``0 .. n - 1``), so decode (T = 1) and MTP verify (T <= 8) read it in place on
the token-tile kernel (key-parallel where tensor ops are unavailable), whose
per-token bits do not depend on T — verify equals plain decode. The cache
contents, APC snapshots and the drafter's shared-KV reads are untouched; no
dense K/V view or copy is needed. The drafter's own one-row caches take the
same route (same math, only draft acceptance can move).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .ragged_attention import quantize_kv_pair, ragged_decode_attention, tile_ready

STEP = 256


def _round_up(n: int) -> int:
    return -(-max(int(n), 1) // STEP) * STEP


def _pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


# Per-step host-built index arrays, shared by every attention layer of a step
# (same slots, lengths and buffer shape).
_SHARED: dict = {}


def _shared(key: tuple, build):
    hit = _SHARED.get(key)
    if hit is None:
        if len(_SHARED) > 64:
            _SHARED.clear()
        hit = _SHARED[key] = build()
    return hit


def _scatter_rows(buf: mx.array, index: mx.array, x: mx.array) -> mx.array:
    """``buf`` [S, H, CAP, W] with token rows ``index`` (flat ``(slot * H + h)
    * CAP + pos``, shape [B, H, T]) set to ``x`` [B, H, T, W]: one in-place
    scatter for every row of the batch."""
    flat = buf.reshape(-1, buf.shape[-1])
    flat[index] = x
    return flat.reshape(buf.shape)


class RaggedKVCache:
    """Batch KV with per-row lengths in slot-addressed buffers (see module
    docstring)."""

    def __init__(self, kv_format: str = "bf16"):
        if kv_format not in ("bf16", "int8"):
            raise ValueError(f"RaggedKVCache: unknown kv_format {kv_format!r}")
        self.kv_format = kv_format
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        # int8 only: fp16 scales per (slot, KV head, token, 32-dim group),
        # [S, HKV, CAP, D/32].
        self.k_scales: mx.array | None = None
        self.v_scales: mx.array | None = None
        self.lengths: list[int] = []
        self.slots: list[int] = []
        self._sync()

    # ── bookkeeping ──────────────────────────────────────────────────────
    def _sync(self) -> None:
        # ``offset``: per-row positions of the next token (the model builds
        # rotary positions from it before update_and_fetch). ``_idx``: the
        # longest row, read by the model where it expects a scalar offset.
        lengths, slots = tuple(self.lengths), tuple(self.slots)
        self.offset = _shared(
            ("offset", lengths), lambda: mx.array(lengths, dtype=mx.int32)
        )
        self.slot_ids = _shared(
            ("slots", slots), lambda: mx.array(slots, dtype=mx.int32)
        )
        self._idx = max(self.lengths) if self.lengths else 0
        self.metadata_revision = getattr(self, "metadata_revision", 0) + 1

    @property
    def capacity(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])

    @property
    def rows(self) -> int:
        """Buffer slots (live rows plus free ones)."""
        return 0 if self.keys is None else int(self.keys.shape[0])

    @property
    def state(self):
        return [
            a
            for a in (self.keys, self.values, self.k_scales, self.v_scales)
            if a is not None
        ]

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.state)

    @property
    def quantized(self) -> bool:
        return self.kv_format == "int8"

    def _buffers(self) -> list[mx.array]:
        return [self.keys, self.values] + (
            [self.k_scales, self.v_scales] if self.quantized else []
        )

    def _set_buffers(self, bufs: list[mx.array]) -> None:
        self.keys, self.values = bufs[0], bufs[1]
        if self.quantized:
            self.k_scales, self.v_scales = bufs[2], bufs[3]

    def empty(self) -> bool:
        return self.keys is None

    def size(self) -> int:
        return self._idx

    def make_mask(self, n: int, **_kw):
        return None  # masking is by per-row length inside the kernel

    def row(self, b: int) -> tuple[mx.array, mx.array]:
        """Batch row ``b``'s stored keys and values, [HKV, lengths[b], D]."""
        s, n = self.slots[b], self.lengths[b]
        return self.keys[s, :, :n], self.values[s, :, :n]

    # ── storage ──────────────────────────────────────────────────────────
    def _reserve(self, rows: int, keys: int, like: list[mx.array]) -> None:
        """Make room for ``rows`` slots of ``keys`` keys; ``like`` gives the
        per-buffer (heads, width, dtype) when nothing is allocated yet."""
        if self.keys is None:
            S, cap = _pow2(rows), _round_up(keys + 1)
            self._set_buffers(
                [mx.zeros((S, a.shape[1], cap, a.shape[-1]), a.dtype) for a in like]
            )
            return
        S, cap = self.rows, self.capacity
        if rows <= S and keys <= cap:
            return
        S2 = _pow2(rows) if rows > S else S
        cap2 = _round_up(keys + 1) if keys > cap else cap
        # one pass per buffer: live data copied, new slots / keys zero-filled
        self._set_buffers(
            [
                mx.pad(a, [(0, S2 - S), (0, 0), (0, cap2 - cap), (0, 0)])
                for a in self._buffers()
            ]
        )

    def _compact(self, keys: int) -> None:
        """Gather the live slots into a fresh ``[B, HKV, keys, D]`` buffer."""
        sel = self.slot_ids
        self._set_buffers(
            [mx.take(a[:, :, :keys, :], sel, axis=0) for a in self._buffers()]
        )
        self.slots = list(range(len(self.lengths)))

    def _encode(self, k: mx.array, v: mx.array) -> list[mx.array]:
        """bf16 K, V [.., T, D] -> stored arrays in buffer order: [k, v] or
        [k codes, v codes, k scales, v scales] (one quantize launch)."""
        if not self.quantized:
            return [k, v]
        kq, ks, vq, vs = quantize_kv_pair(k, v)
        return [kq, vq, ks, vs]

    def _append(self, rows: list[tuple[int, list[mx.array]]]) -> None:
        """Place new rows ``(length, [k, v(, k_scales, v_scales)])`` (each
        array [1, HKV, >= length, W], stored format) into free slots."""
        if not rows:
            return
        like = next((a for _, a in rows if a), None)
        if self.keys is None and like is None:
            # rows without keys yet: slots are assigned, buffers come with
            # the first update_and_fetch
            for n, _ in rows:
                self.slots.append(len(self.slots))
                self.lengths.append(n)
            self._sync()
            return
        live = len(self.lengths)
        longest = max([n for n, _ in rows] + self.lengths)
        self._reserve(live + len(rows), longest, like)
        free = sorted(set(range(self.rows)) - set(self.slots))
        bufs = self._buffers()
        for (n, arrays), slot in zip(rows, free, strict=False):
            if n > 0:
                for i, a in enumerate(arrays):
                    bufs[i][slot : slot + 1, :, :n, :] = a[:, :, :n, :]
            self.slots.append(slot)
            self.lengths.append(n)
        self._set_buffers(bufs)
        self._sync()

    def _rows_of(self, cache: Any) -> list[tuple[int, list[mx.array]]]:
        """Each row of a stock ``KVCache`` / ``BatchKVCache`` or another
        ``RaggedKVCache``, in this cache's stored format."""
        if isinstance(cache, RaggedKVCache):
            if cache.kv_format != self.kv_format:
                raise ValueError("RaggedKVCache.extend: kv_format mismatch")
            if cache.keys is None:
                return [(n, []) for n in cache.lengths]
            bufs = cache._buffers()
            return [
                (n, [a[s : s + 1] for a in bufs])
                for s, n in zip(cache.slots, cache.lengths, strict=True)
            ]
        if cache.keys is None:
            B = int(cache.offset.shape[0]) if isinstance(cache.offset, mx.array) else 1
            return [(0, [])] * max(B, 1)
        if hasattr(cache, "left_padding"):
            idx = int(cache._idx)
            spans = [(int(p), idx) for p in cache.left_padding.tolist()]
        else:  # KVCache: every row holds keys 0 .. offset - 1
            spans = [(0, int(cache.offset))] * int(cache.keys.shape[0])
        out = []
        for b, (p, e) in enumerate(spans):
            k = cache.keys[b : b + 1, :, p:e]
            v = cache.values[b : b + 1, :, p:e]
            out.append((e - p, self._encode(k, v)))
        return out

    # ── construction ─────────────────────────────────────────────────────
    @classmethod
    def from_cache(cls, cache: Any, kv_format: str = "bf16") -> RaggedKVCache:
        """Build from a stock ``KVCache`` / left-padded ``BatchKVCache``
        (valid keys of row b at ``left_padding[b] .. _idx - 1``)."""
        rc = cls(kv_format)
        rc.extend(cache)
        return rc

    # ── model interface ──────────────────────────────────────────────────
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, T, _ = keys.shape
        if len(self.lengths) != B:
            raise ValueError(
                f"RaggedKVCache: {B} rows of keys for {len(self.lengths)} rows"
            )
        arrays = self._encode(keys, values)
        self._reserve(B, max(self.lengths) + T, arrays)
        cap = self.capacity
        slots, lengths = tuple(self.slots), tuple(self.lengths)

        def index():
            return mx.array(
                [
                    [[(s * H + h) * cap + n + t for t in range(T)] for h in range(H)]
                    for s, n in zip(slots, lengths, strict=True)
                ],
                dtype=mx.int32,
            )

        idx = _shared(("write", slots, lengths, H, cap, T), index)
        self._set_buffers(
            [
                _scatter_rows(b, idx, a)
                for b, a in zip(self._buffers(), arrays, strict=True)
            ]
        )
        self.lengths = [n + T for n in self.lengths]
        self._sync()
        return self.keys, self.values

    def attend(self, queries: mx.array, scale: float) -> mx.array:
        """Attention of this step's queries over each row's keys (after
        ``update_and_fetch`` stored them)."""
        return ragged_decode_attention(
            queries,
            self.keys,
            self.values,
            self.offset,
            scale,
            max_length=self._idx,
            k_scales=self.k_scales,
            v_scales=self.v_scales,
            row_lengths=self.lengths,
            slots=self.slot_ids,
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
        self.slots = [self.slots[i] for i in idx]
        self._sync()
        if self.keys is not None and self.lengths:
            need = _round_up(max(self.lengths) + 1)
            if need * 2 <= self.capacity:
                # the longest row left: give its keys back
                self._compact(need)
                self._sync()

    def extend(self, other: Any) -> None:
        self._append(self._rows_of(other))

    def trim(self, n: int) -> int:
        """Speculative rollback: drop the last ``n`` keys of every row."""
        n = min(int(n), min(self.lengths, default=0))
        self.lengths = [x - n for x in self.lengths]
        self._sync()
        return n


# ── single-row lane: the kernel over a stock KVCache ─────────────────────────


_CLASSES: dict = {}


def _dense_kv_classes() -> tuple:
    """Plain one-sequence caches (mlx-lm's and mlx-vlm's ``KVCache``)."""
    hit = _CLASSES.get("dense")
    if hit is None:
        from mlx_lm.models.cache import KVCache as LmKVCache

        classes = [LmKVCache]
        try:
            from mlx_vlm.models.cache import KVCache as VlmKVCache

            classes.append(VlmKVCache)
        except ImportError:
            pass
        hit = _CLASSES["dense"] = tuple(classes)
    return hit


def _stock_batch_kv_classes() -> tuple:
    """Both left-padded batch caches: mlx-vlm's BatchGenerator builds its own
    ``mlx_vlm.models.cache.BatchKVCache`` (same fields as mlx-lm's)."""
    hit = _CLASSES.get("batch")
    if hit is None:
        from mlx_lm.models.cache import BatchKVCache as LmBatchKVCache

        classes = [LmBatchKVCache]
        try:
            from mlx_vlm.models.cache import BatchKVCache as VlmBatchKVCache

            classes.append(VlmBatchKVCache)
        except ImportError:
            pass
        hit = _CLASSES["batch"] = tuple(classes)
    return hit


_STATE: dict = {"installed": False, "format": None, "dense_lane": False}


def set_format(kv_format: str | None) -> None:
    """KV format the generation-batch merge builds (None: stock caches). The
    runner sets it only while its own model steps, so another model served in
    the same process keeps upstream's caches."""
    if kv_format not in (None, "bf16", "int8"):
        raise ValueError(f"ragged KV: unknown format {kv_format!r}")
    _STATE["format"] = kv_format


def supports(language_model: Any) -> bool:
    """True when the model's attention is qwen3_5's (what ``install`` routes
    through the ragged kernels); any other family keeps stock caches."""
    try:
        from mlx_vlm.models.qwen3_5 import language as lang
    except ImportError:  # pragma: no cover - older mlx-vlm
        return False
    modules = getattr(language_model, "modules", None)
    if not callable(modules):
        return False
    return any(isinstance(m, lang.Qwen3_5Attention) for m in modules())


def set_dense_lane(active: bool) -> None:
    """Route single-row ``KVCache`` attention (decode and speculative verify)
    through the ragged kernel while the speculative lane steps. Set on the
    single MLX thread, before the step's graphs are built."""
    _STATE["dense_lane"] = bool(active)


def _lane_length(cache: Any) -> int | None:
    """Stored keys of a one-row stock cache whose buffer already has the
    ragged layout (keys at ``0 .. n - 1`` of ``cache.keys[0]``): a ``KVCache``,
    or a one-row ``BatchKVCache`` without left padding (the speculative lane's
    verify cache). None for anything else."""
    kind = type(cache)
    if cache is None or cache.keys is None or cache.keys.shape[0] != 1:
        return None
    if kind in _dense_kv_classes():
        return int(cache.offset)
    if kind in _stock_batch_kv_classes():
        lp = cache.left_padding
        info = getattr(cache, "_yunshu_lane_pad", None)
        if info is None or info[0] is not lp:
            # one host read per left_padding array (it changes on filter /
            # extend, not per step)
            info = (lp, int(lp.reshape(-1)[0].item()) if lp.size == 1 else -1)
            cache._yunshu_lane_pad = info
        return int(cache._idx) if info[1] == 0 else None
    return None


def dense_lane_attention(queries: mx.array, cache: Any, scale: float):
    """The ragged kernel over a one-row stock cache (``_lane_length``) after
    ``update_and_fetch`` stored this step's keys, or None when the call is not
    the lane's (lane off, other cache, shape or dtype)."""
    if not _STATE["dense_lane"]:
        return None
    n = _lane_length(cache)
    if n is None:
        return None
    keys, values = cache.keys, cache.values
    if (
        queries.ndim != 4
        or queries.shape[0] != 1
        or queries.shape[2] > 8
        or queries.dtype != mx.bfloat16
        or keys.dtype != mx.bfloat16
        or values.dtype != mx.bfloat16
        or queries.shape[-1] != keys.shape[-1]
        or keys.shape[-1] % 32
    ):
        return None
    # tile: every token x head of the KV head per pass over the keys (verify
    # reads K/V once); the key-parallel fallback (no tensor ops) keeps the
    # same per-token bits for any T but reads K/V once per token.
    tile = queries.shape[1] // keys.shape[1] <= 8 and tile_ready()
    if tile and keys.shape[2] % 64:
        # The tile kernel walks keys in 64-key windows at absolute positions,
        # but pulls the last window back inside a buffer whose capacity is not
        # a multiple of 64 — which moves keys to other columns and changes the
        # bits. Stock caches grow to ``offset + 256`` after a rollback, so the
        # verify cache would have such a capacity while plain decode's does
        # not: pad the buffer (once per growth) so windows never move.
        pad = [(0, 0), (0, 0), (0, _round_up(keys.shape[2]) - keys.shape[2]), (0, 0)]
        cache.keys = keys = mx.pad(keys, pad)
        cache.values = values = mx.pad(values, pad)
    return ragged_decode_attention(
        queries,
        keys,
        values,
        _shared(("offset", (n,)), lambda: mx.array([n], dtype=mx.int32)),
        scale,
        max_length=n,
        row_lengths=(n,),
        impl="tile" if tile else "auto",
    )


# ── install ──────────────────────────────────────────────────────────────────


def _attention(self, x, cache, position_ids, position_embeddings, linears=None):
    """qwen3_5 attention over a ragged cache, or a dense-lane ``KVCache``;
    None hands the call back to the caller's stock path."""
    B, L, _ = x.shape
    ragged = isinstance(cache, RaggedKVCache)
    if not ragged and not (
        _STATE["dense_lane"]
        and B == 1
        and L <= 8  # prefill keeps the stock path
        and x.dtype == mx.bfloat16
        and _lane_length(cache) is not None
        and cache.keys.dtype == mx.bfloat16
    ):
        return None
    if linears is None:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    else:
        q, k, v = linears((self.q_proj, self.k_proj, self.v_proj), x)
    queries, _keys, _values, gate, _ = self._prepare_projected_qkv(
        q, k, v, cache, position_ids, position_embeddings, None
    )
    if ragged:
        out = cache.attend(queries, self.scale)
    else:
        out = dense_lane_attention(queries, cache, self.scale)
        if out is None:  # a head size the kernel does not take
            out = mx.fast.scaled_dot_product_attention(
                queries,
                _keys,
                _values,
                scale=self.scale,
                mask="causal" if L > 1 else None,
            )
    return out.transpose(0, 2, 1, 3).reshape(B, L, -1), gate


def install() -> bool:
    """Route qwen3_5 attention through the ragged kernel when its cache is a
    ``RaggedKVCache`` (and, while ``set_dense_lane`` is on, a one-row
    ``KVCache``) — both the model's own attention and mlx-vlm's batch-invariant
    speculative forward. Any other cache keeps the stock path."""
    if _STATE["installed"]:
        return True
    from mlx_vlm.models.qwen3_5 import language as lang

    cls = lang.Qwen3_5Attention
    orig = cls.__call__

    def __call__(
        self, x, mask=None, cache=None, position_ids=None, position_embeddings=None
    ):
        res = _attention(self, x, cache, position_ids, position_embeddings)
        if res is None:
            return orig(self, x, mask, cache, position_ids, position_embeddings)
        out, gate = res
        return self.o_proj(out * mx.sigmoid(gate))

    cls.__call__ = __call__

    try:
        from mlx_vlm.models.qwen3_5 import speculative_verifier as sv
    except ImportError:  # pragma: no cover - older mlx-vlm
        sv = None
    if sv is not None:
        vcls = sv.Qwen3_5BatchInvariantForward
        vorig = vcls._attention

        def _v_attention(
            self, attention, x, mask, cache, position_ids, position_embeddings
        ):
            res = _attention(
                attention,
                x,
                cache,
                position_ids,
                position_embeddings,
                linears=self._linears,
            )
            if res is None:
                return vorig(
                    self, attention, x, mask, cache, position_ids, position_embeddings
                )
            out, gate = res
            return self._linear(attention.o_proj, out * mx.sigmoid(gate))

        vcls._attention = _v_attention
    _STATE["installed"] = True
    return True


def enable(kv_format: str | None) -> None:
    """Make upstream's generation-batch merge (``_extend_cache``) build ragged
    attention caches in ``kv_format`` (None: stock behavior). A lone request
    keeps its stock cache; the first join converts the decoding row once and
    each later join copies only the joining row."""
    set_format(kv_format)
    from mlx_vlm.generate import ar

    if getattr(ar._extend_cache, "_yunshu_ragged", False):
        return
    orig = ar._extend_cache

    def _extend_cache(cache_a, cache_b):
        fmt = _STATE["format"]
        if fmt is None or not cache_a or not cache_b:
            return orig(cache_a, cache_b)
        kv = _dense_kv_classes() + _stock_batch_kv_classes()
        out = []
        for ca, cb in zip(cache_a, cache_b, strict=False):
            ok_a = isinstance(ca, RaggedKVCache) or type(ca) in kv
            if ok_a and (isinstance(cb, RaggedKVCache) or type(cb) in kv):
                if not isinstance(ca, RaggedKVCache):
                    rc = RaggedKVCache(fmt)
                    rc._append(rc._rows_of(ca) + rc._rows_of(cb))
                    out.append(rc)
                    continue
                ca.extend(cb)
                out.append(ca)
            else:
                out.extend(orig([ca], [cb]))
        return out

    _extend_cache._yunshu_ragged = True
    ar._extend_cache = _extend_cache


__all__ = [
    "RaggedKVCache",
    "dense_lane_attention",
    "enable",
    "install",
    "set_dense_lane",
]
