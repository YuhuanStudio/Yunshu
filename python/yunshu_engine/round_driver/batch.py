"""Decoding rows in shared slot buffers: one launch per layer for every row.

The round driver's decode step used to run each row's attention and GDN
recurrence as its own call per layer. Here the decoding rows share

- **attention KV** in slot buffers ``[S, HKV, CAP, D]`` per attention layer
  (row ``b`` owns slot ``slots[b]``, keys ``0 .. n - 1``): one scatter writes
  every row's new keys, one ragged-attention launch reads them;
- **GDN state** as batch arrays ``[B, Hv, Dv, Dk]`` / ``[B, K - 1, C]`` per
  layer, in decode-set order: ``gdn_rows`` advances every row with its own
  window length, and its output *is* the next step's input (no gather).

A step runs the rows' windows (the pending token plus drafts, ``w_b`` tokens)
right-padded to ``T = max w_b`` tokens: every weight-bearing op sees ``B * T``
rows (the lane matmuls are flat in the row count up to 128 rows), padded
positions carry a copy of the row's last token and are never read — they
write keys past the row's length, and the recurrence skips them.

Per-row arithmetic does not depend on the other rows or on ``T``: projections
are row-invariant lane matmuls, attention is the ragged kernel whose token
bits do not depend on the window length or on what shares the launch, the
recurrence is the upstream step kernel per (row, head, value dim). A row keeps
its first ``used`` window positions; KV needs only its length set, the GDN
state / conv window continue from the position ``used`` (``commit``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import mlx.core as mx
import numpy as np

from ..blocks.kv import _scatter_rows
from ..kernels import ragged_attention as ra
from .decode_set import DecodeSet

CAP_STEP = 512  # key capacity granularity (a multiple of the tile's 64)
MAX_WINDOW = 8  # the tile kernel's token window
DECODE_GROWTH = 2048  # most keys a slot buffer grows by at once (tree_verify.DECODE_STEP_CAP's headroom)


def _round_cap(n: int) -> int:
    return -(-max(int(n), 1) // CAP_STEP) * CAP_STEP


class Slots:
    """KV buffers of ``n_layers`` layers ``[S, HKV, CAP, D]`` sharing one slot
    pool (layers without keys keep ``None``).

    ``kv_format="int8"``: the buffers hold int8 codes and ``ks`` / ``vs`` fp16 scales per
    (slot, KV head, token, 32-dim group) ``[S, HKV, CAP, D/32]`` (ragged_attention.quantize_kv_pair);
    rows are quantized once on write, the kernel reads codes + scales directly."""

    def __init__(self, n_layers: int, kv_format: str = "bf16"):
        if kv_format not in ("bf16", "int8"):
            raise ValueError(f"Slots: unknown kv_format {kv_format!r}")
        self.kv_format = kv_format
        self.k: list[mx.array | None] = [None] * n_layers
        self.v: list[mx.array | None] = [None] * n_layers
        self.ks: list[mx.array | None] = [None] * n_layers  # int8 only: key scales
        self.vs: list[mx.array | None] = [None] * n_layers  # int8 only: value scales
        self.S = 0
        self.cap = 0
        self.free: list[int] = []

    @property
    def int8(self) -> bool:
        return self.kv_format == "int8"

    def alloc(self) -> int:
        if not self.free:
            # doubling while buffers are small; one slot at a time once a
            # slot is large (every slot is as long as the longest row)
            big = self.cap > 4096
            S2 = self.S + 1 if big else max(1, self.S * 2)
            self._resize(S2, self.cap)
            self.free = list(range(self.S, S2))[::-1]
            self.S = S2
        return self.free.pop()

    def release(self, slot: int) -> None:
        self.free.append(int(slot))

    def _resize(self, S2: int, cap2: int) -> None:
        for buf in (self.k, self.v, self.ks, self.vs):
            for i, a in enumerate(buf):
                if a is None:
                    continue
                grown = mx.pad(
                    a, [(0, S2 - a.shape[0]), (0, 0), (0, cap2 - a.shape[2]), (0, 0)]
                )
                mx.eval(grown)  # one layer's copy at a time
                buf[i] = grown
        self.cap = cap2

    def reserve(self, keys: int) -> None:
        """Room for ``keys`` keys per slot."""
        if keys > self.cap:
            # growth is a quarter of the capacity up to DECODE_GROWTH keys: a 32K row asks for 2K more keys,
            # not 8K (a 25% jump left ~0.5 GiB of unused KV per 32K row, and the pooled buffer of a finished
            # row kept it)
            grow = min(max(self.cap // 4, CAP_STEP), DECODE_GROWTH)
            self._resize(self.S, _round_cap(max(keys, self.cap + grow)))

    def _buf(self, i: int, like: mx.array) -> None:
        if self.k[i] is None:
            shape = (self.S, like.shape[1], self.cap, like.shape[-1])
            if self.int8:
                scale_shape = (*shape[:-1], shape[-1] // ra.GROUP)
                self.k[i] = mx.zeros(shape, mx.int8)
                self.v[i] = mx.zeros(shape, mx.int8)
                self.ks[i] = mx.zeros(scale_shape, mx.float16)
                self.vs[i] = mx.zeros(scale_shape, mx.float16)
            else:
                self.k[i] = mx.zeros(shape, like.dtype)
                self.v[i] = mx.zeros(shape, like.dtype)

    def write_prefix(self, i: int, slot: int, k: mx.array, v: mx.array, n: int):
        """Store a row's first ``n`` keys (``k``, ``v`` [1, HKV, >= n, D], bf16): quantized
        here once when the format is int8."""
        self._buf(i, k)
        # _buf allocated this layer's buffers (codes + scales when int8)
        k_buf, v_buf = self.k[i], self.v[i]
        assert k_buf is not None and v_buf is not None
        if self.int8:
            kq, ks, vq, vs = ra.quantize_kv_pair(k[:, :, :n, :], v[:, :, :n, :])
            ks_buf, vs_buf = self.ks[i], self.vs[i]
            assert ks_buf is not None and vs_buf is not None
            k_buf[slot, :, :n, :] = kq[0]
            v_buf[slot, :, :n, :] = vq[0]
            ks_buf[slot, :, :n, :] = ks[0]
            vs_buf[slot, :, :n, :] = vs[0]
            return
        k_buf[slot, :, :n, :] = k[0, :, :n, :]
        v_buf[slot, :, :n, :] = v[0, :, :n, :]

    def write_codes(self, i: int, slot: int, c: Any, n: int) -> None:
        """Store a row's first ``n`` keys from an int8 row cache (codes + scales) as they are."""
        if not self.int8:
            raise ValueError("Slots: int8 row cache in a bf16 slot pool")
        self._buf(i, c.keys)
        for buf, src in (
            (self.k, c.keys),
            (self.v, c.values),
            (self.ks, c.k_scales),
            (self.vs, c.v_scales),
        ):
            b = buf[i]
            assert b is not None
            b[slot, :, :n, :] = src[0, :, :n, :]

    def read_prefix(self, i: int, slot: int, n: int) -> tuple | None:
        """Detached copies of a slot's first ``n`` keys of layer ``i`` (``[1, HKV, n, D]``): bf16 (k, v) or
        int8 (codes, codes, key scales, value scales). Evaluated here: the slot is reused after the row
        leaves, and a one-slot pool's buffers go to the restore pool."""
        if self.k[i] is None:
            return None
        bufs = (self.k, self.v, self.ks, self.vs) if self.int8 else (self.k, self.v)
        out = tuple(
            mx.contiguous(cast(mx.array, b[i])[slot : slot + 1, :, :n, :]) for b in bufs
        )
        mx.eval(*out)
        return out

    def arrays(self) -> list[mx.array]:
        return [a for a in self.k + self.v + self.ks + self.vs if a is not None]


@dataclass
class KVPlan:
    """Host-built per-step layout of the rows' windows (shared by all layers)."""

    slot_ids: mx.array  # [B] int32
    n0: np.ndarray  # keys already stored per row
    T: int  # window width (rows padded to it)
    positions: mx.array  # [3, B, T]
    lengths: mx.array  # [B] keys visible to the row's last window token
    length_list: tuple
    max_length: int
    slots: list[int]
    index: dict = field(default_factory=dict)  # (H, cap) -> scatter index

    @classmethod
    def make(
        cls, slots: list[int], n0: list[int], T: int, deltas: list[int] | None = None
    ) -> KVPlan:
        """``deltas``: per-row mRoPE offsets added to the window's rotary positions only (a media
        prompt's ``rope_deltas``); keys, lengths and masks stay in cache indices."""
        n = np.asarray(n0, dtype=np.int64)
        shift = n if deltas is None else n + np.asarray(deltas, dtype=np.int64)
        pos = shift[:, None] + np.arange(T)[None, :]
        lengths = n + T
        return cls(
            slot_ids=mx.array(slots, dtype=mx.int32),
            n0=n,
            T=T,
            positions=mx.array(np.broadcast_to(pos, (3, *pos.shape)), dtype=mx.int32),
            lengths=mx.array(lengths, dtype=mx.int32),
            length_list=tuple(int(x) for x in lengths),
            max_length=int(lengths.max()),
            slots=list(slots),
        )

    def scatter_index(self, H: int, cap: int) -> mx.array:
        key = (H, cap)
        hit = self.index.get(key)
        if hit is None:
            s = np.asarray(self.slots, dtype=np.int64)[:, None, None]
            idx = (
                (s * H + np.arange(H)[None, :, None]) * cap
                + self.n0[:, None, None]
                + np.arange(self.T)[None, None, :]
            )
            hit = self.index[key] = mx.array(idx, dtype=mx.int32)
        return hit


class Pack:
    """Rows' windows as ``N = sum(w_b)`` packed tokens (projections, MLP and
    norms run on real tokens only) and as ``[B, T]`` right-padded tokens (the
    sequence mixers). ``pad`` / ``unpad`` convert; identity when every window
    has the same length."""

    def __init__(self, lens: list[int], T: int):
        self.B, self.T = len(lens), T
        self.uniform = all(w == T for w in lens)
        if not self.uniform:
            cum = np.concatenate([[0], np.cumsum(lens)[:-1]])
            pad = cum[:, None] + np.minimum(
                np.arange(T)[None, :], np.asarray(lens)[:, None] - 1
            )
            real = np.arange(T)[None, :] < np.asarray(lens)[:, None]
            self.pad_idx = mx.array(pad.reshape(-1), dtype=mx.int32)
            self.unpad_idx = mx.array(np.flatnonzero(real.reshape(-1)), dtype=mx.int32)

    def pad(self, a: mx.array) -> mx.array:
        """[N, W] -> [B, T, W] (padding repeats each row's last token)."""
        if not self.uniform:
            a = mx.take(a, self.pad_idx, axis=0)
        return a.reshape(self.B, self.T, -1)

    def unpad(self, a: mx.array) -> mx.array:
        """[B, T, ...] -> [N, W]."""
        a = a.reshape(self.B * self.T, -1)
        return a if self.uniform else mx.take(a, self.unpad_idx, axis=0)


class _Cell(list):
    """A GDN layer's ``[conv, state]`` pair as ``tree_commit`` reads and writes a cache entry."""

    def __init__(self, conv, state):
        super().__init__([conv, state])
        self._omlx_gdn_pending = None


def attend(
    at: Any,
    xn: mx.array,
    slots: Slots,
    i: int,
    plan: KVPlan,
    pack: Pack | None = None,
    verifier: Any = None,
) -> mx.array:
    """One full-attention layer over the rows' windows: keys are written to
    the row's slot at ``n0 .. n0 + T - 1`` and the ragged kernel reads
    ``0 .. n0 + t`` for window token ``t``. ``xn``: [N, D] packed tokens with
    ``pack``, else [B, T, D]."""
    if verifier is not None:
        q, k, v = (
            t[0] for t in verifier._linears((at.q_proj, at.k_proj, at.v_proj), xn[None])
        )
    else:
        q, k, v = at.q_proj(xn), at.k_proj(xn), at.v_proj(xn)
    if pack is not None:
        q, k, v = pack.pad(q), pack.pad(k), pack.pad(v)
    queries, keys, values, gate, _ = at._prepare_projected_qkv(
        q, k, v, None, plan.positions, None, None
    )
    B, H, T, D = (int(s) for s in queries.shape)
    HKV = int(keys.shape[1])
    slots._buf(i, keys)
    idx = plan.scatter_index(HKV, slots.cap)
    # _buf allocated this layer's buffers (codes + scales when int8)
    k_buf, v_buf = slots.k[i], slots.v[i]
    assert k_buf is not None and v_buf is not None
    if slots.int8:
        # new rows are quantized on write; the kernel reads codes + scales
        kq, ks, vq, vs = ra.quantize_kv_pair(keys, values)
        ks_buf, vs_buf = slots.ks[i], slots.vs[i]
        assert ks_buf is not None and vs_buf is not None
        slots.k[i] = _scatter_rows(k_buf, idx, kq)
        slots.v[i] = _scatter_rows(v_buf, idx, vq)
        slots.ks[i] = _scatter_rows(ks_buf, idx, ks)
        slots.vs[i] = _scatter_rows(vs_buf, idx, vs)
    else:
        slots.k[i] = _scatter_rows(k_buf, idx, keys)
        slots.v[i] = _scatter_rows(v_buf, idx, values)
    if not slots.int8 and (queries.dtype != mx.bfloat16 or keys.dtype != mx.bfloat16):
        # The ragged Metal kernels accept bf16 only. MTP heads may retain
        # fp16/fp32 weights; keep their arithmetic instead of rounding to bf16.
        stored_k, stored_v = slots.k[i], slots.v[i]
        assert stored_k is not None and stored_v is not None
        ks = mx.take(stored_k, plan.slot_ids, axis=0)[..., : plan.max_length, :]
        vs = mx.take(stored_v, plan.slot_ids, axis=0)[..., : plan.max_length, :]
        positions = mx.array(plan.n0, dtype=mx.int32)[:, None] + mx.arange(T)[None, :]
        mask = mx.arange(plan.max_length)[None, None, :] <= positions[:, :, None]
        out = mx.fast.scaled_dot_product_attention(
            queries, ks, vs, scale=at.scale, mask=mask[:, None, :, :]
        )
    else:
        # int8 rows run the same token-tile family as bf16 (the int8 tile kernel dequantizes in-kernel), so
        # plain decode, MTP verify and tree verify share one arithmetic
        tile = H // HKV <= 8 and ra.tile_ready()
        k_now, v_now = slots.k[i], slots.v[i]
        assert k_now is not None and v_now is not None
        out = ra.ragged_decode_attention(
            queries,
            k_now,
            v_now,
            plan.lengths,
            at.scale,
            max_length=plan.max_length,
            k_scales=slots.ks[i] if slots.int8 else None,
            v_scales=slots.vs[i] if slots.int8 else None,
            row_lengths=plan.length_list,
            slots=plan.slot_ids,
            impl="tile" if tile else "auto",
        )
    out = out.transpose(0, 2, 1, 3).reshape(B, T, -1) * mx.sigmoid(gate)
    out = out if pack is None else pack.unpad(out)
    if verifier is not None:
        return cast(mx.array, verifier._linear(at.o_proj, out[None])[0])
    return cast(mx.array, at.o_proj(out))


class DecodeBatch(DecodeSet):
    """The decoding rows of one target model (see the module docstring)."""

    def __init__(self, language_model: Any, kv_format: str = "bf16"):
        self.lm = language_model
        self.kv_format = kv_format
        layers = language_model.model.layers
        self.linear = [bool(layer.is_linear) for layer in layers]
        self.slots = Slots(len(layers), kv_format)
        self.rows: list = []  # decode order: index b of every batch array
        # GDN layers: one [1, Hv, Dv, Dk] state and one [1, K-1, C] conv window per row, in decode order
        self.state: list[list[mx.array]] = [[] for _ in layers]
        self.conv: list[list[mx.array]] = [[] for _ in layers]
        self._last: dict | None = None

    # ── membership ───────────────────────────────────────────────────────
    def join(self, rows: list) -> None:
        """Move prefilled rows (their ``cache`` / ``n`` set) into the batch."""
        if not rows:
            return
        need = max(r.n for r in rows) + MAX_WINDOW + 1
        for row in rows:
            row.slot = self.slots.alloc()
        self.slots.reserve(need)
        for i, lin in enumerate(self.linear):
            if lin:
                for r in rows:
                    self.state[i].append(r.cache[i][1])
                    self.conv[i].append(r.cache[i][0])
            else:
                for row in rows:
                    c = row.cache[i]
                    if getattr(c, "k_scales", None) is not None:
                        # a restored int8 row cache (a stored reply): its codes are the slot's
                        self.slots.write_codes(i, row.slot, c, row.n)
                    else:
                        self.slots.write_prefix(i, row.slot, c.keys, c.values, row.n)
        for row in rows:
            row.cache = None
        self.rows.extend(rows)

    def cut_out(self, row: Any) -> None:
        """Give ``row`` its own cache back (cut out of the slots / state lists, copied) before it leaves:
        the layout its prefill had (bf16 ``KVCache`` / int8 ``Int8KVCache`` per attention layer, the
        model's GDN cache objects), holding ``row.n`` positions."""
        if (
            row.cache is not None
            or row.slot is None
            or row.n < 1
            or row not in self.rows
        ):
            return
        b = self.rows.index(row)
        cache = self.lm.make_cache()
        for i, lin in enumerate(self.linear):
            if lin:
                cache[i][0], cache[i][1] = self.conv[i][b], self.state[i][b]
                continue
            got = self.slots.read_prefix(i, row.slot, row.n)
            if got is None:
                raise RuntimeError("decode batch: attention layer without keys")
            c = cache[i]
            if self.slots.int8:
                from ..kernels.int8_kv import Int8KVCache

                c = cache[i] = Int8KVCache()
                c.keys, c.values, c.k_scales, c.v_scales = got
            else:
                c.keys, c.values = got
            c.offset = int(row.n)
        gdn = [a for i, lin in enumerate(self.linear) if lin for a in cache[i]]
        mx.eval(
            *[a for a in gdn if a is not None]
        )  # copies, not views of the batch lists
        row.cache = cache

    def leave(self, gone: list) -> None:
        """Drop rows (finished or cancelled) from the batch."""
        ids = {id(r) for r in gone}
        keep = [b for b, r in enumerate(self.rows) if id(r) not in ids]
        if len(keep) == len(self.rows):
            return
        for r in self.rows:
            if id(r) in ids:
                self.slots.release(r.slot)
                r.slot = None
        self.rows = [self.rows[b] for b in keep]
        if not keep:
            self.state = [[] for _ in self.linear]
            self.conv = [[] for _ in self.linear]
            # an idle batch holds no KV: the next rows size the buffers to
            # themselves, not to the longest row ever served. A one-slot pool's buffers
            # go to the restore pool (the next request's cache hit overwrites them in
            # place instead of allocating a fresh context-sized buffer), as the other
            # runner paths offer a finished request's buffers.
            if self.slots.S == 1 and not self.slots.int8:
                from ..kernels import buffer_cache

                buffer_cache.offer_arrays(self.slots.arrays())
            self.slots = Slots(len(self.linear), self.slots.kv_format)
            return
        for i, lin in enumerate(self.linear):
            if lin:
                self.state[i] = [self.state[i][b] for b in keep]
                self.conv[i] = [self.conv[i][b] for b in keep]

    def arrays(self) -> list[mx.array]:
        """Everything the batch keeps between steps (evaluated each step so no
        lazy graph carries over)."""
        out = self.slots.arrays()
        out += [a for rows in self.state + self.conv for a in rows if a is not None]
        return out

    # ── one decode step ──────────────────────────────────────────────────
    def forward(self, lens: list[int], tokens: mx.array) -> mx.array:
        """Run every row's window (``lens[b]`` tokens: the row's pending token
        and drafts, back to back in ``tokens`` [N] int32, which may still be a
        lazy graph) through the decoder; returns final-norm hidden states
        ``[N, D]`` (the rows' tokens back to back, ``N = sum(len(w))``). KV is appended and the GDN state advanced optimistically
        (all of every window kept); ``commit`` corrects rows that keep less."""
        model = self.lm.model
        lens = list(lens)
        T = max(lens)
        if T > MAX_WINDOW:
            raise ValueError("decode window longer than the tile window")
        n0 = [r.n for r in self.rows]
        self.slots.reserve(max(n0) + T + 1)
        plan = KVPlan.make(
            [r.slot for r in self.rows],
            n0,
            T,
            [r.req.rope_delta for r in self.rows],
        )
        pack = Pack(lens, T)
        from mlx_vlm.models.qwen3_5 import language as q35

        # The arithmetic of tree_forward / the packed verify: the verifier's row-invariant projections and
        # feed-forward, the fused GDN prework + tree GDN kernel per row, the fused norm-gate. A plain
        # one-token round here is therefore the same bits as a DFlash-batch plain-verify round, so spec
        # on == spec off holds whichever batch a row is in.
        verifier = q35._EXACT_SPECULATIVE_VERIFIER
        x = model.embed_tokens(tokens)[None]  # [1, N, D]
        recs: list[dict] = [
            {} for _ in lens
        ]  # per row: layer -> GDN record the commit replays
        for i, layer in enumerate(model.layers):
            xn = layer.input_layernorm(x)
            if self.linear[i]:
                r = self._gdn(verifier, layer.linear_attn, xn, i, lens, recs)
            else:
                r = attend(layer.self_attn, xn[0], self.slots, i, plan, pack, verifier)[
                    None
                ]
            h = x + r
            x = h + verifier._feed_forward(layer.mlp, layer.post_attention_layernorm(h))
        self._last = {"recs": recs, "lens": lens, "T": T}
        return cast(mx.array, model.norm(x)[0])

    def _gdn(self, verifier, g, xn, i, lens, recs) -> mx.array:
        """One GDN layer over the rows' windows, row by row through ``tree_verify._gdn_layer`` (a chain
        window); the state advance is the commit's replay, as in the DFlash batch."""
        from .. import tree_verify as tv

        projected = verifier._linears(
            (g.in_proj_qkv, g.in_proj_z, g.in_proj_b, g.in_proj_a), xn
        )
        mixed, at = [], 0
        for b, w in enumerate(lens):
            out, rec = tv._gdn_layer(
                verifier,
                g,
                xn[:, at : at + w],
                (self.conv[i][b], self.state[i][b]),
                tv.TreeShape(list(range(-1, w - 1))),
                projected=tuple(p[:, at : at + w] for p in projected),
                project_output=False,
            )
            recs[b][i] = rec
            mixed.append(out)
            at += w
        return cast(
            mx.array, verifier._linear(g.out_proj, mx.concatenate(mixed, axis=1))
        )

    def commit(self, used: list[int]) -> None:
        """Rows keep their first ``used[b]`` window positions (``1 <= used <=
        len(window)``): KV lengths advance, and every row's GDN state / conv window advance by the
        accepted chain through the tree commit's replay kernels (the state was not touched by the
        forward, as in the DFlash batch)."""
        from .. import tree_verify as tv

        last, self._last = self._last, None
        assert last is not None
        for r, u in zip(self.rows, used, strict=True):
            r.n += u
        for b, (u, rec) in enumerate(zip(used, last["recs"], strict=True)):
            if not rec:
                continue
            cells = {i: _Cell(self.conv[i][b], self.state[i][b]) for i in rec}
            res = tv.TreeResult(
                shape=tv.TreeShape(list(range(-1, last["lens"][b] - 1))),
                n0=0,
                hidden=None,
            )
            res.records = dict(rec)
            tv.tree_commit(self.lm, cells, res, list(range(u)))
            for i, cell in cells.items():
                self.conv[i][b], self.state[i][b] = cell[0], cell[1]
