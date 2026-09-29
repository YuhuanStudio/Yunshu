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
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..kernels import ragged_attention as ra
from ..kernels.gdn_rows import gated_delta_rows
from ..kernels.ragged_kv import _scatter_rows

CAP_STEP = 512  # key capacity granularity (a multiple of the tile's 64)
MAX_WINDOW = 8  # the tile kernel's token window


def _round_cap(n: int) -> int:
    return -(-max(int(n), 1) // CAP_STEP) * CAP_STEP


class Slots:
    """KV buffers of ``n_layers`` layers ``[S, HKV, CAP, D]`` sharing one slot
    pool (layers without keys keep ``None``)."""

    def __init__(self, n_layers: int):
        self.k: list[mx.array | None] = [None] * n_layers
        self.v: list[mx.array | None] = [None] * n_layers
        self.S = 0
        self.cap = 0
        self.free: list[int] = []

    def alloc(self) -> int:
        if not self.free:
            S2 = max(1, self.S * 2)
            self._resize(S2, self.cap)
            self.free = list(range(self.S, S2))[::-1]
            self.S = S2
        return self.free.pop()

    def release(self, slot: int) -> None:
        self.free.append(int(slot))

    def _resize(self, S2: int, cap2: int) -> None:
        for buf in (self.k, self.v):
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
            self._resize(self.S, _round_cap(max(keys, self.cap * 5 // 4)))

    def _buf(self, i: int, like: mx.array) -> None:
        if self.k[i] is None:
            shape = (self.S, like.shape[1], self.cap, like.shape[-1])
            self.k[i] = mx.zeros(shape, like.dtype)
            self.v[i] = mx.zeros(shape, like.dtype)

    def write_prefix(self, i: int, slot: int, k: mx.array, v: mx.array, n: int):
        """Store a row's first ``n`` keys (``k``, ``v`` [1, HKV, >= n, D])."""
        self._buf(i, k)
        self.k[i][slot, :, :n, :] = k[0, :, :n, :]
        self.v[i][slot, :, :n, :] = v[0, :, :n, :]

    def arrays(self) -> list[mx.array]:
        return [a for a in self.k + self.v if a is not None]


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
    def make(cls, slots: list[int], n0: list[int], T: int) -> KVPlan:
        n = np.asarray(n0, dtype=np.int64)
        pos = n[:, None] + np.arange(T)[None, :]
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


def attend(
    at: Any, xn: mx.array, slots: Slots, i: int, plan: KVPlan, pack: Pack | None = None
) -> mx.array:
    """One full-attention layer over the rows' windows: keys are written to
    the row's slot at ``n0 .. n0 + T - 1`` and the ragged kernel reads
    ``0 .. n0 + t`` for window token ``t``. ``xn``: [N, D] packed tokens with
    ``pack``, else [B, T, D]."""
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
    slots.k[i] = _scatter_rows(slots.k[i], idx, keys)
    slots.v[i] = _scatter_rows(slots.v[i], idx, values)
    tile = H // HKV <= 8 and ra.tile_ready()
    out = ra.ragged_decode_attention(
        queries,
        slots.k[i],
        slots.v[i],
        plan.lengths,
        at.scale,
        max_length=plan.max_length,
        row_lengths=plan.length_list,
        slots=plan.slot_ids,
        impl="tile" if tile else "auto",
    )
    out = out.transpose(0, 2, 1, 3).reshape(B, T, -1) * mx.sigmoid(gate)
    return at.o_proj(out if pack is None else pack.unpad(out))


class DecodeBatch:
    """The decoding rows of one target model (see the module docstring)."""

    def __init__(self, language_model: Any):
        self.lm = language_model
        layers = language_model.model.layers
        self.linear = [bool(layer.is_linear) for layer in layers]
        self.slots = Slots(len(layers))
        self.rows: list = []  # decode order: index b of every batch array
        self.state: list[mx.array | None] = [None] * len(layers)  # [B, Hv, Dv, Dk]
        self.conv: list[mx.array | None] = [None] * len(layers)  # [B, K-1, C]
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
                cs = [r.cache[i] for r in rows]
                states = [c[1] for c in cs]
                convs = [c[0] for c in cs]
                if self.state[i] is not None:
                    states.insert(0, self.state[i])
                    convs.insert(0, self.conv[i])
                self.state[i] = (
                    mx.concatenate(states, axis=0) if len(states) > 1 else states[0]
                )
                self.conv[i] = (
                    mx.concatenate(convs, axis=0) if len(convs) > 1 else convs[0]
                )
            else:
                for row in rows:
                    c = row.cache[i]
                    self.slots.write_prefix(i, row.slot, c.keys, c.values, row.n)
        for row in rows:
            row.cache = None
        self.rows.extend(rows)

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
            self.state = [None] * len(self.linear)
            self.conv = [None] * len(self.linear)
            return
        sel = mx.array(keep, dtype=mx.int32)
        for i, lin in enumerate(self.linear):
            if lin:
                self.state[i] = mx.take(self.state[i], sel, axis=0)
                self.conv[i] = mx.take(self.conv[i], sel, axis=0)

    def arrays(self) -> list[mx.array]:
        """Everything the batch keeps between steps (evaluated each step so no
        lazy graph carries over)."""
        out = self.slots.arrays()
        out += [a for a in self.state + self.conv if a is not None]
        return out

    # ── one decode step ──────────────────────────────────────────────────
    def forward(self, windows: list[list[int]]) -> mx.array:
        """Run every row's window (``windows[b]`` = the row's pending token and
        drafts) through the decoder; returns final-norm hidden states
        ``[N, D]`` (the rows' tokens back to back, ``N = sum(len(w))``). KV is appended and the GDN state advanced optimistically
        (all of every window kept); ``commit`` corrects rows that keep less."""
        model = self.lm.model
        lens = [len(w) for w in windows]
        T = max(lens)
        if T > MAX_WINDOW:
            raise ValueError("decode window longer than the tile window")
        n0 = [r.n for r in self.rows]
        self.slots.reserve(max(n0) + T + 1)
        plan = KVPlan.make([r.slot for r in self.rows], n0, T)
        pack = Pack(lens, T)
        lens_arr = mx.array(lens, dtype=mx.int32)
        toks = np.array([t for w in windows for t in w], dtype=np.int32)
        x = model.embed_tokens(mx.array(toks))
        hist: dict[int, tuple] = {}
        for i, layer in enumerate(model.layers):
            xn = layer.input_layernorm(x)
            if self.linear[i]:
                r = self._gdn(layer.linear_attn, xn, i, lens, lens_arr, pack, hist)
            else:
                r = attend(layer.self_attn, xn, self.slots, i, plan, pack)
            h = x + r
            x = h + layer.mlp(layer.post_attention_layernorm(h))
        self._last = {"hist": hist, "lens": lens, "T": T}
        return model.norm(x)

    def _gdn(self, g, xn, i, lens, lens_arr, pack, hist) -> mx.array:
        from mlx_vlm.models.qwen3_5.gated_delta import _compute_g_beta

        B, T = pack.B, pack.T
        K1 = g.conv_kernel_size - 1
        qkv, z = pack.pad(g.in_proj_qkv(xn)), pack.pad(g.in_proj_z(xn))
        b, a = pack.pad(g.in_proj_b(xn)), pack.pad(g.in_proj_a(xn))
        conv_in = mx.concatenate([self.conv[i], qkv], axis=1)  # [B, K-1+T, C]
        if pack.uniform:
            self.conv[i] = conv_in[:, T:]
        else:
            pos = np.asarray(lens)[:, None] + np.arange(K1)[None, :]
            self.conv[i] = conv_in[mx.arange(B)[:, None], mx.array(pos)]
        conv_out = nn.silu(g.conv1d(conv_in))  # [B, T, C]
        q, k, v = [
            t.reshape(B, T, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [g.key_dim, 2 * g.key_dim], -1),
                [g.num_k_heads, g.num_k_heads, g.num_v_heads],
                [g.head_k_dim, g.head_k_dim, g.head_v_dim],
                strict=True,
            )
        ]
        inv = k.shape[-1] ** -0.5
        q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv * mx.fast.rms_norm(k, None, 1e-6)
        gate, beta = _compute_g_beta(g.A_log, a, b, g.dt_bias)
        y, self.state[i], h = gated_delta_rows(
            q, k, v, gate, beta, self.state[i], lens_arr
        )
        if h is not None:
            hist[i] = (h, conv_in)
        z = z.reshape(B, T, -1, g.head_v_dim)
        return g.out_proj(pack.unpad(g.norm(y, z).reshape(B, T, -1)))

    def commit(self, used: list[int]) -> None:
        """Rows keep their first ``used[b]`` window positions (``1 <= used <=
        len(window)``): KV lengths advance, and rows that verified fewer
        positions than they ran continue their GDN state / conv window from
        there."""
        last, self._last = self._last, None
        lens = last["lens"]
        for r, u in zip(self.rows, used, strict=True):
            r.n += u
        part = [b for b, (u, w) in enumerate(zip(used, lens, strict=True)) if u < w]
        if not part:
            return
        rows = mx.array(part, dtype=mx.int32)
        keep = mx.array([used[b] for b in part], dtype=mx.int32)
        for i, (h, conv_in) in last["hist"].items():
            self.state[i][rows] = h[rows, keep - 1]
            K1 = self.conv[i].shape[1]
            pos = keep[:, None] + mx.arange(K1)[None, :]
            self.conv[i][rows] = conv_in[rows[:, None], pos]
