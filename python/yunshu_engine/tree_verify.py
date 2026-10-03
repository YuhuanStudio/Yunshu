# Upstream (inspired): ashhart/TensorFold (MIT) src/tensorfold/kernels/qwen/dense/v1/lane_tree.py @ 34bae79a
"""Tree-shaped speculative verify for the Qwen3.5 family (bit-exact per path).

A *window* is the pending token (row 0, the root) plus draft nodes whose
``parents`` are earlier rows. One forward runs every row; each row gets exactly
the arithmetic a one-token decode step would give it after the tokens on its
root path, so the accepted path equals plain greedy decode token for token:

- **Projections / MLP / norms**: the batch-invariant projections (``<= 8`` rows)
  are row-independent, so the window is just rows.
- **RoPE**: row position is ``n0 + depth`` (not the row index).
- **Attention**: keys live in the cache at ``n0 + row``. A row attends to the
  prefix ``0 .. n0 - 1`` plus its own ancestors *as if they sat at* ``n0 + j``.
  The ragged tile kernel splits keys into fixed 512-key chunks whose partial
  softmax states merge in chunk order; chunks below the window's first chunk
  are shared by every row (one launch for all rows), the chunk(s) holding the
  window are recomputed per row from a gathered copy whose key ``j`` is the
  row's ``j``-th ancestor, and a merge kernel combines the two sets in chunk
  order. Same kernel, same operands, same order: same bits as a decode step.
- **Gated DeltaNet**: the 4-tap causal convolution reads each row's last three
  ancestor inputs (the fused prework kernel with one row per batch entry), and
  the recurrence starts each row from its parent's state (a tree variant of the
  fused verify kernel with identical per-step arithmetic). After the walk the
  accepted path's rows are replayed onto the start state (the replay kernel the
  chain verify uses) and the conv state is rebuilt from the path's inputs.

Rows must be in parent-before-child order. ``tree_forward`` never touches the
GDN caches and appends the window's keys to the attention caches;
``tree_commit`` keeps the accepted path (compacting its keys to ``n0 + j``),
``tree_abort`` drops the window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .kernels import ragged_attention as ra
from .kernels import ragged_kv
from .kernels.omlx import qwen35_gdn_prework as gp
from .kernels.omlx import qwen35_gdn_verify_fused as gv
from .kernels.omlx import qwen35_verify_qmm as vq

logger = logging.getLogger(__name__)

MAX_ROWS = 32  # lane projections (<= 128 rows), the fused GDN kernels (<= 32), the tile kernel's token groups
NARROW_ROWS = (
    8  # without lane projections only the sg8 / packed kernels' window is row-invariant
)
CK = ra.CHUNK


def _tree_tail_capacity(live_keys: int) -> int:
    """Physical tail rows covering every live 64-key tile."""
    return ((live_keys + 63) // 64) * 64


class TreeShape:
    """Parent structure of a window (host side) with its cached index tables."""

    def __init__(self, parents):
        self.parents = tuple(int(p) for p in parents)
        w = len(self.parents)
        if w < 1 or self.parents[0] != -1:
            raise ValueError("tree window: row 0 is the root (parent -1)")
        depths = [0]
        paths = [[0]]
        for r in range(1, w):
            p = self.parents[r]
            if not 0 <= p < r:
                raise ValueError("tree window: parents must precede children")
            depths.append(depths[p] + 1)
            paths.append(paths[p] + [r])
        self.width = w
        self.depths = depths
        self.paths = paths
        self.max_depth = max(depths)
        self.is_chain = self.parents == tuple(range(-1, w - 1))
        self.dynamic = False
        self._arrays: dict[str, mx.array] = {}

    def _const(self, name, make):
        arr = self._arrays.get(name)
        if arr is None:
            arr = self._arrays[name] = make()
        return arr

    def parents_array(self) -> mx.array:
        return self._const("parents", lambda: mx.array(self.parents, dtype=mx.int32))

    def path_table(self) -> mx.array:
        """[W, max_depth + 1]: row ``r``'s ``j``-th ancestor (``j <= depth``), 0 padded."""

        def make():
            rows = [p + [0] * (self.max_depth + 1 - len(p)) for p in self.paths]
            return mx.array(rows, dtype=mx.int32)

        return self._const("paths", make)

    def depth_array(self) -> mx.array:
        return self._const("depth", lambda: mx.array(self.depths, dtype=mx.int32))

    def conv_index(self) -> mx.array:
        """[W * 3]: per row, its last three conv inputs in ``[state (3); window (W)]``."""

        def make():
            out = []
            for path in self.paths:
                seq = [0, 1, 2] + [3 + r for r in path[:-1]]
                out += seq[-3:]
            return mx.array(out, dtype=mx.int32)

        return self._const("conv", make)


class DynamicShape:
    """A window whose parents are only known on the GPU (``parents`` [W] int32,
    root -1, parents before children): the index tables are computed by array
    ops, so the verify graph builds before the draft tree is read back.
    ``max_depth`` bounds the tree's depth (host side, sizes the tail gather)."""

    dynamic = True
    is_chain = False

    def __init__(self, parents: mx.array, max_depth: int):
        self.width = int(parents.shape[0])
        self.max_depth = int(max_depth)
        self._parents = parents.astype(mx.int32)
        w = self.width
        up = mx.maximum(self._parents, 0)  # the root points at itself
        anc = [mx.arange(w, dtype=mx.int32)]
        for _ in range(self.max_depth):
            anc.append(mx.take(up, anc[-1]))
        self._anc = mx.stack(anc)  # [maxd + 1, W]: k-th ancestor of each row
        depth = mx.zeros((w,), dtype=mx.int32)
        cur = mx.arange(w, dtype=mx.int32)
        for _ in range(self.max_depth):
            step = (mx.take(self._parents, cur) >= 0).astype(mx.int32)
            depth = depth + step
            cur = mx.take(up, cur)
        self._depth = depth
        self.depths = None

    def parents_array(self) -> mx.array:
        return self._parents

    def depth_array(self) -> mx.array:
        return self._depth

    def path_table(self) -> mx.array:
        """[W, max_depth + 1]: row ``r``'s ``j``-th ancestor (``j <= depth``)."""
        j = mx.arange(self.max_depth + 1, dtype=mx.int32)[None]  # [1, M]
        k = mx.maximum(self._depth[:, None] - j, 0)  # [W, M]
        rows = mx.arange(self.width, dtype=mx.int32)[:, None]
        return self._anc[k, rows]

    def conv_index(self) -> mx.array:
        """[W * 3] as ``TreeShape.conv_index``."""
        out = []
        for m in (3, 2, 1):  # oldest to newest of the last three inputs
            from_window = 3 + self._anc[min(m, self.max_depth)]
            from_state = 3 + self._depth - m
            use_window = self._depth >= m
            out.append(mx.where(use_window, from_window, from_state))
        return mx.stack(out, axis=1).reshape(-1)


@dataclass
class TreeResult:
    shape: TreeShape
    n0: int
    hidden: mx.array  # [1, W, D] final-norm hidden
    captured: list = field(default_factory=list)  # [1, W, D] per captured layer
    records: dict = field(default_factory=dict)  # layer index -> record


# ── recurrence: a tree variant of the fused verify kernel ───────────────────

_TREE_GDN = """
    constexpr int NK = Dk / 32;
    uint lane = thread_position_in_threadgroup.x;
    uint n = thread_position_in_grid.z;
    uint hv = n % Hv;
    uint hk = hv / (Hv / Hk);
    uint dv = thread_position_in_grid.y;
    float st0[NK];
    {
        auto s = state_in + (n * Dv + dv) * Dk + lane * NK;
        for (int i = 0; i < NK; ++i)
            st0[i] = static_cast<float>(s[i]);
    }
    float neg_a = -metal::precise::exp(static_cast<float>(A_log[hv]));
    InT dtb = dt_bias[hv];
    float lane_g = 0.0f, lane_b = 0.0f;
    if (int(lane) < T) {
        int gi = lane * Hv + hv;
        lane_g = gdn_decay(a[gi], dtb, neg_a);
        lane_b = gdn_beta(b[gi]);
    }
    float states[T][NK];
    for (int t = 0; t < T; ++t) {
        int par = parents[t];
        float st[NK];
        for (int i = 0; i < NK; ++i)
            st[i] = par < 0 ? st0[i] : states[par][i];
        float gt = simd_shuffle(lane_g, ushort(t));
        float bt = simd_shuffle(lane_b, ushort(t));
        auto kp = k + (t * Hk + hk) * Dk + lane * NK;
        auto qp = q + (t * Hk + hk) * Dk + lane * NK;
        float kv = 0.0f;
        for (int i = 0; i < NK; ++i) {
            st[i] = st[i] * gt;
            kv += st[i] * kp[i];
        }
        kv = simd_sum(kv);
        float delta = (v[(t * Hv + hv) * Dv + dv] - kv) * bt;
        float acc = 0.0f;
        for (int i = 0; i < NK; ++i) {
            st[i] = st[i] + kp[i] * delta;
            acc += st[i] * qp[i];
        }
        acc = simd_sum(acc);
        if (lane == 0)
            y[(t * Hv + hv) * Dv + dv] = static_cast<InT>(acc);
        for (int i = 0; i < NK; ++i)
            states[t][i] = st[i];
    }
"""

_KERNELS: dict = {}


def _gdn_kernel():
    kernel = _KERNELS.get("gdn")
    if kernel is None:
        kernel = _KERNELS["gdn"] = mx.fast.metal_kernel(
            name="yunshu_gdn_tree",
            input_names=[
                "state_in",
                "A_log",
                "dt_bias",
                "q",
                "k",
                "v",
                "a",
                "b",
                "parents",
            ],
            output_names=["y"],
            source=_TREE_GDN,
            header=gv._HELPERS,
        )
    return kernel


_REPLAY_PATH = (
    gv._PROLOGUE
    + """
    {
        int keep = count[0];
        float lane_g = 0.0f, lane_b = 0.0f;
        if (int(lane) < keep) {
            int gi = (b_idx * P + path[lane]) * Hv + hv;
            lane_g = gdn_decay(pa[gi], dtb, neg_a);
            lane_b = gdn_beta(pb[gi]);
        }
        for (int t = 0; t < keep; ++t) {
            int row = path[t];
            float gt = simd_shuffle(lane_g, ushort(t));
            float bt = simd_shuffle(lane_b, ushort(t));
            auto kp = pk + ((b_idx * P + row) * Hk + hk) * Dk + lane * NK;
            float kv = 0.0f;
            for (int i = 0; i < NK; ++i) {
                st[i] = st[i] * gt;
                kv += st[i] * kp[i];
            }
            kv = simd_sum(kv);
            float delta = (pv[((b_idx * P + row) * Hv + hv) * Dv + dv] - kv) * bt;
            for (int i = 0; i < NK; ++i)
                st[i] = st[i] + kp[i] * delta;
        }
        auto o = state_out + (n * Dv + dv) * Dk + lane * NK;
        for (int i = 0; i < NK; ++i)
            o[i] = st[i];
    }
"""
)


def _replay_kernel():
    kernel = _KERNELS.get("replay")
    if kernel is None:
        kernel = _KERNELS["replay"] = mx.fast.metal_kernel(
            name="yunshu_gdn_replay_path",
            input_names=[
                "state_in",
                "A_log",
                "dt_bias",
                "pk",
                "pv",
                "pa",
                "pb",
                "path",
                "count",
            ],
            output_names=["state_out"],
            source=_REPLAY_PATH,
            header=gv._HELPERS,
        )
    return kernel


def replay_path(layer, state, rows, path_arr, count_arr):
    """The state after the window rows ``path_arr[:count]`` in order, from
    ``state`` (the verify kernel's per-step arithmetic, rows read in place)."""
    pk, pv, pa, pb = rows
    geo = gv._geometry(layer, 1)
    (out,) = _replay_kernel()(
        inputs=[state, layer.A_log, layer.dt_bias, pk, pv, pa, pb, path_arr, count_arr],
        template=[
            ("InT", pk.dtype),
            ("Hk", geo["Hk"]),
            ("Hv", geo["Hv"]),
            ("Dk", geo["Dk"]),
            ("Dv", geo["Dv"]),
            ("P", pk.shape[1]),
        ],
        grid=geo["grid"],
        threadgroup=(32, 4, 1),
        output_shapes=[state.shape],
        output_dtypes=[mx.float32],
    )
    return out


def _gdn_layer(verifier, layer, x, cache, shape: TreeShape):
    """One GDN layer over the window; returns the out-projection output and
    the record ``tree_commit`` needs."""
    w = shape.width
    mixed, z, b, a = verifier._linears(
        (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a), x
    )
    dtype = x.dtype
    hk, hv = layer.num_k_heads, layer.num_v_heads
    dk, dv = layer.head_k_dim, layer.head_v_dim
    conv_prev = cache[0]
    state = cache[1]
    if state is None:
        state = mx.zeros((1, hv, dv, dk), dtype=mx.float32)
    seq = mx.concatenate([conv_prev, mixed], axis=1)[0]  # [3 + W, C]
    c_dim = seq.shape[-1]
    windows = mx.take(seq, shape.conv_index(), axis=0).reshape(w, 3, c_dim)
    inv = dk**-0.5
    q, k, v, _ = gp.gdn_prework_fused(
        mixed.reshape(w, 1, c_dim),
        windows,
        layer.conv1d.weight,
        mx.array(inv * inv, dtype=dtype),
        mx.array(inv, dtype=dtype),
        hk,
        hv,
        dk,
        dv,
    )
    q = q.reshape(1, w, hk, dk)
    k = k.reshape(1, w, hk, dk)
    v = v.reshape(1, w, hv, dv)
    a = a.reshape(1, w, hv)
    b = b.reshape(1, w, hv)
    (y,) = _gdn_kernel()(
        inputs=[
            state,
            layer.A_log,
            layer.dt_bias,
            q,
            k,
            v,
            a,
            b,
            shape.parents_array(),
        ],
        template=[
            ("InT", dtype),
            ("Hk", hk),
            ("Hv", hv),
            ("Dk", dk),
            ("Dv", dv),
            ("T", w),
        ],
        grid=(32, dv, hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(1, w, hv, dv)],
        output_dtypes=[dtype],
    )
    z = z.reshape(1, w, hv, dv)
    out, sums = gv._norm_gate_kernel(layer.norm.eps)(
        inputs=[y, z, layer.norm.weight],
        template=[("InT", dtype)],
        grid=(32, w * hv, 1),
        threadgroup=(32, 8, 1),
        output_shapes=[(1, w, hv * dv), (w, hv * dv // 64)],
        output_dtypes=[dtype, mx.float32],
    )
    vq.register_group_sums(out, sums)
    record = ("gdn", layer, state, conv_prev, mixed, (k, v, a, b))
    return verifier._linear(layer.out_proj, out), record


# ── attention ───────────────────────────────────────────────────────────────

_MERGE2 = """
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.y;             // node * H + head
    constexpr int DPL = D / 32;
    const uint node = r / H, h = r % H;
    const int nkeys = lengths[node];
    const int nc = min(NC, max(0, (nkeys + CK - 1) / CK));
    float m = -INFINITY, l = 0.0f, o[DPL];
    for (int i = 0; i < DPL; i++) o[i] = 0.0f;
    for (int c = 0; c < nc; c++) {                             // chunk order
        const bool shared = c < CS;
        const int64_t row = shared ? ((int64_t)startsA[node] + c) * H + h
                                   : ((int64_t)startsB[node] + (c - CS)) * H + h;
        const float mc = shared ? PMA[row] : PMB[row];
        if (mc == -INFINITY) continue;
        const float lc = shared ? PLA[row] : PLB[row];
        const float nm = max(m, mc);
        const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
        l = l * f1 + lc * f2;
        for (int i = 0; i < DPL; i++) {
            const float po = shared ? POA[row * D + lane * DPL + i] : POB[row * D + lane * DPL + i];
            o[i] = o[i] * f1 + po * f2;
        }
        m = nm;
    }
    device bfloat16_t* dst = out + ((int64_t)h * W + node) * D + lane * DPL;
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (int i = 0; i < DPL; i++) dst[i] = static_cast<bfloat16_t>(o[i] * inv);
"""


def _merge_kernel():
    kernel = _KERNELS.get("merge")
    if kernel is None:
        kernel = _KERNELS["merge"] = mx.fast.metal_kernel(
            name="yunshu_tree_attn_merge",
            input_names=[
                "POA",
                "PMA",
                "PLA",
                "POB",
                "PMB",
                "PLB",
                "startsA",
                "startsB",
                "lengths",
            ],
            output_names=["out"],
            source=_MERGE2,
        )
    return kernel


def _tile_partials(q_fused, keys, values, lengths, slots, scale, plan, t_tokens, nc):
    """The tile kernel's partial launch (``ragged_decode_attention``'s tile
    branch) on a work list ``plan`` = ``ra._work_list(...)``: returns PO, PM,
    PL and the per-(row, token) first slot."""
    hkv, cap, d = int(keys.shape[1]), int(keys.shape[2]), int(keys.shape[3])
    h = int(q_fused.shape[2]) // 8 * hkv
    g = h // hkv
    work, wstart, nslot = plan
    po, pm, pl = ra._kernel("tile")(
        inputs=[q_fused, keys, values, lengths, slots, scale, work, wstart],
        template=[
            ("D", d),
            ("G", g),
            ("T", t_tokens),
            ("H", h),
            ("HKV", hkv),
            ("CAP", cap),
            ("CK", CK),
            ("NC", nc),
            ("WL", True),
        ],
        grid=(256, hkv * int(work.size), 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(nslot * h * d,), (nslot * h,), (nslot * h,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return po, pm, pl, wstart


def _fuse_tokens(q, hkv, g, tokens):
    """[B, H, T, D] -> the tile kernel's [B, HKV, 8 * G, D] (row = t * G + g)."""
    bsz, _, t, d = q.shape
    f = q.reshape(bsz, hkv, g, t, d).transpose(0, 1, 3, 2, 4)
    if t < 8:
        f = mx.pad(f, [(0, 0), (0, 0), (0, 8 - t), (0, 0), (0, 0)])
    return mx.contiguous(f.reshape(bsz, hkv, 8 * g, d))


_SCALES: dict = {}
_ZEROS: dict = {}


def _scale_array(scale: float) -> mx.array:
    arr = _SCALES.get(scale)
    if arr is None:
        arr = _SCALES[scale] = mx.array([float(scale)], dtype=mx.float32)
    return arr


def _zeros(shape, dtype):
    key = (shape, dtype)
    arr = _ZEROS.get(key)
    if arr is None:
        if len(_ZEROS) > 32:
            _ZEROS.clear()
        arr = _ZEROS[key] = mx.zeros(shape, dtype=dtype)
    return arr


class RoundContext:
    """Tables of one window at one position, built once and shared by every
    attention layer of the forward. A dynamic shape sizes its work lists for
    the deepest possible row (rows shorter than that skip their empty chunks
    from their true lengths, which are arrays)."""

    def __init__(self, shape, n0: int):
        w = shape.width
        self.shape, self.n0 = shape, n0
        self.cstar = n0 // CK
        self.tail_start = self.cstar * CK
        self.ntail = (n0 + shape.max_depth - self.tail_start) // CK + 1
        self.nc_total = self.cstar + self.ntail
        self.m = n0 - self.tail_start
        self.win_idx = shape.path_table().reshape(-1)
        self.slots_b = mx.arange(w, dtype=mx.int32)
        if shape.dynamic:
            depth = shape.depth_array()
            worst = n0 + shape.max_depth + 1 - self.tail_start
            self.local_arr = n0 + depth + 1 - self.tail_start
            self.abs_lengths = n0 + depth + 1
            self.plan_b = ra._work_list((worst,) * w, 1, self.ntail, True)
        else:
            local = [n0 + dep + 1 - self.tail_start for dep in shape.depths]
            self.local_arr = mx.array(local, dtype=mx.int32)
            self.abs_lengths = mx.array(
                [n0 + dep + 1 for dep in shape.depths], dtype=mx.int32
            )
            self.plan_b = ra._work_list(tuple(local), 1, self.ntail, True)
        if self.cstar:
            # shared chunks, one tile launch per 8 consecutive tokens (the
            # kernel's fused token rows; a group's last token sees ``off``
            # fewer window keys than the window's last)
            self.slot0 = mx.array([0], dtype=mx.int32)
            self.groups_a = []
            bases, base = [], 0
            for lo in range(0, w, ra.TILE_TOKENS):
                tg = min(ra.TILE_TOKENS, w - lo)
                length = self.tail_start + lo + tg - 1
                nc = -(-length // CK)
                plan = ra._work_list((length,), tg, nc, True)
                self.groups_a.append(
                    (lo, tg, mx.array([length], dtype=mx.int32), nc, plan)
                )
                bases.append(plan[1] + base)
                base += plan[2]
            self.starts_a = bases[0] if len(bases) == 1 else mx.concatenate(bases)
        self.pos = None


def tree_attention(
    queries: mx.array,
    cache: Any,
    scale: float,
    shape: TreeShape,
    n0: int,
    rc: RoundContext | None = None,
) -> mx.array:
    """Attention of the window's rows (``queries`` [1, H, W, D], keys already
    appended at ``n0 + row``) with each row seeing the prefix and its own
    ancestors at logical positions ``n0 + j``. Returns [1, H, W, D] bf16."""
    w = shape.width
    rc = rc or RoundContext(shape, n0)
    keys, values = cache.keys, cache.values
    if keys.shape[2] % 64:
        pad = [
            (0, 0),
            (0, 0),
            (0, ragged_kv._round_up(keys.shape[2]) - keys.shape[2]),
            (0, 0),
        ]
        cache.keys = keys = mx.pad(keys, pad)
        cache.values = values = mx.pad(values, pad)
    _, h, _, d = (int(s) for s in queries.shape)
    hkv = int(keys.shape[1])
    g = h // hkv
    cstar, tail_start, ntail, m = rc.cstar, rc.tail_start, rc.ntail, rc.m
    scale_arr = _scale_array(scale)

    # shared chunks 0 .. cstar - 1: every row sees them whole
    if cstar:
        parts = [
            _tile_partials(
                _fuse_tokens(queries[:, :, lo : lo + tg], hkv, g, tg),
                keys,
                values,
                len_arr,
                rc.slot0,
                scale_arr,
                plan,
                tg,
                nc,
            )[:3]
            for lo, tg, len_arr, nc, plan in rc.groups_a
        ]
        if len(parts) == 1:
            po_a, pm_a, pl_a = parts[0]
        else:
            po_a, pm_a, pl_a = (
                mx.concatenate(list(x)) for x in zip(*parts, strict=True)
            )
        starts_a = rc.starts_a
    else:
        po_a = pm_a = pl_a = _zeros((1,), mx.float32)
        starts_a = _zeros((w,), mx.int32)

    # window chunk(s): one row per node over a gathered copy of the tail
    # Keep CK-sized partials and their reduction order, but only materialize
    # complete 64-key tiles containing live keys. The tile kernel's last load
    # still begins at the same address; masked CK padding needs no backing rows.
    cap2 = _tree_tail_capacity(m + shape.max_depth + 1)
    hkv_keys = keys[0]  # [HKV, CAP, D]
    hkv_vals = values[0]
    win_k = (
        mx.take(hkv_keys[:, n0 : n0 + w], rc.win_idx, axis=1)
        .reshape(hkv, w, shape.max_depth + 1, d)
        .transpose(1, 0, 2, 3)
    )
    win_v = (
        mx.take(hkv_vals[:, n0 : n0 + w], rc.win_idx, axis=1)
        .reshape(hkv, w, shape.max_depth + 1, d)
        .transpose(1, 0, 2, 3)
    )
    parts_k, parts_v = [], []
    if m:
        parts_k.append(
            mx.broadcast_to(hkv_keys[None, :, tail_start:n0], (w, hkv, m, d))
        )
        parts_v.append(
            mx.broadcast_to(hkv_vals[None, :, tail_start:n0], (w, hkv, m, d))
        )
    parts_k.append(win_k)
    parts_v.append(win_v)
    fill = cap2 - m - (shape.max_depth + 1)
    if fill:
        z = _zeros((w, hkv, fill, d), keys.dtype)
        parts_k.append(z)
        parts_v.append(z)
    tail_k = mx.concatenate(parts_k, axis=2)
    tail_v = mx.concatenate(parts_v, axis=2)
    q_b = queries[0].reshape(hkv, g, w, d).transpose(2, 0, 1, 3)  # [W, HKV, G, D]
    q_b = q_b[:, :, None]  # [W, HKV, 1, G, D]
    q_b = mx.pad(q_b, [(0, 0), (0, 0), (0, 7), (0, 0), (0, 0)]).reshape(
        w, hkv, 8 * g, d
    )
    po_b, pm_b, pl_b, starts_b = _tile_partials(
        mx.contiguous(q_b),
        tail_k,
        tail_v,
        rc.local_arr,
        rc.slots_b,
        scale_arr,
        rc.plan_b,
        1,
        ntail,
    )
    (out,) = _merge_kernel()(
        inputs=[po_a, pm_a, pl_a, po_b, pm_b, pl_b, starts_a, starts_b, rc.abs_lengths],
        template=[
            ("D", d),
            ("H", h),
            ("W", w),
            ("NC", rc.nc_total),
            ("CK", CK),
            ("CS", cstar),
        ],
        grid=(32, w * h, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, h, w, d)],
        output_dtypes=[mx.bfloat16],
    )
    return out


def supported(language_model: Any) -> bool:
    """Tile kernel present, dense Qwen3.5 layers the fused GDN kernels take."""
    if not ra.tile_ready():
        return False
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:  # pragma: no cover
        return False
    inner = getattr(language_model, "model", None)
    if not isinstance(inner, q35.Qwen3_5Model):
        return False
    if not gv._PATCHED:
        return False
    from .kernels import batch_invariant

    if not batch_invariant.is_installed():
        return False
    for layer in inner.layers:
        if layer.is_linear:
            g = layer.linear_attn
            if g.head_k_dim != 128 or g.head_v_dim != 128 or g.conv_kernel_size != 4:
                return False
        else:
            a = layer.self_attn
            if a.head_dim != 256 or a.num_attention_heads // a.num_key_value_heads > 8:
                return False
    return True


# ── the forward ─────────────────────────────────────────────────────────────


def lane_projections(lm: Any) -> bool:
    """The decoder's projections are ``LaneLinear`` (row-invariant at any row count)."""
    from .kernels.lane_linear import LaneLinear

    layer = lm.model.layers[0]
    proj = layer.linear_attn.in_proj_qkv if layer.is_linear else layer.self_attn.q_proj
    return isinstance(proj, LaneLinear)


def lane_ready(lm: Any, cache: list) -> bool:
    """The prompt cache is the single-row lane layout ``tree_forward`` reads."""
    return ragged_kv._lane_length(cache[lm.model.fa_idx]) is not None


def _rope_delta(lm) -> int:
    delta = getattr(lm, "_rope_deltas", None)
    if delta is None:
        return 0
    return int(delta.reshape(-1)[0].item())


def tree_forward(
    lm, tokens: mx.array, shape: TreeShape, cache: list, capture_ids=()
) -> TreeResult:
    """Run the window through the decoder. ``tokens`` [1, W]."""
    from mlx_vlm.models.qwen3_5 import language as q35

    verifier = q35._EXACT_SPECULATIVE_VERIFIER
    model = lm.model
    w = shape.width
    if w > MAX_ROWS:
        raise ValueError(f"tree window of {w} rows (limit {MAX_ROWS})")
    if w > NARROW_ROWS and not lane_projections(lm):
        raise ValueError(f"tree window of {w} rows needs the lane projections")
    n0 = ragged_kv._lane_length(cache[model.fa_idx])
    if n0 is None:
        raise ValueError("tree verify needs the single-row lane cache")
    base = n0 + _rope_delta(lm)
    if shape.dynamic:
        pos = mx.broadcast_to((base + shape.depth_array())[None, None], (3, 1, w))
    else:
        pos = mx.array(
            [[[base + d for d in shape.depths]]] * 3, dtype=mx.int32
        ).reshape(3, 1, w)
    rc = RoundContext(shape, n0)
    h = model.embed_tokens(tokens)
    res = TreeResult(shape=shape, n0=n0, hidden=h)
    capture = set(capture_ids)
    layers = model.layers
    nxt_norm = {i: layers[i + 1].input_layernorm for i in range(len(layers) - 1)}
    normed = layers[0].input_layernorm(h)
    for i, (layer, c) in enumerate(zip(layers, cache, strict=True)):
        if layer.is_linear:
            r, rec = _gdn_layer(verifier, layer.linear_attn, normed, c, shape)
        else:
            at = layer.self_attn
            q, k, v = verifier._linears((at.q_proj, at.k_proj, at.v_proj), normed)
            queries, _k, _v, gate, _ = at._prepare_projected_qkv(
                q, k, v, c, pos, None, None
            )
            out = tree_attention(queries, c, at.scale, shape, n0, rc)
            out = out.transpose(0, 2, 1, 3).reshape(1, w, -1)
            r = verifier._linear(at.o_proj, out * mx.sigmoid(gate))
            rec = ("kv",)
        res.records[i] = rec
        # fused residual add + RMSNorm (bit-exact to the separate ops)
        post = layer.post_attention_layernorm
        if vq._add_rms_eligible(h, r, post):
            h, normed, _ = vq.add_rms_norm(h, r, post)
        else:
            h = h + r
            normed = post(h)
        ff = verifier._feed_forward(layer.mlp, normed)
        nxt = nxt_norm.get(i)
        if nxt is not None and vq._add_rms_eligible(h, ff, nxt):
            h, normed, _ = vq.add_rms_norm(h, ff, nxt)
        else:
            h = h + ff
            if nxt is not None:
                normed = nxt(h)
        if i in capture:
            res.captured.append(h)
        # Submit the unchanged graph while later layers are being built. The
        # first layer starts the drafter/verify dependency promptly; four-layer
        # chunks avoid a separate submission for every decoder layer.
        if (i == 0 or (i + 1) % 4 == 0) and i + 1 < len(layers):
            try:
                mx.async_eval(h, normed)
            except BaseException:
                # Early evaluation can fail before the round loop receives
                # TreeResult. Restore appended KV lengths here in that case.
                tree_abort(cache, res)
                raise
    res.hidden = model.norm(h)
    return res


def tree_commit(lm, cache: list, res: TreeResult, path: list[int]) -> None:
    """Keep the window rows in ``path`` (root first, each row a child of the
    previous one): the caches end as if those tokens had been decoded one by one."""
    from mlx_vlm.models.qwen3_5 import language as q35

    w, n0, m = res.shape.width, res.n0, len(path)
    path_arr = mx.array(path + [0] * (w - m), dtype=mx.int32)
    count_arr = mx.array([m], dtype=mx.int32)
    conv_idx = mx.array(([0, 1, 2] + [3 + r for r in path])[-3:], dtype=mx.int32)
    compact = path != list(range(m))
    if compact:
        src = mx.array([n0 + r for r in path[1:]], dtype=mx.int32)
    for i, rec in res.records.items():
        c = cache[i]
        if rec[0] == "kv":
            if compact and m > 1:
                c.keys[..., n0 + 1 : n0 + m, :] = mx.take(c.keys, src, axis=2)
                c.values[..., n0 + 1 : n0 + m, :] = mx.take(c.values, src, axis=2)
            c.trim(w - m)
        else:
            _, layer, state, conv_prev, mixed, (k, v, a, b) = rec
            c[1] = replay_path(layer, state, (k, v, a, b), path_arr, count_arr)
            c._omlx_gdn_pending = None
            seq = mx.concatenate([conv_prev, mixed], axis=1)[0]
            c[0] = mx.take(seq, conv_idx, axis=0)[None]
            if hasattr(c, "advance"):
                c.advance(m)
                q35._qwen3_5_advance_lengths_info(c, m)


def tree_abort(cache: list, res: TreeResult) -> None:
    """Drop the window: attention caches lose its keys, GDN caches never changed."""
    for i, rec in res.records.items():
        if rec[0] == "kv":
            cache[i].trim(res.shape.width)


__all__ = [
    "MAX_ROWS",
    "TreeResult",
    "TreeShape",
    "supported",
    "tree_abort",
    "tree_attention",
    "tree_commit",
    "tree_forward",
]
