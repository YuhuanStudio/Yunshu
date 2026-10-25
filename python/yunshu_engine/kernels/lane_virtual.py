# Upstream (derived): ashhart/TensorFold (MIT) src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py @ 34bae79a
"""Own address-only grouping: several original buffers, one unchanged lane op.

Select the projection pointer and local column at each column tile. No packed
weight copy of a wide projection, no module mutation, no model-owned cache is
needed. The original per-projection split-K, row block, fma and slice-add order
remain fixed, so a grouped column is bit-identical to the same column computed
by its own ``LaneLinear`` call: members share a group only when their K split
(and weight format) already agree. A projection whose width is not a multiple of
32 (GDN ``in_proj_a`` / ``in_proj_b``, 48 wide) is read from a private copy
tiled with zero padding to the next 32 (a few KiB per layer); padded columns are
never stored.

Members of one call that cannot share a kernel (different bits or split) run as
single lane matmuls on the same input group sums.
"""

import logging

import mlx.core as mx

from . import gdn_prefill
from . import lane_linear as _ll
from .lane_linear import LaneLinear
from .tensorfold import lane_qmm as q

logger = logging.getLogger(__name__)
MAX_MEMBERS = 4
STATS: dict = {"launches": 0, "members": 0}
FAMILIES = {4: "main_tiled", 5: "bytes"}
_KERNELS: dict = {}


def _source(count: int, bits: int, pipe: bool = False) -> str:
    q._resolve_variant()
    from .tensorfold import lane_widen

    if pipe and bits == 4:
        from .tensorfold import lane_pipe

        body = lane_pipe._PIPE
    else:
        body = q._MAIN_TILED if bits == 4 else lane_widen.BYTES
    needle = "  const int n0 = threadgroup_position_in_grid.x * NT;"
    assert body.count(needle) == 1

    def pick(stem: str) -> str:
        chain = " : ".join(f"mi == {i} ? {stem}{i}" for i in range(count - 1))
        return f"{chain} : {stem}{count - 1}"

    steps = "".join(
        f"  if (gcol >= O{i}) {{ mi = {i}; offset = O{i}; }}\n" for i in range(1, count)
    )
    select = (
        "  const int gcol = threadgroup_position_in_grid.x * NT;\n"
        "  int mi = 0, offset = 0;\n" + steps + f"  const int N = {pick('V')};\n"
        "  const int n0 = gcol - offset;\n"
        f"  auto Wq = {pick('W')};\n"
        f"  auto SBt = {pick('S')};\n"
        f"  auto Y = {pick('Y')};\n"
    )
    body = body.replace(needle, select)
    assert body.count("threadgroup_position_in_grid.x * KG") >= 1
    return body.replace("threadgroup_position_in_grid.x * KG", "(n0 / NT) * KG")


def _kernel(count: int, bits: int):
    pipe = bool(q.PIPE[0] and bits == 4 and q._resolve_variant() == "m5")
    key = (count, bits, pipe)
    if key not in _KERNELS:
        inputs = ["X", "XS"]
        for i in range(count):
            inputs += [f"W{i}", f"S{i}"]
        _KERNELS[key] = q._Baked(
            f"yunshu_virtual_columns_{count}_b{bits}{'_pipe' if pipe else ''}",
            _source(count, bits, pipe),
            inputs + ["mdims"],
            [f"Y{i}" for i in range(count)],
        )
    return _KERNELS[key]


def _padded_tiles(member: LaneLinear):
    """(weight, padded width) in 32-column tiles; a narrow projection gets a private padded copy."""
    n = member.output_dims
    pad = -(-n // q.NT) * q.NT
    if member.tiled:
        return member.weight, pad
    copy = member.__dict__.get("_lane_padded")
    if copy is None:
        kw = int(member.weight.shape[1])
        rows = mx.concatenate(
            [member.weight, mx.zeros((pad - n, kw), dtype=member.weight.dtype)]
        )
        copy = mx.contiguous(
            q.tile_weight(rows, q.NT, member.group_size, bits=member.bits)
        )
        mx.eval(copy)
        object.__setattr__(member, "_lane_padded", copy)
    return copy, pad


def _eligible(m) -> bool:
    return (
        m.bits in FAMILIES
        and (m.bits == 4 or m.group_size == 64)
        and "bias" not in m
        and getattr(m, "sk", None) is None
        and m.output_dims % 4 == 0
    )


def _block(rows: int, n: int):
    return 16 if 16 < rows <= 48 and n < 100_000 else None


def _group_sums(x, x2, group: int, mp: int, rows: int):
    k = int(x2.shape[-1])
    for probe in (x, x2):
        key = id(probe) if group == 64 else (id(probe), group)
        hit = q._xs_cache.get(key)
        if hit is not None and hit[0] is probe and hit[1].shape == (k // group, mp):
            return hit[1]
    mdims = q._mdims(rows, mp)
    xs = q._kernel("xsum")(
        inputs=[x2, mdims],
        template=[("K", k), ("GS", group)],
        grid=(k // group, mp, 1),
        threadgroup=(min(k // group, 256), 1, 1),
        output_shapes=[(k // group, mp)],
        output_dtypes=[mx.float32],
    )[0]
    key = id(x2) if group == 64 else (id(x2), group)
    q._xs_cache[key] = (x2, xs)
    while len(q._xs_cache) > 4:
        q._xs_cache.pop(next(iter(q._xs_cache)))
    return xs


def _run_group(members, x2, xs, rows, mp):
    a = members[0]
    split = q.split_k(a.output_dims, a.input_dims)
    block = _block(rows, a.output_dims) or min(mp, q.ROW_BLOCK)
    weights, pads = [], []
    for m in members:
        w, pad = _padded_tiles(m)
        weights.append(w)
        pads.append(pad)
    offsets = [0]
    for pad in pads[:-1]:
        offsets.append(offsets[-1] + pad)
    inputs = [x2, xs]
    for m, w in zip(members, weights, strict=True):
        inputs += [w, m.sbt]
    template = [
        ("TMR", block // 16),
        ("K", a.input_dims),
        ("NT", q.NT),
        ("SK", split),
        ("GS", a.group_size),
        ("BITS", a.bits),
        ("TILED", 1),
        ("EDGE", int(mp % block != 0)),
    ]
    for i, m in enumerate(members):
        template += [(f"V{i}", m.output_dims), (f"O{i}", offsets[i])]
    total = sum(pads) // q.NT
    return _kernel(len(members), a.bits)(
        inputs=inputs + [q._mdims(rows, mp)],
        template=template,
        grid=(total * 32 * split, -(-mp // block), 1),
        threadgroup=(32 * split, 1, 1),
        output_shapes=[(rows, m.output_dims) for m in members],
        output_dtypes=[mx.bfloat16] * len(members),
    )


def grouped_linears(members, x):
    """The members' outputs for ``x`` with compatible ones in one launch; None when nothing groups.

    Every output is bit-identical to ``member(x)``: grouped columns keep their own split-K and
    fma order, and the remaining members run their own lane matmul on the same group sums.
    """
    members = tuple(members)
    if not 2 <= len(members) <= MAX_MEMBERS or not all(
        isinstance(m, LaneLinear) for m in members
    ):
        return None
    a = members[0]
    rows = x.size // a.input_dims
    if not 0 < rows <= 128 or any(m.input_dims != a.input_dims for m in members):
        return None
    if gdn_prefill.split_enabled() and gdn_prefill.in_prefill():
        return None
    if _ll.STOCK_ROWS and rows > _ll.STOCK_ROWS:
        return None
    parts: dict = {}
    for i, m in enumerate(members):
        if _eligible(m):
            key = (
                m.bits,
                m.group_size,
                q.split_k(m.output_dims, m.input_dims),
                _block(rows, m.output_dims),
            )
            parts.setdefault(key, []).append(i)
    groups = [idx for idx in parts.values() if len(idx) >= 2]
    if not groups:
        return None
    dtype = x.dtype
    x2 = x.reshape(rows, a.input_dims)
    if dtype != mx.bfloat16:
        x2 = x2.astype(mx.bfloat16)
    mp = 16 * ((rows + 15) // 16)
    results: list = [None] * len(members)
    # cache_input: a producer kernel may already have written the sums under id(x)
    cache_input = x if _ll._SUM_REUSE and dtype == mx.bfloat16 else None
    for idx in groups:
        if not STATS["launches"]:
            logger.info(
                "lane columns grouped: %d members of %d, %d-bit",
                len(idx),
                len(members),
                members[idx[0]].bits,
            )
        STATS["launches"] += 1
        STATS["members"] += len(idx)
        sub = [members[i] for i in idx]
        xs = _group_sums(x, x2, sub[0].group_size, mp, rows)
        for i, y in zip(idx, _run_group(sub, x2, xs, rows, mp), strict=True):
            results[i] = y
    for i, m in enumerate(members):
        if results[i] is None:
            results[i] = m._rows(x2, cache_input=cache_input)
            if "bias" in m:
                results[i] = results[i] + m["bias"]
    return tuple(
        y.reshape(*x.shape[:-1], m.output_dims).astype(dtype)
        for y, m in zip(results, members, strict=True)
    )
