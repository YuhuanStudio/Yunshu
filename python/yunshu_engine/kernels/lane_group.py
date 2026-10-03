"""Group column-aligned lane projections without changing their split-K.

A concatenated output width must never choose a new split-K: its reduction
order could change target logits. Packed buffers replace the member buffers
with views, so there is only one resident copy. The cache belongs to the first
member, rather than a process-wide verifier that would retain unloaded models.
"""

from __future__ import annotations

import weakref

import mlx.core as mx

from . import lane_linear
from .tensorfold import lane_qmm


class _Group:
    def __init__(self, members, split):
        self.members = tuple(weakref.ref(member) for member in members)
        self.split = split
        self.weight = mx.concatenate([member.weight for member in members], axis=0)
        self.sbt = mx.concatenate([member.sbt for member in members], axis=1)
        mx.eval(self.weight, self.sbt)
        offset = 0
        for member in members:
            end = offset + member.output_dims
            member.weight = self.weight[offset:end]
            member.sbt = self.sbt[:, offset:end]
            offset = end
        mx.eval([member.parameters() for member in members])
        self.buffers = tuple((member.weight, member.sbt) for member in members)

    def matches(self, members):
        return len(members) == len(self.members) and all(
            ref() is member and member.weight is weight and member.sbt is sbt
            for ref, member, (weight, sbt) in zip(
                self.members, members, self.buffers, strict=True
            )
        )


def grouped_linears(members, x):
    """Return grouped outputs, or None when independent calls are required.

    The caller owns its normal fallback. Prefill and biased/unaligned/mixed
    projections stay there. This function does not change sampling policy.
    """
    members = tuple(members)
    if len(members) < 2 or not all(
        isinstance(member, lane_linear.LaneLinear) for member in members
    ):
        return None
    first = members[0]
    rows = x.size // first.input_dims
    if not 0 < rows <= 128:
        return None
    if lane_linear.STOCK_ROWS and rows > lane_linear.STOCK_ROWS:
        return None
    if not all(
        member.bits == first.bits
        and member.group_size == first.group_size
        and member.input_dims == first.input_dims
        and member.tiled == first.tiled
        and member.output_dims % lane_qmm.NT == 0
        and "bias" not in member
        for member in members
    ):
        return None
    splits = {
        lane_qmm.split_k(member.output_dims, member.input_dims) for member in members
    }
    blocks = {
        16 if 16 < rows <= 48 and member.output_dims < 100_000 else None
        for member in members
    }
    if len(splits) != 1 or len(blocks) != 1:
        return None
    group = getattr(first, "_yunshu_column_group", None)
    if group is not None and not group.matches(members):
        return None
    if group is None and any(
        getattr(member, "_yunshu_column_group", None) is not None for member in members
    ):
        return None
    if group is None:
        group = _Group(members, splits.pop())
        for member in members:
            object.__setattr__(member, "_yunshu_column_group", group)
    dtype = x.dtype
    x2 = x.reshape(rows, first.input_dims).astype(mx.bfloat16)
    out = lane_qmm.lane_matmul(
        x2,
        group.weight,
        group.sbt,
        tiled=first.tiled,
        group=first.group_size,
        sk=group.split,
        row_block=blocks.pop(),
        row_limit=lane_linear.PIECE,
    ).reshape(*x.shape[:-1], sum(member.output_dims for member in members))
    offset = 0
    outputs = []
    for member in members:
        end = offset + member.output_dims
        outputs.append(out[..., offset:end].astype(dtype))
        offset = end
    return tuple(outputs)
