"""Non-bf16 attention fallback (fp16 MTP head): the gather slices the stored row before ``take``.

Output must equal the reference that gathers the whole slot row and slices after it, and every
fallback call is counted in ``FALLBACK_HITS``."""

import pytest

mx = pytest.importorskip("mlx.core")


class _Attn:
    """Projections on [B, T, 64] with 2 query heads and 1 KV head of width 32, no rotary."""

    scale = 32**-0.5

    def q_proj(self, x):
        return x

    def k_proj(self, x):
        return x[..., :32]

    def v_proj(self, x):
        return x[..., 32:]

    def o_proj(self, x):
        return x

    def _prepare_projected_qkv(self, q, k, v, *_):
        b, t, _d = q.shape
        queries = q.reshape(b, t, 2, 32).transpose(0, 2, 1, 3)
        keys = k.reshape(b, t, 1, 32).transpose(0, 2, 1, 3)
        values = v.reshape(b, t, 1, 32).transpose(0, 2, 1, 3)
        return queries, keys, values, mx.zeros((b, t, 64), q.dtype), None


def test_fp16_fallback_slices_before_gather_and_counts_hits():
    from yunshu_engine.round_driver import batch as bt

    at = _Attn()
    slots = bt.Slots(1)
    slot = slots.alloc()
    slots.reserve(16)  # CAP 512: the stored row is far longer than the attended prefix
    hits0 = bt.FALLBACK_HITS[0]

    mx.random.seed(0)
    xn0 = mx.random.normal((1, 4, 64)).astype(mx.float16)
    bt.attend(at, xn0, slots, 0, bt.KVPlan.make([slot], [0], 4))

    xn1 = mx.random.normal((1, 1, 64)).astype(mx.float16)
    plan = bt.KVPlan.make([slot], [4], 1)
    out = bt.attend(at, xn1, slots, 0, plan)
    mx.eval(out)

    assert bt.FALLBACK_HITS[0] - hits0 == 2
    assert slots.k[0].shape[2] == 512

    # reference: gather the whole row, then slice (the order before this change)
    queries = xn1.reshape(1, 1, 2, 32).transpose(0, 2, 1, 3)
    length = plan.max_length
    ref_k = mx.take(slots.k[0], plan.slot_ids, axis=0)[..., :length, :]
    ref_v = mx.take(slots.v[0], plan.slot_ids, axis=0)[..., :length, :]
    positions = mx.array(plan.n0, dtype=mx.int32)[:, None] + mx.arange(1)[None, :]
    mask = mx.arange(length)[None, None, :] <= positions[:, :, None]
    ref = mx.fast.scaled_dot_product_attention(
        queries, ref_k, ref_v, scale=at.scale, mask=mask[:, None, :, :]
    )
    ref = ref.transpose(0, 2, 1, 3).reshape(1, 1, -1) * mx.sigmoid(
        mx.zeros((1, 1, 64), mx.float16)
    )
    mx.eval(ref)
    assert out.dtype == mx.float16
    assert mx.allclose(out, ref, atol=1e-3).item()
