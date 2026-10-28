"""Tensor-unit QSA prefill attention vs a fp32 dense reference and upstream's indexed kernel (M5 GPU only)."""

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.kernels import qsa_sparse_prefill as qsp

pytestmark = pytest.mark.skipif(
    not (mx.metal.is_available() and qsp.ready()), reason="needs an M5-class GPU"
)

H, KV, D, TOPK, BLOCK = 24, 2, 256, 512, 4


def make(rows=192, keys=2700, seed=0):
    rng = np.random.default_rng(seed)
    cap = keys + 100  # keys live in a larger buffer, as in the cache
    q = mx.array(rng.normal(size=(1, H, rows, D)).astype(np.float32)).astype(
        mx.bfloat16
    )
    k = mx.array(rng.normal(size=(1, KV, cap, D)).astype(np.float32)).astype(
        mx.bfloat16
    )
    v = mx.array(rng.normal(size=(1, KV, cap, D)).astype(np.float32)).astype(
        mx.bfloat16
    )
    k_view, v_view = k[:, :, :keys, :], v[:, :, :keys, :]
    ends = np.arange(keys - rows + 1, keys + 1, dtype=np.int32)
    blocks = np.full((1, rows, TOPK), -1, dtype=np.int32)
    for i, end in enumerate(ends):
        complete = int(end) // BLOCK
        pick = rng.permutation(complete)[:TOPK]
        blocks[0, i, : len(pick)] = pick
    return q, k_view, v_view, mx.array(blocks), mx.array(ends[None])


def reference(q, k, v, blocks, ends, scale):
    """fp32 attention over each row's selected blocks and tail tokens."""
    rows = q.shape[2]
    blocks_np, ends_np = np.array(blocks)[0], np.array(ends)[0]
    out = np.zeros((1, H, rows, D), dtype=np.float32)
    kf, vf = np.array(k.astype(mx.float32))[0], np.array(v.astype(mx.float32))[0]
    qf = np.array(q.astype(mx.float32))[0]
    for r in range(rows):
        sel = blocks_np[r][blocks_np[r] >= 0]
        ids = np.concatenate(
            [
                (sel[:, None] * BLOCK + np.arange(BLOCK)).ravel(),
                np.arange((ends_np[r] // BLOCK) * BLOCK, ends_np[r]),
            ]
        )
        for h in range(H):
            s = (qf[h, r] @ kf[h // (H // KV)][ids].T) * scale
            p = np.exp(s - s.max())
            out[0, h, r] = (p / p.sum()) @ vf[h // (H // KV)][ids]
    return out


def test_matches_reference_as_well_as_upstream_and_agrees_on_argmax():
    from mlx_vlm.models.qwen4_exp.qsa_kernel import qsa_sparse_attention

    scale = D**-0.5
    q, k, v, blocks, ends = make()
    got = np.array(qsp.attend(q, k, v, blocks, ends, scale).astype(mx.float32))
    ref = reference(q, k, v, blocks, ends, scale)
    up = np.array(
        qsa_sparse_attention(
            q, k, v, blocks, ends, scale=scale, block_size=BLOCK
        ).astype(mx.float32)
    )
    err, up_err = np.abs(got - ref).max(), np.abs(up - ref).max()
    print(
        f"max abs diff vs fp32 reference: kernel {err:.4g}, upstream indexed kernel {up_err:.4g}"
    )
    assert err <= 2 * up_err + 2e-3
    proj = np.random.default_rng(3).normal(size=(H * D, 64)).astype(np.float32)
    flat = lambda a: a.transpose(0, 2, 1, 3).reshape(a.shape[2], H * D) @ proj  # noqa: E731
    agree = float((flat(got).argmax(-1) == flat(ref).argmax(-1)).mean())
    print(f"argmax agreement with the reference through a random head: {agree:.3f}")
    assert agree >= 0.98


def test_row_result_does_not_depend_on_the_other_rows():
    scale = D**-0.5
    q, k, v, blocks, ends = make(rows=96, keys=2400, seed=1)
    full = qsp.attend(q, k, v, blocks, ends, scale)
    half = qsp.attend(q[:, :, 40:], k, v, blocks[:, 40:], ends[:, 40:], scale)
    assert mx.array_equal(full[:, :, 40:], half).item()


def test_short_prefix_rows_attend_every_key():
    scale = D**-0.5
    q, k, v, blocks, ends = make(
        rows=64, keys=900, seed=2
    )  # every row has < 512 complete blocks
    got = np.array(qsp.attend(q, k, v, blocks, ends, scale).astype(mx.float32))
    ref = reference(q, k, v, blocks, ends, scale)
    assert np.abs(got - ref).max() < 3e-2


def test_dispatch_routes_eligible_prefill_and_falls_back_otherwise(monkeypatch):
    from mlx_vlm.models.qwen4_exp import language as lang

    seen = []

    def original(queries, keys, values, block_indices, query_ends, **kw):
        seen.append(kw)
        return queries

    monkeypatch.setattr(lang, "dispatch_qsa_attention", original)
    monkeypatch.setitem(qsp.COUNTS, "engaged", 0)
    monkeypatch.setitem(qsp.COUNTS, "fallback", 0)
    assert qsp.install() and qsp.install()
    scale = D**-0.5
    q, k, v, blocks, ends = make(rows=64, keys=1200, seed=4)
    kw = dict(
        cache=None,
        scale=scale,
        block_size=BLOCK,
        causal=True,
        mask=None,
        mask_factory=None,
    )
    got = lang.dispatch_qsa_attention(q, k, v, blocks, ends, **kw)
    assert mx.array_equal(got, qsp.attend(q, k, v, blocks, ends, scale)).item()
    assert qsp.COUNTS == {"engaged": 1, "fallback": 0} and not seen
    # a quantized cache hands over tuples; a decode step is not a prefill call: both keep upstream's path
    lang.dispatch_qsa_attention(q, (k, k), (v, v), blocks, ends, **kw)
    lang.dispatch_qsa_attention(
        q[:, :, :1], k, v, blocks[:, :1], ends[:, :1], allow_sparse_decode=True, **kw
    )
    assert qsp.COUNTS == {"engaged": 1, "fallback": 1} and len(seen) == 2


def make_exact(rows=64, keys=2048, seed=7):
    """Keys in a buffer of exactly ``keys`` rows (nothing readable past them); the last row's query end equals
    the key count and is a multiple of 4, and the rows cover every tail length 0..3."""
    rng = np.random.default_rng(seed)
    q = mx.array(rng.normal(size=(1, H, rows, D)).astype(np.float32)).astype(
        mx.bfloat16
    )
    k = mx.array(rng.normal(size=(1, KV, keys, D)).astype(np.float32)).astype(
        mx.bfloat16
    )
    v = mx.array(rng.normal(size=(1, KV, keys, D)).astype(np.float32)).astype(
        mx.bfloat16
    )
    ends = np.arange(keys - rows + 1, keys + 1, dtype=np.int32)
    assert ends[-1] % 4 == 0 and {int(e) % 4 for e in ends} == {0, 1, 2, 3}
    blocks = np.full((1, rows, TOPK), -1, dtype=np.int32)
    for i, end in enumerate(ends):
        pick = rng.permutation(int(end) // BLOCK)[:TOPK]
        blocks[0, i, : len(pick)] = pick
    return q, k, v, mx.array(blocks), mx.array(ends[None])


def test_chunk_end_on_a_block_boundary_reads_nothing_past_the_keys():
    import sys
    from pathlib import Path

    from mlx_vlm.models.qwen4_exp.qsa_kernel import qsa_sparse_attention

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify.kld import capture

    scale = D**-0.5
    q, k, v, blocks, ends = make_exact()
    ref = np.array(
        capture.reference_attend(q, k, v, blocks, ends, scale).astype(mx.float32)
    )
    new = np.array(qsp.attend(q, k, v, blocks, ends, scale).astype(mx.float32))
    up = np.array(
        qsa_sparse_attention(
            q, k, v, blocks, ends, scale=scale, block_size=BLOCK
        ).astype(mx.float32)
    )
    for name, got in (("tensor-unit kernel", new), ("upstream kernel", up)):
        assert np.isfinite(got).all(), name
        assert np.abs(got - ref).max() < 2e-3, name
    assert np.isfinite(ref).all()
