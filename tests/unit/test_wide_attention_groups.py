"""Windows wider than 8 tokens run the tile kernel in groups of 8 with per-group causal limits."""

import mlx.core as mx

from yunshu_engine.kernels import ragged_attention as ra


def test_wide_window_groups(monkeypatch):
    calls = []

    def fake(
        q, keys, values, lengths, scale, max_length, k, v, impl, row_lengths, slots
    ):
        calls.append((int(q.shape[2]), int(lengths[0].item()), max_length, row_lengths))
        return mx.zeros(q.shape, dtype=mx.bfloat16)

    monkeypatch.setattr(ra, "_attend", fake)
    q = mx.zeros((1, 4, 19, 8), dtype=mx.bfloat16)
    keys = mx.zeros((1, 2, 64, 8), dtype=mx.bfloat16)
    out = ra.ragged_decode_attention(
        q,
        keys,
        keys,
        mx.array([40]),
        1.0,
        max_length=40,
        impl="tile",
        row_lengths=(40,),
    )
    assert out.shape == (1, 4, 19, 8)
    # groups of 8, 8, 3 tokens; a group's last token sees 11 / 3 / 0 fewer keys
    assert calls == [(8, 29, 29, (29,)), (8, 37, 37, (37,)), (3, 40, 40, (40,))]
