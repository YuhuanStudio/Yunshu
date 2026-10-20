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
    monkeypatch.setattr(ra, "GROUPED_LAUNCH", False)
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
    # one grouped launch: groups of 3 (padded first), 8, 8 as batch rows of buffer row 0
    calls.clear()
    monkeypatch.setattr(ra, "GROUPED_LAUNCH", True)
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
    assert calls == [(8, 29, 40, (29, 32, 40))]


def test_grouped_launch_bits_equal_per_group_launches(monkeypatch):
    import pytest

    if not mx.metal.is_available() or not ra.tile_ready():
        pytest.skip("needs M5-class tensor ops")
    mx.random.seed(7)
    hkv, h, d, cap = 4, 16, 256, 1536
    keys = mx.random.normal((1, hkv, cap, d)).astype(mx.bfloat16)
    values = mx.random.normal((1, hkv, cap, d)).astype(mx.bfloat16)
    for t, n in ((9, 1000), (17, 1300), (19, 777), (32, 1500)):
        q = mx.random.normal((1, h, t, d)).astype(mx.bfloat16)
        outs = []
        for grouped in (False, True):
            monkeypatch.setattr(ra, "GROUPED_LAUNCH", grouped)
            outs.append(
                ra.ragged_decode_attention(
                    q,
                    keys,
                    values,
                    mx.array([n]),
                    d**-0.5,
                    max_length=n,
                    impl="tile",
                    row_lengths=(n,),
                )
            )
        assert mx.array_equal(outs[0], outs[1]), (t, n)
