"""dflash_ring: the fixed-width ring drafter matches the vendor drafter's proposal hidden states."""

import mlx.core as mx
import mlx.nn as nn
import pytest

pytest.importorskip("mlx_vlm")

from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config  # noqa: E402
from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel  # noqa: E402

from yunshu_engine import dflash_ring  # noqa: E402

H, D_TARGET = 128, 2 * 128


def _drafter(window):
    mx.random.seed(5)
    cfg = DFlash2Config(
        hidden_size=H,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=1000,
        mask_token_id=7,
        max_position_embeddings=8192,
        block_size=8,
        num_target_layers=8,
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=8,
        selector_top_k=4,
        layer_types=["sliding_attention"] * 2,
        sliding_window=window,
        target_layer_ids=[1, 2],
    )
    d = DFlash2DraftModel(cfg)
    d.embed_tokens = nn.Embedding(1000, H)
    mx.eval(d.parameters())
    d.set_dtype(mx.bfloat16)
    return d


def _vendor_rounds(d, rounds, positions):
    cache = d.make_cache()
    outs = []
    for hidden, count, bonus in rounds:
        masks = [7] * positions
        inputs = mx.array([[bonus] + masks], dtype=mx.int32)
        outs.append(d._hidden(inputs, hidden[:, :count], cache)[:, 1:])
    return outs


def _ring_rounds(d, rounds, positions, cap):
    keep = int(d.config.sliding_window) - 1
    ring = dflash_ring.Ring(d, cap, keep)
    outs = []
    for hidden, count, bonus in rounds:
        ring.append(d, hidden, mx.array([count], dtype=mx.int32))
        outs.append(
            dflash_ring.draft_hidden(
                d, ring, mx.array([bonus], dtype=mx.int32), positions
            )
        )
    return outs


def _rounds(width, counts, first=9):
    mx.random.seed(9)
    rounds = [(mx.random.normal((1, first, D_TARGET)).astype(mx.bfloat16), first, 11)]
    for i, c in enumerate(counts):
        h = mx.random.normal((1, width, D_TARGET)).astype(mx.bfloat16)
        rounds.append((h, c, 100 + i))
    return rounds


def test_ring_matches_vendor_while_the_window_holds_everything():
    d = _drafter(window=4096)
    rounds = _rounds(16, [1, 5, 16, 3, 2, 9])
    want = _vendor_rounds(d, rounds, 15)
    # the first round appends 9 rows: the ring's width per append is the array's width
    got = _ring_rounds(d, rounds, 15, cap=4096 + 64)
    for w, g in zip(want, got, strict=True):
        assert w.shape == g.shape
        err = mx.abs(w.astype(mx.float32) - g.astype(mx.float32)).max().item()
        scale = mx.abs(w.astype(mx.float32)).max().item()
        assert err <= 0.02 * scale, (err, scale)


def test_ring_keeps_only_the_last_window_rows():
    keep = 12
    d = _drafter(window=keep + 1)
    ring = dflash_ring.Ring(d, 64, keep)
    for c in (9, 4, 16, 2):
        ring.append(
            d,
            mx.zeros((1, 16, D_TARGET), dtype=mx.bfloat16),
            mx.array([c], dtype=mx.int32),
        )
    total = 9 + 4 + 16 + 2
    assert ring.pos.item() == total
    live = (ring.mask(3)[0, 0, 0, : ring.cap] == 0).sum().item()
    assert live == keep
