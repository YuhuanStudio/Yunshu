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


def test_lattice_runs_outside_profiling_and_from_the_ring():
    # 33cb10b4 split the lattice tail off and left t0 unset outside profile mode: every
    # fast-tree round raised UnboundLocalError (no unit test ran the real lattice)
    from yunshu_engine import dflash_tree as dt

    d = _drafter(window=4096)
    d.lm_head = d.embed_tokens.as_linear
    cache = d.make_cache()
    hidden = mx.random.normal((1, 9, D_TARGET)).astype(mx.bfloat16)
    lat = dt.compute_lattice_gpu(d, 11, hidden, cache, 15)
    mx.eval(lat.cands, lat.unary, lat.anchor)
    assert lat.cands.shape == (15, d.candidate_selector.top_k)
    ring = dflash_ring.Ring(d, 4096 + 64, 4095)
    ring.append(d, hidden, mx.array([9], dtype=mx.int32))
    anchor = mx.array([11], dtype=mx.int32)
    lat2 = dt.lattice_from_hidden(
        d, dflash_ring.draft_hidden(d, ring, anchor, 15), anchor
    )
    mx.eval(lat2.cands)
    assert lat2.cands.shape == lat.cands.shape


def test_a_tight_ring_drafts_as_a_roomy_one_after_the_window_drops_rows():
    # the served ring has keep + 64 slots: the first append fills the window at once and
    # later appends overwrite only rows the window has already dropped. (The ring keeps the
    # last ``keep`` rows; the vendor's rotating cache keeps keep - 1 older rows plus the
    # round's own, one row apart at keep = 2047; drafts only propose.)
    window = 25
    d = _drafter(window=window)
    keep = window - 1
    rounds = _rounds(16, [3, 16, 5, 9, 16, 2, 7], first=keep)
    roomy = _ring_rounds(d, rounds, 15, cap=2 * keep + 64)
    tight = _ring_rounds(d, rounds, 15, cap=keep + 16)
    for r, t in zip(roomy, tight, strict=True):
        assert mx.array_equal(r, t)


def test_only_the_first_append_may_fill_past_the_window():
    keep = 12
    d = _drafter(window=keep + 1)
    ring = dflash_ring.Ring(d, keep + 16, keep)
    ring.append(
        d, mx.zeros((1, keep + 16, D_TARGET), dtype=mx.bfloat16), mx.array([keep])
    )
    with pytest.raises(ValueError):
        ring.append(d, mx.zeros((1, 17, D_TARGET), dtype=mx.bfloat16), mx.array([1]))


@pytest.mark.parametrize("bits", [None, 8])
def test_fused_context_projection_matches_each_layers_own(bits):
    # the served drafter is 8-bit: every layer's k_proj / v_proj stacked into one matmul
    d = _drafter(window=4096)
    if bits:
        nn.quantize(
            d,
            group_size=32,
            bits=bits,
            class_predicate=lambda _, m: (
                isinstance(m, nn.Linear) and m.weight.shape[-1] % 32 == 0
            ),
        )
    rounds = _rounds(16, [1, 5, 16, 3])
    fused = _ring_rounds(d, rounds, 15, cap=4096 + 64)
    assert d._yunshu_ring_kv is not False
    d._yunshu_ring_kv = False
    split = _ring_rounds(d, rounds, 15, cap=4096 + 64)
    for f, s in zip(fused, split, strict=True):
        err = mx.abs(f.astype(mx.float32) - s.astype(mx.float32)).max().item()
        scale = mx.abs(s.astype(mx.float32)).max().item()
        assert err <= 0.02 * scale, (err, scale)
