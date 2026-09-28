"""DFlash drafter context window (python/yunshu_engine/dflash_context.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm.speculative.drafters.dflash2")

from yunshu_engine import dflash_context  # noqa: E402

VOCAB = 512
HIDDEN = 64


def _drafter(layer_types=("sliding_attention", "sliding_attention"), window=9):
    import mlx.nn as nn
    from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel, ModelConfig

    cfg = ModelConfig.from_dict(
        {
            "model_type": "qwen3",
            "hidden_size": HIDDEN,
            "intermediate_size": 128,
            "num_hidden_layers": len(layer_types),
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": VOCAB,
            "layer_types": list(layer_types),
            "sliding_window": window,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "num_target_layers": 4,
            "hidden_act": "silu",
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "dflash_config": {
                "block_size": 4,
                "mask_token_id": VOCAB - 1,
                "target_layer_ids": [0, 2],
                "conv_kernel_size": 2,
                "conv_group_size": 16,
                "selector_rank": 8,
                "selector_top_k": 4,
            },
        }
    )
    d = DFlash2DraftModel(cfg)
    mx.random.seed(0)
    d.update(
        nn.utils.tree_map(
            lambda p: mx.random.normal(p.shape, dtype=mx.float32) * 0.05,
            d.parameters(),
        )
    )
    d.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
    d.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    return d


def test_context_window_only_for_all_sliding_drafters():
    assert dflash_context.context_window(_drafter()) == 8
    assert (
        dflash_context.context_window(
            _drafter(layer_types=("sliding_attention", "full_attention"))
        )
        is None
    )
    assert dflash_context.context_window(SimpleNamespace()) is None


def test_trimmed_hidden_matches_upstream_window():
    """The patched ``_hidden`` sees only the window yet leaves the drafter
    caches and outputs as upstream's in-layer trim does."""
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

    d = _drafter()
    mx.random.seed(1)
    target = mx.random.normal((1, 30, 2 * HIDDEN)) * 0.5
    inputs = mx.array([[5, VOCAB - 1, VOCAB - 1, VOCAB - 1]])

    dflash_context.install()
    patched = DFlashDraftModel._hidden
    # Upstream: the original, wrapped function (closure cell of the patch).
    upstream = patched.__closure__[
        patched.__code__.co_freevars.index("orig_hidden")
    ].cell_contents

    ref_cache = d.make_cache()
    ref = upstream(d, inputs, target, ref_cache)
    got_cache = d.make_cache()
    got = patched(d, inputs, target, got_cache)
    mx.eval(ref, got)
    assert mx.allclose(ref, got, atol=1e-5).item()
    for a, b in zip(ref_cache, got_cache, strict=True):
        assert a.offset == b.offset == 30
        ka, va = a.state
        kb, vb = b.state
        assert mx.allclose(ka, kb, atol=1e-5).item()
        assert mx.allclose(va, vb, atol=1e-5).item()
    # A later round with a short context is untouched.
    step = mx.random.normal((1, 3, 2 * HIDDEN)) * 0.5
    ref2 = upstream(d, inputs, step, ref_cache)
    got2 = patched(d, inputs, step, got_cache)
    assert mx.allclose(ref2, got2, atol=1e-5).item()


def test_prefill_retains_only_the_window():
    from mlx_vlm.speculative.utils import SpeculativePrefill

    dflash_context.install()
    d = _drafter(window=9)  # keep 8 positions
    pre = SpeculativePrefill("dflash", d)
    assert pre._yunshu_keep == 8

    def chunk(n):
        return SimpleNamespace(hidden_states=[mx.zeros((1, n, 4)), mx.ones((1, n, 4))])

    for n in (5, 5, 5, 5):
        pre.append(chunk(n))
    # Newest chunks covering >= 8 positions stay: the last two (10 positions).
    assert [c[0].shape[1] for c in pre.chunks] == [5, 5]
    out = pre.finish(chunk(3))
    assert [h.shape[1] for h in out.hidden_states] == [13, 13]

    # Other drafter kinds keep everything.
    mtp = SpeculativePrefill("mtp", d)
    assert mtp._yunshu_keep is None
