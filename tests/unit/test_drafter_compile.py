"""The compiled DFlash2 decoder-layer halves equal the eager layer bit for bit."""

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm")

from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config  # noqa: E402
from mlx_vlm.speculative.drafters.dflash2.dflash2 import (
    DFlash2DecoderLayer,  # noqa: E402
)

from yunshu_engine import dflash_tree  # noqa: E402


def _config():
    return DFlash2Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=1000,
        mask_token_id=7,
        max_position_embeddings=4096,
        block_size=8,
        num_target_layers=8,
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=8,
        selector_top_k=4,
        layer_types=["full_attention"],
        target_layer_ids=[1, 2],
    )


def _pair():
    mx.random.seed(11)
    cfg = _config()
    layer = DFlash2DecoderLayer(cfg, 0)
    mx.eval(layer.parameters())
    layer.set_dtype(mx.bfloat16)
    compiled = dflash_tree.compiled_layer_class()
    twin = DFlash2DecoderLayer(cfg, 0)
    twin.update(layer.parameters())
    twin.__class__ = compiled
    return layer, twin, cfg


def test_compiled_layer_is_bit_identical_to_eager():
    layer, twin, cfg = _pair()
    from mlx_lm.models.rope_utils import initialize_rope

    rope = initialize_rope(cfg.head_dim, 10000.0, True)
    for rows in (1, 16):
        x = mx.random.normal((1, rows, 128)).astype(mx.bfloat16)
        ctx = mx.random.normal((1, 12, 128)).astype(mx.bfloat16)
        from mlx_lm.models.cache import KVCache

        want = layer(x, ctx, rope, KVCache())
        got = twin(x, ctx, rope, KVCache())
        got2 = twin(x, ctx, rope, KVCache())  # second call reuses the compiled graphs
        assert mx.array_equal(want, got).item() and mx.array_equal(want, got2).item()


def test_compile_layers_switches_only_plain_dflash2_layers():
    layer, _, cfg = _pair()

    class View:
        layers = [layer]

    assert dflash_tree.compile_layers(View) == 1
    assert type(View.layers[0]).__name__ == "CompiledDecoderLayer"
    assert dflash_tree.compile_layers(View) == 0  # already switched
