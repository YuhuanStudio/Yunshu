"""Prefill-only fused conv1d+silu is bit-identical to MLX's concat + conv1d + silu and engages only
inside split-invariant prefill. Needs Metal (run through gpuq)."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_vlm.models.qwen3_5.language")

from mlx_vlm.models.cache import ArraysCache  # noqa: E402
from mlx_vlm.models.qwen3_5.config import TextConfig  # noqa: E402
from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet  # noqa: E402

from yunshu_engine.kernels import gdn_prefill  # noqa: E402

pytestmark = pytest.mark.skipif(
    mx.default_device() != mx.gpu, reason="Metal kernel needs the GPU"
)

C = 1024


def _reference(state, x, conv):
    return nn.silu(conv(mx.concatenate([state, x], axis=1)))


def _conv(seed=1):
    mx.random.seed(seed)
    conv = nn.Conv1d(C, C, kernel_size=4, groups=C, padding=0, bias=False)
    conv.weight = (mx.random.normal(conv.weight.shape) * 0.3).astype(mx.bfloat16)
    return conv


@pytest.mark.parametrize("rows", [3, 4, 5, 64, 1984, 2048])
def test_kernel_bit_equal_to_mlx(rows):
    conv = _conv(rows)
    state = mx.random.normal((1, 3, C)).astype(mx.bfloat16)
    x = (mx.random.normal((1, rows, C)) * 2).astype(mx.bfloat16)
    got = gdn_prefill.fused_conv_silu(state, x, conv.weight)
    ref = _reference(state, x, conv)
    assert got.dtype == mx.bfloat16 and got.shape == ref.shape
    assert mx.array_equal(got.view(mx.uint16), ref.view(mx.uint16)).item()


def test_split_in_two_cuts_equals_one_pass():
    conv = _conv(7)
    state = mx.random.normal((1, 3, C)).astype(mx.bfloat16)
    x = mx.random.normal((1, 200, C)).astype(mx.bfloat16)
    whole = gdn_prefill.fused_conv_silu(state, x, conv.weight)
    first = gdn_prefill.fused_conv_silu(state, x[:, :77], conv.weight)
    carried = mx.concatenate([state, x[:, :77]], axis=1)[:, -3:]
    second = gdn_prefill.fused_conv_silu(carried, x[:, 77:], conv.weight)
    cut = mx.concatenate([first, second], axis=1)
    assert mx.array_equal(whole.view(mx.uint16), cut.view(mx.uint16)).item()


def _layer():
    cfg = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=16,
        num_key_value_heads=1,
        max_position_embeddings=128,
    )
    mx.random.seed(3)
    layer = Qwen3_5GatedDeltaNet(cfg)
    from mlx.utils import tree_map

    layer.update(
        tree_map(
            lambda v: v.astype(mx.bfloat16) if v.ndim > 1 else v, layer.parameters()
        )
    )
    return layer


def _cache():
    c = ArraysCache(2)
    c.cache = [
        (mx.random.normal((1, 3, 1024)) * 0.5).astype(mx.bfloat16),
        (mx.random.normal((1, 4, 128, 128)) * 0.1).astype(mx.float32),
    ]
    return c


def test_layer_engages_only_in_split_prefill_and_matches_upstream(monkeypatch):
    class Layer(Qwen3_5GatedDeltaNet):
        pass

    monkeypatch.setitem(gdn_prefill._STATE, "enabled", True)
    monkeypatch.setitem(gdn_prefill._STATE, "split", True)
    gdn_prefill.install_fused_conv(Layer)
    layer = _layer()
    layer.__class__ = Layer
    layer.eval()
    x = (mx.random.normal((1, 130, 64)) * 0.5).astype(mx.bfloat16)
    ref_cache = _cache()
    ref = Qwen3_5GatedDeltaNet.__call__(layer, x, cache=ref_cache)
    mx.eval(ref, *ref_cache.state)
    before = gdn_prefill.fused_conv_calls()
    # outside a prefill scope: upstream path, counter untouched
    out0 = layer(x, cache=_cache())
    mx.eval(out0)
    assert gdn_prefill.fused_conv_calls() == before
    base = _cache()
    ref_cache = ArraysCache(2)
    ref_cache.cache = list(base.cache)
    ref = Qwen3_5GatedDeltaNet.__call__(layer, x, cache=ref_cache)
    cache = ArraysCache(2)
    cache.cache = list(base.cache)
    with gdn_prefill.prefill_scope():
        out = layer(x, cache=cache)
    mx.eval(ref, out, *ref_cache.cache, *cache.cache)
    assert gdn_prefill.fused_conv_calls() == before + 1
    assert mx.array_equal(out.view(mx.uint16), ref.view(mx.uint16)).item()
    assert mx.array_equal(cache.cache[0], ref_cache.cache[0]).item()
    assert mx.array_equal(cache.cache[1], ref_cache.cache[1]).item()
    # S < 3 never routes
    with gdn_prefill.prefill_scope():
        layer(x[:, :2], cache=_cache())
    assert gdn_prefill.fused_conv_calls() == before + 1
