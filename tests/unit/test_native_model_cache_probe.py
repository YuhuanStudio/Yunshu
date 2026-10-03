"""A real Qwen singleton forward keeps prefix bytes, outputs, and capacity."""

import importlib.util
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_vlm.models.cache import ArraysCache, BatchKVCache  # noqa: E402
from mlx_vlm.models.qwen3_5.config import TextConfig  # noqa: E402
from mlx_vlm.models.qwen3_5.language import (  # noqa: E402
    Qwen3_5Model,
    _extract_row_cache,
)

spec = importlib.util.spec_from_file_location(
    "native_model_cache_probe",
    Path(__file__).resolve().parents[2] / "scripts/research/native_model_cache.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def cpu_arrays():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)


def _model():
    cfg = TextConfig(
        model_type="qwen3_5",
        hidden_size=16,
        intermediate_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        head_dim=8,
        full_attention_interval=2,
        rope_parameters=dict(
            type="default",
            mrope_section=[1, 1, 2],
            rope_theta=10000,
            partial_rotary_factor=1.0,
        ),
    )
    return Qwen3_5Model(cfg)


@pytest.mark.parametrize("suffix", [1, 8, 66])
def test_real_model_outputs_and_valid_cache_are_bit_equal(suffix):
    mx.random.seed(123)
    model = _model()
    baseline, candidate = (
        [ArraysCache(2), BatchKVCache([0])],
        [ArraysCache(2), BatchKVCache([0])],
    )
    optimized = probe.wrap(Qwen3_5Model.__call__)
    for tokens in (mx.array([[1, 2, 3, 4, 5]]), mx.arange(suffix).reshape(1, -1) % 32):
        left = Qwen3_5Model.__call__(model, tokens, cache=baseline)
        right = optimized(model, tokens, cache=candidate)
        mx.eval(left, right, [c.state for c in baseline + candidate])
        assert mx.array_equal(left, right).item()
        assert baseline[1]._idx == candidate[1]._idx
        assert mx.array_equal(
            baseline[1].keys, candidate[1].keys[..., : candidate[1]._idx, :]
        ).item()
        assert mx.array_equal(
            baseline[1].values, candidate[1].values[..., : candidate[1]._idx, :]
        ).item()
        for a, b in zip(baseline[0].state, candidate[0].state, strict=True):
            assert mx.array_equal(a, b).item()
        assert baseline[0].lengths is candidate[0].lengths is None
        assert baseline[0].left_padding is candidate[0].left_padding is None
        assert candidate[1].keys.shape[2] == 256


def test_views_reset_extract_and_merge_metadata_and_isolate_mutation():
    kv = BatchKVCache([0])
    kv.update_and_fetch(mx.ones((1, 1, 32, 4)), mx.ones((1, 1, 32, 4)) * 3)
    rec = ArraysCache(2)
    rec.cache = [mx.ones((1, 2, 3)), None]
    rec.lengths = mx.array([7])
    rec.left_padding = mx.array([0])
    rows = probe.extract_rows([rec, kv])
    expected = _extract_row_cache(rec, 0)
    assert rows[0].lengths.tolist() == expected.lengths.tolist()
    assert rows[0].left_padding is expected.left_padding is None
    kv.keys[..., :32, :] = 9
    rec.cache[0][:] = 8
    assert mx.all(rows[1].keys[..., :32, :] == 1).item()
    assert mx.all(rows[0].cache[0] == 1).item()
    merged = probe.merge_rows(rows)
    assert merged[0].lengths is None and merged[0].left_padding is None
    assert merged[1].keys.shape[2] == 256
    assert merged[1]._idx == 32 and merged[1].offset.tolist() == [32]
    merged[1].values[..., :32, :] = 7
    assert mx.all(rows[1].values[..., :32, :] == 3).item()
    assert mx.all(kv.values[..., :32, :] == 3).item()


@pytest.mark.parametrize(
    "unsupported", ["padding", "multi", "right_padding", "speculation", "custom"]
)
def test_unsupported_cache_contracts_fall_back(unsupported):
    kv, rec = BatchKVCache([0]), ArraysCache(2)
    if unsupported == "padding":
        kv.left_padding = mx.array([1])
    elif unsupported == "multi":
        kv = BatchKVCache([0, 0])
    elif unsupported == "right_padding":
        kv._right_padding = mx.array([1])
    elif unsupported == "speculation":
        rec._speculation = {}
    else:

        class Custom(ArraysCache):
            pass

        rec = Custom(2)
    assert probe.extract_rows([rec, kv]) is None


def test_failed_forward_keeps_callers_original_snapshot():
    from types import SimpleNamespace

    rec, kv = ArraysCache(2), BatchKVCache([0])
    rec.cache = [mx.ones((1, 2, 3)), None]
    kv.update_and_fetch(mx.ones((1, 1, 32, 4)), mx.ones((1, 1, 32, 4)) * 3)
    cache = [rec, kv]
    mx.eval([c.state for c in cache])

    def fail(self, inputs, **kwargs):
        rows = kwargs["cache"]
        rows[0].cache[0][:] = 8
        rows[1].keys[:] = 9
        rows[1].update_and_fetch(mx.ones((1, 1, 1, 4)), mx.ones((1, 1, 1, 4)))
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        probe.wrap(fail)(SimpleNamespace(fa_idx=1), mx.array([[1]]), cache=cache)
    assert cache[0] is rec and cache[1] is kv
    assert kv._idx == 32
    assert mx.all(kv.keys[..., :32, :] == 1).item()
    assert mx.all(rec.cache[0] == 1).item()


def test_row_keeps_legacy_dtype_promotion_without_a_reference_cycle():
    import weakref

    source = BatchKVCache([0])
    source.update_and_fetch(
        mx.ones((1, 1, 5, 4), mx.float16), mx.ones((1, 1, 5, 4), mx.float16)
    )
    source.left_padding = mx.array([0])
    row = probe.extract_rows([source])[0]
    expected = _extract_row_cache(source, 0)
    keys = mx.full((1, 1, 1, 4), 1.0001, dtype=mx.float32)
    values = keys * 3
    actual_state = row.update_and_fetch(keys, values)
    expected_state = expected.update_and_fetch(keys, values)
    mx.eval(actual_state, expected_state)
    assert row.keys.dtype == expected.keys.dtype == mx.float32
    assert mx.array_equal(actual_state[0], expected_state[0]).item()
    assert mx.array_equal(actual_state[1], expected_state[1]).item()
    reference = weakref.ref(row)
    del row
    assert reference() is None


def test_shipped_installer_is_idempotent_and_preserves_callable_signature(monkeypatch):
    import inspect

    from yunshu_engine.kernels import singleton_cache

    original = Qwen3_5Model.__call__
    monkeypatch.setattr(Qwen3_5Model, "__call__", original)
    assert singleton_cache.install()
    installed = Qwen3_5Model.__call__
    assert installed is not original
    assert singleton_cache.install()
    assert Qwen3_5Model.__call__ is installed
    assert list(inspect.signature(installed).parameters) == list(
        inspect.signature(original).parameters
    )
