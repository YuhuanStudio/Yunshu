"""score_tokens scores with the draft's captured queries, not the key-magnitude fallback."""

import mlx.core as mx
import pytest


def _tiny_llama(kv_heads):
    from mlx_lm.models import llama

    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        rms_norm_eps=1e-5,
        vocab_size=64,
    )
    mx.random.seed(0)
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("kv_heads", [4, 2])
def test_queries_are_captured_and_scored(monkeypatch, kv_heads):
    from yunshu_engine import spec_prefill

    model = _tiny_llama(kv_heads)
    captured = []
    real_patch = spec_prefill._patch_attention_capture

    def spy(m, buf):
        captured.append(buf)
        return real_patch(m, buf)

    monkeypatch.setattr(spec_prefill, "_patch_attention_capture", spy)
    scores = spec_prefill.score_tokens(model, list(range(1, 41)), n_lookahead=3)
    assert scores.shape == (40,)
    assert sorted(captured[0]) == [0, 1]
    assert all(len(v) == 3 for v in captured[0].values())  # one per lookahead step
