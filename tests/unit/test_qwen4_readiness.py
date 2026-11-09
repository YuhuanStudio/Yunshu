"""A Qwen4 release that differs from the Flash-Next stand-in must not degrade silently."""

import json
import logging

import mlx.core as mx
import pytest

from yunshu_engine import fallbacks
from yunshu_engine.mlxvlm_mtp import is_mtp_capable, mtp_head_problem
from yunshu_engine.qwen4_norms import infer_direct_gamma


@pytest.fixture(autouse=True)
def _clean():
    fallbacks.reset()
    yield
    fallbacks.reset()


def test_missing_mtp_head_is_named_for_a_qwen4_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen4_exp", "text_config": {"mtp_num_hidden_layers": 1}}
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"language_model.layers.0.w": "a.safetensors"}})
    )
    assert not is_mtp_capable(str(tmp_path))
    problem = mtp_head_problem(str(tmp_path))
    assert problem is not None
    for prefix in ("mtp.", "language_model.mtp.", "model.mtp."):
        assert prefix in problem
    assert "no indexed tensor" in problem

    (tmp_path / "a.safetensors").touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc.weight": "a.safetensors"}})
    )
    assert mtp_head_problem(str(tmp_path)) is None


def test_fallback_is_counted_and_logged_once_with_the_shape(caplog):
    with caplog.at_level(logging.WARNING, logger="yunshu_engine.fallbacks"):
        assert fallbacks.record("qsa_sparse_prefill", "queries=(1, 32, 9, 128)")
        assert not fallbacks.record("qsa_sparse_prefill", "queries=(1, 32, 10, 128)")
    lines = [r.getMessage() for r in caplog.records]
    assert lines == [
        "Kernel fallback (qsa_sparse_prefill): queries=(1, 32, 9, 128); "
        "using the upstream path"
    ]
    got = fallbacks.counts()["qsa_sparse_prefill"]
    assert got == {"count": 2, "first": "queries=(1, 32, 9, 128)"}


def test_qsa_dispatch_fallback_counts_the_unsupported_shape(monkeypatch):
    from types import ModuleType

    from yunshu_engine import qwen4_family
    from yunshu_engine.kernels import qsa_sparse_prefill as qsp

    lang = ModuleType("fake_lang")
    lang.dispatch_qsa_attention = lambda *a, **k: "upstream"  # type: ignore[attr-defined]
    monkeypatch.setattr(qwen4_family, "qwen4_first_module", lambda sub="language": lang)
    assert qsp.install()
    q = mx.zeros((1, 16, 9, 128), dtype=mx.bfloat16)  # not the 24-head / 256-dim shape
    k = mx.zeros((1, 2, 64, 128), dtype=mx.bfloat16)
    out = lang.dispatch_qsa_attention(
        q,
        k,
        k,
        mx.zeros((1, 9, 512), dtype=mx.int32),
        None,
        causal=True,
        block_size=4,
        scale=1.0,
    )
    assert out == "upstream"
    info = fallbacks.counts()["qsa_sparse_prefill"]
    assert info["count"] == 1
    assert "(1, 16, 9, 128)" in str(info["first"])


def test_ambiguous_or_missing_norm_anchors_fail_the_load():
    with mx.stream(mx.cpu):
        with pytest.raises(ValueError, match="cannot be determined"):
            infer_direct_gamma({"language_model.model.embed.weight": mx.zeros((4,))})
        mixed = {
            f"language_model.model.layers.{i}.attn_hyper_connection.hc_norm.weight": mx.full(
                (4,), 0.0 if i % 2 else 1.0
            )
            for i in range(8)
        }
        with pytest.raises(ValueError, match="ambiguous"):
            infer_direct_gamma(mixed)


def test_model_card_speculative_follows_the_family(tmp_path):
    from yunshu_engine import model_card

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    got = model_card._speculative("vlm", tmp_path, {"model_type": "qwen4_exp"})
    assert got.get("available") is not None
    import inspect

    assert "is_qwen4_family" in inspect.getsource(model_card._speculative)
