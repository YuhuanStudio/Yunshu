"""Console-facing status fields: decode rate per request, single-model size_gb, last.model."""

from types import SimpleNamespace
from unittest import mock

from yunshu_gateway import x_yunshu
from yunshu_gateway.routers import yunshu as yr


def test_decode_row_has_tokens_per_second():
    info = x_yunshu.RequestInfo(
        request_id="d1", method="POST", path="/v1/chat/completions"
    )
    info.gen = SimpleNamespace(
        model="m",
        stats=SimpleNamespace(
            phase="decode",
            prompt_tokens=5,
            prefill_total=5,
            prefill_done=5,
            cached_tokens=0,
            cache_tier=None,
            generated=11,
            t_first=100.0,
            t_last=102.0,
        ),
    )
    assert x_yunshu.progress_payload(info)["tokens_per_second"] == 5.0


def test_decode_row_without_two_tokens_has_no_rate():
    info = x_yunshu.RequestInfo(
        request_id="d2", method="POST", path="/v1/chat/completions"
    )
    info.gen = SimpleNamespace(
        model="m",
        stats=SimpleNamespace(
            phase="decode",
            prompt_tokens=5,
            prefill_total=5,
            prefill_done=5,
            cached_tokens=0,
            cache_tier=None,
            generated=1,
            t_first=100.0,
            t_last=100.0,
        ),
    )
    assert "tokens_per_second" not in x_yunshu.progress_payload(info)


def test_single_model_size_gb_cached(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"0" * 1_500_000_000)
    yr._WEIGHT_GB.clear()
    eng = SimpleNamespace(_model_path=str(tmp_path), model_name="m", is_loaded=True)
    with (
        mock.patch.object(yr, "get_model_manager", lambda: None),
        mock.patch.object(yr, "get_engine", lambda: eng),
    ):
        assert yr._models()[0]["size_gb"] == 1.5
        (tmp_path / "b.safetensors").write_bytes(b"0" * 1_000_000_000)
        assert yr._models()[0]["size_gb"] == 1.5  # cached, not re-stat'ed


def test_weights_gb_sums_safetensors(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"0" * 1_500_000_000)
    (tmp_path / "b.safetensors").write_bytes(b"0" * 1_000_000_000)
    yr._WEIGHT_GB.clear()
    assert yr._weights_gb(str(tmp_path)) == 2.5


def test_last_has_model():
    x_yunshu.registry.clear()
    info = x_yunshu.RequestInfo(
        request_id="m1", method="POST", path="/v1/chat/completions"
    )
    info.gen = SimpleNamespace(model="qwen")
    with mock.patch("yunshu_gateway.serve_log.record", lambda *a, **k: None):
        x_yunshu.record_done(info, {"prompt_tokens": 1, "completion_tokens": 1})
    assert x_yunshu.registry.last()["model"] == "qwen"
    x_yunshu.registry.clear()
