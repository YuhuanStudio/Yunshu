"""`GET /v1/yunshu/status` `last` carries the finished request's speculative stats,
so the console can show acceptance without parsing a stream."""

from unittest import mock

from yunshu_gateway import x_yunshu


def test_record_done_keeps_speculative_block():
    x_yunshu.registry.clear()
    info = x_yunshu.RequestInfo(request_id="r1", method="POST", path="/v1/chat/completions")
    spec = {"mode": "dflash", "drafted": 90, "accepted": 54, "acceptance_rate": 0.6, "rounds": 15}
    with mock.patch("yunshu_gateway.serve_log.record", lambda *a, **k: None):
        x_yunshu.record_done(
            info,
            {"prompt_tokens": 10, "completion_tokens": 70, "ttft_ms": 12.0, "speculative": spec},
        )
    last = x_yunshu.registry.last()
    assert last is not None and last["speculative"] == spec
    x_yunshu.registry.clear()


def test_record_done_without_spec_is_none():
    x_yunshu.registry.clear()
    info = x_yunshu.RequestInfo(request_id="r2", method="POST", path="/v1/chat/completions")
    with mock.patch("yunshu_gateway.serve_log.record", lambda *a, **k: None):
        x_yunshu.record_done(info, {"prompt_tokens": 1, "completion_tokens": 1})
    assert x_yunshu.registry.last()["speculative"] is None
    x_yunshu.registry.clear()
