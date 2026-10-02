"""The serve log: numbers only, opt-in, bounded, never a prompt, an output or a token id."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yunshu_engine import settings
from yunshu_gateway import serve_log, x_yunshu

MARKER = "ZXQ-SECRET-MARKER-7731 prompt text"
MARKER_TOKEN = "ZXQSECRETMARKER7731"


def _stats(**kw):
    base = {
        "request_id": "req_1",
        "prompt_tokens": 3000,
        "completion_tokens": 128,
        "cached_tokens": 2048,
        "queue_wait_ms": 1.5,
        "ttft_ms": 240.2,
        "prefill_ms": 200.0,
        "decode_ms": 1500.0,
        "decode_tps": 85.1,
        "speculative": {
            "mode": "mtp",
            "drafted": 100,
            "accepted": 72,
            "acceptance_rate": 0.72,
        },
        "cache": {"tier": "ram", "cached_tokens": 2048, "reload_ms": None},
        # content-bearing fields a careless caller might leave in the stats dict
        "text": MARKER,
        "output": MARKER,
        "messages": [{"role": "user", "content": MARKER}],
        "token_ids": [1, 2, 3, 4],
        "timings": {"prompt_n": 952},
    }
    base.update(kw)
    return base


def _ctx(**kw):
    base = {
        "t_start": 1000.0,
        "t_end": 1002.0,
        "route": "/v1/chat/completions",
        "dialect": "chat",
        "stream": True,
        "model": "Qwen3.8-27B",
        "finish_reason": "stop",
        "concurrency_start": 1,
        "concurrency_end": 2,
        "arm": "default",
        "build": "0.1.1+abc1234",
        "prompt": MARKER,
        "headers": {"authorization": MARKER},
    }
    base.update(kw)
    return base


def test_event_has_the_required_fields_and_no_content():
    ev = serve_log.event_from_stats(_stats(), _ctx())
    for key in [
        "t_start",
        "t_end",
        "route",
        "dialect",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "ttft_ms",
        "decode_tps",
        "spec_mode",
        "spec_acceptance",
        "cache_tier",
        "finish_reason",
        "concurrency_start",
        "concurrency_end",
        "arm",
        "build",
    ]:
        assert ev[key] is not None, key
    assert ev["spec_acceptance"] == 0.72 and ev["cache_tier"] == "ram"
    assert ev["ctx_bucket"] == "<=4k"
    blob = json.dumps(ev)
    assert MARKER_TOKEN not in blob and "token_ids" not in blob


@pytest.mark.parametrize("bad", [MARKER, "a b", 'q"uote', "x" * 200, "", None, 5])
def test_labels_reject_free_text(bad):
    ev = serve_log.event_from_stats(
        _stats(), _ctx(model=bad, arm=bad, route=bad, finish_reason=bad)
    )
    assert ev["model"] is None and ev["arm"] is None and ev["finish_reason"] is None


def test_missing_values_stay_missing():
    ev = serve_log.event_from_stats({}, {})
    assert ev["ttft_ms"] is None and ev["decode_tps"] is None
    assert ev["prompt_tokens"] is None and ev["ctx_bucket"] is None


def test_non_finite_numbers_dropped():
    ev = serve_log.event_from_stats(
        _stats(decode_tps=float("nan"), ttft_ms=float("inf")), _ctx()
    )
    assert ev["decode_tps"] is None and ev["ttft_ms"] is None
    json.dumps(ev, allow_nan=False)


def test_rotation_caps_total_size(tmp_path):
    log = serve_log.ServeLog(tmp_path, max_bytes=2048, keep=2)
    ev = serve_log.event_from_stats(_stats(), _ctx())
    for _ in range(200):
        assert log.append(ev)
    sizes = [p.stat().st_size for p in tmp_path.iterdir()]
    assert len(sizes) == 3  # live + 2 rotated
    assert all(s <= 2048 for s in sizes)
    assert sum(sizes) <= log.cap_bytes
    assert all(isinstance(r, dict) for r in log.read())


def test_unwritable_directory_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    log = serve_log.ServeLog(blocker / "sub", 4096, 1)
    assert log.append({"a": 1}) is False


def test_torn_line_skipped(tmp_path):
    log = serve_log.ServeLog(tmp_path, 4096, 1)
    log.append({"a": 1})
    with open(log.path, "ab") as f:
        f.write(b'{"torn":')
    assert log.read() == [{"a": 1}]


@pytest.fixture
def serve_on(tmp_path):
    settings.set_override("YUNSHU_SERVE_LOG", True)
    settings.set_override("YUNSHU_SERVE_LOG_DIR", str(tmp_path))
    yield tmp_path
    settings.clear_overrides()


def test_off_by_default(tmp_path):
    settings.clear_overrides()
    settings.set_override("YUNSHU_SERVE_LOG_DIR", str(tmp_path))
    try:
        info = x_yunshu.RequestInfo("req_1", "POST", "/v1/chat/completions")
        x_yunshu.record_done(info, _stats())
    finally:
        settings.clear_overrides()
    assert list(tmp_path.iterdir()) == []


def test_record_done_writes_and_marker_never_appears(serve_on):
    """Requests whose prompt and output carry a unique marker: it must not reach any file."""
    serve_log.set_arm("armB")
    try:
        for i in range(5):
            info = x_yunshu.RequestInfo(f"req_{i}", "POST", "/v1/messages")
            info.stream = True
            info.gen = SimpleNamespace(
                model="Qwen3.8-27B", prompt=MARKER, output=MARKER
            )
            info.usage = {"prompt_tokens": 10, "completion_tokens": 5, "marker": MARKER}
            info.concurrency_start = 2
            stats = _stats(request_id=f"req_{i}")
            x_yunshu.record_done(info, stats)
    finally:
        serve_log.set_arm(None)
    files = list(serve_on.iterdir())
    assert files
    for f in files:
        raw = f.read_bytes().decode()
        assert MARKER_TOKEN not in raw and "ZXQ" not in raw
        assert "token_ids" not in raw and "content" not in raw
    rows = serve_log.get_log().read()
    assert len(rows) == 5
    assert rows[0]["dialect"] == "anthropic" and rows[0]["arm"] == "armB"
    assert rows[0]["concurrency_start"] == 2 and rows[0]["route"] == "/v1/messages"
    assert rows[0]["build"]


def test_settings_registered_stable():
    assert settings.REGISTRY["YUNSHU_SERVE_LOG"].stability == "stable"
    assert settings.get("YUNSHU_SERVE_LOG") is False
