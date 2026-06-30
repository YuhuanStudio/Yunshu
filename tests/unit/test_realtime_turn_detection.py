"""Server-wide default server-VAD turn detection is env-tunable (conversation feel),
with a client's session.update still overriding per-session."""

from yunshu_gateway.routers.realtime import _default_turn_detection

_ENVS = (
    "YUNSHU_REALTIME_SILENCE_MS",
    "YUNSHU_REALTIME_BARGE_IN_MS",
    "YUNSHU_REALTIME_VAD_THRESHOLD",
    "YUNSHU_REALTIME_PREFIX_PADDING_MS",
)


def test_defaults(monkeypatch):
    for v in _ENVS:
        monkeypatch.delenv(v, raising=False)
    assert _default_turn_detection() == {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "barge_in_min_ms": 120,
    }


def test_env_override(monkeypatch):
    monkeypatch.setenv("YUNSHU_REALTIME_SILENCE_MS", "350")  # snappier
    monkeypatch.setenv("YUNSHU_REALTIME_BARGE_IN_MS", "80")  # easier to interrupt
    monkeypatch.setenv("YUNSHU_REALTIME_VAD_THRESHOLD", "0.6")
    monkeypatch.setenv("YUNSHU_REALTIME_PREFIX_PADDING_MS", "200")
    td = _default_turn_detection()
    assert td["silence_duration_ms"] == 350
    assert td["barge_in_min_ms"] == 80
    assert td["threshold"] == 0.6
    assert td["prefix_padding_ms"] == 200


def test_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("YUNSHU_REALTIME_SILENCE_MS", "not-a-number")
    monkeypatch.setenv("YUNSHU_REALTIME_VAD_THRESHOLD", "")
    td = _default_turn_detection()
    assert td["silence_duration_ms"] == 500
    assert td["threshold"] == 0.5
