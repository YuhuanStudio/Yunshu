"""The fullcmp test stub: OpenAI-compatible responses, usage, stop/max_tokens, and engine log lines
that the audit's engaged-mode probe actually recognises. CPU only, localhost only."""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_stub_server as stub  # noqa: E402
import fullcmp_audit_manifest as fam  # noqa: E402

ENGINES = [
    "mlx-serve-26.11.1",
    "kapitsa-0.2.26",
    "omlx-0.7.3",
    "omlx-0.7.3rc1",
    "yunshu-main",
    "tf-1.2.1",
]


def _post(port, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode()


@pytest.fixture
def server(tmp_path):
    srv, port, _ = stub.start("yunshu-main", "stub-model", tmp_path / "e.log")
    yield port, tmp_path / "e.log"
    srv.shutdown()


def test_non_stream_usage_and_max_tokens(server):
    port, _ = server
    out = json.loads(
        _post(
            port,
            {
                "messages": [{"role": "user", "content": "one two three"}],
                "max_tokens": 5,
            },
        )
    )
    choice = out["choices"][0]
    assert choice["finish_reason"] == "length"
    assert len(choice["message"]["content"].split()) == 5
    assert out["usage"]["prompt_tokens"] == 3
    assert out["usage"]["completion_tokens"] == 5
    assert out["usage"]["total_tokens"] == 8


def test_stop_string_truncates_and_reports_stop(server):
    port, _ = server
    base = json.loads(
        _post(port, {"messages": [{"role": "user", "content": "x"}], "max_tokens": 8})
    )
    text = base["choices"][0]["message"]["content"]
    stop = text.split(" ")[3]  # the fourth token
    out = json.loads(
        _post(
            port,
            {
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 8,
                "stop": [stop],
            },
        )
    )
    assert out["choices"][0]["finish_reason"] == "stop"
    assert stop not in out["choices"][0]["message"]["content"]


def test_deterministic_for_the_same_prompt(server):
    port, _ = server
    body = {"messages": [{"role": "user", "content": "same"}], "max_tokens": 6}
    a = json.loads(_post(port, body))["choices"][0]["message"]["content"]
    b = json.loads(_post(port, body))["choices"][0]["message"]["content"]
    assert a == b


def test_stream_emits_tokens_then_usage_and_done(server):
    port, _ = server
    raw = _post(
        port,
        {
            "messages": [{"role": "user", "content": "s"}],
            "max_tokens": 4,
            "stream": True,
        },
    )
    events = [ln[len("data: ") :] for ln in raw.splitlines() if ln.startswith("data: ")]
    assert events[-1] == "[DONE]"
    last = json.loads(events[-2])
    assert last["usage"]["completion_tokens"] == 4
    assert last["choices"][0]["finish_reason"] == "length"
    pieces = [
        json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-2]
    ]
    assert len(pieces) >= 4


def test_models_endpoint_reports_the_served_model(server):
    port, _ = server
    data = json.loads(
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models").read()
    )
    assert data["data"][0]["id"] == "stub-model"


@pytest.mark.parametrize("engine", ENGINES)
def test_load_and_round_lines_match_the_engaged_mode_probe(tmp_path, engine):
    """Each engine's stub log must contain a line the audit's probe accepts as engagement: the
    high-confidence load line where the probe has one, and a round line matching a medium pattern."""
    srv, port, _ = stub.start(engine, "stub-model", tmp_path / "e.log")
    try:
        _post(port, {"messages": [{"role": "user", "content": "p q"}], "max_tokens": 3})
    finally:
        srv.shutdown()
    text = (tmp_path / "e.log").read_text()
    probe = fam.EXPECTED_MODE_PROBE.get(engine)
    if probe is None:
        assert "round-profile" in text  # yunshu and TensorFold: round profile lines
        return
    pats = [p["regex"] for p in probe["log_patterns"] if "regex" in p]
    assert any(re.search(p, text) for p in pats), (engine, text)
