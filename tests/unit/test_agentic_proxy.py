"""Recording proxy for the agentic benchmark: transparent forwarding plus per-request metrics."""

import http.client
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts" / "research" / "agentic")
)
from fake_server import Fake  # noqa: E402
from proxy import RecordingProxy, Tracker, api_kind  # noqa: E402


@pytest.fixture()
def rig(tmp_path):
    fake = Fake()
    px = RecordingProxy(fake.url, bodies_dir=tmp_path / "b").start()
    yield fake, px
    px.stop()
    fake.stop()


def post(px, path, body):
    c = http.client.HTTPConnection("127.0.0.1", px.port, timeout=10)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    n = len(px.snapshot())
    for _ in range(100):  # the record lands just after the last byte is forwarded
        if len(px.snapshot()) > n or px.pending == 0:
            break
        time.sleep(0.01)
    return r.status, data


TOOLS = {
    "/v1/chat/completions": [{"type": "function", "function": {"name": "bash"}}],
    "/v1/messages": [{"name": "bash", "input_schema": {}}],
    "/v1/responses": [{"type": "function", "name": "bash"}],
}


@pytest.mark.parametrize("path", list(TOOLS))
@pytest.mark.parametrize("stream", [True, False])
def test_usage_and_text(rig, path, stream):
    fake, px = rig
    st, data = post(px, path, {"model": "fake", "stream": stream, "messages": []})
    assert st == 200 and b"fake" in data
    r = px.snapshot()[0]
    assert r["prompt_tokens"] == 100 and r["completion_tokens"] == 5
    assert r["cached_tokens"] == 60 and r["tool_calls"] == 0
    assert "ttft_s" in r and r["total_s"] >= r["ttft_s"]
    if stream:
        assert r["decode_tok_s"] > 0


@pytest.mark.parametrize("path", list(TOOLS))
@pytest.mark.parametrize("stream", [True, False])
def test_tool_calls_and_malformed(rig, path, stream):
    fake, px = rig
    for model in ("fake-tool", "fake-tool-bad"):
        post(px, path, {"model": model, "stream": stream, "tools": TOOLS[path]})
    good, bad = px.snapshot()
    assert good["tool_calls"] == 1 and good["malformed_tool_calls"] == 0
    if path == "/v1/messages" and not stream:
        return  # Messages returns parsed input objects, so it cannot be malformed
    assert bad["malformed_tool_calls"] == 1


def test_error_saved(rig):
    fake, px = rig
    st, _ = post(px, "/v1/chat/completions", {"model": "fake-err"})
    assert st == 500
    r = px.snapshot()[0]
    assert r["error"] and "boom" in r["error_body"]
    assert Path(r["error_req_file"]).exists()


def test_passthrough_get(rig):
    fake, px = rig
    c = http.client.HTTPConnection("127.0.0.1", px.port)
    c.request("GET", "/v1/models")
    assert b"fake-model" in c.getresponse().read()


def test_leak_detection_and_kind():
    t = Tracker("chat", 0.0)
    t.feed_event({"choices": [{"delta": {"content": "x <tool_call> y"}}]}, 0.1)
    assert t.summary(1.0)["leaked_tool_markup"]
    assert api_kind("/v1/messages?beta=true") == "messages"
    assert api_kind("/v1/models") is None


@pytest.mark.parametrize(
    "marker", ["<tool_call>", "<function=", "<|tool_call", "[TOOL_CALLS]"]
)
def test_leak_marker_across_every_chunk_boundary(marker):
    for split in range(1, len(marker)):
        tracker = Tracker("responses", 0)
        tracker.feed_event(
            {"type": "response.output_text.delta", "delta": marker[:split]}, 1
        )
        assert not tracker.leaked
        tracker.feed_event(
            {"type": "response.output_text.delta", "delta": marker[split:]}, 2
        )
        assert tracker.leaked
        assert (
            len(tracker._leak_tail)
            <= max(map(len, __import__("proxy").LEAK_MARKERS)) - 1
        )
