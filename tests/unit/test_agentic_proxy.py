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
    assert px.wait_idle(), "the proxy never listed the record"
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
    assert not r.get("error")
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


def test_stream_record_listed_before_the_response_ends(rig, monkeypatch):
    """A client that read a stream to its end finds the finished record at once (the terminating chunk
    used to go out first, so a slow summary left snapshot() empty or half-built: G1)."""
    fake, px = rig
    real = Tracker.summary

    def slow(self, t_end):
        time.sleep(0.3)
        return real(self, t_end)

    monkeypatch.setattr(Tracker, "summary", slow)
    c = http.client.HTTPConnection("127.0.0.1", px.port, timeout=10)
    body = {"model": "fake", "stream": True, "messages": []}
    c.request("POST", "/v1/responses", json.dumps(body))
    c.getresponse().read()
    c.close()
    # only this request's record: a stray client (a recycled ephemeral port was probed with GET /health
    # in the random-order run) also lands in the proxy's list
    recs = [r for r in px.snapshot() if r["method"] == "POST"]
    assert len(recs) == 1 and recs[0]["prompt_tokens"] == 100


def test_wait_idle_covers_content_length_replies(rig, monkeypatch):
    fake, px = rig
    real = Tracker.summary
    monkeypatch.setattr(
        Tracker, "summary", lambda self, t: (time.sleep(0.3), real(self, t))[1]
    )
    c = http.client.HTTPConnection("127.0.0.1", px.port, timeout=10)
    c.request("POST", "/v1/responses", json.dumps({"model": "fake"}))
    c.getresponse().read()
    c.close()
    assert px.wait_idle()
    recs = [r for r in px.snapshot() if r["method"] == "POST"]
    assert recs and recs[0]["prompt_tokens"] == 100


def test_error_never_saves_prompt_or_response(rig):
    fake, px = rig
    st, _ = post(px, "/v1/chat/completions", {"model": "fake-err"})
    assert st == 500
    r = px.snapshot()[0]
    assert r["error"] is True
    assert "error_body" not in r and "error_req_file" not in r
    assert not list(px.bodies_dir.glob("*"))


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


@pytest.mark.parametrize("error", [True, False])
def test_http_200_error_or_truncated_stream_fails_without_saving_content(
    rig, monkeypatch, error
):
    fake, px = rig
    px.save_bodies = True  # even legacy opt-in must never persist bodies
    event = (
        {"error": {"message": "secret-prompt"}}
        if error
        else {"choices": [{"delta": {"content": "partial"}}]}
    )
    monkeypatch.setattr(
        fake, "_answer", lambda h, body: fake._sse(h, [(None, event), (None, "[DONE]")])
    )
    status, _ = post(
        px,
        "/v1/chat/completions",
        {
            "model": "fake",
            "stream": True,
            "messages": [{"role": "user", "content": "secret-prompt"}],
        },
    )
    record = px.snapshot()[0]
    assert status == 200 and record["error"] is True
    assert "secret-prompt" not in json.dumps(record)
    assert not list(px.bodies_dir.glob("*"))
