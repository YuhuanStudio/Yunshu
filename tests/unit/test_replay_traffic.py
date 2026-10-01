"""Replay measurements distinguish visible output and reject incomplete SSE."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def replay():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/research/agentic/replay_traffic.py"
    )
    spec = importlib.util.spec_from_file_location("replay_traffic_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stream:
    def __init__(self, events):
        self.lines = [
            b"data: " + json.dumps(event).encode() + b"\n"
            if isinstance(event, dict)
            else event
            for event in events
        ]

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *args):
        return False


def event(delta, finish=None):
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


def test_replay_separates_reasoning_and_visible_output(replay, monkeypatch):
    stream = Stream(
        [
            event({"reasoning_content": "thinking"}),
            event({"content": "answer"}),
            event({}, "stop"),
            {"choices": [], "usage": {"completion_tokens": 3}},
            b"data: [DONE]\n",
        ]
    )
    monkeypatch.setattr(
        replay.urllib.request, "urlopen", lambda *args, **kwargs: stream
    )
    clock = iter([10.0, 11.0, 12.0, 12.0, 14.0])
    monkeypatch.setattr(replay.time, "perf_counter", lambda: next(clock))
    result = replay.send("http://localhost", {})
    assert result["ttft_s"] == 1.0
    assert result["content_ttft_s"] == 2.0
    assert result["content"] == "answer"
    assert result["reasoning"] == "thinking"
    assert result["stream_done"] is True


@pytest.mark.parametrize(
    "events",
    [
        [event({"content": "partial"})],
        [event({"content": "partial"}), b"data: [DONE]\n"],
        [event({}, "stop")],
        [{"error": {"message": "generation failed"}}],
    ],
)
def test_replay_rejects_failed_or_incomplete_stream(replay, monkeypatch, events):
    monkeypatch.setattr(
        replay.urllib.request, "urlopen", lambda *args, **kwargs: Stream(events)
    )
    with pytest.raises(RuntimeError):
        replay.send("http://localhost", {})


def test_replay_title_failure_propagates_and_stops_server(
    replay, monkeypatch, tmp_path
):
    class Server:
        url = "http://localhost"
        killed = False

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return self

        def kill(self):
            Server.killed = True

    body = tmp_path / "body.json"
    title = tmp_path / "title.json"
    output = tmp_path / "out.jsonl"
    body.write_text(json.dumps({"model": "m", "messages": [], "kind": "main"}))
    title.write_text(json.dumps({"kind": "title"}))
    monkeypatch.setattr(replay, "Server", Server)
    monkeypatch.setattr(replay, "free_ports", lambda count: [18999])
    monkeypatch.setattr(replay.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        replay.sys,
        "argv",
        [
            "replay",
            "--checkpoint",
            "existing-model",
            "--bodies",
            str(body),
            "--title",
            str(title),
            "--out",
            str(output),
        ],
    )

    def send(url, payload):
        if payload.get("kind") == "title":
            raise RuntimeError("title request failed")
        return {"text": "ok"}

    monkeypatch.setattr(replay, "send", send)
    with pytest.raises(RuntimeError, match="title request failed"):
        replay.main()
    assert Server.killed
    assert not output.exists()


def test_replay_digest_compares_complete_arguments_independent_of_chunks(
    replay, monkeypatch
):
    def run(arguments, split):
        pieces = [arguments[:split], arguments[split:]]
        events = [
            event(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "random-id",
                            "function": {"name": "Bash", "arguments": pieces[0]},
                        }
                    ]
                }
            ),
            event({"tool_calls": [{"index": 0, "function": {"arguments": pieces[1]}}]}),
            event({}, "tool_calls"),
            b"data: [DONE]\n",
        ]
        monkeypatch.setattr(
            replay.urllib.request, "urlopen", lambda *args, **kwargs: Stream(events)
        )
        return replay.send("http://localhost", {})["output_sha256"]

    original = run('{"command":"ls"}', 4)
    assert original == run('{"command":"ls"}', 9)
    assert original != run('{"command":"pwd"}', 4)


def test_replay_blank_content_does_not_count_as_meaningful_output(replay, monkeypatch):
    events = [
        event({"content": "\n\n"}),
        event({"reasoning_content": "thinking"}),
        event({"content": "answer"}),
        event({}, "stop"),
        b"data: [DONE]\n",
    ]
    monkeypatch.setattr(
        replay.urllib.request, "urlopen", lambda *a, **kw: Stream(events)
    )
    clock = iter([10.0, 11.0, 11.0, 15.0, 16.0])
    monkeypatch.setattr(replay.time, "perf_counter", lambda: next(clock))
    result = replay.send("http://localhost", {})
    assert result["content_ttft_s"] == 1.0
    assert result["meaningful_ttft_s"] == 5.0
    assert result["content"] == "\n\nanswer"
