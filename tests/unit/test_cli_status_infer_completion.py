"""`yunshu status`, the infer-style client commands and `completion`: no network, no model.

httpx.get / httpx.post are replaced with canned responses, so the tests check what
each command sends (path, body, auth header) and how it reports success and failure.
"""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from yunshu_cli import _output, app
from yunshu_cli.status import _fmt

runner = CliRunner()
URL = "http://engine.test:8000"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("YUNSHU_GATEWAY_URL", raising=False)
    monkeypatch.setattr(_output, "_json_mode", False)
    yield
    _output._json_mode = False


def _resp(status=200, body=None, text=None):
    req = httpx.Request("GET", URL)
    if text is not None:
        return httpx.Response(status, text=text, request=req)
    return httpx.Response(status, json=body, request=req)


class Net:
    """Records requests and answers from a {path: response-or-exception} table."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, **kw):
        path = url.removeprefix(URL)
        self.calls.append((path, kw))
        out = self.routes[path]
        if isinstance(out, Exception):
            raise out
        return out


def _install(monkeypatch, routes):
    net = Net(routes)
    monkeypatch.setattr(httpx, "get", net)
    monkeypatch.setattr(httpx, "post", net)
    return net


# ── status ────────────────────────────────────────────────────────────────


def test_fmt_scales_bytes_through_the_units():
    assert _fmt(512) == "512.0 B"
    assert _fmt(2048) == "2.0 KB"
    assert _fmt(3 * 1024**3) == "3.0 GB"
    assert _fmt(5 * 1024**5) == "5.0 PB"


STATUS_ROUTES = {
    "/health": _resp(body={"engine": {"loaded": True}}),
    "/debug/system": _resp(
        body={
            "gpu": {"active_bytes": 1024**3, "total_uma_bytes": 8 * 1024**3},
            "cpu": {"percent": 12.5},
        }
    ),
    "/debug/engine": _resp(body={"requests_processed": 7, "active_requests": 1}),
    "/v1/models": _resp(body={"data": [{"id": "qwen-27b"}]}),
}


def test_status_json_collects_health_system_engine_and_models(monkeypatch):
    _install(monkeypatch, dict(STATUS_ROUTES))
    r = runner.invoke(app, ["--json", "status", "--url", URL])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert out["healthy"] is True and out["url"] == URL
    assert out["models"] == ["qwen-27b"]
    assert out["engine"]["requests_processed"] == 7
    assert out["system"]["cpu"]["percent"] == 12.5


def test_status_human_output_shows_a_summary_and_loaded_model(monkeypatch):
    _install(monkeypatch, dict(STATUS_ROUTES))
    r = runner.invoke(app, ["status", "--url", URL])
    assert r.exit_code == 0, r.output
    assert "Healthy" in r.output and "qwen-27b" in r.output
    assert "Loaded" in r.output and "CPU: 12.5%" in r.output


def test_status_sends_the_bearer_token_to_the_debug_routes(monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "s3cret")
    net = _install(monkeypatch, dict(STATUS_ROUTES))
    assert runner.invoke(app, ["status", "--url", URL]).exit_code == 0
    sent = {p: kw.get("headers") for p, kw in net.calls}
    assert sent["/debug/system"] == {"Authorization": "Bearer s3cret"}
    assert sent["/v1/models"] == {"Authorization": "Bearer s3cret"}


def test_status_unreachable_server_exits_2_with_a_hint(monkeypatch):
    _install(monkeypatch, {"/health": httpx.ConnectError("refused")})
    r = runner.invoke(app, ["--json", "status", "--url", URL])
    assert r.exit_code == 2
    assert "yunshu serve" in json.loads(r.stdout)["error"]


def test_status_survives_failing_debug_routes(monkeypatch):
    routes = dict(STATUS_ROUTES) | {
        "/debug/system": _resp(500, text="boom"),
        "/debug/engine": httpx.ReadTimeout("slow"),
    }
    _install(monkeypatch, routes)
    r = runner.invoke(app, ["--json", "status", "--url", URL])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert out["system"] == {} and out["engine"] == {}


# ── infer-style client commands ─────────────────────────────────────────────


def test_complete_posts_a_chat_request_with_system_prompt(monkeypatch):
    net = _install(
        monkeypatch,
        {
            "/v1/chat/completions": _resp(
                body={
                    "model": "m",
                    "choices": [
                        {"message": {"content": "hello"}, "finish_reason": "stop"}
                    ],
                    "usage": {"total_tokens": 5},
                }
            )
        },
    )
    r = runner.invoke(
        app,
        [
            "--json",
            "complete",
            "hi",
            "--url",
            URL,
            "-s",
            "be brief",
            "--max-tokens",
            "9",
        ],
    )
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert out["text"] == "hello" and out["finish_reason"] == "stop"
    (_, kw) = net.calls[0]
    assert kw["json"]["max_tokens"] == 9
    assert kw["json"]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]


def test_tokenize_reports_a_count_and_embed_the_dimension(monkeypatch):
    _install(
        monkeypatch,
        {
            "/v1/tokenize": _resp(body={"count": 4, "tokens": [1, 2, 3, 4]}),
            "/v1/embeddings": _resp(
                body={"model": "e", "data": [{"embedding": [0.5, 0.25, 0.125]}]}
            ),
        },
    )
    r = runner.invoke(app, ["tokenize", "four tokens here", "--url", URL])
    assert r.exit_code == 0 and "tokens: 4" in r.output
    r = runner.invoke(app, ["--json", "embed", "x", "--url", URL])
    assert json.loads(r.stdout)["dim"] == 3


def test_rerank_passes_top_n_only_when_given(monkeypatch):
    net = _install(
        monkeypatch,
        {"/v1/rerank": _resp(body={"results": [{"index": 1, "relevance_score": 0.9}]})},
    )
    runner.invoke(app, ["--json", "rerank", "q", "a", "b", "--url", URL])
    runner.invoke(
        app, ["--json", "rerank", "q", "a", "b", "--url", URL, "--top-n", "1"]
    )
    assert "top_n" not in net.calls[0][1]["json"]
    assert net.calls[1][1]["json"]["top_n"] == 1


def test_server_errors_use_the_error_message_and_a_nonzero_exit(monkeypatch):
    _install(
        monkeypatch,
        {"/v1/chat/completions": _resp(429, body={"error": {"message": "queue full"}})},
    )
    r = runner.invoke(app, ["--json", "complete", "hi", "--url", URL])
    assert r.exit_code == 1
    err = json.loads(r.stdout)
    assert "queue full" in err["error"] and err["status"] == 429


def test_unreachable_server_and_odd_bodies_fail_cleanly(monkeypatch):
    _install(monkeypatch, {"/v1/tokenize": httpx.ConnectError("refused")})
    r = runner.invoke(app, ["--json", "tokenize", "x", "--url", URL])
    assert r.exit_code == 2 and "yunshu serve" in json.loads(r.stdout)["error"]

    _install(monkeypatch, {"/v1/chat/completions": _resp(text="<html>oops</html>")})
    r = runner.invoke(app, ["--json", "complete", "x", "--url", URL])
    assert r.exit_code == 1 and "non-JSON" in json.loads(r.stdout)["error"]

    _install(monkeypatch, {"/v1/chat/completions": _resp(body={"choices": []})})
    r = runner.invoke(app, ["--json", "complete", "x", "--url", URL])
    assert (
        r.exit_code == 1
        and "Unexpected response shape" in json.loads(r.stdout)["error"]
    )


def test_speak_refuses_a_json_error_body_instead_of_writing_it_as_audio(
    monkeypatch, tmp_path
):
    bad = httpx.Response(
        200,
        json={"detail": "no tts"},
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", URL),
    )
    _install(monkeypatch, {"/v1/audio/speech": bad})
    out = tmp_path / "a.wav"
    r = runner.invoke(app, ["--json", "speak", "hi", "-o", str(out), "--url", URL])
    assert r.exit_code == 1 and not out.exists()


def test_speak_writes_audio_and_picks_the_format_from_the_suffix(monkeypatch, tmp_path):
    audio = httpx.Response(
        200,
        content=b"RIFFdata",
        headers={"content-type": "audio/wav"},
        request=httpx.Request("POST", URL),
    )
    net = _install(monkeypatch, {"/v1/audio/speech": audio})
    out = tmp_path / "a.mp3"
    r = runner.invoke(app, ["--json", "speak", "hi", "-o", str(out), "--url", URL])
    assert r.exit_code == 0, r.output
    assert out.read_bytes() == b"RIFFdata"
    assert net.calls[0][1]["json"]["response_format"] == "mp3"
    assert json.loads(r.stdout)["bytes"] == 8


# ── completion ──────────────────────────────────────────────────────────────


def test_completion_script_is_generated_for_the_yunshu_command():
    r = runner.invoke(app, ["--json", "completion", "zsh"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert out["shell"] == "zsh"
    assert "_YUNSHU_COMPLETE" in out["script"]


def test_completion_rejects_other_shells_with_exit_2():
    r = runner.invoke(app, ["--json", "completion", "powershell"])
    assert r.exit_code == 2
    assert "zsh, bash or fish" in json.loads(r.stdout)["error"]
