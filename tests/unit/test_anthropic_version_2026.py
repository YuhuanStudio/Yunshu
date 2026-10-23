"""Anthropic API version 2026-10-21: version-aware 400s, Bearer auth from the 2.x SDK, system messages."""

import os

import anthropic
import pytest
from fastapi.testclient import TestClient

from yunshu_gateway.anthropic_version import (
    is_strict,
    validate_messages,
    validate_request_body,
)

needs_sdk2 = pytest.mark.skipif(
    int(anthropic.__version__.split(".")[0]) < 2,
    reason="needs the anthropic 2.x SDK (Bearer key, 2026-10-21 version header)",
)

DOC = {
    "type": "document",
    "source": {"type": "text", "media_type": "text/plain", "data": "hello"},
}
CITE = {"type": "char_location", "cited_text": "a", "document_index": 0}


def test_is_strict():
    assert is_strict("2026-10-21")
    assert is_strict("2027-01-01")
    assert not is_strict("2023-06-01")
    assert not is_strict(None)
    assert not is_strict("")
    assert not is_strict("garbage")


def test_document_in_assistant_rejected():
    msg = validate_messages(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": [DOC]}]
    )
    assert msg and msg.startswith("messages.1.content.0:") and "assistant" in msg


def test_document_in_user_and_nested_in_tool_result_accepted():
    assert validate_messages([{"role": "user", "content": [DOC]}]) is None
    nested = {"type": "web_fetch_tool_result", "content": {"type": "x", "content": DOC}}
    assert validate_messages([{"role": "assistant", "content": [nested]}]) is None


def test_citations_in_user_text_rejected_but_assistant_and_enabled_ok():
    bad = {"type": "text", "text": "t", "citations": [CITE]}
    assert "citations" in validate_messages([{"role": "user", "content": [bad]}])
    inner = {"type": "search_result", "content": [bad], "source": "s", "title": "t"}
    got = validate_messages([{"role": "user", "content": [inner]}])
    assert got and got.startswith("messages.0.content.0.content.0.citations")
    assert validate_messages([{"role": "assistant", "content": [bad]}]) is None
    empty = {"type": "text", "text": "t", "citations": []}
    assert validate_messages([{"role": "user", "content": [empty]}]) is None
    enabled = dict(DOC, citations={"enabled": True})
    assert validate_messages([{"role": "user", "content": [enabled]}]) is None


def test_batch_body_prefix():
    body = {"messages": [{"role": "assistant", "content": [DOC]}]}
    assert validate_request_body(body, "requests.2.params.").startswith(
        "requests.2.params.messages.0.content.0:"
    )


@pytest.fixture
def server():
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    engine = BatchedEngine()
    engine._model = object()
    engine._loaded = True
    engine._running = True
    engine.model_name = "Qwen3.8-27B"
    captured: dict = {}

    async def _fake(*args, **kwargs):
        captured["messages"] = kwargs.get("messages", kwargs.get("prompt"))
        return GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )

    engine.generate = _fake
    engine.chat = _fake
    set_engine(engine)
    client = TestClient(create_app(), raise_server_exceptions=False)
    yield client, captured
    set_engine(None)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


BODY_DOC = {
    "model": "m",
    "max_tokens": 8,
    "messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [DOC]},
    ],
}
BODY_CITE = {
    "model": "m",
    "max_tokens": 8,
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "t", "citations": [CITE]}],
        }
    ],
}


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
@pytest.mark.parametrize("body", [BODY_DOC, BODY_CITE])
def test_new_version_is_400_old_version_is_not(server, path, body):
    client, _ = server
    r = client.post(path, json=body, headers={"anthropic-version": "2026-10-21"})
    assert r.status_code == 400
    err = r.json()
    assert err["type"] == "error" and err["error"]["type"] == "invalid_request_error"
    for h in ({"anthropic-version": "2023-06-01"}, {}):
        r = client.post(path, json=body, headers=h)
        if path.endswith("count_tokens"):  # the scripted engine has no tokenizer
            assert r.status_code != 400, r.text
        else:
            assert r.status_code == 200, r.text


def test_batch_create_validates_each_request(server):
    client, _ = server
    body = {"requests": [{"custom_id": "a", "params": BODY_DOC}]}
    r = client.post(
        "/v1/messages/batches", json=body, headers={"anthropic-version": "2026-10-21"}
    )
    assert r.status_code == 400
    assert "requests.0.params.messages.1.content.0" in r.json()["error"]["message"]


@needs_sdk2
def test_sdk2_bearer_auth_and_version_header(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "owner-secret")
    from yunshu_gateway.main import create_app

    http = TestClient(create_app(), raise_server_exceptions=False)
    seen = {}

    def spy(request):
        seen["auth"] = request.headers.get("authorization")
        seen["x"] = request.headers.get("x-api-key")
        seen["ver"] = request.headers.get("anthropic-version")

    http.event_hooks = {"request": [spy]}
    ok = anthropic.Anthropic(
        base_url="http://testserver", api_key="owner-secret", http_client=http
    )
    with pytest.raises(anthropic.BadRequestError):  # past auth, into the 2026 rule
        ok.messages.create(model="m", max_tokens=8, messages=BODY_DOC["messages"])
    assert seen["auth"] == "Bearer owner-secret" and not seen["x"]
    assert seen["ver"] >= "2026-10-21"
    bad = anthropic.Anthropic(
        base_url="http://testserver", api_key="nope", http_client=http, max_retries=0
    )
    with pytest.raises(anthropic.AuthenticationError):
        bad.messages.create(model="m", max_tokens=8, messages=BODY_DOC["messages"])


def test_sdk2_system_role_message_is_lifted(server):
    client, captured = server
    sdk = anthropic.Anthropic(
        base_url="http://testserver", api_key="x", http_client=client, max_retries=0
    )
    sdk.messages.create(
        model="m",
        max_tokens=8,
        messages=[
            {"role": "system", "content": "SYS_NOTE"},
            {"role": "user", "content": "hi"},
        ],
    )
    assert [m["role"] for m in captured["messages"]] == ["system", "user"]
    assert "SYS_NOTE" in captured["messages"][0]["content"]


def test_beta_clear_at_next_user_message(server):
    client, captured = server
    body = {
        "model": "m",
        "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "system", "content": "ONCE", "clear_at": "next_user_message"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "ALWAYS", "clear_at": "never"},
            {"role": "user", "content": "q2"},
        ],
    }
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 200, r.text
    text = " ".join(str(m["content"]) for m in captured["messages"])
    assert "ONCE" not in text and "ALWAYS" in text
    # still pending (no later user turn): rendered
    body["messages"] = body["messages"][:2]
    body["messages"].insert(0, {"role": "user", "content": "q0"})
    body["messages"] = [m for m in body["messages"] if m["content"] != "q1"]
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 200, r.text
    assert "ONCE" in " ".join(str(m["content"]) for m in captured["messages"])


def test_beta_system_message_without_content_is_dropped(server):
    client, captured = server
    body = {
        "model": "m",
        "max_tokens": 8,
        "messages": [
            {"role": "system", "output_config": {"effort": "low"}},
            {"role": "user", "content": "hi"},
        ],
    }
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 200, r.text
