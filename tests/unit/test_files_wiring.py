"""Files API references inside generation requests (Anthropic documents, Responses input_file)."""

from __future__ import annotations

import base64

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway import files_store
from yunshu_gateway.routers import anthropic, responses
from yunshu_gateway.server_tools import search

from .server_tools_helpers import FakeSearch, ScriptedInner


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path / "files"))
    files_store.reset_store()
    yield files_store.get_store()
    files_store.reset_store()
    search.set_provider_for_tests(None)


def test_anthropic_document_file_id_is_inlined_before_generation(store, monkeypatch):
    meta = store.put(
        b"The launch code is TANGERINE-7.", "notes.txt", mime_type="text/plain"
    )
    inner = ScriptedInner([([{"type": "text", "text": "ok"}], "end_turn")])

    async def non_stream_inner(req, request):
        from fastapi.responses import JSONResponse

        from yunshu_gateway.server_tools.anthropic_loop import assemble_message

        result = await inner(req, request)
        return JSONResponse(await assemble_message(result.body_iterator))

    monkeypatch.setattr(anthropic, "create_message", non_stream_inner)
    search.set_provider_for_tests(FakeSearch())
    app = FastAPI()
    app.include_router(anthropic.router, prefix="/v1")
    r = TestClient(app).post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 50,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": meta["id"]},
                        },
                        {"type": "text", "text": "what is the code?"},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 200, r.text
    doc = inner.requests[0].messages[0].content[0]
    assert doc["type"] == "text"
    assert "The launch code is TANGERINE-7." in doc["text"]
    assert "[Document 0" in doc["text"]


def test_anthropic_unknown_file_id_is_a_404(store):
    app = FastAPI()
    app.include_router(anthropic.router, prefix="/v1")
    r = TestClient(app).post(
        "/v1/messages",
        json={
            "model": "m",
            "max_tokens": 50,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": "file_" + "0" * 24},
                        }
                    ],
                }
            ],
        },
    )
    assert r.status_code == 404
    assert (
        r.json()["type"] == "error" and r.json()["error"]["type"] == "not_found_error"
    )


def test_input_file_text_is_inlined():
    data = "data:text/plain;base64," + base64.b64encode(b"alpha beta").decode()
    out = responses._extract_input_text(
        [
            {"type": "input_text", "text": "summarize"},
            {"type": "input_file", "filename": "n.txt", "file_data": data},
        ]
    )
    assert "summarize" in out and "[file: n.txt]" in out and "alpha beta" in out
    pdf = "data:application/pdf;base64," + base64.b64encode(b"%PDF").decode()
    note = responses._extract_input_text(
        [{"type": "input_file", "filename": "a.pdf", "file_data": pdf}]
    )
    assert "a.pdf" in note and "cannot be read" in note


def test_responses_input_file_id_is_resolved(store, monkeypatch):
    meta = store.put(b"hello from a file", "f.txt", mime_type="text/plain")
    seen = {}

    async def fake_inner(req, request):
        seen["messages"] = responses._convert_to_messages(req)
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {"id": "resp_x", "object": "response", "status": "completed", "output": []}
        )

    monkeypatch.setattr(responses, "_prewarm_response", fake_inner)
    app = FastAPI()
    app.include_router(responses.router, prefix="/v1")
    r = TestClient(app).post(
        "/v1/responses",
        json={
            "model": "m",
            "generate": False,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": meta["id"]},
                        {"type": "input_text", "text": "read it"},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 200, r.text
    content = seen["messages"][0]["content"]
    assert "hello from a file" in content and "read it" in content
