"""Client tool wire formats, constrained inputs, documents and native helper routes."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from yunshu_gateway.responses_client_tools import (
    call_item,
    create_client_tools,
    declarations,
    replay,
)
from yunshu_gateway.routers import responses as r
from yunshu_gateway.server_tools.responses_loop import (
    function_tools,
    input_item_to_messages,
)


def events(text):
    return [
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")
    ]


def req(**kwargs):
    return r.ResponsesRequest(
        model="local", input="Use the tool.", max_output_tokens=128, **kwargs
    )


def fake_body(output):
    return {
        "id": "resp_test",
        "object": "response",
        "model": "local",
        "created_at": 1,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "completed_at": 2,
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def test_all_client_tools_become_callable_functions():
    q = req(
        tools=[
            {
                "type": "namespace",
                "name": "files",
                "tools": [{"type": "custom", "name": "apply_patch"}],
            },
            {"type": "local_shell"},
            {"type": "tool_search", "execution": "client"},
        ]
    )
    tools = function_tools(q.tools)
    assert [t.name for t in tools] == ["apply_patch", "local_shell", "tool_search"]
    assert tools[0].parameters["properties"]["input"]["type"] == "string"
    assert declarations(q.tools)["apply_patch"]["namespace"] == "files"


@pytest.mark.parametrize(
    "kind,fields,name,args",
    [
        ("custom_tool_call", {"input": "raw\n"}, "patch", {"input": "raw\n"}),
        (
            "local_shell_call",
            {"action": {"type": "exec", "command": ["pwd"]}},
            "local_shell",
            {"command": ["pwd"]},
        ),
        (
            "tool_search_call",
            {"arguments": {"query": "files"}, "execution": "client"},
            "tool_search",
            {"query": "files"},
        ),
    ],
)
def test_call_id_roundtrip(kind, fields, name, args):
    item = {
        "type": kind,
        "id": "different_item_id",
        "call_id": "call_exact",
        "name": name,
        **fields,
    }
    messages = input_item_to_messages(item)
    assert messages[0]["tool_calls"][0]["id"] == "call_exact"
    actual = json.loads(messages[0]["tool_calls"][0]["function"]["arguments"])
    assert actual == (fields["action"] if kind == "local_shell_call" else args)
    result_kind = {
        "custom_tool_call": "custom_tool_call_output",
        "local_shell_call": "local_shell_call_output",
        "tool_search_call": "tool_search_output",
    }[kind]
    msg = input_item_to_messages(
        {"type": result_kind, "call_id": "call_exact", "output": "done", "tools": []}
    )
    assert msg[0]["tool_call_id"] == "call_exact"


@pytest.mark.parametrize(
    "syntax,definition,spec",
    [
        ("regex", "PATCH\\n", {"type": "regex", "pattern": "PATCH\\n"}),
        ("lark", 'start: "PATCH\\n"', {"type": "cfg", "grammar": 'start: "PATCH\\n"'}),
    ],
)
def test_custom_raw_input_uses_decoder_constraint_and_preserves_whitespace(
    syntax, definition, spec
):
    seen = []

    async def inner(q, request):
        seen.append(q)
        return JSONResponse(
            fake_body(
                [
                    {
                        "type": "message",
                        "id": "m",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "PATCH\n",
                                "annotations": [],
                            }
                        ],
                    }
                ]
            )
        )

    q = req(
        tools=[
            {
                "type": "custom",
                "name": "patch",
                "format": {
                    "type": "grammar",
                    "syntax": syntax,
                    "definition": definition,
                },
            }
        ],
        tool_choice={"type": "custom", "name": "patch"},
        stream=True,
        store=True,
    )

    async def run():
        response = await create_client_tools(
            q, SimpleNamespace(state=SimpleNamespace()), inner
        )
        return "".join([chunk async for chunk in response.body_iterator])

    ev = events(asyncio.run(run()))
    assert seen[0].grammar == spec and seen[0]._custom_raw_input
    assert seen[0].tools is None and seen[0].enable_thinking is False
    added = next(e["item"] for e in ev if e["type"] == "response.output_item.added")
    final = ev[-1]["response"]
    item = final["output"][0]
    assert item["type"] == "custom_tool_call" and item["input"] == "PATCH\n"
    assert added["call_id"] == item["call_id"] and added["input"] == ""
    assert (
        next(
            e["delta"]
            for e in ev
            if e["type"] == "response.custom_tool_call_input.delta"
        )
        == "PATCH\n"
    )
    assert [e["sequence_number"] for e in ev] == list(range(len(ev)))
    assert r._get_stored_response(final["id"])["output"] == final["output"]


def test_special_calls_replay_valid_sdk_events():
    from openai.types.responses import ResponseStreamEvent
    from pydantic import TypeAdapter

    items = []
    for kind, args in [
        ("custom", {"input": "abc"}),
        ("local_shell", {"command": ["pwd"]}),
    ]:
        items.append(
            call_item(
                {
                    "id": f"item_{kind}",
                    "call_id": f"call_{kind}",
                    "name": kind,
                    "arguments": json.dumps(args),
                    "status": "completed",
                },
                {"type": kind},
            )
        )

    async def run():
        return "".join([chunk async for chunk in replay(fake_body(items))])

    for ev in events(asyncio.run(run())):
        TypeAdapter(ResponseStreamEvent).validate_python(ev)


def test_custom_route_does_not_drop_tool(monkeypatch):
    from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    engine = BatchedEngine()
    engine._model, engine._loaded, engine._running = object(), True, True
    engine.model_name = "compat-unit"
    seen = []

    async def chat(messages, **kwargs):
        seen.append(kwargs)
        return GenerationOutput(
            text="PATCH\n",
            new_text="PATCH\n",
            prompt_tokens=4,
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
        )

    monkeypatch.setattr(engine, "chat", chat)
    set_engine(engine)
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "local",
                    "input": "Patch.",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "apply_patch",
                            "format": {
                                "type": "grammar",
                                "syntax": "lark",
                                "definition": 'start: "PATCH\\n"',
                            },
                        }
                    ],
                    "tool_choice": {"type": "custom", "name": "apply_patch"},
                    "store": True,
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["output"][0]["type"] == "custom_tool_call"
            assert body["output"][0]["input"] == "PATCH\n"
            assert seen[0]["json_schema"]["type"] == "cfg"
            got = client.get("/v1/responses/" + body["id"])
            assert got.json()["output"] == body["output"]
    finally:
        set_engine(None)


def test_loaded_search_tools_are_promoted_without_recursion():
    seen = []

    async def inner(q, request):
        seen.append(q)
        return JSONResponse(fake_body([]))

    q = r.ResponsesRequest(
        model="local",
        input=[
            {
                "type": "tool_search_output",
                "call_id": "call_s",
                "execution": "client",
                "status": "completed",
                "tools": [
                    {
                        "type": "function",
                        "name": "read",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        ],
    )
    response = asyncio.run(
        create_client_tools(q, SimpleNamespace(state=SimpleNamespace()), inner)
    )
    assert response.status_code == 200
    assert seen[0].tools[0].name == "read"
    assert seen[0].input[0].type == "function_call_output"


def test_hosted_search_is_rejected_explicitly():
    with pytest.raises(HTTPException, match="Hosted"):
        function_tools(req(tools=[{"type": "tool_search"}]).tools)


def test_documents_and_checked_citations():
    from yunshu_gateway.anthropic_documents import attach_citations, read_document

    doc = asyncio.run(
        read_document(
            {
                "type": "document",
                "title": "Fact",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "The code is BLUE.",
                },
                "citations": {"enabled": True},
            },
            0,
        )
    )
    content = attach_citations(
        [{"type": "text", "text": "BLUE.[[cite:0:12:16]]"}], [doc]
    )
    citation = content[0]["citations"][0]
    assert citation == {
        "type": "char_location",
        "document_index": 0,
        "document_title": "Fact",
        "cited_text": "BLUE",
        "start_char_index": 12,
        "end_char_index": 16,
    }
    with pytest.raises(HTTPException, match="out-of-range"):
        attach_citations([{"type": "text", "text": "Oops[[cite:0:0:999]]"}], [doc])


def test_pdf_text_layer_and_page_citation():
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    from yunshu_gateway.anthropic_documents import read_document

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (BLUE) Tj ET")
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    data = io.BytesIO()
    writer.write(data)
    doc = asyncio.run(
        read_document(
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(data.getvalue()).decode(),
                },
                "citations": {"enabled": True},
            },
            0,
        )
    )
    assert "BLUE" in doc.text
    assert doc.citation(0, 4)["type"] == "page_location"
    assert doc.citation(0, 4)["end_page_number"] == 2


def test_document_route_renders_source_instead_of_base64(monkeypatch):
    from yunshu_gateway.anthropic_documents import create_documents
    from yunshu_gateway.routers.anthropic import (
        AnthropicMessagesRequest,
        _convert_anthropic_messages,
    )

    seen = []

    async def inner(q, request):
        seen.extend(_convert_anthropic_messages(q.messages)[0])
        return JSONResponse(
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "local",
                "content": [{"type": "text", "text": "BLUE.[[cite:0:0:4]]"}],
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "stop_reason": "end_turn",
                "stop_sequence": None,
            }
        )

    q = AnthropicMessagesRequest(
        model="local",
        max_tokens=16,
        stream=True,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "BLUE",
                        },
                        "citations": {"enabled": True},
                    }
                ],
            }
        ],
    )

    async def run():
        result = await create_documents(q, None, inner)
        return "".join([chunk async for chunk in result.body_iterator])

    ev = events(asyncio.run(run()))
    assert "BLUE" in seen[0]["content"]
    citation = next(
        e["delta"]["citation"]
        for e in ev
        if e.get("delta", {}).get("type") == "citations_delta"
    )
    assert citation["cited_text"] == "BLUE"


def test_computer_schema_includes_display_and_does_not_mutate_shared_schema():
    from yunshu_gateway.anthropic_client_tools import fill_client_tool_schemas
    from yunshu_gateway.routers.anthropic import AnthropicTool

    tool = AnthropicTool(
        type="computer_20250124",
        name="computer",
        display_width_px=1280,
        display_height_px=720,
    )
    assert fill_client_tool_schemas([tool])
    assert "coordinate" in tool.input_schema["properties"]
    assert "1280 x 720" in tool.description


def test_video_https_uses_guarded_download(monkeypatch, tmp_path):
    from yunshu_engine import vlm_engine
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine.__new__(VLMEngine)
    paths, calls = [], []
    engine._register_temp_file = paths.append

    async def resolve(url):
        return None

    async def download(url, dest, **kwargs):
        calls.append(kwargs)
        from pathlib import Path

        Path(dest).write_bytes(b"video")

    async def frames(path, *args, **kwargs):
        assert path == paths[0]
        return ["frame.png"]

    monkeypatch.setattr(vlm_engine, "_resolve_media_target", resolve)
    monkeypatch.setattr(vlm_engine.netguard, "download_to_file", download)
    monkeypatch.setattr(engine, "_extract_frames_from_file", frames)
    try:
        result = asyncio.run(
            engine._extract_video_frames(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": "https://example.org/v.mp4"},
                            }
                        ],
                    }
                ]
            )
        )
        assert result == ["frame.png"]
        assert (
            calls[0]["allow_private"] is False
            and calls[0]["max_bytes"] > 0
            and calls[0]["verify"] is True
        )
    finally:
        from pathlib import Path

        for path in paths:
            Path(path).unlink(missing_ok=True)


def test_continuous_usage_has_counts_on_each_chunk_and_is_request_local():
    from yunshu_gateway.continuous_usage import update_usage, with_continuous_usage
    from yunshu_gateway.routers.chat import ChatCompletionRequest
    from yunshu_gateway.streaming import format_openai_chunk

    @with_continuous_usage
    async def stream(req):
        for count in (1, 2):
            update_usage(req, 7, count)
            yield format_openai_chunk("id", "local", "x")

    async def run(enabled):
        q = ChatCompletionRequest(
            model="local",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            stream_options={"include_usage": enabled, "continuous_usage_stats": True},
        )
        return "".join([chunk async for chunk in stream(q)])

    ev = events(asyncio.run(run(True)))
    assert [e["usage"]["completion_tokens"] for e in ev] == [1, 2]
    assert all(e["usage"]["prompt_tokens"] == 7 for e in ev)
    assert all("usage" not in e for e in events(asyncio.run(run(False))))


def test_apply_template_and_props_routes(monkeypatch):
    from yunshu_gateway.routers import tokenize

    monkeypatch.setattr(
        tokenize,
        "_resolve_tokenizer",
        lambda _: SimpleNamespace(
            chat_template="template",
            apply_chat_template=lambda messages, **kw: (
                "rendered:" + messages[0]["content"]
            ),
        ),
    )
    monkeypatch.setattr(
        tokenize,
        "get_engine",
        lambda: SimpleNamespace(is_loaded=True, model_name="local"),
    )
    monkeypatch.setattr(tokenize, "_resolve_context_limit", lambda _: 8192)
    monkeypatch.setattr(
        "yunshu_gateway.routers.models._check_permission", lambda *a: None
    )
    app = FastAPI()
    app.include_router(tokenize.router)
    with TestClient(app) as client:
        result = client.post(
            "/apply-template", json={"messages": [{"role": "user", "content": "hello"}]}
        )
        assert result.json()["prompt"] == "rendered:hello"
        assert (
            client.get("/props").json()["default_generation_settings"]["n_ctx"] == 8192
        )


def test_document_messages_route_ingests_document_and_returns_citation(monkeypatch):
    from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    engine = BatchedEngine()
    engine._model, engine._loaded, engine._running = object(), True, True
    engine.model_name = "compat-unit"
    seen = []

    async def chat(messages, **kwargs):
        seen.extend(messages)
        text = "BLUE.[[cite:0:0:4]]"
        return GenerationOutput(
            text=text,
            new_text=text,
            prompt_tokens=20,
            completion_tokens=6,
            finished=True,
            finish_reason="stop",
        )

    monkeypatch.setattr(engine, "chat", chat)
    set_engine(engine)
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "local",
                    "max_tokens": 64,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "text",
                                        "media_type": "text/plain",
                                        "data": "BLUE",
                                    },
                                    "citations": {"enabled": True},
                                }
                            ],
                        }
                    ],
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["content"][0]["citations"][0]["cited_text"] == "BLUE"
            prompt = str(seen)
            assert "[Document 0" in prompt and "[[cite:0:" in prompt
    finally:
        set_engine(None)


def test_document_tool_result_and_token_count_use_same_rendering():
    from yunshu_gateway.anthropic_documents import has_documents, prepare_documents
    from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest

    q = AnthropicMessagesRequest(
        model="local",
        max_tokens=8,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "text",
                                    "media_type": "text/plain",
                                    "data": "BLUE",
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    )
    assert has_documents([m.content for m in q.messages])
    adapted, docs = asyncio.run(prepare_documents(q))
    assert docs[0].text == "BLUE"
    assert adapted.messages[0].content[0]["content"][0]["type"] == "text"


def test_editor_and_computer_version_specific_schemas():
    from yunshu_gateway.anthropic_client_tools import schema_for

    assert (
        "undo_edit"
        not in schema_for("text_editor_20250728")["input_schema"]["properties"][
            "command"
        ]["enum"]
    )
    assert (
        "undo_edit"
        in schema_for("text_editor_20241022")["input_schema"]["properties"]["command"][
            "enum"
        ]
    )
    assert (
        "zoom"
        in schema_for("computer_20251124")["input_schema"]["properties"]["action"][
            "enum"
        ]
    )
    assert (
        "zoom"
        not in schema_for("computer_20250124")["input_schema"]["properties"]["action"][
            "enum"
        ]
    )


def test_pdf_page_rendering_preserves_visual_input_and_scan_support():
    pypdf = pytest.importorskip("pypdf")
    pytest.importorskip("pypdfium2")
    from yunshu_gateway.anthropic_documents import read_document

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=120, height=120)
    data = io.BytesIO()
    writer.write(data)
    block = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(data.getvalue()).decode(),
        },
    }
    with pytest.raises(HTTPException, match="vision model"):
        asyncio.run(read_document(block, 0))
    doc = asyncio.run(read_document(block, 0, render_pdf_images=True))
    assert doc.images[1]["type"] == "image"
    assert base64.b64decode(doc.images[1]["source"]["data"]).startswith(b"\x89PNG")


def test_local_shell_legacy_output_id_is_call_id():
    message = input_item_to_messages(
        {"type": "local_shell_call_output", "id": "call_exact", "output": "done"}
    )[0]
    assert message["tool_call_id"] == "call_exact"


def test_search_call_json_arguments_validate_as_response_input():
    q = r.ResponsesRequest(
        model="local",
        input=[
            {
                "type": "tool_search_call",
                "call_id": "call_s",
                "execution": "client",
                "arguments": {"query": "files"},
            }
        ],
    )
    message = r._convert_to_messages(q)[0]
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "query": "files"
    }


def test_custom_output_images_remain_multimodal():
    message = input_item_to_messages(
        {
            "type": "custom_tool_call_output",
            "call_id": "call_img",
            "output": [
                {"type": "input_text", "text": "screenshot"},
                {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="},
            ],
        }
    )[0]
    assert message["content"][1]["type"] == "image_url"
    assert message["tool_call_id"] == "call_img"


def test_auto_custom_grammar_regenerates_constrained_input_with_one_budget():
    seen = []

    async def inner(q, request):
        seen.append(q)
        if len(seen) == 1:
            return JSONResponse(
                fake_body(
                    [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "patch",
                            "arguments": json.dumps({"input": "wrong"}),
                            "status": "completed",
                        }
                    ]
                )
            )
        assert q.max_output_tokens == 126 and q.grammar == {
            "type": "regex",
            "pattern": "PATCH",
        }
        return JSONResponse(
            fake_body(
                [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "PATCH"}],
                    }
                ]
            )
        )

    q = req(
        tools=[
            {
                "type": "custom",
                "name": "patch",
                "format": {"type": "grammar", "syntax": "regex", "definition": "PATCH"},
            }
        ]
    )
    response = asyncio.run(
        create_client_tools(q, SimpleNamespace(state=SimpleNamespace()), inner)
    )
    body = json.loads(response.body)
    assert (
        body["output"][0]["input"] == "PATCH"
        and body["output"][0]["call_id"] == "call_1"
    )
    assert body["usage"]["output_tokens"] == 4 and len(seen) == 2
    assert not seen[0].store


def test_multimodal_responses_keep_function_tools_and_calls(monkeypatch):
    from yunshu_gateway.routers import chat

    async def fake(chat_req, messages, request, json_schema=None):
        assert chat_req.tools[0].function.name == "read"
        assert chat_req.tool_choice.function.name == "read"
        return JSONResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_read",
                                    "function": {"name": "read", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        )

    monkeypatch.setattr(chat, "_handle_vlm_chat", fake)
    q = req(
        tools=[{"type": "function", "name": "read"}],
        tool_choice={"type": "function", "name": "read"},
        store=False,
    )
    request = SimpleNamespace(state=SimpleNamespace())
    response = asyncio.run(
        r._vlm_to_responses(q, [{"role": "user", "content": "x"}], request, None, [])
    )
    item = json.loads(response.body)["output"][-1]
    assert item["type"] == "function_call" and item["call_id"] == "call_read"


def test_multimodal_responses_forward_raw_custom_grammar(monkeypatch):
    from yunshu_gateway.routers import chat

    async def fake(chat_req, messages, request, json_schema=None):
        assert json_schema == {"type": "cfg", "grammar": 'start: "PATCH\\n"'}
        return JSONResponse(
            {
                "choices": [
                    {"message": {"content": "PATCH\n"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        )

    monkeypatch.setattr(chat, "_handle_vlm_chat", fake)
    q = req(grammar={"type": "cfg", "grammar": 'start: "PATCH\\n"'}, store=False)
    q._custom_raw_input = True
    request = SimpleNamespace(state=SimpleNamespace())
    response = asyncio.run(
        r._vlm_to_responses(q, [{"role": "user", "content": "x"}], request, None, [])
    )
    assert json.loads(response.body)["output"][0]["content"][0]["text"] == "PATCH\n"


def test_client_route_probe_cpu_parser_and_verdict(monkeypatch, capsys):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
    import client_compat_routes as probe

    monkeypatch.setattr(probe.subprocess, "check_output", lambda *a, **kw: "tree123\n")
    assert (
        probe.main(
            [
                "--model",
                "small",
                "--tree-sha",
                "tree123",
                "--device",
                "m3",
                "--out",
                "unused.json",
                "--dry-run",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["complete"]
    good = {
        "complete": True,
        "pass": True,
        "checks": {k: {"status": "pass"} for k in probe.CHECKS},
    }
    assert probe.judge(good)[0]
    good["checks"]["agent-custom-tools"]["status"] = "fail"
    assert not probe.judge(good)[0]


@pytest.mark.parametrize("status", ["completed", "incomplete"])
def test_forced_custom_stream_is_incremental_and_marks_truncation(status):
    from fastapi.responses import StreamingResponse
    from openai.types.responses import ResponseStreamEvent
    from pydantic import TypeAdapter

    state = {"finished": False}

    def event(kind, **data):
        return (
            "event: "
            + kind
            + "\ndata: "
            + json.dumps({"type": kind, "sequence_number": 0, **data})
            + "\n\n"
        )

    async def source():
        body = fake_body([])
        body.update(id="resp_stream_" + status, status=status)
        if status == "incomplete":
            body["incomplete_details"] = {"reason": "max_output_tokens"}
        initial = {**body, "output": [], "status": "in_progress", "usage": None}
        yield event("response.created", response=initial)
        yield event(
            "response.output_item.added",
            output_index=0,
            item={
                "type": "message",
                "id": "msg_raw",
                "role": "assistant",
                "content": [],
                "status": "in_progress",
            },
        )
        for text in ("PA", "TCH\n"):
            yield event(
                "response.output_text.delta",
                item_id="msg_raw",
                output_index=0,
                content_index=0,
                delta=text,
                logprobs=[],
            )
        yield event(
            "response.output_text.done",
            item_id="msg_raw",
            output_index=0,
            content_index=0,
            text="PATCH\n",
            logprobs=[],
        )
        item = {
            "type": "message",
            "id": "msg_raw",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "PATCH\n", "annotations": []}],
            "status": "completed",
        }
        yield event("response.output_item.done", output_index=0, item=item)
        body["output"] = [item]
        state["finished"] = True
        yield event("response." + status, response=body)

    async def inner(q, request):
        assert q.stream and q._custom_raw_input and not q.store
        return StreamingResponse(source(), media_type="text/event-stream")

    q = req(
        tools=[
            {
                "type": "custom",
                "name": "patch",
                "format": {
                    "type": "grammar",
                    "syntax": "regex",
                    "definition": "PATCH\\n",
                },
            }
        ],
        tool_choice={"type": "custom", "name": "patch"},
        stream=True,
        store=True,
    )

    async def run():
        response = await create_client_tools(
            q, SimpleNamespace(state=SimpleNamespace()), inner
        )
        collected = []
        async for chunk in response.body_iterator:
            ev = events(chunk)[0]
            if ev["type"] == "response.custom_tool_call_input.delta":
                assert not state["finished"]
            if ev["type"] == "response.output_item.done":
                assert state["finished"]
                assert ev["item"]["status"] == status
            TypeAdapter(ResponseStreamEvent).validate_python(ev)
            collected.append(ev)
        return collected

    ev = asyncio.run(run())
    assert [e["sequence_number"] for e in ev] == list(range(len(ev)))
    assert (
        "".join(
            e["delta"]
            for e in ev
            if e["type"] == "response.custom_tool_call_input.delta"
        )
        == "PATCH\n"
    )
    final = ev[-1]["response"]
    assert final["output"][0]["type"] == "custom_tool_call"
    assert r._get_stored_response(final["id"])["output"] == final["output"]


@pytest.mark.parametrize("grammar", [False, True])
def test_auto_stream_preserves_text_and_custom_call_id(grammar):
    from fastapi.responses import StreamingResponse
    from openai.types.responses import ResponseStreamEvent
    from pydantic import TypeAdapter

    state, seen = {"selected": False}, []
    text_item = {
        "type": "message",
        "id": "msg_text",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        "status": "completed",
    }
    function_item = {
        "type": "function_call",
        "id": "fc_orig",
        "call_id": "call_orig",
        "name": "patch",
        "arguments": json.dumps({"input": "RAW"}),
        "status": "completed",
    }

    async def selection():
        async for chunk in replay(fake_body([text_item, function_item])):
            if "event: response.completed" in chunk:
                state["selected"] = True
            yield chunk

    async def inner(q, request):
        seen.append(q)
        if len(seen) == 1:
            assert q.stream
            return StreamingResponse(selection(), media_type="text/event-stream")
        assert q.grammar == {"type": "regex", "pattern": "PATCH\\n"} and q.stream
        body = fake_body(
            [
                {
                    **text_item,
                    "id": "msg_raw",
                    "content": [
                        {"type": "output_text", "text": "PATCH\n", "annotations": []}
                    ],
                }
            ]
        )
        return StreamingResponse(replay(body), media_type="text/event-stream")

    tool = {"type": "custom", "name": "patch"}
    if grammar:
        tool["format"] = {
            "type": "grammar",
            "syntax": "regex",
            "definition": "PATCH\\n",
        }
    q = req(tools=[tool, {"type": "function", "name": "noop"}], stream=True)

    async def run():
        response = await create_client_tools(
            q, SimpleNamespace(state=SimpleNamespace()), inner
        )
        evs = []
        async for chunk in response.body_iterator:
            ev = events(chunk)[0]
            if ev["type"] == "response.output_text.delta":
                assert not state["selected"]
            TypeAdapter(ResponseStreamEvent).validate_python(ev)
            evs.append(ev)
        return evs

    ev = asyncio.run(run())
    added = [
        e["item"]
        for e in ev
        if e["type"] == "response.output_item.added"
        and e["item"]["type"] == "custom_tool_call"
    ]
    assert len(added) == 1 and added[0]["call_id"] == "call_orig"
    final = ev[-1]["response"]["output"][-1]
    assert final["call_id"] == "call_orig" and final["id"] == "fc_orig"
    assert final["input"] == ("PATCH\n" if grammar else "RAW")
    assert [e["sequence_number"] for e in ev] == list(range(len(ev)))
    assert len(seen) == (2 if grammar else 1)


def test_deferred_tool_schema_is_hidden_until_search_output_loads_it():
    seen = []

    async def inner(q, request):
        seen.append(q)
        return JSONResponse(fake_body([]))

    tool = {
        "type": "function",
        "name": "read",
        "description": "Read a file.",
        "defer_loading": True,
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
    q = req(tools=[tool, {"type": "tool_search", "execution": "client"}], store=True)
    request = SimpleNamespace(state=SimpleNamespace())
    asyncio.run(create_client_tools(q, request, inner))
    assert [t.name for t in seen[-1].tools] == ["tool_search"]
    assert "read: Read a file." in seen[-1].tools[0].description
    q = q.model_copy(
        update={
            "input": [
                r.ResponseInputText(
                    type="tool_search_output", call_id="call_s", tools=[tool]
                )
            ],
            "previous_response_id": None,
        }
    )
    asyncio.run(create_client_tools(q, request, inner))
    assert "read" in [t.name for t in seen[-1].tools]
    following = req(
        previous_response_id="resp_test",
        tools=[{"type": "tool_search", "execution": "client"}],
    )
    asyncio.run(create_client_tools(following, request, inner))
    assert "read" in [t.name for t in seen[-1].tools]


def test_document_cache_breakpoint_follows_page_images():
    from yunshu_gateway.anthropic_documents import prepare_documents
    from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest

    q = AnthropicMessagesRequest(
        model="local",
        max_tokens=8,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "content",
                            "content": [
                                {"type": "text", "text": "BLUE"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "abc",
                                    },
                                },
                            ],
                        },
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    )
    adapted, docs = asyncio.run(prepare_documents(q))
    blocks = adapted.messages[0].content
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}
    assert blocks[-2]["type"] == "image" and "cache_control" not in blocks[0]
    assert docs[0].citation(0, 4)["end_block_index"] == 1


def test_editor_view_limit_is_tool_configuration_and_zoom_is_opt_in():
    from yunshu_gateway.anthropic_client_tools import fill_client_tool_schemas
    from yunshu_gateway.routers.anthropic import AnthropicTool

    editor = AnthropicTool(
        type="text_editor_20250728", name="editor", max_characters=10000
    )
    computer = AnthropicTool(type="computer_20251124", name="computer")
    fill_client_tool_schemas([editor, computer])
    assert "10000" in editor.description
    assert "max_characters" not in editor.input_schema["properties"]
    assert "zoom" not in computer.input_schema["properties"]["action"]["enum"]


def test_function_output_content_array_preserves_images():
    q = r.ResponsesRequest(
        model="local",
        input=[
            {
                "type": "function_call_output",
                "call_id": "call_image",
                "output": [
                    {"type": "input_text", "text": "A screenshot"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,aGVsbG8=",
                    },
                ],
            }
        ],
    )
    message = r._convert_to_messages(q)[0]
    assert message["content"][1]["type"] == "image_url"
    assert message["tool_call_id"] == "call_image"


@pytest.mark.parametrize(
    "kind,args",
    [("local_shell", {"command": "pwd"}), ("custom", {"bad": "missing input"})],
)
def test_invalid_client_tool_arguments_end_in_response_failed(kind, args):
    from fastapi.responses import StreamingResponse
    from openai.types.responses import ResponseStreamEvent
    from pydantic import TypeAdapter

    name = "patch" if kind == "custom" else "local_shell"
    tool = {"type": kind, **({"name": name} if kind == "custom" else {})}
    item = {
        "type": "function_call",
        "id": "fc_bad",
        "call_id": "call_bad",
        "name": name,
        "arguments": json.dumps(args),
        "status": "completed",
    }

    async def inner(q, request):
        return StreamingResponse(
            replay(fake_body([item])), media_type="text/event-stream"
        )

    q = req(tools=[tool], stream=True, store=True)

    async def run():
        response = await create_client_tools(
            q, SimpleNamespace(state=SimpleNamespace()), inner
        )
        return "".join([chunk async for chunk in response.body_iterator])

    ev = events(asyncio.run(run()))
    assert ev[-1]["type"] == "response.failed"
    assert ev[-1]["response"]["error"]["code"] == "server_error"
    assert r._get_stored_response(ev[-1]["response"]["id"])["status"] == "failed"
    TypeAdapter(ResponseStreamEvent).validate_python(ev[-1])
