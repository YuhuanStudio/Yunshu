"""Real-server agent-client probes; registered by route_checks, exercised by m3sweep routes."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from route_checks import Ctx, check, expect, skip


def _events(c, path, body):
    events = []
    with c.http.stream(
        "POST", path, headers=c.auth(), json=body, timeout=240
    ) as response:
        expect(
            response.status_code == 200,
            f"{path}: {response.status_code} {response.read()[:500]}",
        )
        for line in response.iter_lines():
            if line.startswith("data: {"):
                events.append(json.loads(line[6:]))
    expect(events, "No SSE events")
    return events


@check(
    "agent-custom-tools",
    "POST /v1/responses",
    "GET /v1/responses/{response_id}",
    served=True,
)
def custom_tools(c: Ctx):
    patch = "*** Begin Patch\n*** Add File: compat.txt\n+BLUE\n*** End Patch\n"
    grammar = (
        Path(__file__).resolve().parents[2] / "tests/fixtures/codex_apply_patch.lark"
    ).read_text()
    for syntax, definition in [("lark", grammar), ("regex", r"PATCH\n")]:
        tool = {
            "type": "custom",
            "name": "apply_patch",
            "description": "Return exactly the patch requested, preserving newlines.",
            "format": {"type": "grammar", "syntax": syntax, "definition": definition},
        }
        body = {
            "model": c.model,
            "input": "Return this exact tool input:\n"
            + (patch if syntax == "lark" else "PATCH\n"),
            "tools": [tool],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
            "max_output_tokens": 128,
            "stream": True,
            "store": True,
            "temperature": 0,
            "enable_thinking": False,
        }
        events = _events(c, "/v1/responses", body)
        expect(
            events[-1]["type"] == "response.completed",
            f"custom {syntax} incomplete: {events[-1]}",
        )
        item = events[-1]["response"]["output"][0]
        expect(
            item["type"] == "custom_tool_call"
            and item["name"] == "apply_patch"
            and item["call_id"],
            str(item),
        )
        text = "".join(
            e["delta"]
            for e in events
            if e["type"] == "response.custom_tool_call_input.delta"
        )
        expect(text == item["input"], "Custom delta != completed input")
        expect(
            item["input"] == "PATCH\n"
            if syntax == "regex"
            else (
                "*** Add File:" in item["input"]
                and item["input"].startswith("*** Begin Patch\n")
                and item["input"].rstrip().endswith("*** End Patch")
            ),
            f"Invalid {syntax} input: {item['input']!r}",
        )
        expect(
            [e["sequence_number"] for e in events] == list(range(len(events))),
            "SSE sequence is not monotonic",
        )
        rid = events[-1]["response"]["id"]
        stored = c.req("GET", "/v1/responses/" + rid)
        expect(
            stored.status_code == 200
            and stored.json()["output"] == events[-1]["response"]["output"],
            "Stored tool items differ",
        )
        follow = c.req(
            "POST",
            "/v1/responses",
            json={
                "model": c.model,
                "previous_response_id": rid,
                "input": [
                    {
                        "type": "custom_tool_call_output",
                        "call_id": item["call_id"],
                        "output": "Tool finished. Secret confirmation: MAGENTA.",
                    },
                    {
                        "role": "user",
                        "content": "What is the secret confirmation? Reply with that word only.",
                    },
                ],
                "max_output_tokens": 64,
                "enable_thinking": False,
            },
            timeout=240,
        )
        expect(
            follow.status_code == 200 and "MAGENTA" in str(follow.json()["output"]),
            f"custom call_id followup failed: {follow.text[:500]}",
        )


@check("agent-shell-search", "POST /v1/responses", served=True)
def shell_search(c: Ctx):
    for tool, args in [
        ({"type": "local_shell"}, {"command": ["pwd"]}),
        (
            {
                "type": "tool_search",
                "execution": "client",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "enum": ["files"]}},
                    "required": ["query"],
                },
            },
            {"query": "files"},
        ),
    ]:
        name = tool["type"]
        events = _events(
            c,
            "/v1/responses",
            {
                "model": c.model,
                "input": f"Use {name} with {json.dumps(args)}.",
                "tools": [tool],
                "tool_choice": {"type": name},
                "enable_thinking": False,
                "max_output_tokens": 192,
                "stream": True,
                "store": True,
                "temperature": 0,
            },
        )
        expect(events[-1]["type"] == "response.completed", f"{name} incomplete")
        item = next(
            (
                i
                for i in events[-1]["response"]["output"]
                if i["type"] == name + "_call"
            ),
            None,
        )
        expect(item and item.get("call_id"), f"No {name}_call: {events[-1]}")
        if name == "local_shell":
            expect(
                item["action"]["type"] == "exec"
                and isinstance(item["action"]["command"], list)
                and isinstance(item["action"]["env"], dict),
                str(item),
            )
            follow = c.req(
                "POST",
                "/v1/responses",
                json={
                    "model": c.model,
                    "previous_response_id": events[-1]["response"]["id"],
                    "input": [
                        {
                            "type": "local_shell_call_output",
                            "id": item["call_id"],
                            "output": "Secret confirmation: COBALT.",
                        },
                        {
                            "role": "user",
                            "content": "Repeat the secret confirmation word only.",
                        },
                    ],
                    "max_output_tokens": 64,
                    "enable_thinking": False,
                },
                timeout=240,
            )
            expect(
                follow.status_code == 200 and "COBALT" in str(follow.json()["output"]),
                follow.text[:500],
            )
        else:
            expect(
                item["execution"] == "client" and item["arguments"]["query"] == "files",
                str(item),
            )
            loaded = {
                "type": "function",
                "name": "compat_read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "enum": ["compat.txt"]}},
                    "required": ["path"],
                },
            }
            follow = c.req(
                "POST",
                "/v1/responses",
                json={
                    "model": c.model,
                    "input": [
                        item,
                        {
                            "type": "tool_search_output",
                            "call_id": item["call_id"],
                            "execution": "client",
                            "status": "completed",
                            "tools": [loaded],
                        },
                        {
                            "role": "user",
                            "content": "Call compat_read with path compat.txt.",
                        },
                    ],
                    "tool_choice": {"type": "function", "name": "compat_read"},
                    "max_output_tokens": 128,
                    "enable_thinking": False,
                },
                timeout=240,
            )
            expect(follow.status_code == 200, follow.text[:500])
            expect(
                any(
                    i.get("type") == "function_call" and i.get("name") == "compat_read"
                    for i in follow.json()["output"]
                ),
                f"Loaded tool unavailable: {follow.text[:500]}",
            )


def _pdf():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
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
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (BLUE) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return base64.b64encode(buffer.getvalue()).decode()


@check(
    "agent-documents-citations",
    "POST /v1/messages",
    "POST /v1/messages/count_tokens",
    served=True,
)
def documents(c: Ctx):
    upload = c.req(
        "POST",
        "/v1/files",
        files={"file": ("compat.txt", b"BLUE", "text/plain")},
        data={"purpose": "assistants"},
        timeout=120,
    )
    expect(upload.status_code == 200, upload.text[:500])
    file_id = upload.json()["id"]
    sources = [
        {"type": "text", "media_type": "text/plain", "data": "BLUE"},
        {"type": "base64", "media_type": "application/pdf", "data": _pdf()},
        {"type": "content", "content": [{"type": "text", "text": "BLUE"}]},
        {"type": "file", "file_id": file_id},
    ]
    try:
        for source in sources:
            body = {
                "model": c.model,
                "max_tokens": 128,
                "thinking": {"type": "disabled"},
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": source,
                                "title": "Secret",
                                "citations": {"enabled": True},
                            },
                            {
                                "type": "text",
                                "text": "Return exactly BLUE.[[cite:0:0:4]] to cite the first four characters of document 0. No other text.",
                            },
                        ],
                    }
                ],
            }
            counted = c.req("POST", "/v1/messages/count_tokens", json=body, timeout=120)
            expect(
                counted.status_code == 200 and counted.json()["input_tokens"] > 0,
                counted.text[:500],
            )
            events = _events(c, "/v1/messages", {**body, "stream": True})
            citations = [
                e["delta"]["citation"]
                for e in events
                if e.get("delta", {}).get("type") == "citations_delta"
            ]
            expect(
                citations and citations[0]["cited_text"] == "BLUE",
                f"Missing checked citation: {events}",
            )
            want = {
                "text": "char_location",
                "base64": "page_location",
                "content": "content_block_location",
                "file": "char_location",
            }[source["type"]]
            expect(citations[0]["type"] == want, str(citations))
            # Feed the public citation block back to the client conversation.
            follow = c.req(
                "POST",
                "/v1/messages",
                json={
                    **body,
                    "messages": body["messages"]
                    + [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "BLUE.",
                                    "citations": citations,
                                }
                            ],
                        },
                        {"role": "user", "content": "Repeat the secret word only."},
                    ],
                },
                timeout=240,
            )
            expect(
                follow.status_code == 200 and "BLUE" in str(follow.json()["content"]),
                follow.text[:500],
            )

    finally:
        removed = c.req("DELETE", "/v1/files/" + file_id)
        expect(removed.status_code == 200, removed.text[:500])


@check("agent-anthropic-client-tools", "POST /v1/messages", served=True)
def anthropic_tools(c: Ctx):
    for tool, instruction, key in [
        ({"type": "bash_20250124", "name": "bash"}, "Run printf BLUE.", "command"),
        (
            {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
            "View /tmp/compat.txt.",
            "path",
        ),
        (
            {
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": 1280,
                "display_height_px": 720,
            },
            "Take a screenshot.",
            "action",
        ),
    ]:
        result = c.req(
            "POST",
            "/v1/messages",
            json={
                "model": c.model,
                "max_tokens": 192,
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": tool["name"]},
                "messages": [{"role": "user", "content": instruction}],
                "temperature": 0,
            },
            timeout=240,
        )
        expect(result.status_code == 200, result.text[:500])
        body = result.json()
        item = next((b for b in body["content"] if b["type"] == "tool_use"), None)
        expect(
            item
            and item["name"] == tool["name"]
            and key in item["input"]
            and body["stop_reason"] == "tool_use",
            result.text[:500],
        )
        follow = c.req(
            "POST",
            "/v1/messages",
            json={
                "model": c.model,
                "max_tokens": 64,
                "thinking": {"type": "disabled"},
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": body["content"]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": item["id"],
                                "content": "Secret confirmation: BLUE.",
                            },
                            {
                                "type": "text",
                                "text": "Repeat the secret confirmation word only.",
                            },
                        ],
                    },
                ],
            },
            timeout=240,
        )
        expect(
            follow.status_code == 200 and "BLUE" in str(follow.json()["content"]),
            follow.text[:500],
        )


@check(
    "agent-continuous-usage",
    "POST /v1/chat/completions",
    "POST /v1/completions",
    served=True,
)
def continuous_usage(c: Ctx):
    events = _events(
        c,
        "/v1/chat/completions",
        {
            "model": c.model,
            "messages": [{"role": "user", "content": "Count from one to five."}],
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "max_tokens": 48,
            "enable_thinking": False,
        },
    )
    chunks = [e for e in events if e.get("choices")]
    expect(
        chunks and all("usage" in e for e in chunks), "A chunk omitted continuous usage"
    )
    counts = [e["usage"]["completion_tokens"] for e in chunks]
    expect(
        counts == sorted(counts) and counts[-1] > 0, f"Noncumulative counts: {counts}"
    )
    expect(
        events[-1]["usage"]["completion_tokens"] == counts[-1],
        "Final usage differs from latest continuous usage",
    )

    completion = _events(
        c,
        "/v1/completions",
        {
            "model": c.model,
            "prompt": "Count from one to five:",
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "max_tokens": 32,
            "enable_thinking": False,
        },
    )
    chunks = [e for e in completion if e.get("choices")]
    expect(
        chunks and all("usage" in e for e in chunks),
        "A completion chunk omitted continuous usage",
    )
    counts = [e["usage"]["completion_tokens"] for e in chunks]
    expect(
        counts == sorted(counts) and counts[-1] > 0,
        f"Noncumulative completion counts: {counts}",
    )
    expect(
        completion[-1]["usage"]["completion_tokens"] == counts[-1],
        "Final completion usage differs",
    )


@check(
    "agent-template-props",
    "POST /apply-template",
    "POST /v1/apply-template",
    "GET /props",
    "GET /v1/props",
    served=True,
)
def template_props(c: Ctx):
    for prefix in ("", "/v1"):
        result = c.req(
            "POST",
            prefix + "/apply-template",
            json={"model": c.model, "messages": [{"role": "user", "content": "BLUE"}]},
        )
        expect(
            result.status_code == 200 and "BLUE" in result.json()["prompt"],
            result.text[:500],
        )
        result = c.req("GET", prefix + "/props")
        expect(
            result.status_code == 200 and result.json().get("chat_template"),
            result.text[:500],
        )


@check("agent-http-video", "POST /v1/chat/completions", served=True)
def video(c: Ctx):
    if c.kind != "vlm":
        skip("HTTP video requires a VLM")
    result = c.req(
        "POST",
        "/v1/chat/completions",
        json={
            "model": c.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4"
                            },
                        },
                        {"type": "text", "text": "Briefly describe the video."},
                    ],
                }
            ],
            "max_tokens": 64,
            "enable_thinking": False,
        },
        timeout=240,
    )
    expect(
        result.status_code == 200
        and result.json()["choices"][0]["message"].get("content"),
        result.text[:500],
    )
    blocked = c.req(
        "POST",
        "/v1/chat/completions",
        json={
            "model": c.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "http://127.0.0.1:8000/private.mp4"},
                        }
                    ],
                }
            ],
            "max_tokens": 8,
        },
        timeout=120,
    )
    expect(
        blocked.status_code == 400
        and any(word in blocked.text.lower() for word in ("ssrf", "blocked", "unsafe")),
        blocked.text[:500],
    )
