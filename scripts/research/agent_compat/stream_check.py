"""Spec checks for recorded responses: SSE event order / fields and non-stream bodies.

Each checker takes parsed SSE events `[(event_name | None, data)]` (data = parsed JSON or str) and returns a
list of problem strings; an empty list is valid. Pure Python (no server, no SDK) so it unit-tests on CPU.
`check_body` additionally validates non-stream bodies with the official SDK models when they are installed.

References: Anthropic Messages streaming (message_start, content_block_*, message_delta, message_stop; `ping`
and `error` may appear), OpenAI chat.completion.chunk (role delta first, finish_reason chunk, optional usage
chunk with empty choices, `[DONE]`), OpenAI Responses streaming (`sequence_number`, output_item / content_part
pairs, one terminal response.completed|incomplete|failed).
"""

from __future__ import annotations

import json

ANTH_STOP = {"end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"}
ANTH_DELTAS = {
    "text": {"text_delta", "citations_delta"},
    "thinking": {"thinking_delta", "signature_delta"},
    "tool_use": {"input_json_delta"},
    "server_tool_use": {"input_json_delta"},
}
CHAT_FINISH = {"stop", "length", "tool_calls", "content_filter", "function_call"}


def _is_json_obj(s: str) -> bool:
    try:
        return isinstance(json.loads(s or "{}"), dict)
    except ValueError:
        return False


def check_anthropic_stream(events) -> list[str]:
    p: list[str] = []
    if not events:
        return ["empty stream"]
    evs = [(n, d) for n, d in events if n != "ping" and not (isinstance(d, dict) and d.get("type") == "ping")]
    if evs and evs[-1][0] == "error" or (evs and isinstance(evs[-1][1], dict) and evs[-1][1].get("type") == "error"):
        return ["stream ended with an error event: " + json.dumps(evs[-1][1])[:200]]
    for n, d in evs:
        if not isinstance(d, dict):
            p.append(f"non-JSON data for event {n!r}")
            return p
        if n != d.get("type"):
            p.append(f"event name {n!r} != data.type {d.get('type')!r}")
    types = [d["type"] for _, d in evs if isinstance(d, dict)]
    if not types or types[0] != "message_start":
        p.append(f"first event is {types[:1]}, expected message_start")
        return p
    if types[-1] != "message_stop":
        p.append(f"last event is {types[-1]!r}, expected message_stop")
    ms = evs[0][1].get("message") or {}
    for k in ("id", "type", "role", "model", "content", "usage"):
        if k not in ms:
            p.append(f"message_start.message lacks {k}")
    if ms.get("type") != "message" or ms.get("role") != "assistant":
        p.append("message_start.message type/role wrong")
    if "input_tokens" not in (ms.get("usage") or {}):
        p.append("message_start.message.usage lacks input_tokens")
    open_blocks: dict[int, dict] = {}
    next_index = 0
    kinds: list[str] = []
    seen_delta = seen_stop = False
    stop_reason = None
    for _, d in evs[1:]:
        t = d["type"]
        if seen_stop:
            p.append(f"event {t} after message_stop")
            break
        if t == "content_block_start":
            if seen_delta:
                p.append("content_block_start after message_delta")
            i = d.get("index")
            if i != next_index:
                p.append(f"content_block_start index {i}, expected {next_index}")
            if open_blocks:
                p.append("content_block_start while another block is open")
            cb = d.get("content_block") or {}
            if "type" not in cb:
                p.append("content_block_start lacks content_block.type")
            open_blocks[i] = {"type": cb.get("type"), "json": "", "block": cb}
            kinds.append(cb.get("type"))
            next_index += 1
        elif t == "content_block_delta":
            b = open_blocks.get(d.get("index"))
            if b is None:
                p.append(f"content_block_delta for closed/unknown block {d.get('index')}")
                continue
            dt = (d.get("delta") or {}).get("type")
            allowed = ANTH_DELTAS.get(b["type"])
            if allowed is not None and dt not in allowed:
                p.append(f"{dt} delta in a {b['type']} block")
            if dt == "input_json_delta":
                b["json"] += d["delta"].get("partial_json", "")
        elif t == "content_block_stop":
            b = open_blocks.pop(d.get("index"), None)
            if b is None:
                p.append(f"content_block_stop for unknown block {d.get('index')}")
            elif b["type"] in ("tool_use", "server_tool_use") and not _is_json_obj(b["json"]):
                p.append(f"tool_use input is not a JSON object: {b['json'][:80]!r}")
        elif t == "message_delta":
            if open_blocks:
                p.append("message_delta with an open content block")
            stop_reason = (d.get("delta") or {}).get("stop_reason")
            if stop_reason not in ANTH_STOP:
                p.append(f"stop_reason {stop_reason!r} not in the spec")
            if "output_tokens" not in (d.get("usage") or {}):
                p.append("message_delta.usage lacks output_tokens")
            seen_delta = True
        elif t == "message_stop":
            seen_stop = True
            if not seen_delta:
                p.append("message_stop without message_delta")
        elif t == "message_start":
            p.append("second message_start")
    if open_blocks:
        p.append("stream ended with an open content block")
    if (stop_reason == "tool_use") != ("tool_use" in kinds):
        p.append(f"stop_reason {stop_reason!r} disagrees with blocks {kinds}")
    if not kinds and stop_reason != "refusal":
        p.append("assistant message with no content block")
    return p


def check_chat_stream(events, expect_usage: bool = False) -> list[str]:
    p: list[str] = []
    if not events:
        return ["empty stream"]
    if events[-1][1] != "[DONE]":
        p.append("stream does not end with data: [DONE]")
    chunks = [d for _, d in events if d != "[DONE]"]
    if any(not isinstance(c, dict) for c in chunks):
        return p + ["non-JSON chunk"]
    for c in chunks:
        if "error" in c and "choices" not in c:
            return p + ["error chunk: " + json.dumps(c["error"])[:200]]
        for k in ("id", "object", "created", "model", "choices"):
            if k not in c:
                p.append(f"chunk lacks {k}")
                break
        if c.get("object") != "chat.completion.chunk":
            p.append(f"object {c.get('object')!r}")
    ids = {c.get("id") for c in chunks}
    if len(ids) > 1:
        p.append(f"chunk id changes inside one stream: {sorted(map(str, ids))[:3]}")
    body = [c for c in chunks if c.get("choices")]
    if not body:
        return p + ["no choice chunk"]
    first = body[0]["choices"][0].get("delta") or {}
    if first.get("role") != "assistant":
        p.append("first delta lacks role=assistant")
    tool_args: dict[tuple, str] = {}
    tool_seen = False
    finish = {}
    for c in body:
        for ch in c["choices"]:
            i = ch.get("index", 0)
            if i in finish and (ch.get("delta") or {}):
                if any(v for v in ch["delta"].values()):
                    p.append("delta content after finish_reason")
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                tool_seen = True
                key = (i, tc.get("index"))
                if tc.get("index") is None:
                    p.append("tool_call delta lacks index")
                fn = tc.get("function") or {}
                if key not in tool_args:
                    if not tc.get("id") or not fn.get("name"):
                        p.append("first tool_call delta lacks id or function.name")
                    tool_args[key] = ""
                tool_args[key] += fn.get("arguments") or ""
            if ch.get("finish_reason"):
                if ch["finish_reason"] not in CHAT_FINISH:
                    p.append(f"finish_reason {ch['finish_reason']!r}")
                finish[i] = ch["finish_reason"]
    if not finish:
        p.append("no finish_reason")
    for a in tool_args.values():
        if not _is_json_obj(a):
            p.append(f"tool_call arguments not a JSON object: {a[:80]!r}")
    if tool_seen != any(f == "tool_calls" for f in finish.values()):
        p.append(f"finish_reason {sorted(finish.values())} disagrees with tool_calls={tool_seen}")
    usage = [c for c in chunks if c.get("usage") and not c.get("choices")]
    if expect_usage:
        if not usage:
            p.append("stream_options.include_usage but no usage chunk with empty choices")
        else:
            u = usage[-1]["usage"]
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if k not in u:
                    p.append(f"usage lacks {k}")
    return p


def check_responses_stream(events) -> list[str]:
    p: list[str] = []
    if not events:
        return ["empty stream"]
    seq = -1
    names = []
    for n, d in events:
        if not isinstance(d, dict):
            return p + [f"non-JSON data for {n!r}"]
        if n and n != d.get("type"):
            p.append(f"event name {n!r} != data.type {d.get('type')!r}")
        names.append(d.get("type"))
        if "sequence_number" not in d:
            p.append(f"{d.get('type')} lacks sequence_number")
        elif d["sequence_number"] <= seq:
            p.append(f"sequence_number not increasing at {d.get('type')}")
        else:
            seq = d["sequence_number"]
    if names[0] != "response.created":
        p.append(f"first event {names[0]!r}, expected response.created")
    terminal = [n for n in names if n in ("response.completed", "response.incomplete", "response.failed", "error")]
    if len(terminal) != 1 or names[-1] != terminal[0]:
        p.append(f"expected one terminal event at the end, got {terminal}")
    if terminal and terminal[0] in ("response.failed", "error"):
        p.append("terminal event " + terminal[0])
    added, done = {}, {}
    final = None
    for _, d in events:
        t = d.get("type")
        if t == "response.output_item.added":
            added[d.get("output_index")] = d.get("item") or {}
        elif t == "response.output_item.done":
            done[d.get("output_index")] = d.get("item") or {}
        elif t in ("response.completed", "response.incomplete"):
            final = d.get("response") or {}
        elif t == "response.output_text.delta" and "delta" not in d:
            p.append("output_text.delta lacks delta")
    if set(added) != set(done):
        p.append(f"output_item added {sorted(added)} vs done {sorted(done)}")
    if final is not None:
        for k in ("id", "object", "status", "output", "model"):
            if k not in final:
                p.append(f"final response lacks {k}")
        if final.get("object") != "response":
            p.append("final response.object != response")
        if len(final.get("output") or []) != len(done):
            p.append(f"final output has {len(final.get('output') or [])} items, stream produced {len(done)}")
        if "usage" not in final:
            p.append("final response lacks usage")
    for item in done.values():
        if item.get("type") == "function_call" and not _is_json_obj(item.get("arguments", "")):
            p.append("function_call arguments not a JSON object")
    return p


def check_body(path: str, body) -> list[str]:
    """Non-stream response body vs the official SDK model for the route (skipped when no SDK)."""
    p = path.split("?")[0]
    try:
        if p == "/v1/messages":
            from anthropic.types import Message as M

            if body.get("type") == "error":
                return ["error body: " + json.dumps(body)[:200]]
        elif p == "/v1/chat/completions":
            from openai.types.chat import ChatCompletion as M
        elif p == "/v1/responses":
            from openai.types.responses import Response as M
        else:
            return []
    except ImportError:
        return []
    if not isinstance(body, dict):
        return ["body is not a JSON object"]
    if "error" in body and p != "/v1/messages":
        return ["error body: " + json.dumps(body)[:200]]
    try:
        M.model_validate(body)
    except Exception as e:  # noqa: BLE001
        return [f"{M.__name__} validation: {str(e)[:300]}"]
    return []
