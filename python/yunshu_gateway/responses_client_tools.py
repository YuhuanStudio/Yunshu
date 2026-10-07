"""Responses client tools adapted to local models' function-call templates.

Execution stays with the client. Grammar-bearing custom inputs use a dedicated
constrained generation after tool selection; replay streams expose only that input.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

SHELL_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "array", "items": {"type": "string"}},
        "timeout_ms": {"type": "integer"},
        "working_directory": {"type": "string"},
        "env": {"type": "object", "additionalProperties": {"type": "string"}},
        "user": {"type": "string"},
    },
    "required": ["command"],
}


def declarations(tools):
    """Flatten namespaces while retaining the outward tool kind and namespace."""
    out = {}
    for tool in tools or []:
        d = (
            tool.model_dump(exclude_none=True)
            if hasattr(tool, "model_dump")
            else dict(tool)
        )
        if d.get("type") == "namespace":
            for name, child in declarations(d.get("tools")).items():
                child = {**child, "namespace": d["name"]}
                if name in out:
                    raise HTTPException(
                        400, "Duplicate tool names across namespaces are unsupported"
                    )
                out[name] = child
        else:
            name = d.get("name") or d.get("type")
            if name in out:
                raise HTTPException(400, f"Duplicate tool name: {name}")
            out[name] = d
    return out


def as_function(d):
    from .routers.responses import ResponseTool

    kind = d.get("type")
    name = d.get("name") or kind
    if kind == "custom":
        schema = {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
            "additionalProperties": False,
        }
        description = (
            d.get("description") or ""
        ) + " Pass the complete raw tool input in the input string."
    elif kind == "local_shell":
        schema, description = (
            SHELL_SCHEMA,
            "Request a command to execute in the client's local shell.",
        )
    elif kind == "tool_search":
        if d.get("execution", "server") != "client":
            raise HTTPException(
                400, "Hosted tool_search is unsupported; use execution='client'"
            )
        schema = d.get("parameters") or {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        description = (
            d.get("description") or "Search for tools available to the client."
        )
    elif kind == "function":
        return ResponseTool(**d)
    else:
        return None
    return ResponseTool(
        type="function", name=name, description=description, parameters=schema
    )


def constraint(d):
    fmt = d.get("format") or {"type": "text"}
    if fmt.get("type") == "text":
        return None
    if (
        fmt.get("type") != "grammar"
        or fmt.get("syntax") not in ("regex", "lark")
        or not isinstance(fmt.get("definition"), str)
    ):
        raise HTTPException(
            400, "custom tool format must be text or a regex/lark grammar"
        )
    spec = (
        {"type": "cfg", "grammar": fmt["definition"]}
        if fmt["syntax"] == "lark"
        else {"type": "regex", "pattern": fmt["definition"]}
    )
    from yunshu_engine.grammar_constraint import validate_constraint_spec

    try:
        validate_constraint_spec(spec)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return spec


def call_item(item, d):
    """Convert a function template's call to the public Responses item."""
    kind = d.get("type")
    if kind == "function":
        return {**item, **({"namespace": d["namespace"]} if d.get("namespace") else {})}
    try:
        args = json.loads(item["arguments"])
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            502, "Model produced malformed client tool arguments"
        ) from exc
    common = {k: item[k] for k in ("id", "call_id", "status")}
    if kind == "custom":
        if not isinstance(args, dict) or not isinstance(args.get("input"), str):
            raise HTTPException(
                502, "Model produced a custom call without string input"
            )
        return {
            **common,
            "type": "custom_tool_call",
            "name": item["name"],
            "input": args["input"],
            **({"namespace": d["namespace"]} if d.get("namespace") else {}),
        }
    if kind == "local_shell":
        if (
            not isinstance(args, dict)
            or not isinstance(args.get("command"), list)
            or not all(isinstance(a, str) for a in args["command"])
        ):
            raise HTTPException(502, "Model produced an invalid local shell command")
        return {
            **common,
            "type": "local_shell_call",
            "action": {"env": {}, **args, "type": "exec"},
        }
    return {
        **common,
        "type": "tool_search_call",
        "execution": "client",
        "arguments": args,
    }


async def create_client_tools(req, request, inner):
    from .routers import responses as r

    if req.stream and req.n > 1:
        raise HTTPException(
            400, "n>1 is not supported with stream=True for the Responses API"
        )
    started = time.monotonic()
    defs = declarations(req.tools)
    # Schemas discovered on a previous client tool-search turn become callable.
    for item in req.input if isinstance(req.input, list) else []:
        if item.type in ("tool_search_output", "additional_tools"):
            defs.update(declarations(getattr(item, "tools", [])))
    formats = {
        name: constraint(d) for name, d in defs.items() if d.get("type") == "custom"
    }
    forced = req.tool_choice.get("name") if isinstance(req.tool_choice, dict) else None
    if isinstance(req.tool_choice, dict) and req.tool_choice.get("type") in (
        "local_shell",
        "tool_search",
    ):
        forced = req.tool_choice["type"]
    if req.tool_choice == "required" and len(defs) == 1:
        forced = next(iter(defs))

    async def raw_input(name, budget=None, response_id=None):
        d = defs[name]
        guidance = f"Produce only the raw input for the {name} tool. {d.get('description') or ''}"
        fmt = d.get("format") or {}
        if fmt.get("definition"):
            guidance += "\nThe input must follow this grammar:\n" + fmt["definition"]
        raw_req = req.model_copy(
            update={
                "tools": None,
                "tool_choice": "none",
                "stream": False,
                "store": False,
                "background": False,
                "grammar": formats[name],
                "response_format": None,
                "text": None,
                "enable_thinking": False,
                "thinking_budget": 0,
                "reasoning": None,
                "reasoning_effort": None,
                "n": 1,
                "max_output_tokens": budget
                if budget is not None
                else req.max_output_tokens,
                "timeout": max(0.001, req.timeout - (time.monotonic() - started))
                if req.timeout is not None
                else None,
                "instructions": (req.instructions or "") + "\n" + guidance,
            }
        )
        raw_req._custom_raw_input = True
        from yunshu_engine.batched_engine import _REQUEST_TOOL_USE, _REQUEST_TOOLS

        tools_token = _REQUEST_TOOLS.set(None)
        use_token = _REQUEST_TOOL_USE.set(None)
        previous_id = getattr(request.state, "_forced_response_id", None)
        if response_id:
            request.state._forced_response_id = response_id
        try:
            response = await inner(raw_req, request)
        finally:
            if response_id:
                request.state._forced_response_id = previous_id
            _REQUEST_TOOLS.reset(tools_token)
            _REQUEST_TOOL_USE.reset(use_token)
        if response.status_code != 200:
            return response, None
        body = json.loads(response.body)
        text = "".join(
            p.get("text", "")
            for i in body["output"]
            if i.get("type") == "message"
            for p in i.get("content", [])
        )
        return body, text

    if forced in formats and req.n == 1:
        body, text = await raw_input(forced)
        if text is None:
            return body
        body["output"] = [
            {
                "type": "custom_tool_call",
                "id": "ctc_" + uuid.uuid4().hex[:24],
                "call_id": "call_" + uuid.uuid4().hex[:24],
                "name": forced,
                "input": text,
                "status": "completed"
                if body["status"] == "completed"
                else "incomplete",
                **(
                    {"namespace": defs[forced]["namespace"]}
                    if defs[forced].get("namespace")
                    else {}
                ),
            }
        ]
        body["_input_messages"] = [
            m for m in r._convert_to_messages(req) if m.get("role") != "system"
        ]
    else:
        tools = [t for d in defs.values() if (t := as_function(d)) is not None]
        choice = {"type": "function", "name": forced} if forced else req.tool_choice
        adapted_input = req.input
        if isinstance(req.input, list):
            adapted_input = [
                i.model_copy(
                    update={
                        "type": "function_call_output",
                        "output": getattr(i, "tools", []),
                    }
                )
                if i.type == "tool_search_output"
                else i
                for i in req.input
                if i.type != "additional_tools"
            ]
        adapted = req.model_copy(
            update={
                "tools": tools,
                "stream": False,
                "store": False,
                "tool_choice": choice,
                "input": adapted_input,
            }
        )
        response = await inner(adapted, request)
        if response.status_code != 200:
            return response
        body = json.loads(response.body)
        for idx, item in enumerate(body.get("output", [])):
            if item.get("type") != "function_call" or item.get("name") not in defs:
                continue
            d = defs[item["name"]]
            converted = call_item(item, d)
            if formats.get(item["name"]) is not None:
                budget = req.max_output_tokens - body["usage"]["output_tokens"]
                if budget <= 0 or body["status"] != "completed":
                    converted.update(input="", status="incomplete")
                    body.update(
                        status="incomplete",
                        incomplete_details={"reason": "max_output_tokens"},
                    )
                    body["output"][idx] = converted
                    continue
                raw, text = await raw_input(item["name"], budget, body["id"])
                if text is None:
                    return raw
                converted["input"] = text
                if raw["status"] != "completed":
                    body["status"] = raw["status"]
                    body["incomplete_details"] = raw.get("incomplete_details")
                    converted["status"] = "incomplete"
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    body["usage"][key] += raw["usage"][key]
            body["output"][idx] = converted
    body["_input_messages"] = [
        m for m in r._convert_to_messages(req) if m.get("role") != "system"
    ]
    if isinstance(req.input, list):
        raw_items = []
        for idx, item in enumerate(req.input):
            if item.type in ("message", ""):
                if item.role in ("system", "developer"):
                    continue
                messages = r._convert_to_messages(
                    req.model_copy(update={"input": [item], "instructions": None})
                )
                for message in messages:
                    raw_items.extend(r._input_item_of(body["id"], idx, message))
            else:
                raw = item.model_dump(exclude_none=True)
                raw.pop("role", None)
                raw.setdefault("id", f"{body['id']}:input:{idx}")
                raw_items.append(raw)
        body["_client_input_items"] = raw_items
    body.update(r._config_echo(req))
    if req.store:
        stored = r._get_stored_response(body["id"]) or {}
        r._store_response(
            body["id"], {**stored, **body, "_owner": r._resolve_owner(request)}
        )
    public = r._public_stored(body)
    if not req.stream:
        return JSONResponse(public)
    return StreamingResponse(
        replay(public),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def replay(body):
    """Completed buffered client-tool response as a spec-shaped SSE lifecycle."""
    seq = 0

    def event(kind, **data):
        nonlocal seq
        payload = {"type": kind, "sequence_number": seq, **data}
        seq += 1
        return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    initial = {
        **body,
        "output": [],
        "status": "in_progress",
        "usage": None,
        "completed_at": None,
    }
    yield event("response.created", response=initial)
    yield event("response.in_progress", response=initial)
    for idx, item in enumerate(body["output"]):
        kind = item["type"]
        added = {**item, "status": "in_progress"}
        if kind == "custom_tool_call":
            added["input"] = ""
        elif kind == "function_call":
            added["arguments"] = ""
        elif kind == "message":
            added["content"] = []
        elif kind == "reasoning":
            added["summary"] = []
        yield event("response.output_item.added", output_index=idx, item=added)
        coords = {"item_id": item["id"], "output_index": idx}
        if kind in ("custom_tool_call", "function_call"):
            field = "input" if kind == "custom_tool_call" else "arguments"
            family = (
                "custom_tool_call_input"
                if kind == "custom_tool_call"
                else "function_call_arguments"
            )
            yield event(f"response.{family}.delta", **coords, delta=item[field])
            yield event(f"response.{family}.done", **coords, **{field: item[field]})
        elif kind == "message":
            for ci, part in enumerate(item["content"]):
                yield event(
                    "response.content_part.added",
                    **coords,
                    content_index=ci,
                    part={**part, "text": ""},
                )
                yield event(
                    "response.output_text.delta",
                    **coords,
                    content_index=ci,
                    delta=part.get("text", ""),
                    logprobs=[],
                )
                yield event(
                    "response.output_text.done",
                    **coords,
                    content_index=ci,
                    text=part.get("text", ""),
                    logprobs=[],
                )
                yield event(
                    "response.content_part.done", **coords, content_index=ci, part=part
                )
        elif kind == "reasoning":
            for si, part in enumerate(item.get("summary", [])):
                yield event(
                    "response.reasoning_summary_part.added",
                    **coords,
                    summary_index=si,
                    part={**part, "text": ""},
                )
                yield event(
                    "response.reasoning_summary_text.delta",
                    **coords,
                    summary_index=si,
                    delta=part["text"],
                )
                yield event(
                    "response.reasoning_summary_text.done",
                    **coords,
                    summary_index=si,
                    text=part["text"],
                )
                yield event(
                    "response.reasoning_summary_part.done",
                    **coords,
                    summary_index=si,
                    part=part,
                )
        yield event("response.output_item.done", output_index=idx, item=item)
    yield event("response." + body["status"], response=body)
