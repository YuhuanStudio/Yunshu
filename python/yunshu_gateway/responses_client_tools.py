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


def computer_schema():
    """Independent schema matching the SDK's discriminated computer actions."""
    xy = {"x": {"type": "integer"}, "y": {"type": "integer"}}
    keys = {"keys": {"type": "array", "items": {"type": "string"}}}
    variants = []
    for kind in (
        "click",
        "double_click",
        "drag",
        "move",
        "scroll",
        "keypress",
        "type",
        "wait",
        "screenshot",
    ):
        props = {"type": {"const": kind}}
        required = ["type"]
        if kind in ("click", "double_click", "move", "scroll"):
            props.update(xy)
            required += ["x", "y"]
        if kind in ("click", "double_click", "drag", "move", "scroll", "keypress"):
            props.update(keys)
        if kind == "click":
            props["button"] = {"enum": ["left", "right", "wheel", "back", "forward"]}
            required.append("button")
        if kind == "scroll":
            props.update(
                {"scroll_x": {"type": "integer"}, "scroll_y": {"type": "integer"}}
            )
            required += ["scroll_x", "scroll_y"]
        if kind == "keypress":
            required.append("keys")
        if kind == "type":
            props["text"] = {"type": "string"}
            required.append("text")
        if kind == "drag":
            props["path"] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": xy,
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            }
            required.append("path")
        variants.append(
            {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            }
        )
    return {
        "type": "object",
        "properties": {"actions": {"type": "array", "items": {"oneOf": variants}}},
        "required": ["actions"],
        "additionalProperties": False,
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
    elif kind == "computer":
        schema = computer_schema()
        description = "Request ordered computer actions on the client. Return actions, then wait for the client screenshot. All execution is client-side."
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
    if kind == "computer":
        import jsonschema

        try:
            jsonschema.validate(args, computer_schema())
        except jsonschema.ValidationError as exc:
            raise HTTPException(502, "Model produced invalid computer actions") from exc
        return {
            **common,
            "type": "computer_call",
            "actions": args["actions"],
            "pending_safety_checks": [],
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
    loaded = {}
    previous = req.previous_response_id
    visited = set()
    while previous and previous not in visited:
        visited.add(previous)
        stored = r._get_stored_response(previous)
        if stored is None or not r._owns_stored(request, stored):
            break
        for name, declaration in (stored.get("_loaded_tools") or {}).items():
            loaded.setdefault(name, declaration)
        previous = stored.get("previous_response_id")
    for item in req.input if isinstance(req.input, list) else []:
        if item.type in ("tool_search_output", "additional_tools"):
            loaded.update(declarations(getattr(item, "tools", [])))
    for name, declaration in loaded.items():
        defs.setdefault(name, declaration)
    deferred = {
        name: d
        for name, d in defs.items()
        if d.get("defer_loading") and name not in loaded
    }
    defs = {name: d for name, d in defs.items() if name not in deferred}
    if deferred:
        catalog = "\nDeferred tools (search to load their schemas):\n" + "\n".join(
            f"{name}: {d.get('description') or ''}" for name, d in deferred.items()
        )
        defs = {
            name: {**d, "description": (d.get("description") or "") + catalog}
            if d.get("type") == "tool_search"
            else d
            for name, d in defs.items()
        }
    req._loaded_client_tools = loaded
    formats = {
        name: constraint(d) for name, d in defs.items() if d.get("type") == "custom"
    }
    forced = req.tool_choice.get("name") if isinstance(req.tool_choice, dict) else None
    if isinstance(req.tool_choice, dict) and req.tool_choice.get("type") in (
        "local_shell",
        "tool_search",
        "computer",
    ):
        forced = req.tool_choice["type"]
    if req.tool_choice == "required" and len(defs) == 1:
        forced = next(iter(defs))

    async def raw_input(name, budget=None, response_id=None, stream=False):
        d = defs[name]
        guidance = f"Produce only the raw input for the {name} tool. {d.get('description') or ''}"
        fmt = d.get("format") or {}
        if fmt.get("definition"):
            guidance += "\nThe input must follow this grammar:\n" + fmt["definition"]
        raw_req = req.model_copy(
            update={
                "tools": None,
                "tool_choice": "none",
                "stream": stream,
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
        if response.status_code != 200 or isinstance(response, StreamingResponse):
            return response, None
        body = json.loads(response.body)
        text = "".join(
            p.get("text", "")
            for i in body["output"]
            if i.get("type") == "message"
            for p in i.get("content", [])
        )
        return body, text

    if forced and forced in deferred:
        raise HTTPException(
            400, f"Tool {forced} is deferred; load it through tool_search first"
        )
    if forced in formats and req.n == 1:
        body, text = await raw_input(forced, stream=req.stream)
        if isinstance(body, StreamingResponse):
            return stream_custom_input(body, req, request, defs[forced])
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
                        "output": json.dumps(
                            getattr(i, "tools", []), ensure_ascii=False
                        ),
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
                "stream": req.stream,
                "store": False,
                "tool_choice": choice,
                "input": adapted_input,
            }
        )
        response = await inner(adapted, request)
        if response.status_code != 200:
            return response
        if isinstance(response, StreamingResponse):
            return stream_client_calls(response, req, request, defs, formats, raw_input)
        body = json.loads(response.body)
        for idx, item in enumerate(body.get("output", [])):
            if item.get("type") != "function_call" or item.get("name") not in defs:
                continue
            d = defs[item["name"]]
            converted = call_item(
                {**item, "arguments": json.dumps({"input": ""})}
                if formats.get(item["name"]) is not None
                else item,
                d,
            )
            if formats.get(item["name"]) is not None:
                budget = req.max_output_tokens - body["usage"]["output_tokens"]
                if budget <= 0 or body["status"] != "completed":
                    converted.update(input="", status="incomplete")
                    if budget <= 0:
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
    public = finish_body(body, req, request)
    if not req.stream:
        return JSONResponse(public)
    return StreamingResponse(
        replay(public),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def finish_body(body, req, request):
    from .routers import responses as r

    body["_loaded_tools"] = getattr(req, "_loaded_client_tools", {})
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
                if item.type != "additional_tools":
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
    return r._public_stored(body)


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


def stream_failure(body, req, request, exc):
    from .routers.responses import _config_echo

    failed = {
        "id": body.get("id") or "resp_" + uuid.uuid4().hex[:24],
        "object": "response",
        "created_at": body.get("created_at", int(time.time())),
        "model": req.model,
        **body,
        **_config_echo(req),
        "status": "failed",
        "completed_at": int(time.time()),
        "output": [],
        "error": {
            "code": "server_error",
            "message": str(
                getattr(exc, "detail", None) or "Client tool stream conversion failed"
            ),
        },
    }
    return finish_body(failed, req, request)


def stream_custom_input(response, req, request, declaration):
    """Translate a forced raw generation as it arrives, preserving cancellation.

    Tool execution sees item.done only after the engine's terminal status is known,
    so a token-limited patch cannot be mistaken for a completed tool input.
    """
    from .server_tools.responses_loop import _parse_sse

    call = {
        "type": "custom_tool_call",
        "id": "ctc_" + uuid.uuid4().hex[:24],
        "call_id": "call_" + uuid.uuid4().hex[:24],
        "name": declaration["name"],
        "input": "",
        "status": "in_progress",
    }
    if declaration.get("namespace"):
        call["namespace"] = declaration["namespace"]

    async def generate():
        from yunshu_engine.batched_engine import _REQUEST_TOOL_USE, _REQUEST_TOOLS

        from .routers.responses import _config_echo

        tools_token, use_token = _REQUEST_TOOLS.set(None), _REQUEST_TOOL_USE.set(None)
        seq, pending_done = 0, None

        def event(data):
            nonlocal seq
            data = {**data, "sequence_number": seq}
            seq += 1
            return f"event: {data['type']}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        latest_response = {}
        try:
            async for kind, data in _parse_sse(response.body_iterator):
                if kind == "__comment__":
                    yield data + "\n\n"
                    continue
                if isinstance(data.get("response"), dict):
                    latest_response = data["response"]
                if kind in (
                    "response.content_part.added",
                    "response.content_part.done",
                    "response.output_text.annotation.added",
                ):
                    continue
                if (
                    kind == "response.output_item.added"
                    and data["item"].get("type") == "message"
                ):
                    data = {**data, "item": dict(call)}
                elif kind == "response.output_text.delta":
                    call["input"] += data["delta"]
                    data = {
                        "type": "response.custom_tool_call_input.delta",
                        "item_id": call["id"],
                        "output_index": data["output_index"],
                        "delta": data["delta"],
                    }
                elif kind == "response.output_text.done":
                    call["input"] = data["text"]
                    data = {
                        "type": "response.custom_tool_call_input.done",
                        "item_id": call["id"],
                        "output_index": data["output_index"],
                        "input": data["text"],
                    }
                elif (
                    kind == "response.output_item.done"
                    and data["item"].get("type") == "message"
                ):
                    pending_done = data
                    continue
                elif kind in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    body = data["response"]
                    call["status"] = (
                        "completed"
                        if body.get("status") == "completed"
                        else "incomplete"
                    )
                    for idx, item in enumerate(body.get("output", [])):
                        if item.get("type") == "message":
                            call["input"] = "".join(
                                p.get("text", "") for p in item.get("content", [])
                            )
                            body["output"][idx] = dict(call)
                    if pending_done:
                        yield event({**pending_done, "item": dict(call)})
                    data = {**data, "response": finish_body(body, req, request)}
                if "response" in data and kind not in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    data = {
                        **data,
                        "response": {**data["response"], **_config_echo(req)},
                    }
                yield event(data)
        except Exception as exc:
            yield event(
                {
                    "type": "response.failed",
                    "response": stream_failure(latest_response, req, request, exc),
                }
            )
        finally:
            close = getattr(response.body_iterator, "aclose", None)
            if close:
                await close()
            _REQUEST_TOOLS.reset(tools_token)
            _REQUEST_TOOL_USE.reset(use_token)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def stream_client_calls(response, req, request, defs, formats, raw_input):
    """Keep ordinary text streaming; adapt client calls without leaking their JSON envelopes."""
    from .server_tools.responses_loop import _parse_sse

    async def generate():
        from .routers.responses import _config_echo

        seq, pending, blocked = 0, {}, set()

        def event(data):
            nonlocal seq
            data = {**data, "sequence_number": seq}
            seq += 1
            return f"event: {data['type']}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        def prototype(item, declaration):
            args = (
                {"input": ""}
                if declaration["type"] == "custom"
                else {"actions": []}
                if declaration["type"] == "computer"
                else {"command": []}
                if declaration["type"] == "local_shell"
                else {}
            )
            return call_item(
                {**item, "arguments": json.dumps(args), "status": "in_progress"},
                declaration,
            )

        def full_input_events(item, idx, declaration):
            proto = prototype(item, declaration)
            return [
                {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "item": proto,
                },
                {
                    "type": "response.custom_tool_call_input.delta",
                    "output_index": idx,
                    "item_id": item["id"],
                    "delta": item["input"],
                },
                {
                    "type": "response.custom_tool_call_input.done",
                    "output_index": idx,
                    "item_id": item["id"],
                    "input": item["input"],
                },
                {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": item,
                },
            ]

        def add_usage(body, raw):
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                body["usage"][key] += raw["usage"][key]
            for detail in ("input_tokens_details", "output_tokens_details"):
                target = body["usage"].setdefault(detail, {})
                for key, value in raw["usage"].get(detail, {}).items():
                    if isinstance(value, int):
                        target[key] = target.get(key, 0) + value
            if raw.get("status") != "completed":
                body["status"] = raw["status"]
                for key in ("error", "incomplete_details"):
                    if raw.get(key):
                        body[key] = raw[key]

        latest_response = {}
        try:
            async for kind, data in _parse_sse(response.body_iterator):
                if kind == "__comment__":
                    yield data + "\n\n"
                    continue
                if isinstance(data.get("response"), dict):
                    latest_response = data["response"]
                idx = data.get("output_index")
                if (
                    kind == "response.output_item.added"
                    and data["item"].get("type") == "function_call"
                ):
                    item = data["item"]
                    declaration = defs.get(item.get("name"))
                    if declaration and declaration["type"] != "function":
                        pending[idx] = (item, declaration)
                        if formats.get(item["name"]):
                            blocked.add(idx)
                            continue
                        data = {**data, "item": prototype(item, declaration)}
                    elif declaration:
                        data = {**data, "item": call_item(item, declaration)}
                elif (
                    kind.startswith("response.function_call_arguments.")
                    and idx in pending
                ):
                    item, declaration = pending[idx]
                    if idx in blocked or declaration["type"] != "custom":
                        continue
                    if kind.endswith(".delta"):
                        continue  # JSON string escaping is decoded once arguments are complete.
                    converted = call_item(
                        {**item, "arguments": data["arguments"]}, declaration
                    )
                    coords = {"output_index": idx, "item_id": item["id"]}
                    yield event(
                        {
                            "type": "response.custom_tool_call_input.delta",
                            **coords,
                            "delta": converted["input"],
                        }
                    )
                    data = {
                        "type": "response.custom_tool_call_input.done",
                        **coords,
                        "input": converted["input"],
                    }
                elif (
                    kind == "response.output_item.done"
                    and data["item"].get("type") == "function_call"
                ):
                    if idx in blocked:
                        continue
                    declaration = defs.get(data["item"].get("name"))
                    if declaration:
                        data = {**data, "item": call_item(data["item"], declaration)}
                elif kind in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    body = data["response"]
                    for index, item in enumerate(body.get("output", [])):
                        if (
                            item.get("type") != "function_call"
                            or item.get("name") not in defs
                        ):
                            continue
                        declaration = defs[item["name"]]
                        converted = call_item(
                            {**item, "arguments": json.dumps({"input": ""})}
                            if formats.get(item["name"])
                            else item,
                            declaration,
                        )
                        if formats.get(item["name"]):
                            budget = (
                                req.max_output_tokens - body["usage"]["output_tokens"]
                            )
                            if budget <= 0 or body["status"] != "completed":
                                converted.update(input="", status="incomplete")
                                if budget <= 0:
                                    body.update(
                                        status="incomplete",
                                        incomplete_details={
                                            "reason": "max_output_tokens"
                                        },
                                    )
                                for ev in full_input_events(
                                    converted, index, declaration
                                ):
                                    yield event(ev)
                            else:
                                raw, text = await raw_input(
                                    item["name"], budget, body["id"], stream=True
                                )
                                if isinstance(raw, StreamingResponse):
                                    translated = stream_custom_input(
                                        raw,
                                        req.model_copy(update={"store": False}),
                                        request,
                                        declaration,
                                    )
                                    async for raw_kind, ev in _parse_sse(
                                        translated.body_iterator
                                    ):
                                        if raw_kind == "__comment__":
                                            yield ev + "\n\n"
                                            continue
                                        if raw_kind in (
                                            "response.created",
                                            "response.in_progress",
                                        ):
                                            continue
                                        if raw_kind in (
                                            "response.completed",
                                            "response.incomplete",
                                            "response.failed",
                                        ):
                                            raw_body = ev["response"]
                                            raw_call = next(
                                                (
                                                    o
                                                    for o in raw_body.get("output", [])
                                                    if o.get("type")
                                                    == "custom_tool_call"
                                                ),
                                                None,
                                            )
                                            if raw_call:
                                                converted = {
                                                    **raw_call,
                                                    "id": item["id"],
                                                    "call_id": item["call_id"],
                                                }
                                            else:
                                                converted.update(
                                                    input="", status="incomplete"
                                                )
                                            add_usage(body, raw_body)
                                            continue
                                        if "item" in ev:
                                            ev["item"] = {
                                                **ev["item"],
                                                "id": item["id"],
                                                "call_id": item["call_id"],
                                            }
                                        if "item_id" in ev:
                                            ev["item_id"] = item["id"]
                                        ev["output_index"] = index
                                        yield event(ev)
                                elif text is not None:
                                    converted.update(
                                        input=text,
                                        status="completed"
                                        if raw["status"] == "completed"
                                        else "incomplete",
                                    )
                                    add_usage(body, raw)
                                    for ev in full_input_events(
                                        converted, index, declaration
                                    ):
                                        yield event(ev)
                                else:
                                    body.update(
                                        status="failed",
                                        error={
                                            "message": "Custom tool input generation failed",
                                            "type": "server_error",
                                            "code": "server_error",
                                        },
                                    )
                                    converted.update(input="", status="incomplete")
                        body["output"][index] = converted
                    data = {
                        **data,
                        "type": "response." + body["status"],
                        "response": finish_body(body, req, request),
                    }
                if "response" in data and kind not in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    data = {
                        **data,
                        "response": {**data["response"], **_config_echo(req)},
                    }
                yield event(data)
        except Exception as exc:
            yield event(
                {
                    "type": "response.failed",
                    "response": stream_failure(latest_response, req, request, exc),
                }
            )
        finally:
            close = getattr(response.body_iterator, "aclose", None)
            if close:
                await close()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
