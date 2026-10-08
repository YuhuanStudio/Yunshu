"""Responses freeform tools over the engine's native function-call transport."""

from __future__ import annotations

import codecs
import contextlib
import json

from fastapi import HTTPException


def custom_names(tools):
    names = set()
    for tool in tools or []:
        d = tool.model_dump() if hasattr(tool, "model_dump") else tool
        if d.get("type") == "namespace":
            names.update(custom_names(d.get("tools")))
        elif d.get("type") == "custom":
            if not d.get("name"):
                raise HTTPException(400, "custom tool name is required")
            fmt = d.get("format") or {"type": "text"}
            if not isinstance(fmt, dict) or fmt.get("type") != "text":
                raise HTTPException(
                    400,
                    "Custom tool grammar formats cannot be guaranteed; use format.type='text'",
                )
            names.add(d["name"])
    return names


def custom_function(tool):
    d = tool.model_dump() if hasattr(tool, "model_dump") else tool
    custom_names([d])
    return {
        "type": "function",
        "name": d["name"],
        "description": (d.get("description") or "")
        + "\nSend the exact freeform tool input in the input string parameter.",
        "parameters": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
            "additionalProperties": False,
        },
    }


def custom_item(item, names):
    if item.get("type") != "function_call" or item.get("name") not in names:
        return item
    args = item.get("arguments") or ""
    if isinstance(args, str):
        with contextlib.suppress(ValueError):
            args = json.loads(args)
    if isinstance(args, dict):
        # The outer parameter name is a private function transport detail.
        # An XML model may use 'patch' instead of 'input' for a freeform patch.
        values = list(args.values())
        args = (
            args.get("input")
            if "input" in args
            else (values[0] if len(values) == 1 else None)
        )
    if not isinstance(args, str):
        raise ValueError("Custom tool call must contain one text input")
    return {k: v for k, v in item.items() if k != "arguments"} | {
        "type": "custom_tool_call",
        "input": args,
    }


async def custom_stream(source, names):
    """Translate function transport events to the official custom input events."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    items = {}
    seq = 0

    def encode(event):
        nonlocal seq
        seq += 1
        event = {**event, "sequence_number": seq}
        return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()

    async for chunk in source:
        buffer += decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            data = "\n".join(
                line[6:] for line in block.splitlines() if line.startswith("data: ")
            )
            if not data or data == "[DONE]":
                yield (block + "\n\n").encode()
                continue
            event = json.loads(data)
            ty = event.get("type")
            if ty == "response.output_item.added":
                item = custom_item(event["item"], names)
                if item.get("type") == "custom_tool_call":
                    items[item["id"]] = item
                    event = {**event, "item": {**item, "input": ""}}
                    yield encode(event)
                    yield encode(
                        {
                            "type": "response.custom_tool_call_input.delta",
                            "item_id": item["id"],
                            "output_index": event["output_index"],
                            "delta": item["input"],
                        }
                    )
                    continue
            elif (
                ty == "response.function_call_arguments.done"
                and event.get("item_id") in items
            ):
                item = items[event["item_id"]]
                event = {
                    "type": "response.custom_tool_call_input.done",
                    "item_id": item["id"],
                    "output_index": event["output_index"],
                    "input": item["input"],
                }
            elif ty == "response.output_item.done":
                event = {**event, "item": custom_item(event["item"], names)}
            elif isinstance(event.get("response"), dict):
                response = dict(event["response"])
                if isinstance(response.get("output"), list):
                    response["output"] = [
                        custom_item(i, names) for i in response["output"]
                    ]
                event = {**event, "response": response}
            yield encode(event)
    buffer += decoder.decode(b"", final=True)
    if buffer:
        yield buffer.encode()
