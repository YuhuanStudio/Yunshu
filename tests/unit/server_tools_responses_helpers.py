"""Scripted generation handler for the Responses server-tool tests: answers each round with
Responses SSE shaped like the engine's own stream (created, in_progress, output items, deltas,
completed with usage)."""

from __future__ import annotations

import json

from fastapi.responses import StreamingResponse


def sse(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


def text(t: str) -> dict:
    return {"kind": "text", "text": t}


def call(name: str, args: dict, call_id: str = "call_m1") -> dict:
    return {"kind": "call", "name": name, "args": args, "call_id": call_id}


def reasoning(t: str) -> dict:
    return {"kind": "reasoning", "text": t}


def response_events(
    items: list[dict], rid="resp-inner", status="completed"
) -> list[bytes]:
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 2},
        "output_tokens_details": {"reasoning_tokens": 1},
    }
    base = {"id": rid, "object": "response", "model": "m", "output": []}
    seq = [0]

    def ev(_n, **f):
        seq[0] += 1
        return sse(_n, {"type": _n, **f, "sequence_number": seq[0]})

    out = [
        ev("response.created", response={**base, "status": "in_progress"}),
        ev("response.in_progress", response={**base, "status": "in_progress"}),
    ]
    done_items = []
    for i, it in enumerate(items):
        if it["kind"] == "reasoning":
            item = {
                "type": "reasoning",
                "id": f"rs_{i}",
                "summary": [{"type": "summary_text", "text": it["text"]}],
            }
            out.append(
                ev(
                    "response.output_item.added",
                    output_index=i,
                    item={**item, "summary": [], "status": "in_progress"},
                )
            )
            out.append(ev("response.output_item.done", output_index=i, item=item))
        elif it["kind"] == "text":
            mid = f"msg_{i}"
            out.append(
                ev(
                    "response.output_item.added",
                    output_index=i,
                    item={
                        "type": "message",
                        "id": mid,
                        "role": "assistant",
                        "content": [],
                        "status": "in_progress",
                    },
                )
            )
            part = {"type": "output_text", "text": "", "annotations": []}
            out.append(
                ev(
                    "response.content_part.added",
                    item_id=mid,
                    output_index=i,
                    content_index=0,
                    part=part,
                )
            )
            out.append(
                ev(
                    "response.output_text.delta",
                    item_id=mid,
                    output_index=i,
                    content_index=0,
                    delta=it["text"],
                    logprobs=[],
                )
            )
            out.append(
                ev(
                    "response.output_text.done",
                    item_id=mid,
                    output_index=i,
                    content_index=0,
                    text=it["text"],
                    logprobs=[],
                )
            )
            full = {"type": "output_text", "text": it["text"], "annotations": []}
            out.append(
                ev(
                    "response.content_part.done",
                    item_id=mid,
                    output_index=i,
                    content_index=0,
                    part=full,
                )
            )
            item = {
                "type": "message",
                "id": mid,
                "role": "assistant",
                "content": [full],
                "status": "completed",
            }
            out.append(ev("response.output_item.done", output_index=i, item=item))
        else:
            args = json.dumps(it["args"])
            fid = f"fc_{i}"
            out.append(
                ev(
                    "response.output_item.added",
                    output_index=i,
                    item={
                        "type": "function_call",
                        "id": fid,
                        "call_id": it["call_id"],
                        "name": it["name"],
                        "arguments": "",
                        "status": "in_progress",
                    },
                )
            )
            out.append(
                ev(
                    "response.function_call_arguments.delta",
                    item_id=fid,
                    output_index=i,
                    delta=args,
                )
            )
            out.append(
                ev(
                    "response.function_call_arguments.done",
                    item_id=fid,
                    output_index=i,
                    name=it["name"],
                    arguments=args,
                )
            )
            item = {
                "type": "function_call",
                "id": fid,
                "call_id": it["call_id"],
                "name": it["name"],
                "arguments": args,
                "status": "completed",
            }
            out.append(ev("response.output_item.done", output_index=i, item=item))
        done_items.append(item)
    out.append(
        ev(
            "response.completed",
            response={**base, "status": status, "output": done_items, "usage": usage},
        )
    )
    out.append(b"data: [DONE]\n\n")
    return out


class ScriptedResponses:
    """Stands in for ``create_response``: round N answers with ``rounds[N]`` (a list of items)."""

    def __init__(self, rounds: list[list[dict]]):
        self.rounds = list(rounds)
        self.requests: list = []
        self.forced_ids: list = []

    async def __call__(self, req, request):
        self.requests.append(req)
        self.forced_ids.append(getattr(request.state, "_forced_response_id", None))
        items = self.rounds.pop(0) if self.rounds else [text("done")]

        async def gen():
            for e in response_events(items):
                yield e

        return StreamingResponse(gen(), media_type="text/event-stream")


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    out = []
    for block in raw.split("\n\n"):
        name, data = None, None
        for ln in block.split("\n"):
            if ln.startswith("event:"):
                name = ln[6:].strip()
            elif ln.startswith("data:") and ln[5:].strip() != "[DONE]":
                data = json.loads(ln[5:])
        if name and data is not None:
            out.append((name, data))
    return out
