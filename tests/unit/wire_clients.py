"""Per-dialect client adapters for the wire-contract tests.

Each adapter drives the official SDK (openai, anthropic) or, for Ollama, raw HTTP against the
real router, and folds the response into one canonical ``Out`` so the same scripted generation
can be compared across dialects, stream against non-stream. The raw event/chunk sequence is kept
in ``Out.events`` for ordering assertions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai

MODEL = "scripted"
DIALECTS = [
    "chat",
    "completions",
    "messages",
    "responses",
    "ollama_chat",
    "ollama_generate",
]


@dataclass
class Out:
    text: str = ""
    thinking: str = ""
    tools: list[tuple[str, dict]] = field(default_factory=list)
    finish: str | None = None  # canonical: stop | length | tool_calls
    prompt: int | None = None  # total prompt tokens, cached included
    completion: int | None = None
    reasoning: int | None = None
    cached: int | None = None
    events: list[str] = field(default_factory=list)
    raw: Any = None


def tool_text(name: str, args: dict) -> str:
    return (
        "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"
    )


def chunks(text: str, n: int = 4) -> list[str]:
    return [text[i : i + n] for i in range(0, len(text), n)]


WEATHER = {
    "name": "get_weather",
    "description": "weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}},
    "required": ["a"],
}


class Clients:
    def __init__(self, http):
        self.http = http
        self.oa = openai.OpenAI(
            base_url="http://testserver/v1",
            api_key="x",
            http_client=http,
            max_retries=0,
        )
        self.an = anthropic.Anthropic(
            base_url="http://testserver", api_key="x", http_client=http, max_retries=0
        )


_FIN_CHAT = {"stop": "stop", "length": "length", "tool_calls": "tool_calls"}
_FIN_ANTH = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


def _tool_choice_chat(tc):
    if tc in (None, "auto", "none", "required"):
        return tc
    return {"type": "function", "function": {"name": tc}}


def run(cl: Clients, dialect: str, *, stream: bool, **kw) -> Out:
    return globals()[f"_run_{dialect}"](cl, stream=stream, **kw)


# ── OpenAI chat ──────────────────────────────────────────────────────────────────
def _run_chat(
    cl,
    *,
    stream,
    max_tokens=None,
    stop=None,
    tools=False,
    tool_choice=None,
    schema=False,
    include_usage=True,
    n_prompt="hi",
    parallel=None,
    **_,
):
    args: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": n_prompt}],
    }
    if max_tokens is not None:
        args["max_tokens"] = max_tokens
    if stop:
        args["stop"] = stop
    if tools:
        args["tools"] = [{"type": "function", "function": WEATHER}]
        if tool_choice:
            args["tool_choice"] = _tool_choice_chat(tool_choice)
        if parallel is not None:
            args["parallel_tool_calls"] = parallel
    if schema:
        args["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "s", "schema": SCHEMA, "strict": True},
        }
    out = Out()
    if not stream:
        r = cl.oa.chat.completions.create(**args)
        out.raw = r
        ch = r.choices[0]
        out.text = ch.message.content or ""
        out.thinking = getattr(ch.message, "reasoning_content", None) or ""
        for tc in ch.message.tool_calls or []:
            out.tools.append((tc.function.name, json.loads(tc.function.arguments)))
        out.finish = _FIN_CHAT.get(ch.finish_reason, ch.finish_reason)
        _chat_usage(out, r.usage)
        return out
    if include_usage:
        args["stream_options"] = {"include_usage": True}
    acc: dict[int, dict] = {}
    for ev in cl.oa.chat.completions.create(stream=True, **args):
        out.raw = out.raw or []
        out.raw.append(ev)
        out.events.append("usage" if not ev.choices and ev.usage else "chunk")
        if ev.usage:
            _chat_usage(out, ev.usage)
        for ch in ev.choices:
            d = ch.delta
            out.text += d.content or ""
            out.thinking += getattr(d, "reasoning_content", None) or ""
            for tc in d.tool_calls or []:
                a = acc.setdefault(tc.index, {"name": "", "args": ""})
                a["name"] += tc.function.name or "" if tc.function else ""
                a["args"] += (tc.function.arguments or "") if tc.function else ""
            if ch.finish_reason:
                out.finish = _FIN_CHAT.get(ch.finish_reason, ch.finish_reason)
    for k in sorted(acc):
        out.tools.append((acc[k]["name"], json.loads(acc[k]["args"] or "{}")))
    return out


def _chat_usage(out, u):
    out.prompt = u.prompt_tokens
    out.completion = u.completion_tokens
    cd = u.completion_tokens_details
    out.reasoning = cd.reasoning_tokens if cd else None
    pd = u.prompt_tokens_details
    out.cached = pd.cached_tokens if pd else None


# ── OpenAI completions ───────────────────────────────────────────────────────────
def _run_completions(
    cl, *, stream, max_tokens=None, stop=None, include_usage=True, **_
):
    args: dict[str, Any] = {"model": MODEL, "prompt": "hi"}
    if max_tokens is not None:
        args["max_tokens"] = max_tokens
    if stop:
        args["stop"] = stop
    out = Out()
    if not stream:
        r = cl.oa.completions.create(**args)
        out.text = r.choices[0].text
        out.finish = _FIN_CHAT.get(r.choices[0].finish_reason)
        _chat_usage(out, r.usage)
        return out
    if include_usage:
        args["stream_options"] = {"include_usage": True}
    for ev in cl.oa.completions.create(stream=True, **args):
        out.events.append("usage" if not ev.choices and ev.usage else "chunk")
        if ev.usage:
            _chat_usage(out, ev.usage)
        for ch in ev.choices:
            out.text += ch.text or ""
            if ch.finish_reason:
                out.finish = _FIN_CHAT.get(ch.finish_reason)
    return out


# ── Anthropic messages ───────────────────────────────────────────────────────────
def _run_messages(
    cl,
    *,
    stream,
    max_tokens=None,
    stop=None,
    tools=False,
    tool_choice=None,
    schema=False,
    parallel=None,
    **_,
):
    args: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": max_tokens if max_tokens is not None else 256,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if stop:
        args["stop_sequences"] = stop
    if tools:
        args["tools"] = [
            {
                "name": WEATHER["name"],
                "description": "weather",
                "input_schema": WEATHER["parameters"],
            }
        ]
        if tool_choice or parallel is False:
            tc = (
                {"type": "any"}
                if tool_choice == "required"
                else {"type": tool_choice}
                if tool_choice in ("auto", "none")
                else {"type": "tool", "name": tool_choice}
                if tool_choice
                else {"type": "auto"}
            )
            if parallel is False and tc["type"] != "none":
                tc["disable_parallel_tool_use"] = True
            args["tool_choice"] = tc
    out = Out()
    if not stream:
        r = cl.an.messages.create(**args)
        out.raw = r
        for b in r.content:
            if b.type == "text":
                out.text += b.text
            elif b.type == "thinking":
                out.thinking += b.thinking
            elif b.type == "tool_use":
                out.tools.append((b.name, b.input))
        out.finish = _FIN_ANTH.get(r.stop_reason, r.stop_reason)
        out.raw_stop_reason = r.stop_reason  # type: ignore[attr-defined]
        out.raw_stop_sequence = r.stop_sequence  # type: ignore[attr-defined]
        _anth_usage(out, r.usage)
        return out
    partial: dict[int, str] = {}
    names: dict[int, str] = {}
    last_usage = None
    for ev in cl.an.messages.create(stream=True, **args):
        out.events.append(ev.type)
        out.raw = out.raw or []
        out.raw.append(ev)
        if ev.type == "message_start":
            out.start_usage = ev.message.usage  # type: ignore[attr-defined]
        elif ev.type == "content_block_start":
            if ev.content_block.type == "tool_use":
                names[ev.index] = ev.content_block.name
                partial[ev.index] = ""
        elif ev.type == "content_block_delta":
            d = ev.delta
            if d.type == "text_delta":
                out.text += d.text
            elif d.type == "thinking_delta":
                out.thinking += d.thinking
            elif d.type == "input_json_delta":
                partial[ev.index] += d.partial_json
        elif ev.type == "message_delta":
            out.raw_stop_reason = ev.delta.stop_reason  # type: ignore[attr-defined]
            out.raw_stop_sequence = ev.delta.stop_sequence  # type: ignore[attr-defined]
            out.finish = _FIN_ANTH.get(ev.delta.stop_reason, ev.delta.stop_reason)
            last_usage = ev.usage
    for k in sorted(names):
        out.tools.append((names[k], json.loads(partial[k] or "{}")))
    out.delta_usage = last_usage  # type: ignore[attr-defined]
    # What the SDK's own accumulator makes of message_start + message_delta.
    start = out.start_usage  # type: ignore[attr-defined]
    out.prompt = (start.input_tokens or 0) + (start.cache_read_input_tokens or 0)
    if last_usage is not None:
        out.completion = last_usage.output_tokens
        d = getattr(last_usage, "output_tokens_details", None)
        out.reasoning = getattr(d, "reasoning_tokens", None) if d else None
        if last_usage.input_tokens is not None:
            out.prompt = (last_usage.input_tokens or 0) + (
                last_usage.cache_read_input_tokens or 0
            )
    out.cached = start.cache_read_input_tokens or 0
    return out


def _anth_usage(out, u):
    out.prompt = (u.input_tokens or 0) + (u.cache_read_input_tokens or 0)
    out.completion = u.output_tokens
    d = getattr(u, "output_tokens_details", None)
    out.reasoning = getattr(d, "reasoning_tokens", None) if d else None
    out.cached = u.cache_read_input_tokens or 0


# ── OpenAI responses ─────────────────────────────────────────────────────────────
def _run_responses(
    cl,
    *,
    stream,
    max_tokens=None,
    stop=None,
    tools=False,
    tool_choice=None,
    schema=False,
    parallel=None,
    **_,
):
    args: dict[str, Any] = {"model": MODEL, "input": "hi"}
    if max_tokens is not None:
        args["max_output_tokens"] = max_tokens
    if stop:
        args["extra_body"] = {"stop": stop}
    if tools:
        args["tools"] = [{"type": "function", **WEATHER}]
        if tool_choice:
            args["tool_choice"] = (
                tool_choice
                if tool_choice in ("auto", "none", "required")
                else {"type": "function", "name": tool_choice}
            )
        if parallel is not None:
            args["parallel_tool_calls"] = parallel
    if schema:
        args["text"] = {
            "format": {
                "type": "json_schema",
                "name": "s",
                "schema": SCHEMA,
                "strict": True,
            }
        }
    out = Out()
    if not stream:
        r = cl.oa.responses.create(**args)
        out.raw = r
        _resp_fold(out, r)
        return out
    final = None
    for ev in cl.oa.responses.create(stream=True, **args):
        out.events.append(ev.type)
        out.raw = out.raw or []
        out.raw.append(ev)
        if ev.type == "response.output_text.delta":
            out.text += ev.delta
        elif ev.type == "response.reasoning_summary_text.delta":
            out.thinking += ev.delta
        elif ev.type in (
            "response.completed",
            "response.incomplete",
            "response.failed",
        ):
            final = ev.response
    assert final is not None, out.events
    _resp_fold(out, final, stream=True)
    return out


def _resp_fold(out, r, stream=False):
    saw_tool = False
    for item in r.output:
        if item.type == "message":
            if not stream:
                out.text += "".join(
                    c.text for c in item.content if c.type == "output_text"
                )
        elif item.type == "reasoning":
            if not stream:
                out.thinking += "".join(s.text for s in (item.summary or []))
        elif item.type == "function_call":
            saw_tool = True
            out.tools.append((item.name, json.loads(item.arguments or "{}")))
    out.raw_status = r.status  # type: ignore[attr-defined]
    out.raw_incomplete = r.incomplete_details  # type: ignore[attr-defined]
    if r.status == "incomplete":
        out.finish = "length"
    else:
        out.finish = "tool_calls" if saw_tool else "stop"
    u = r.usage
    if u is not None:
        out.prompt = u.input_tokens
        out.completion = u.output_tokens
        d = u.output_tokens_details
        out.reasoning = d.reasoning_tokens if d else None
        i = u.input_tokens_details
        out.cached = i.cached_tokens if i else None


# ── Ollama (raw HTTP, NDJSON) ────────────────────────────────────────────────────
def _ollama(
    cl,
    path,
    *,
    stream,
    max_tokens=None,
    stop=None,
    tools=False,
    tool_choice=None,
    schema=False,
    **_,
):
    body: dict[str, Any] = {"model": MODEL, "stream": stream}
    if path == "/api/chat":
        body["messages"] = [{"role": "user", "content": "hi"}]
    else:
        body["prompt"] = "hi"
    opts: dict[str, Any] = {}
    if max_tokens is not None:
        opts["num_predict"] = max_tokens
    if stop:
        opts["stop"] = stop
    if opts:
        body["options"] = opts
    if tools and path == "/api/chat":
        body["tools"] = [{"type": "function", "function": WEATHER}]
    if schema:
        body["format"] = SCHEMA
    out = Out()
    r = cl.http.post(path, json=body)
    assert r.status_code == 200, r.text
    lines = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    if not stream:
        assert len(lines) == 1
    out.raw = lines
    for ln in lines:
        msg = ln.get("message") or {}
        out.events.append("done" if ln.get("done") else "chunk")
        out.text += msg.get("content", "") or ln.get("response", "")
        out.thinking += msg.get("thinking", "") or ln.get("thinking", "")
        for tc in msg.get("tool_calls") or []:
            out.tools.append((tc["function"]["name"], tc["function"]["arguments"]))
        if ln.get("done"):
            dr = ln.get("done_reason")
            out.finish = "tool_calls" if out.tools and dr == "stop" else dr
            out.prompt = ln.get("prompt_eval_count")
            out.completion = ln.get("eval_count")
    return out


def _run_ollama_chat(cl, **kw):
    return _ollama(cl, "/api/chat", **kw)


def _run_ollama_generate(cl, **kw):
    return _ollama(cl, "/api/generate", **kw)
