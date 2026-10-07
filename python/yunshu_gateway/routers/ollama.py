"""Ollama-compatible API (`/api/*`): a thin translation layer over the OpenAI routes.

Ollama clients (the official `ollama` SDKs, Open WebUI, Continue, ...) speak NDJSON
`/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, `/api/ps`, `/api/version`.
Each request is translated to the matching OpenAI request and sent back to this same server over
loopback (so streaming is real, and auth, model routing, prefix cache, tools and constrained
decoding behave exactly as on the OpenAI routes). Model management supports native MLX
repositories and persistent names without duplicating checkpoints; registry uploads and
GGUF conversion are unsupported.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from yunshu_engine.version import yunshu_version

router = APIRouter(prefix="/api", tags=["ollama"])

_NS = 1_000_000_000


class OllamaError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _client(request: Request) -> httpx.AsyncClient:
    headers = {}
    rid = getattr(request.state, "request_id", None)
    if (
        rid
    ):  # same id on the loopback call: cancel / DELETE /v1/requests/{id} still work
        headers["X-Request-Id"] = rid
    if request.headers.get("authorization"):
        headers["Authorization"] = request.headers["authorization"]
    base = f"{request.url.scheme}://{request.url.netloc}"
    return httpx.AsyncClient(base_url=base, headers=headers, timeout=None)


def _err_from(resp_text: str, status: int) -> OllamaError:
    try:
        j = json.loads(resp_text)
        e = j.get("error", j)
        msg = e.get("message") if isinstance(e, dict) else str(e)
    except Exception:
        msg = resp_text[:300]
    return OllamaError(status, msg or "request failed")


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        raise OllamaError(400, "invalid JSON body") from None
    if not isinstance(body, dict):
        raise OllamaError(400, "request body must be a JSON object")
    return body


def _model_name(body: dict) -> str:
    m = body.get("model") or body.get("name")
    if not m or not isinstance(m, str):
        raise OllamaError(400, "model is required")
    return m


def _details(model_id: str) -> dict:
    low = model_id.lower()
    quant = next((q for q in ("4bit", "8bit", "bf16", "fp16", "mxfp4") if q in low), "")
    return {
        "parent_model": "",
        "format": "safetensors",
        "family": low.split("-")[0],
        "families": [low.split("-")[0]],
        "parameter_size": "",
        "quantization_level": quant.upper() if quant else "",
    }


def _card_details(m: dict) -> tuple[dict, int]:
    """(details, size in bytes) from a `/v1/models` item; name heuristics when it has no card."""
    card = m.get("yunshu")
    if not isinstance(card, dict):
        return _details(m["id"]), 0
    from ..model_card_formats import ollama_show

    return ollama_show(card)["details"], int(
        (card.get("memory") or {}).get("weights_bytes") or 0
    )


def _digest(model_id: str) -> str:
    return hashlib.sha256(model_id.encode()).hexdigest()


def _sampling(options: dict | None, fmt: Any, body: dict) -> dict:
    """Map Ollama `options` / `format` / `think` / `keep_alive` onto OpenAI-route fields."""
    o = options or {}
    out: dict[str, Any] = {}
    for src, dst in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("min_p", "min_p"),
        ("seed", "seed"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
        ("repeat_penalty", "repetition_penalty"),
    ):
        if o.get(src) is not None:
            out[dst] = o[src]
    n = o.get("num_predict")
    if isinstance(n, int) and n > 0:
        out["max_tokens"] = n
    elif n in (-1, -2):
        out["max_tokens"] = 8192
    else:
        out["max_tokens"] = 8192
    stop = o.get("stop")
    if stop:
        out["stop"] = [stop] if isinstance(stop, str) else list(stop)
    if fmt == "json":
        out["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        out["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": fmt, "strict": True},
        }
    think = body.get("think")
    if isinstance(think, bool):
        out["enable_thinking"] = think
    elif isinstance(think, str):  # "low" | "medium" | "high"
        out["reasoning_effort"] = think
    return out


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        imgs = m.get("images") or []
        msg: dict[str, Any] = {"role": role}
        if imgs:
            parts: list[dict] = [{"type": "text", "text": content}]
            for img in imgs:
                if isinstance(img, (bytes, bytearray)):
                    img = base64.b64encode(img).decode()
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    }
                )
            msg["content"] = parts
        else:
            msg["content"] = content
        if m.get("tool_calls"):
            msg["tool_calls"] = [
                {
                    "id": tc.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.dumps(tc["function"].get("arguments") or {}),
                    },
                }
                for i, tc in enumerate(m["tool_calls"])
            ]
        if role == "tool":
            msg["tool_call_id"] = m.get("tool_call_id") or "call_0"
            if m.get("tool_name") or m.get("name"):
                msg["name"] = m.get("tool_name") or m.get("name")
        out.append(msg)
    return out


def _tool_calls_out(tcs: list[dict] | None) -> list[dict]:
    res = []
    for tc in tcs or []:
        fn = tc.get("function", {})
        try:
            raw = fn.get("arguments")
            args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        except Exception:
            args = {}
        res.append({"function": {"name": fn.get("name"), "arguments": args}})
    return res


def _done_reason(fr: str | None) -> str:
    return {"length": "length", "stop": "stop", "tool_calls": "stop"}.get(
        fr or "stop", fr or "stop"
    )


def _timing(
    t0: float, usage: dict | None, t_first: float | None, xy: dict | None = None
) -> dict:
    """Ollama's own timing fields (ns). Engine-side ``x_yunshu`` numbers win when the
    upstream sent them; otherwise the gateway clock is used."""
    xy = xy or {}

    def ns(ms: float | None) -> int | None:
        return None if ms is None else int(ms * 1_000_000)

    total = ns(xy.get("total_ms"))
    if total is None:
        total = int((time.perf_counter() - t0) * _NS)
    pe = ns(xy.get("prefill_ms"))
    if pe is None:
        pe = ns(xy.get("ttft_ms"))
    if pe is None:
        pe = int(((t_first or time.perf_counter()) - t0) * _NS)
    ev = ns(xy.get("decode_ms"))
    if ev is None:
        ev = max(total - pe, 0)
    u = usage or {}
    return {
        "total_duration": total,
        "load_duration": ns(xy.get("load_ms")) or 0,
        "prompt_eval_count": xy.get("prompt_tokens") or u.get("prompt_tokens", 0),
        "prompt_eval_duration": pe,
        "eval_count": xy.get("completion_tokens") or u.get("completion_tokens", 0),
        "eval_duration": ev,
    }


async def _sse(resp: httpx.Response) -> AsyncIterator[dict]:
    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except Exception:
            continue


def _ndjson(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode()


async def _run_chat(request: Request, body: dict, *, generate: bool):
    """Shared by /api/chat and /api/generate: returns a JSONResponse or StreamingResponse."""
    model = _model_name(body)
    stream = body.get("stream", True)
    if generate:
        msgs: list[dict] = []
        if body.get("system"):
            msgs.append({"role": "system", "content": body["system"]})
        msgs.append(
            {
                "role": "user",
                "content": body.get("prompt", ""),
                **({"images": body["images"]} if body.get("images") else {}),
            }
        )
    else:
        msgs = body.get("messages")
        if not isinstance(msgs, list):
            raise OllamaError(400, "messages is required")
    oa: dict[str, Any] = {
        "model": model,
        "messages": _to_openai_messages(msgs),
        "stream": bool(stream),
        **_sampling(body.get("options"), body.get("format"), body),
    }
    if stream:
        oa["stream_options"] = {"include_usage": True}
    if body.get("tools"):
        oa["tools"] = body["tools"]
    if body.get("keep_alive") is not None:
        oa["keep_alive"] = body["keep_alive"]
    t0 = time.perf_counter()
    client = _client(request)

    def shape(
        content: str, thinking: str, tcs: list[dict], done: bool, extra: dict
    ) -> dict:
        if generate:
            r = {
                "model": model,
                "created_at": _now(),
                "response": content,
                "done": done,
            }
            if thinking:
                r["thinking"] = thinking
        else:
            m: dict[str, Any] = {"role": "assistant", "content": content}
            if thinking:
                m["thinking"] = thinking
            if tcs:
                m["tool_calls"] = tcs
            r = {"model": model, "created_at": _now(), "message": m, "done": done}
        r.update(extra)
        return r

    if not stream:
        try:
            resp = await client.post("/v1/chat/completions", json=oa)
        finally:
            await client.aclose()
        if resp.status_code != 200:
            raise _err_from(resp.text, resp.status_code)
        j = resp.json()
        ch = j["choices"][0]
        msg = ch["message"]
        extra = {
            "done_reason": _done_reason(ch.get("finish_reason")),
            **_timing(t0, j.get("usage"), None, j.get("x_yunshu")),
        }
        return JSONResponse(
            shape(
                msg.get("content") or "",
                msg.get("reasoning_content") or "",
                _tool_calls_out(msg.get("tool_calls")),
                True,
                extra,
            )
        )

    # Streaming: open the upstream first so a 4xx/5xx becomes a proper error status.
    req = client.build_request("POST", "/v1/chat/completions", json=oa)
    resp = await client.send(req, stream=True)
    if resp.status_code != 200:
        text = (await resp.aread()).decode(errors="replace")
        await resp.aclose()
        await client.aclose()
        raise _err_from(text, resp.status_code)

    async def gen() -> AsyncIterator[bytes]:
        usage = None
        xy = None
        finish = None
        t_first = None
        tool_acc: dict[int, dict] = {}
        try:
            async for ev in _sse(resp):
                if ev.get("error"):
                    # The chat stream failed after its headers: Ollama's terminal error is a
                    # {"error": ...} line, never a "done" line that reads as a clean finish.
                    err = ev["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    yield _ndjson({"error": msg or "generation failed"})
                    return
                if ev.get("usage"):
                    usage = ev["usage"]
                if ev.get("x_yunshu"):
                    xy = ev["x_yunshu"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    for tc in d.get("tool_calls") or []:
                        acc = tool_acc.setdefault(
                            tc.get("index", 0),
                            {"function": {"name": "", "arguments": ""}},
                        )
                        f = tc.get("function") or {}
                        if f.get("name"):
                            acc["function"]["name"] = f["name"]
                        acc["function"]["arguments"] += f.get("arguments") or ""
                    content = d.get("content") or ""
                    thinking = d.get("reasoning_content") or ""
                    if content or thinking:
                        t_first = t_first or time.perf_counter()
                        yield _ndjson(shape(content, thinking, [], False, {}))
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
            tcs = _tool_calls_out([tool_acc[k] for k in sorted(tool_acc)])
            extra = {
                "done_reason": _done_reason(finish),
                **_timing(t0, usage, t_first, xy),
            }
            yield _ndjson(shape("", "", tcs, True, extra))
        except Exception as e:  # noqa: BLE001
            yield _ndjson({"error": str(e)})
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _wrap(fn):
    async def inner(request: Request):
        try:
            return await fn(request)
        except OllamaError as e:
            return error_response(e.status, e.message)
        except HTTPException as e:
            return error_response(e.status_code, str(e.detail))

    inner.__name__ = fn.__name__
    return inner


@router.post("/chat")
@_wrap
async def chat(request: Request):
    return await _run_chat(request, await _json_body(request), generate=False)


@router.post("/generate")
@_wrap
async def generate(request: Request):
    body = await _json_body(request)
    if not body.get("prompt") and not body.get("images"):
        # Ollama: an empty prompt just loads the model.
        _model_name(body)
        return JSONResponse(
            {
                "model": body["model"],
                "created_at": _now(),
                "response": "",
                "done": True,
                "done_reason": "load",
            }
        )
    return await _run_chat(request, body, generate=True)


@router.post("/embed")
@_wrap
async def embed(request: Request):
    body = await _json_body(request)
    model = _model_name(body)
    inp = body.get("input")
    if inp is None:
        raise OllamaError(400, "input is required")
    t0 = time.perf_counter()
    async with _client(request) as c:
        payload: dict[str, Any] = {"model": model, "input": inp}
        if body.get("dimensions"):
            payload["dimensions"] = body["dimensions"]
        resp = await c.post("/v1/embeddings", json=payload)
    if resp.status_code != 200:
        raise _err_from(resp.text, resp.status_code)
    j = resp.json()
    return {
        "model": model,
        "embeddings": [d["embedding"] for d in j["data"]],
        "total_duration": int((time.perf_counter() - t0) * _NS),
        "load_duration": 0,
        "prompt_eval_count": j.get("usage", {}).get("prompt_tokens", 0),
    }


@router.post("/embeddings")
@_wrap
async def embeddings_legacy(request: Request):
    body = await _json_body(request)
    model = _model_name(body)
    async with _client(request) as c:
        resp = await c.post(
            "/v1/embeddings", json={"model": model, "input": body.get("prompt", "")}
        )
    if resp.status_code != 200:
        raise _err_from(resp.text, resp.status_code)
    return {"embedding": resp.json()["data"][0]["embedding"]}


async def _list_models(request: Request) -> list[dict]:
    async with _client(request) as c:
        resp = await c.get("/v1/models")
    if resp.status_code != 200:
        raise _err_from(resp.text, resp.status_code)
    return resp.json().get("data", [])


@router.get("/tags")
@_wrap
async def tags(request: Request):
    out = []
    for m in await _list_models(request):
        mid = m["id"]
        details, size = _card_details(m)
        out.append(
            {
                "name": mid,
                "model": mid,
                "modified_at": datetime.fromtimestamp(
                    m.get("created") or time.time(), UTC
                )
                .isoformat()
                .replace("+00:00", "Z"),
                "size": size,
                "digest": _digest(mid),
                "details": details,
            }
        )
    return {"models": out}


@router.get("/ps")
@_wrap
async def ps(request: Request):
    from ..engine import get_model_manager

    manager = get_model_manager()
    loaded = (
        {e.model_id for e in manager.list_entries() if e.is_loaded} if manager else None
    )
    out = []
    for m in await _list_models(request):
        if loaded is not None and m["id"] not in loaded:
            continue
        if m.get("loaded") is False or m.get("state") == "not-loaded":
            continue
        mid = m["id"]
        details, size = _card_details(m)
        out.append(
            {
                "name": mid,
                "model": mid,
                "size": size,
                "digest": _digest(mid),
                "details": details,
                "expires_at": "0001-01-01T00:00:00Z",
                "size_vram": size,
            }
        )
    return {"models": out}


@router.post("/show")
@_wrap
async def show(request: Request):
    body = await _json_body(request)
    model = _model_name(body)
    base = model.removesuffix(":latest")
    found = [m for m in await _list_models(request) if m["id"] in (model, base)]
    if not found:
        raise OllamaError(404, f"model '{model}' not found")
    mid = found[0]["id"]
    card = found[0].get("yunshu")
    if not isinstance(card, dict):  # upstream without model cards
        return {
            "modelfile": f"FROM {mid}\n",
            "parameters": "",
            "template": "",
            "details": _details(mid),
            "model_info": {"general.architecture": _details(mid)["family"]},
            "capabilities": ["embedding"] if "embed" in mid.lower() else ["completion"],
            "modified_at": _now(),
        }
    from ..model_card_formats import ollama_show

    out = ollama_show(card)
    out["modified_at"] = _now()
    return out


@router.get("/version")
async def version():
    return {"version": yunshu_version()}


@router.post("/copy")
@_wrap
async def copy(request: Request):
    from ..ollama_models import copy_model

    body = await _json_body(request)
    source, destination = body.get("source"), body.get("destination")
    if (
        not isinstance(source, str)
        or not source
        or not isinstance(destination, str)
        or not destination
    ):
        raise OllamaError(400, "source and destination are required")
    await copy_model(request, source, destination)
    return Response(status_code=200)


@router.delete("/delete")
@_wrap
async def delete(request: Request):
    from ..ollama_models import delete_model

    await delete_model(request, _model_name(await _json_body(request)))
    return Response(status_code=200)


def _status_success(body):
    if body.get("stream", True):

        async def events():
            yield _ndjson({"status": "success"})

        return StreamingResponse(events(), media_type="application/x-ndjson")
    return {"status": "success"}


@router.post("/create")
@_wrap
async def create(request: Request):
    from ..ollama_models import copy_model

    body = await _json_body(request)
    model = _model_name(body)
    source = body.get("from")
    if not isinstance(source, str) or not source:
        raise OllamaError(
            400, "from is required: create supports naming an existing native MLX model"
        )
    unsupported = [
        key
        for key in (
            "files",
            "adapters",
            "quantize",
            "parameters",
            "template",
            "system",
            "messages",
        )
        if body.get(key)
    ]
    if unsupported:
        raise OllamaError(400, "Unsupported create fields: " + ", ".join(unsupported))
    await copy_model(request, source, model)
    return _status_success(body)


@router.post("/pull")
@_wrap
async def pull(request: Request):
    from ..ollama_models import pull_model

    body = await _json_body(request)
    await pull_model(request, _model_name(body))
    return _status_success(body)


@router.post("/push")
async def push_na(request: Request):
    return error_response(
        501,
        "Ollama registry uploads are unsupported: Yunshu serves native MLX repositories",
    )
