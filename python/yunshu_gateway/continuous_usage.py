"""Request-local cumulative usage on every chat SSE chunk (vLLM extension)."""

from __future__ import annotations

import inspect
import json
from contextvars import ContextVar
from functools import wraps

from .usage_shapes import openai_usage

_usage: ContextVar[dict | None] = ContextVar("continuous_usage", default=None)


def update_usage(req, prompt, completion, reasoning=0, cached=0):
    opts = req.stream_options
    if opts and opts.include_usage and opts.continuous_usage_stats:
        _usage.set(openai_usage(prompt, completion, reasoning, cached))


def with_continuous_usage(fn):
    signature = inspect.signature(fn)

    @wraps(fn)
    async def wrapped(*args, **kwargs):
        req = signature.bind(*args, **kwargs).arguments["req"]
        opts = req.stream_options
        enabled = opts and opts.include_usage and opts.continuous_usage_stats
        token = _usage.set(openai_usage(0, 0) if enabled else None)
        try:
            async for chunk in fn(*args, **kwargs):
                if enabled and isinstance(chunk, str) and chunk.startswith("data: {"):
                    data = json.loads(chunk[6:])
                    if "choices" in data and data["choices"]:
                        data["usage"] = _usage.get()
                        chunk = "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
                yield chunk
        finally:
            _usage.reset(token)

    return wrapped
