"""Actionable hints for API errors (what to do next, the way `yunshu doctor` reports problems).

`hint_for` maps an HTTP status + message to a one-line fix. `add_hint` puts it on an
OpenAI-style error object twice: appended to ``message`` (the only field many SDKs show) and
in the namespaced ``x_yunshu`` object next to the request id, so a client can parse it.
"""

from __future__ import annotations

import re

_RULES: list[tuple[int | None, re.Pattern[str], str]] = [
    (
        None,
        re.compile(r"exceeds max context window|context length|context_length", re.I),
        "shorten the messages or lower max_tokens; the model's window is in "
        "`yunshu show <model>` (YUNSHU_MAX_PREFILL_TOKENS can cap prompts earlier)",
    ),
    (
        None,
        re.compile(r"out of gpu memory|out of memory|\boom\b|memory_error", re.I),
        "free unified memory: lower max_tokens or the prompt length, quantize the KV cache, "
        "or serve a smaller model; `yunshu doctor` shows the memory budget",
    ),
    (
        401,
        re.compile(r".*", re.S),
        "send `Authorization: Bearer <key>` (or `x-api-key`); the server key is "
        "YUNSHU_AUTH_TOKEN (`yunshu config` shows whether one is set)",
    ),
    (
        404,
        re.compile(r"model", re.I),
        "GET /v1/models lists the model ids this server can serve; "
        "pull one with `yunshu pull <repo>`",
    ),
    (
        413,
        re.compile(r".*", re.S),
        "send a smaller body (fewer or lower-resolution images, a shorter prompt) or raise "
        "YUNSHU_MAX_REQUEST_SIZE",
    ),
    (
        429,
        re.compile(r"queue_full|requests in flight", re.I),
        "the server is full: retry after the Retry-After header; the limit is "
        "YUNSHU_QUEUE_LIMIT (0 turns it off)",
    ),
    (
        429,
        re.compile(r".*", re.S),
        "wait for the Retry-After header and retry; the limit is YUNSHU_RATE_LIMIT_RPM "
        "(0 turns it off)",
    ),
    (
        503,
        re.compile(r"memory_pressure|under memory pressure", re.I),
        "memory is nearly full while other requests run: retry after the Retry-After header, "
        "or lower max_tokens / the prompt length; the threshold is YUNSHU_MEMORY_PRESSURE_REJECT",
    ),
    (
        504,
        re.compile(r"deadline", re.I),
        "the request ran past its X-Yunshu-Deadline-Ms; raise the deadline or shorten the "
        "prompt / max_tokens",
    ),
    (
        503,
        re.compile(r"shut|drain", re.I),
        "the server is shutting down; retry against the restarted server",
    ),
    (
        503,
        re.compile(r".*", re.S),
        "the model may still be loading: poll GET /health/ready (or GET /v1/yunshu/status) and "
        "retry after the Retry-After header",
    ),
    (
        400,
        re.compile(
            r"validation_error|field required|input should be|extra inputs", re.I
        ),
        "the request body does not match the API schema; docs/guides/API_SURFACE.md lists "
        "every accepted field",
    ),
    (
        500,
        re.compile(r".*", re.S),
        "an internal error: run `yunshu doctor` and look for the request id "
        "(X-Request-Id) in the server log",
    ),
]


def hint_for(status: int, message: str = "", code: str | None = None) -> str | None:
    text = f"{message} {code or ''}"
    for st, pat, hint in _RULES:
        if (st is None or st == status) and pat.search(text):
            return hint
    return None


def add_hint(err: dict, status: int, request_id: str | None = None) -> dict:
    """Attach the hint (and request id) to an OpenAI-style ``error`` object, in place."""
    hint = hint_for(status, str(err.get("message", "")), err.get("code"))
    ext: dict = {}
    if hint:
        ext["hint"] = hint
        msg = str(err.get("message", ""))
        if "(hint:" not in msg:
            err["message"] = f"{msg} (hint: {hint})"
    if request_id:
        ext["request_id"] = request_id
    if ext:
        err["x_yunshu"] = ext
    return err
