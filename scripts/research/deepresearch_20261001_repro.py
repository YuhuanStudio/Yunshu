"""Small, model-free repros for the 2026-10-01 research handoff.

Run: uv run python scripts/research/deepresearch_20261001_repro.py
These print observations, not a passing regression suite.
"""

import asyncio
import hmac
import json
import re
from unittest.mock import patch

import httpx
import mlx.core as mx

# Apply before importing engine modules. This research must never schedule GPU work.
mx.set_default_device(mx.cpu)

from yunshu_engine import settings
from yunshu_engine import vlm_batch_runner as runner
from yunshu_engine.context_window import ContextWindowManager
from yunshu_engine.grammar_constraint import (
    ChoiceConstraint,
    LarkGrammarConstraint,
    RegexConstraint,
)
from yunshu_engine.keyed_sampling import KeyedSampler
from yunshu_engine.request_tracker import RequestTracker, current_request_id
from yunshu_engine.vlm_engine import _VALIDATE_URL
from yunshu_gateway.server_tools.mcp_connector import McpConnection
from yunshu_gateway.server_tools.netguard import Target, resolve_target


def context_repro():
    mgr = ContextWindowManager(token_counter=len)
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "A" * 20},
        {"role": "assistant", "content": "B" * 20},
        {"role": "user", "content": "latest"},
    ]
    result = mgr.compute_truncation(messages, 40, "truncate_oldest")
    print(
        "context", json.dumps(result.messages), "tokens", result.truncated_token_count
    )
    for strategy in ("sliding_window", "importance_aware", "summary_compression"):
        r = mgr.compute_truncation(
            [
                {"role": "system", "content": "one"},
                {"role": "user", "content": "hi"},
                {"role": "developer", "content": "two"},
                {"role": "user", "content": "latest" * 10},
            ],
            35,
            strategy,
        )
        print(
            "context-strategy",
            strategy,
            json.dumps(r.messages),
            r.truncated_token_count,
        )


def regex_repro():
    for pattern, text in (
        (".+", "Ж"),
        ("[^x]+", "龍"),
        ("[\\s\\S]+", "Ж"),
        ("(?=a)b", "b"),
        ("(?i)abc", "ABC"),
        (r"(a)\1", "a"),
    ):
        c = RegexConstraint(pattern)
        c.advance(text)
        print(
            "regex",
            repr(pattern),
            repr(text),
            "python_match",
            bool(re.fullmatch(pattern, text)),
            "dfa_accept",
            c._dfa.is_accepting(c._dfa_state),
            "done",
            c.is_done,
        )

    class IntEosTokenizer:
        eos_token_ids = 2

    choice = ChoiceConstraint(["yes"])
    choice.advance("yes")
    try:
        choice.get_allowed_tokens(IntEosTokenizer(), [])
    except Exception as exc:
        print("constraint-int-eos", type(exc).__name__)
    cfg = LarkGrammarConstraint("start: /[一-龥]+/")
    cfg.advance("龍")
    print("cfg-unicode", cfg.state, "premature_done", cfg.is_done)


def sampling_repro():
    # CPU avoids contention with any serving GPU workload.
    mx.set_default_device(mx.cpu)
    lp = mx.array([[0.0] * 32])
    params = runner.RowParams(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=7)
    row = runner.RowSampler()
    row.add(0, params)
    keyed = KeyedSampler(params, 7)
    with patch.object(runner, "_STEP_UIDS", [0]):
        ar = [int(row(lp)[0]) for _ in range(16)]
    spec = [int(keyed(lp)[0]) for _ in range(16)]
    print("same-seed", "AR", ar, "spec", spec, "equal", ar == spec)
    from mlx_lm.sample_utils import apply_top_k, apply_top_p

    tiny = apply_top_p(mx.log(mx.array([[0.7, 0.2, 0.1]])), 1e-10)
    print("tiny-top-p", tiny.tolist())
    try:
        apply_top_k(lp, 33)
    except Exception as exc:
        print("top-k-over-vocab", type(exc).__name__, str(exc))


def auth_alias_repro():
    try:
        hmac.compare_digest("é", "configured-secret")
    except Exception as exc:
        print("auth-non-ascii", type(exc).__name__)
    tracker = RequestTracker()
    token = current_request_id.set("same-client-id")
    try:
        tracker.register("first", "model")
        tracker.register("second", "model")
        print("alias-before", tracker.resolve("same-client-id"))
        tracker.unregister("first")
        print(
            "alias-after-first-finishes",
            tracker.resolve("same-client-id"),
            "second-still-active",
            tracker.get("second") is not None,
        )
    finally:
        current_request_id.reset(token)
    for value in ("nan", "inf"):
        print(
            "setting-nonfinite",
            value,
            settings._parse(settings.REGISTRY["YUNSHU_MCP_CONNECTOR_TIMEOUT"], value),
        )
    for ip in ("::ffff:127.0.0.1", "100.64.0.1", "224.0.0.1"):
        with patch(
            "socket.getaddrinfo", return_value=[(None, None, None, None, (ip, 80))]
        ):
            try:
                _VALIDATE_URL("http://example.com/image.png")
                print("media-private-ip-accepted", ip)
            except ValueError as exc:
                print("media-private-ip-rejected", ip, str(exc))


async def mcp_repro():
    seen = []

    async def checked(url, **kwargs):
        return Target(url, "example.com", 443, "https", "93.184.216.34")

    def transport(request):
        seen.append(str(request.url))
        body = json.loads(request.content)
        if "id" in body:
            # Deliberately wrong id: the client should reject it.
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 999,
                    "result": {"serverInfo": {"name": "wrong-id"}},
                },
            )
        return httpx.Response(202)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        conn = McpConnection(
            "https://example.com/mcp", client=client, allow_private=False
        )
        with patch("yunshu_gateway.server_tools.mcp_connector.resolve_target", checked):
            await conn.connect()
        print(
            "mcp",
            "checked_ip",
            "93.184.216.34",
            "actual_urls",
            seen,
            "accepted_wrong_id",
            conn.server_info,
        )
        await conn.close()
    for url in ("https://example.com:abc/a", "http://[broken/a"):
        try:
            await resolve_target(url, allow_private=False)
        except Exception as exc:
            print("url-error", url, type(exc).__name__)


async def queue_repro():
    q = asyncio.Queue(maxsize=2)
    loop = asyncio.get_running_loop()
    errors = []
    old = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda loop, context: errors.append(type(context.get("exception")).__name__)
    )
    try:
        for item in range(5):
            if not q.full():
                loop.call_soon_threadsafe(q.put_nowait, item)
        await asyncio.sleep(0.01)
        print("queue", "size", q.qsize(), "callback_errors", errors)
    finally:
        loop.set_exception_handler(old)


async def main():
    context_repro()
    regex_repro()
    sampling_repro()
    auth_alias_repro()
    await mcp_repro()
    await queue_repro()


if __name__ == "__main__":
    asyncio.run(main())
