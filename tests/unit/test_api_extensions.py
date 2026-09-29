"""Yunshu API extensions: request ids, progress comments, x_yunshu stats, queue headers,
cancel by id, keep_alive, error hints, status. The engine is faked: a tiny ASGI app
registers a real ``RequestTracker`` generation and drives real ``RunStats``."""

import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from yunshu_engine import settings
from yunshu_engine.model_manager import ModelEntry, ModelManager, ModelType
from yunshu_engine.request_tracker import get_request_tracker
from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner
from yunshu_gateway import x_yunshu
from yunshu_gateway.error_hints import add_hint, hint_for
from yunshu_gateway.routers import yunshu as yunshu_router
from yunshu_gateway.x_yunshu import (
    YunshuExtensionsMiddleware,
    build_stats,
    parse_keep_alive,
    sanitize_request_id,
)


@pytest.fixture(autouse=True)
def _clean():
    tracker = get_request_tracker()

    def drop_all():
        x_yunshu.registry.clear()
        for gen in tracker.all_active():
            tracker.unregister(gen.request_id)

    drop_all()
    yield
    drop_all()
    settings.clear_overrides()


def _register_fake_generation(completion_id: str, model: str = "m"):
    """What the chat router does: register with the tracker and let the runner attach stats."""
    gen = get_request_tracker().register(completion_id, model)
    stats = RunStats()
    stats.prompt_tokens = 1000
    stats.t_submit = time.perf_counter()
    gen.cancel_event.run_stats = stats
    return gen, stats


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        completion_id = "chatcmpl-test"
        gen, stats = _register_fake_generation(completion_id)
        try:
            if body.get("boom"):
                raise HTTPException(status_code=404, detail="Model 'x' not found")
            if not body.get("stream"):
                await asyncio.sleep(0.05)
                stats.t_admit = stats.t_submit + 0.01
                stats.t_first = stats.t_admit + 0.04
                stats.t_last = stats.t_first + 0.5
                stats.generated = 26
                stats.cached_tokens = 200
                return JSONResponse(
                    {
                        "id": completion_id,
                        "choices": [{"message": {"content": "hi"}}],
                        "usage": {
                            "prompt_tokens": 1000,
                            "completion_tokens": 26,
                            "prompt_tokens_details": {"cached_tokens": 200},
                        },
                    }
                )

            async def events():
                # queued, then prefill 40% -> 80%, then decode
                await asyncio.sleep(0.12)
                stats.t_admit = time.perf_counter()
                stats.prefill_total = 800
                stats.prefill_done = 320
                await asyncio.sleep(0.12)
                stats.prefill_done = 640
                await asyncio.sleep(0.12)
                stats.prefill_done = 800
                stats.t_first = time.perf_counter()
                stats.generated = 1
                for i in range(3):
                    delta = {"content": f"t{i}"}
                    chunk = {"choices": [{"index": 0, "delta": delta}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.01)
                    stats.generated += 1
                    stats.t_last = time.perf_counter()
                usage = {"prompt_tokens": 1000, "completion_tokens": 4}
                if body.get("include_usage"):
                    yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        finally:
            # like the router, the tracker entry is dropped when the stream ends
            if not body.get("stream"):
                get_request_tracker().unregister(completion_id)

    app.include_router(yunshu_router.router, prefix="/v1")
    app.add_middleware(YunshuExtensionsMiddleware)
    return app


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


# ── request ids ────────────────────────────────────────────────────────────


def test_sanitize_request_id():
    assert sanitize_request_id("abc-123_x.y:z") == "abc-123_x.y:z"
    assert sanitize_request_id(b"abc") == "abc"
    assert sanitize_request_id("bad id with spaces") is None
    assert sanitize_request_id("x" * 200) is None
    assert sanitize_request_id(None) is None


async def test_request_id_is_echoed_or_generated_on_every_response():
    async with _client(_make_app()) as c:
        r = await c.get("/v1/requests", headers={"X-Request-Id": "client-42"})
        assert r.headers["x-request-id"] == "client-42"
        r = await c.get("/v1/requests")
        assert r.headers["x-request-id"].startswith("req_")
        r = await c.get("/v1/requests", headers={"X-Request-Id": "bad id"})
        assert r.headers["x-request-id"].startswith("req_")
        # errors carry it too
        r = await c.post(
            "/v1/chat/completions",
            json={"boom": True},
            headers={"X-Request-Id": "err-1"},
        )
        assert r.status_code == 404
        assert r.headers["x-request-id"] == "err-1"


# ── non-streaming stats ────────────────────────────────────────────────────


async def test_non_stream_response_gets_x_yunshu_and_headers():
    async with _client(_make_app()) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "m"},
            headers={"X-Request-Id": "ns-1"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hi"  # spec fields untouched
    xy = body["x_yunshu"]
    assert xy["request_id"] == "ns-1"
    assert xy["prompt_tokens"] == 1000 and xy["cached_tokens"] == 200
    assert xy["completion_tokens"] == 26
    assert xy["queue_wait_ms"] == pytest.approx(10.0, abs=1.0)
    assert xy["decode_tps"] == pytest.approx(25 / 0.5, rel=0.02)
    # prefill: 800 uncached tokens in 40 ms
    assert xy["prefill_tps"] == pytest.approx(800 / 0.04, rel=0.02)
    assert xy["timings"]["predicted_n"] == 26
    assert xy["timings"]["cache_n"] == 200
    assert r.headers["x-yunshu-decode-tps"] == str(xy["decode_tps"])
    assert r.headers["x-yunshu-cached-tokens"] == "200"
    assert r.headers["x-yunshu-queue-position"] == "0"
    assert int(r.headers["content-length"]) == len(r.content)


# ── streaming: progress comments and stats ─────────────────────────────────


async def test_stream_emits_progress_comments_then_stats():
    settings.set_override("YUNSHU_PROGRESS_INTERVAL_S", 0.05)
    async with _client(_make_app()) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"stream": True, "include_usage": True},
            headers={"X-Request-Id": "st-1"},
        )
    assert r.headers["x-request-id"] == "st-1"
    assert r.headers["x-yunshu-queue-position"] == "0"
    text = r.text
    comments = [
        json.loads(line[len(": yunshu-progress ") :])
        for line in text.split("\n")
        if line.startswith(": yunshu-progress ")
    ]
    assert comments, text
    phases = [p["phase"] for p in comments]
    assert "prefill" in phases
    prefill = [p for p in comments if p["phase"] == "prefill"]
    assert all(p["request_id"] == "st-1" for p in prefill)
    assert 0 < prefill[-1]["percent"] <= 100
    assert prefill[-1]["prompt_tokens"] == 1000
    assert "eta_s" in prefill[-1]
    # a strict SSE reader sees only data lines: every one is valid JSON or [DONE]
    data = [ln[6:] for ln in text.split("\n") if ln.startswith("data: ")]
    assert data[-1] == "[DONE]"
    events = [json.loads(d) for d in data[:-1]]
    usage = [e for e in events if e.get("usage")]
    assert len(usage) == 1 and usage[0]["x_yunshu"]["completion_tokens"] == 4
    assert (
        usage[0]["x_yunshu"]["ttft_ms"] > 300
    )  # queued + prefill before the first token
    assert usage[0]["x_yunshu"]["prefill_ms"] > 0


async def test_stream_without_usage_gets_stats_comment_and_no_progress_when_off():
    settings.set_override("YUNSHU_PROGRESS_INTERVAL_S", 0)
    async with _client(_make_app()) as c:
        r = await c.post("/v1/chat/completions", json={"stream": True})
    assert ": yunshu-progress" not in r.text
    stats_lines = [ln for ln in r.text.split("\n") if ln.startswith(": yunshu-stats ")]
    assert len(stats_lines) == 1
    stats = json.loads(stats_lines[0][len(": yunshu-stats ") :])
    assert stats["ttft_ms"] is not None
    assert r.text.rstrip().endswith("data: [DONE]")


# ── status / requests / cancel by id ───────────────────────────────────────


async def test_requests_endpoint_and_cancel_by_client_request_id():
    app = _make_app()
    settings.set_override("YUNSHU_PROGRESS_INTERVAL_S", 0)
    async with _client(app) as c:
        post = asyncio.create_task(
            c.post(
                "/v1/chat/completions",
                json={"stream": True},
                headers={"X-Request-Id": "live-1"},
            )
        )
        await asyncio.sleep(0.08)
        r = await c.get("/v1/requests/live-1")
        assert r.status_code == 200
        assert r.json()["phase"] in ("queued", "prefill", "starting")
        listing = (await c.get("/v1/requests")).json()
        assert any(x["request_id"] == "live-1" for x in listing["data"])
        st = (await c.get("/v1/yunshu/status")).json()
        assert st["object"] == "yunshu.status"
        assert st["requests"]["active"] >= 1
        assert {"models", "memory", "throughput", "uptime_s", "version"} <= set(st)
        gen = get_request_tracker().get("live-1")
        assert gen is not None and gen.client_request_id == "live-1"
        d = await c.delete("/v1/requests/live-1")
        assert d.status_code == 200 and d.json()["status"] == "cancelled"
        assert gen.cancel_event.is_set()
        await post
        missing = await c.delete("/v1/requests/nope")
        assert missing.status_code == 404


async def test_second_request_sees_the_queue():
    settings.set_override("YUNSHU_PROGRESS_INTERVAL_S", 0)
    async with _client(_make_app()) as c:
        first = asyncio.create_task(
            c.post("/v1/chat/completions", json={"stream": True})
        )
        await asyncio.sleep(0.05)
        second = await c.post("/v1/chat/completions", json={"stream": True})
        await first
    assert second.headers["x-yunshu-queue-position"] == "1"
    assert float(second.headers["x-yunshu-queue-est-wait-ms"]) >= 0


# ── stats math ─────────────────────────────────────────────────────────────


def test_build_stats_speculative_and_null_when_unknown():
    info = x_yunshu.RequestInfo("r", "POST", "/v1/chat/completions")
    stats = build_stats(info, {"prompt_tokens": 10, "completion_tokens": 5})
    assert stats["ttft_ms"] is None and stats["decode_tps"] is None
    assert stats["speculative"] is None

    gen = SimpleNamespace(cancel_event=SimpleNamespace(run_stats=RunStats()))
    st = gen.cancel_event.run_stats
    st.t_submit, st.t_admit = info.arrived, info.arrived + 0.1
    st.t_first, st.t_last = info.arrived + 0.5, info.arrived + 1.5
    st.spec_mode, st.spec_drafted, st.spec_accepted = "mtp", 100, 80
    info.gen = SimpleNamespace(stats=st)
    out = build_stats(info, {"prompt_tokens": 500, "completion_tokens": 101})
    assert out["speculative"] == {
        "mode": "mtp",
        "drafted": 100,
        "accepted": 80,
        "acceptance_rate": 0.8,
    }
    assert out["queue_wait_ms"] == pytest.approx(100.0)
    assert out["prefill_tps"] == pytest.approx(500 / 0.4, rel=0.01)
    assert out["decode_tps"] == pytest.approx(100.0, rel=0.01)


def test_runstats_phase():
    st = RunStats()
    assert st.phase == "queued"
    st.t_admit = 1.0
    assert st.phase == "prefill"
    st.t_first = 2.0
    assert st.phase == "decode"
    st.finish_reason = "stop"
    assert st.phase == "done"


def test_note_prefill_reads_upstream_prompt_batch():
    import numpy as np

    st = RunStats()
    st.prefill_total = 4096
    job = SimpleNamespace(stats=st, ids=list(range(4096)))
    pb = SimpleNamespace(
        _processed_prompt_columns=2048,
        _input_ids=np.zeros((1, 2048)),
        _cached_tokens_per_row=[0],
        _prompt_uids=[7],
    )
    group = SimpleNamespace(gen=SimpleNamespace(_prompt_batch=pb), jobs={7: job})
    VLMBatchRunner._note_prefill(group)
    assert (st.prefill_done, st.prefill_total) == (2048, 4096)
    # after the prompt batch is gone the row counts as fully prefilled
    group.gen._prompt_batch = None
    group.gen._unprocessed_sequences = []
    VLMBatchRunner._note_prefill(group)
    assert st.prefill_done == 4096


# ── keep_alive ─────────────────────────────────────────────────────────────


def test_parse_keep_alive():
    assert parse_keep_alive(None) is None
    assert parse_keep_alive("5m") == 300
    assert parse_keep_alive("30s") == 30
    assert parse_keep_alive("1h") == 3600
    assert parse_keep_alive(120) == 120
    assert parse_keep_alive(0) == 0
    assert parse_keep_alive(-1) == float("inf")
    assert parse_keep_alive("-1") == float("inf")
    with pytest.raises(ValueError):
        parse_keep_alive("soon")
    with pytest.raises(ValueError):
        parse_keep_alive(True)


class _Engine:
    def has_active_requests(self):
        return False

    async def stop(self):
        pass


async def test_keep_alive_controls_idle_unload():
    mgr = ModelManager()
    for name in ("short", "forever", "default"):
        mgr._entries[name] = ModelEntry(
            model_id=name,
            model_path="/x",
            model_type=ModelType.LLM,
            engine=_Engine(),
            is_loaded=True,
            last_access=time.monotonic() - 100,
        )
    assert await mgr.check_ttl() == []  # no server TTL, nothing set
    assert mgr.set_keep_alive("short", 10.0)
    mgr._entries["short"].last_access = time.monotonic() - 100
    assert mgr.set_keep_alive("forever", float("inf"))
    mgr._entries["forever"].last_access = time.monotonic() - 1e6
    assert not mgr.set_keep_alive("missing", 1.0)
    assert await mgr.check_ttl() == ["short"]
    assert mgr._entries["forever"].is_loaded and mgr._entries["default"].is_loaded
    # a server TTL applies to entries without their own keep_alive
    mgr.ttl_seconds = 50.0
    assert await mgr.check_ttl() == ["default"]
    assert mgr.expires_in(mgr._entries["forever"]) is None


# ── errors ─────────────────────────────────────────────────────────────────


def test_error_hints():
    assert "YUNSHU_AUTH_TOKEN" in hint_for(401, "bad key")
    assert "/v1/models" in hint_for(404, "Model 'x' not found")
    assert hint_for(404, "Not Found") is None
    assert "max_tokens" in hint_for(400, "prompt exceeds max context window")
    assert "Retry-After" in hint_for(429, "slow down")
    assert "ready" in hint_for(503, "model not loaded")
    err = add_hint({"message": "Invalid API key", "type": "x"}, 401, "req_1")
    assert err["message"].startswith("Invalid API key (hint: ")
    assert err["x_yunshu"]["request_id"] == "req_1"
    assert err["x_yunshu"]["hint"]
    # idempotent
    again = add_hint(err, 401)
    assert again["message"].count("(hint:") == 1


def test_full_app_errors_carry_hint_and_request_id():
    from fastapi.testclient import TestClient

    from yunshu_gateway.main import create_app

    client = TestClient(create_app())
    r = client.post(
        "/v1/chat/completions",
        json={"model": "m"},  # messages missing
        headers={"X-Request-Id": "val-1"},
    )
    assert r.status_code == 400
    assert r.headers["x-request-id"] == "val-1"
    err = r.json()["error"]
    assert err["x_yunshu"]["request_id"] == "val-1"
    assert err["x_yunshu"]["hint"]
    assert "hint:" in err["message"]
    r = client.get("/v1/nope")
    assert r.status_code == 404 and r.headers["x-request-id"].startswith("req_")


def test_keep_alive_accepted_on_chat_request_model():
    from yunshu_gateway.routers.chat import ChatCompletionRequest

    req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "keep_alive": "5m",
        }
    )
    assert req.keep_alive == "5m"
