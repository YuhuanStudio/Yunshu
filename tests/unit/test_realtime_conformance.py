"""Realtime endpoint vs. the official `openai` SDK (GA + beta clients), fake engine.

The same checks run against a real model in the smoke test
(scripts/dev/realtime_conformance.py --url ...).
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from yunshu_engine import settings
from yunshu_gateway.routers import realtime as rt

from .bound_listener import reserve_listener

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "dev"
    / "realtime_conformance.py"
)


class FakeEngine:
    is_loaded = True

    async def generate_stream(self, prompt=None, cancel_event=None, **kw):
        words = ["Hello", " there", ",", " friend", "."]
        joined = str(prompt)
        n = 400 if "Count from 1" in joined else len(words)
        for i in range(n):
            if cancel_event is not None and cancel_event.is_set():
                return
            await asyncio.sleep(0.005)
            yield SimpleNamespace(
                token_text=words[i % len(words)] if n == len(words) else f"{i} ",
                finish_reason=None,
                prompt_tokens=7,
                completion_tokens=i + 1,
            )
        yield SimpleNamespace(
            token_text="", finish_reason="stop", prompt_tokens=7, completion_tokens=n
        )


@pytest.fixture()
def server(monkeypatch):
    from unittest.mock import AsyncMock

    import uvicorn

    monkeypatch.setattr("yunshu_gateway.engine.get_engine", lambda: None)
    monkeypatch.setattr(
        "yunshu_gateway.engine.get_engine_for_model",
        AsyncMock(return_value=FakeEngine()),
    )
    monkeypatch.setattr(
        rt.RealtimeSession, "_resolve_engine", lambda self: FakeEngine()
    )

    async def fake_tts(self, text, response_id, item_id, voice=None, out_fmt=None):
        delta = base64.b64encode(b"\0" * 4800).decode()
        await self.send_event(
            {"type": "response.audio.delta", "response_id": response_id, "delta": delta}
        )
        await self.send_event(
            {"type": "response.audio.done", "response_id": response_id}
        )

    monkeypatch.setattr(rt.RealtimeSession, "_synthesize_audio_response", fake_tts)
    app = FastAPI()
    app.include_router(rt.router)
    listener = reserve_listener()
    port = listener.getsockname()[1]
    srv = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="error", ws="websockets"
        )
    )
    th = threading.Thread(target=srv.run, kwargs={"sockets": [listener]}, daemon=True)
    th.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.02)
    assert srv.started
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=10)
    listener.close()


def _load_script():
    spec = importlib.util.spec_from_file_location("realtime_conformance", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_openai_sdk_ga_and_beta_conformance(server):
    mod = _load_script()
    args = SimpleNamespace(
        url=server,
        model="fake-model",
        api_key="x",
        audio=False,
        audio_out=True,
        content=False,
        uds=None,
    )
    rc = asyncio.run(mod.main(args))
    failed = [r for r in mod.RESULTS if not r[1]]
    assert rc == 0, failed


def test_auth_rejected_at_handshake(server):
    """Wrong/missing key -> handshake failure (HTTP 403), not an accepted socket."""
    import websockets.sync.client as wsc
    from websockets.exceptions import InvalidStatus

    settings.set_override("YUNSHU_AUTH_TOKEN", "sekrit")
    settings.set_override("YUNSHU_AUTH_DISABLED", False)
    try:
        url = server.replace("http", "ws") + "/v1/realtime?model=m"
        with pytest.raises(InvalidStatus):
            wsc.connect(url)
        with pytest.raises(InvalidStatus):
            wsc.connect(url, additional_headers={"Authorization": "Bearer nope"})
        with wsc.connect(
            url, additional_headers={"Authorization": "Bearer sekrit"}
        ) as ws:
            assert "session.created" in ws.recv()
        # browsers: key in the openai-insecure-api-key.<key> subprotocol
        with wsc.connect(
            url, subprotocols=["realtime", "openai-insecure-api-key.sekrit"]
        ) as ws:
            assert ws.subprotocol == "realtime"
            assert "session.created" in ws.recv()
    finally:
        settings.clear_overrides()
