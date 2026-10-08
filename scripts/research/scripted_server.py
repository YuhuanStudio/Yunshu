"""Real HTTP server (uvicorn, real lifespan) in front of the scripted engine: CPU-only, no model.
Used by tests/unit/test_graceful_shutdown_real_server.py. Usage: scripted_server.py PORT DELAY_S PIECES."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import uvicorn  # noqa: E402
from tests.unit.wire_harness import Script, ScriptedEngine  # noqa: E402

from yunshu_cli.serve import graceful_shutdown_timeout  # noqa: E402
from yunshu_engine import settings  # noqa: E402
from yunshu_gateway import engine as engine_mod  # noqa: E402
from yunshu_gateway.main import create_app  # noqa: E402

port, delay, n = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
engine_mod._engine = ScriptedEngine(
    Script(pieces=[f"w{i} " for i in range(n)], delay=delay)
)
engine_mod._model_manager = None


class ScriptedIdentity:
    """Let CPU tests prove readiness belongs to the process they started."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def identified(message):
            if message["type"] == "http.response.start":
                message = {
                    **message,
                    "headers": [
                        *message.get("headers", []),
                        (b"x-yunshu-scripted-pid", str(os.getpid()).encode()),
                    ],
                }
            await send(message)

        await self.app(scope, receive, identified)


app = create_app()
app.add_middleware(ScriptedIdentity)
uvicorn.run(
    app,
    host="127.0.0.1",
    port=port,
    log_level="warning",
    timeout_graceful_shutdown=graceful_shutdown_timeout(
        settings.get("YUNSHU_DRAIN_TIMEOUT")
    ),
)
