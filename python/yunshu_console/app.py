"""The console process: the web console, a reverse proxy to the engine API, and the history.

One light ASGI app (FastAPI, httpx, SQLite; it never imports MLX):

* ``/console/``     the built web console and docs (revalidated shell, immutable hashed assets);
* ``/v1/yunshu/metrics/history``, ``/v1/yunshu/requests/history``, ``/v1/yunshu/console``
                    answered here from the history store, so they work while the engine is down;
* everything else   proxied to the engine unchanged (methods, bodies, streams, WebSockets), so the
                    browser talks to this one origin. The caller's ``Authorization`` / ``x-api-key``
                    reach the engine as sent; when the engine is unreachable the proxy answers 502
                    ``engine_unreachable`` at once.

The engine does not depend on this process: if it dies, the engine is unaffected, and while the
engine restarts or crashes this process keeps serving the console and recording the gap.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import logging
import time as _time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from starlette.background import BackgroundTask
from starlette.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from .discovery import Discovery
from .pause import Pause
from .poller import FIELDS, Poller
from .store import HistoryStore

logger = logging.getLogger(__name__)

# Hop-by-hop headers (RFC 9110 7.6.1) are for one connection and never forwarded either way.
HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
# Not forwarded to the engine: it would see the browser's origin, not a same-origin caller.
DROP_REQUEST = HOP | {"host", "origin", "referer", "content-length"}
DROP_RESPONSE = HOP | {"content-length"}


def default_static_dir() -> Path:
    """The built console, next to the engine package (found without importing it)."""
    spec = importlib.util.find_spec("yunshu_gateway")
    base = (
        Path(list(spec.submodule_search_locations)[0])
        if spec and spec.submodule_search_locations
        else Path(__file__).parent
    )
    return base / "console_static"


class ConsoleStaticFiles(StaticFiles):
    """Hash-routed Vite assets rooted at the console build directory."""

    def __init__(self, directory: Path) -> None:
        self._root = directory
        super().__init__(
            directory=str(directory), html=True, check_dir=False, follow_symlink=False
        )

    async def check_config(self) -> None:
        """An absent build is a request-time 404, not a startup failure."""

    async def get_response(self, path: str, scope: Scope):
        if not self._root.is_dir() or not (self._root / "index.html").is_file():
            return Response(
                "Yunshu Console is not built. Run `cd frontend && pnpm build` to create python/yunshu_gateway/console_static.",
                status_code=404,
                media_type="text/plain",
            )
        response = await super().get_response(path, scope)
        # The page shell is revalidated on every load so a new build is picked up at once.
        if path in ("", ".", "index.html") or path.endswith("/"):
            response.headers["Cache-Control"] = "no-cache"
        return response


def _presented(headers) -> str | None:
    auth = str(headers.get("authorization", ""))
    if auth.startswith("Bearer "):
        return auth[7:]
    key = headers.get("x-api-key")
    return str(key) if key else None


def check_access(request: Request) -> None:
    """Same rule as the engine's console reads (``can_infer``): open when no token and no API key
    is configured, otherwise the caller must present the token or a key. The console process shares
    the engine's settings and key file, so it knows both."""
    from yunshu_engine import settings
    from yunshu_gateway import api_keys

    token = settings.get("YUNSHU_AUTH_TOKEN")
    if settings.get("YUNSHU_AUTH_DISABLED"):
        return
    store = api_keys.get_store()
    if (token and str(token)) or store.has_keys():
        principal = api_keys.principal_for(_presented(request.headers), token)
        if principal is None:
            raise HTTPException(
                401,
                "Invalid or missing Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )


class StampConsole:
    """Pure ASGI middleware: every answer says it came through the console process.

    The web console reads that header on its status poll and only then asks for the recorded
    history (an engine without a console process has none), so a direct engine connection makes no
    requests that can only fail."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def stamped(message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (b"x-yunshu-console", b"1")]
            await send(message)

        await self.app(scope, receive, stamped)


def create_app(
    engine_url: str,
    *,
    store: HistoryStore | None = None,
    poller: Poller | None = None,
    static_dir: Path | None = None,
    poll: bool = True,
    client: httpx.AsyncClient | None = None,
    discovery: Discovery | None = None,
) -> FastAPI:
    """Build the app. ``poll=False`` serves and proxies without the recorder (tests)."""
    engine_url = engine_url.rstrip("/")
    http = client or httpx.AsyncClient(
        base_url=engine_url,
        timeout=httpx.Timeout(connect=2.0, read=None, write=60.0, pool=None),
        follow_redirects=False,
    )
    owns_client = client is None
    if poller is None and poll:
        poller = Poller(engine_url, store)
    if discovery is not None:
        engine_url = discovery.current_url
        if poller is not None:
            poller.retarget(engine_url)

        def _on_switch(old: str | None, new: str) -> None:
            app.state.engine_url = new
            if poller is not None:
                poller.retarget(new)

        discovery.on_change = _on_switch

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        if store is not None:
            store.start()
        task = None
        found = None
        if discovery is not None:
            found = asyncio.get_running_loop().create_task(
                discovery.run(), name="yunshu-console-discovery"
            )
        if poller is not None and poll:
            task = asyncio.get_running_loop().create_task(
                poller.run(), name="yunshu-console-poller"
            )
        try:
            yield
        finally:
            if found is not None:
                found.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await found
            if discovery is not None:
                await discovery.aclose()
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if poller is not None:
                await poller.aclose()
            if store is not None:
                store.close()
            if owns_client:
                await http.aclose()

    app = FastAPI(
        title="Yunshu Console",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(StampConsole)
    app.state.store = store
    app.state.poller = poller
    app.state.http = http
    app.state.engine_url = engine_url
    app.state.discovery = discovery

    # ── local answers (work with the engine down) ───────────────────────
    @app.get("/v1/yunshu/console")
    async def console_state(request: Request) -> dict:
        check_access(request)
        state = poller.state() if poller is not None else {"up": None}
        return {
            "object": "yunshu.console",
            "recording": store is not None,
            "store": (
                {
                    "path": str(store.path),
                    "bytes": store.size_bytes(),
                    "errors": store.errors,
                }
                if store is not None
                else None
            ),
            "outages": (
                await asyncio.to_thread(store.outage_stats, _time.time() - 86_400)
                if store is not None
                else None
            ),
            **state,
        }

    @app.get("/v1/yunshu/metrics/history")
    async def metrics_history(
        request: Request,
        since: float | None = None,
        until: float | None = None,
        step: float | None = Query(None, ge=0),
    ) -> dict:
        """The recorded history: 1 s for the last hour, 10 s for 24 h, 1 min beyond (the finest
        table that still covers ``since`` and is no finer than ``step``). Spans with no samples
        (the engine was down, or this process was) come back in ``gaps`` and ``events``; they are
        never interpolated."""
        check_access(request)
        if since is not None and until is not None and until <= since:
            raise HTTPException(400, "until must be after since")
        if store is None:
            return {
                "object": "yunshu.metrics_history",
                "enabled": False,
                "tier": None,
                "resolution_s": None,
                "fields": list(FIELDS),
                "series": {"t": [], **{f: [] for f in FIELDS}},
                "gaps": [],
                "events": [],
            }
        out = await asyncio.to_thread(store.read, since, until, step)
        return {"enabled": True, **out}

    @app.get("/v1/yunshu/requests/history")
    async def requests_history(
        request: Request,
        limit: int = 50,
        before: str | None = None,
        model: str | None = None,
    ) -> dict:
        """Finished requests, newest first, from the recorded log (metadata only)."""
        check_access(request)
        if not 1 <= limit <= 512:
            raise HTTPException(400, "limit must be between 1 and 512")
        if store is None:
            return {
                "object": "list",
                "enabled": False,
                "data": [],
                "count": 0,
                "next_cursor": None,
            }
        try:
            page = await asyncio.to_thread(store.requests_page, limit, before, model)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"object": "list", "enabled": True, **page}

    # ── engines (auto-follow) ───────────────────────────────────────────
    def _pause_state() -> dict | None:
        src = poller if poller is not None else discovery
        pause = getattr(src, "pause", None)
        return pause.state() if pause is not None else None

    @app.get("/console/api/engines")
    @app.get("/v1/yunshu/console/engines")
    async def engines(request: Request) -> dict:
        check_access(request)
        paused = _pause_state()
        if discovery is not None:
            return {**discovery.snapshot(), "pause": paused}
        eng = poller.engine if poller is not None else {}
        return {
            "object": "yunshu.console.engines",
            "mode": "fixed",
            "selected": app.state.engine_url,
            "pinned": None,
            "waiting": False,
            "pause": paused,
            "data": [
                {
                    "url": app.state.engine_url,
                    "port": httpx.URL(app.state.engine_url).port,
                    "model": (eng.get("models") or [None])[0],
                    "models": eng.get("models") or [],
                    "uptime_seconds": eng.get("uptime_s"),
                    "state": eng.get("state"),
                    "selected": True,
                    "pinned": False,
                }
            ],
        }

    @app.post("/console/api/engines")
    @app.post("/v1/yunshu/console/engines")
    async def pin_engine(request: Request) -> dict:
        """Body ``{"url": "http://127.0.0.1:18990"}`` (or ``{"port": 18990}``) pins that live
        engine; ``{"url": null}`` unpins and returns to automatic selection."""
        check_access(request)
        if discovery is None:
            raise HTTPException(
                409, "the console follows a fixed engine (not --engine auto)"
            )
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise HTTPException(400, "expected a JSON object")
        url = body.get("url")
        if url is None and isinstance(body.get("port"), int):
            url = f"http://127.0.0.1:{body['port']}"
        if url is not None and not isinstance(url, str):
            raise HTTPException(400, "url must be a string or null")
        if not discovery.pin(url):
            raise HTTPException(404, "that engine is not a live discovered engine")
        return discovery.snapshot()

    # ── the web console ─────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/console/", status_code=307)

    app.mount(
        "/console",
        ConsoleStaticFiles(static_dir or default_static_dir()),
        name="console",
    )

    # ── the engine API, proxied ─────────────────────────────────────────
    def unreachable(reason: str) -> JSONResponse:
        engine_url = app.state.engine_url
        if discovery is not None and discovery.selected is None:
            reason = f"waiting for an engine: {reason}"
        since = poller.since if poller is not None and poller.up is False else None
        return JSONResponse(
            {
                "error": {
                    "message": f"The Yunshu engine at {engine_url} is not reachable ({reason}).",
                    "type": "engine_unreachable",
                    "param": None,
                    "code": "engine_unreachable",
                },
                "since": since,
            },
            status_code=502,
        )

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    async def forward(path: str, request: Request) -> Response:
        headers = [
            (k, v) for k, v in request.headers.items() if k.lower() not in DROP_REQUEST
        ]
        if request.client:
            headers.append(("x-forwarded-for", request.client.host))
        base = discovery.current_url if discovery is not None else engine_url
        url = httpx.URL(base).copy_with(
            path="/" + path, query=request.url.query.encode()
        )
        has_body = (
            request.method not in ("GET", "HEAD", "OPTIONS")
            or "content-length" in request.headers
        )
        upstream_request = http.build_request(
            request.method,
            url,
            headers=headers,
            content=request.stream() if has_body else None,
        )
        try:
            upstream = await http.send(upstream_request, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            return unreachable(type(exc).__name__)
        except httpx.HTTPError as exc:
            return unreachable(type(exc).__name__)
        response = StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            background=BackgroundTask(upstream.aclose),
        )
        # Exactly the engine's headers (repeated ones, such as Set-Cookie, included).
        response.raw_headers = [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in upstream.headers.multi_items()
            if k.lower() not in DROP_RESPONSE
        ]
        return response

    @app.websocket("/{path:path}")
    async def forward_ws(websocket: WebSocket, path: str) -> None:
        await _proxy_websocket(
            websocket,
            discovery.current_url if discovery is not None else engine_url,
            path,
        )

    return app


async def _proxy_websocket(websocket: WebSocket, engine_url: str, path: str) -> None:
    """Pipe a WebSocket both ways (the Realtime and text streaming sockets)."""
    import websockets
    from websockets.typing import Subprotocol

    ws_url = engine_url.replace("http", "ws", 1) + "/" + path
    if websocket.url.query:
        ws_url += "?" + websocket.url.query
    wanted = websocket.headers.get("sec-websocket-protocol")
    subprotocols = [p.strip() for p in wanted.split(",")] if wanted else None
    extra = {
        k: v
        for k, v in websocket.headers.items()
        if k.lower() in ("authorization", "x-api-key", "cookie")
    }
    try:
        upstream = await websockets.connect(
            ws_url,
            additional_headers=extra,
            subprotocols=[Subprotocol(p) for p in subprotocols]
            if subprotocols
            else None,
            max_size=None,
        )
    except Exception:
        await websocket.close(code=1013)  # try again later: the engine is not reachable
        return
    await websocket.accept(subprotocol=upstream.subprotocol)

    async def up() -> None:
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    await upstream.send(message["text"])
                elif message.get("bytes") is not None:
                    await upstream.send(message["bytes"])
        finally:
            await upstream.close()

    async def down() -> None:
        try:
            async for message in upstream:
                if isinstance(message, str):
                    await websocket.send_text(message)
                else:
                    await websocket.send_bytes(message)
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()

    tasks = [asyncio.create_task(up()), asyncio.create_task(down())]
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(BaseException):
            await task


def open_store(path: Path | None = None) -> HistoryStore | None:
    """The history store from settings, or None when ``YUNSHU_CONSOLE_HISTORY`` is off or the disk
    refuses (the console still serves and proxies)."""
    from yunshu_engine import paths, settings

    if not settings.get("YUNSHU_CONSOLE_HISTORY"):
        return None
    try:
        configured = settings.get("YUNSHU_CONSOLE_DB")
        return HistoryStore(
            path
            or (Path(str(configured)).expanduser() if configured else None)
            or paths.home() / "console-history.sqlite",
            FIELDS,
            retention_days=float(settings.get("YUNSHU_CONSOLE_RETENTION_DAYS") or 30),
            max_bytes=int(
                float(settings.get("YUNSHU_CONSOLE_DB_MAX_MB") or 64) * 1048576
            ),
        )
    except Exception:
        logger.warning("console history store unavailable", exc_info=True)
        return None


def build_from_settings(
    engine_url: str | None = None, static_dir: str | Path | None = None
) -> FastAPI:
    """The app as ``yunshu console`` runs it: settings decide the engine, token, store and cadence."""
    from yunshu_engine import settings

    from .discovery import DEFAULT_PORTS, candidate_urls, parse_ports

    url = engine_url or settings.get("YUNSHU_CONSOLE_ENGINE") or "http://127.0.0.1:8000"
    auto = url.strip().lower() == "auto"
    configured = None
    if auto:
        url = "http://127.0.0.1:8000"
    token = settings.get("YUNSHU_CONSOLE_ENGINE_TOKEN") or settings.get(
        "YUNSHU_AUTH_TOKEN"
    )
    store = open_store()
    pause = Pause(settings.get("YUNSHU_CONSOLE_PAUSE_FILE"))
    poller = Poller(
        url,
        store,
        token=token,
        pause=pause,
        interval_s=float(settings.get("YUNSHU_CONSOLE_POLL_S") or 1.0),
    )
    discovery = None
    if auto:
        own = settings.get("YUNSHU_CONSOLE_PORT")
        discovery = Discovery(
            candidate_urls(
                parse_ports(
                    settings.get("YUNSHU_CONSOLE_DISCOVER_PORTS") or DEFAULT_PORTS
                ),
                configured,
                skip_port=int(own) if own else None,
            ),
            default_url=url,
            token=token,
            pause=pause,
            interval_s=float(settings.get("YUNSHU_CONSOLE_DISCOVER_S") or 2.0),
        )
    return create_app(
        url,
        store=store,
        poller=poller,
        static_dir=Path(static_dir) if static_dir else None,
        discovery=discovery,
    )


__all__ = [
    "create_app",
    "build_from_settings",
    "open_store",
    "ConsoleStaticFiles",
    "check_access",
]
