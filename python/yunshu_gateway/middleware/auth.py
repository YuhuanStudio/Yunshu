"""Yunshu Gateway — single optional bearer-token auth middleware.

Authenticates requests via a single static bearer token (``YUNSHU_AUTH_TOKEN``).
This engine serves a single consumer (the digital being Yunmo); the multi-tenant
RBACManager / TenantManager machinery has been removed.

Security posture (unchanged, safe by default):
- Auth is enabled when ``YUNSHU_AUTH_TOKEN`` is set.
- Set ``YUNSHU_AUTH_DISABLED=true`` to disable auth (dev only).
- When no token is configured AND auth is not explicitly disabled, admin
  endpoints remain protected by their own ``_check_auth`` (deny by default);
  inference endpoints stay accessible.
- Health/docs endpoints remain public.

After authentication the caller's identity is stamped on ``request.state.actor`` /
``role`` and on ``current_actor`` (``request_tracker`` ownership, ``engine_core`` dedup,
stored-response / container ownership): the static token and auth-disabled single-user
mode are the owner (constant ``"owner"``, overridable via ``YUNSHU_ACTOR_IDENTITY``,
role ``"owner"``, admin); a managed key is ``key:<id>`` with role ``"admin"`` only when
it holds the admin scope, so one infer key cannot touch another key's objects.
"""

import contextlib
import ipaddress
import logging
import threading
import time
from collections import deque

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from yunshu_engine import settings
from yunshu_gateway import api_keys
from yunshu_gateway.error_envelope import format_error_response
from yunshu_gateway.token_compare import tokens_equal

logger = logging.getLogger(__name__)


# Paths served by the Anthropic router — must use Anthropic error format.
# Use exact matching, not endswith, to avoid overmatching paths like
# /api/v1/admin/messages that merely end in '/messages'.
_ANTHROPIC_PATHS = frozenset(
    {
        "/v1/messages",
        "/messages",
        "/v1/messages/count_tokens",
        "/messages/count_tokens",
    }
)


class _ErrorFormatter:
    """Format errors consistently per API family (OpenAI vs Anthropic)."""

    @staticmethod
    def auth_error(
        request: Request, message: str, status_code: int = 401
    ) -> JSONResponse:
        path = request.url.path
        # WWW-Authenticate header only valid on 401, NOT on 429
        headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else {}
        # Anthropic 429: include Retry-After header
        if status_code == 429:
            headers["Retry-After"] = "1"
        # MCP is JSON-RPC — a middleware-level auth/rate-limit denial on /v1/mcp
        # must return a JSON-RPC 2.0 error object, not the OpenAI envelope a JSON-RPC client
        # can't parse. id is null (no parsed body).
        if path.startswith("/tavily/") and path != "/tavily/mcp":
            return JSONResponse(
                status_code=status_code,
                content={"detail": {"error": message}},
                headers=headers,
            )
        if path in ("/v1/mcp", "/tavily/mcp"):
            return JSONResponse(
                status_code=status_code,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32600, "message": message},  # INVALID_REQUEST
                    "id": None,
                },
                headers=headers,
            )
        if path in _ANTHROPIC_PATHS:
            # Map per Anthropic's error-type contract — 429 → rate_limit_error.
            error_type = {
                401: "authentication_error",
                403: "permission_error",
                429: "rate_limit_error",
            }.get(status_code, "invalid_request_error")
            return JSONResponse(
                status_code=status_code,
                content={
                    "type": "error",
                    "error": {
                        "type": error_type,
                        "message": message,
                    },
                },
                headers=headers,
            )
        code = "invalid_api_key" if status_code == 401 else "rate_limit_exceeded"
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": message,
                    "type": "authentication_error"
                    if status_code == 401
                    else "rate_limit_error",
                    "code": code,
                }
            },
            headers=headers,
        )


def _owner_identity() -> str:
    """Single-consumer owner identity, overridable via YUNSHU_ACTOR_IDENTITY."""
    return settings.get("YUNSHU_ACTOR_IDENTITY") or "owner"


# Failed-auth backoff: a client address that presents a WRONG credential _FAIL_LIMIT times within
# _FAIL_WINDOW_S gets 429 + Retry-After on further wrong credentials until the oldest failure ages
# out. Only presented-but-wrong credentials count (a missing header is not a guess) and a correct
# credential is always admitted and resets that client's counter, so a stale key on one machine
# cannot lock the owner out. Loopback clients are exempt. The peer address is used; X-Forwarded-For
# (its last entry, appended by the proxy) only when the peer is listed in YUNSHU_TRUSTED_PROXIES.
_FAIL_LIMIT = 20
_FAIL_WINDOW_S = 60.0
_FAIL_MAX_CLIENTS = 4096
_failures: dict[str, deque] = {}
_failures_lock = threading.Lock()


def _client_ip(request: Request) -> str | None:
    """The peer IP, or None for loopback and non-IP peers (Unix socket, in-process): never throttled."""
    host = request.client.host if request.client else ""
    if host and host in settings.get("YUNSHU_TRUSTED_PROXIES"):
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded.strip():
            host = forwarded.split(",")[-1].strip()
    try:
        return None if ipaddress.ip_address(host).is_loopback else host
    except ValueError:
        return None


def _auth_succeeded(request: Request) -> None:
    """A correct credential clears this client's failure count."""
    ip = _client_ip(request)
    if ip is not None:
        with _failures_lock:
            _failures.pop(ip, None)


def _auth_blocked(request: Request) -> int | None:
    """Seconds to wait when this client has failed authentication too often, else None."""
    ip = _client_ip(request)
    if ip is None:
        return None
    now = time.monotonic()
    with _failures_lock:
        q = _failures.get(ip)
        if q is None:
            return None
        while q and now - q[0] > _FAIL_WINDOW_S:
            q.popleft()
        if not q:
            del _failures[ip]
            return None
        if len(q) >= _FAIL_LIMIT:
            return int(_FAIL_WINDOW_S - (now - q[0])) + 1
    return None


def _auth_failed(
    request: Request, message: str, status: int = 401, *, count: bool = True
) -> Response:
    """A wrong credential: 429 once the client is over the limit, else count it and answer 401.

    ``count=False`` (no credential presented at all) answers 401 without counting or throttling."""
    if count and (wait := _auth_blocked(request)) is not None:
        return format_error_response(
            request.url.path,
            "Too many failed authentication attempts",
            429,
            code="too_many_auth_failures",
            retry_after=wait,
        )
    ip = _client_ip(request) if count else None
    if ip is not None:
        now = time.monotonic()
        with _failures_lock:
            if len(_failures) >= _FAIL_MAX_CLIENTS and ip not in _failures:
                for k in [
                    k for k, q in _failures.items() if now - q[-1] > _FAIL_WINDOW_S
                ]:
                    del _failures[k]
                if len(_failures) >= _FAIL_MAX_CLIENTS:  # all recent: drop the oldest
                    del _failures[next(iter(_failures))]
            _failures.setdefault(ip, deque(maxlen=_FAIL_LIMIT)).append(now)
    return _ErrorFormatter.auth_error(request, message, status)


def _stamp_identity(request: Request, principal=None) -> None:
    """Record who the request is (see the module docstring); ``None`` = the local owner."""
    if principal is not None and principal.key_id:
        actor = f"key:{principal.key_id}"
        role = "admin" if "admin" in principal.scopes else "key"
    else:
        actor, role = _owner_identity(), "owner"
    with contextlib.suppress(Exception):
        request.state.actor = actor
        request.state.role = role
    with contextlib.suppress(Exception):
        from yunshu_engine.request_tracker import current_actor

        current_actor.set(actor)


def _replay_body(body: bytes, receive):
    """A ``receive`` that hands the already-read request body on once, then defers to ``receive``."""
    pending = [{"type": "http.request", "body": body, "more_body": False}]

    async def replay():
        return pending.pop(0) if pending else await receive()

    return replay


class AuthMiddleware:
    """Authenticate requests: the static bearer token and stored API keys (pure ASGI, so
    streamed chunks pass untouched). WebSockets are not handled here."""

    PUBLIC_PATHS = {
        "/health",
        "/health/live",
        "/health/ready",
        "/version",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/",
        "/console",
        "/favicon.ico",
    }
    # Prefixes that are always public (e.g., static assets)
    PUBLIC_PREFIXES = ("/static/", "/assets/", "/console/")

    def _is_auth_enabled(self) -> bool:
        if settings.get_bool("YUNSHU_AUTH_DISABLED"):
            return False
        _static = settings.get("YUNSHU_AUTH_TOKEN")
        if _static:
            return True
        from yunshu_gateway import api_keys

        return (
            api_keys.get_store().has_keys()
        )  # stored keys switch auth on, like a token

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)
        response, key_id = await self._admit(request)
        if response is not None:
            await response(scope, receive, send)
            return
        body = getattr(request, "_body", None)
        if body is not None:  # read while looking for a body-borne api_key: replay it
            receive = _replay_body(body, receive)

        if key_id is None:
            await self.app(scope, receive, send)
            return
        # The key's concurrency slot is held until the response ends or aborts, unless the
        # route moved it to work that outlives the response (``slot_moved``: a background
        # response releases it when its generation ends).
        try:
            await self.app(scope, receive, send)
        finally:
            if not getattr(request.state, "slot_moved", False):
                api_keys.get_store().release(key_id)

    async def _admit(self, request: Request) -> tuple[Response | None, str | None]:
        """``(error response, None)`` to refuse, else ``(None, key id whose slot to release or None)``."""
        # Public paths never need auth
        path = request.url.path
        if path in self.PUBLIC_PATHS:
            return None, None
        if any(path.startswith(p) for p in self.PUBLIC_PREFIXES):
            return None, None

        # CORS preflight (OPTIONS) must pass through without auth —
        # browsers send OPTIONS without Authorization headers, and the
        # CORSMiddleware (inner) needs to respond before any auth check.
        if request.method == "OPTIONS":
            return None, None

        # Check if auth is enabled; without it the one local user is the owner.
        if not self._is_auth_enabled():
            _stamp_identity(request)
            return None, None

        auth_token = settings.get("YUNSHU_AUTH_TOKEN") or ""
        # A queued job's line (batches) runs as the key that submitted it.
        act_as = (
            request.headers.get(api_keys.INTERNAL_KEY_HEADER, "")
            if tokens_equal(
                request.headers.get(api_keys.INTERNAL_HEADER, ""),
                api_keys.INTERNAL_SECRET,
            )
            else ""
        )
        if act_as == api_keys.ADMIN_TOKEN.name:
            with contextlib.suppress(Exception):
                request.state.principal = api_keys.ADMIN_TOKEN
            _stamp_identity(request)
            return None, None
        auth = request.headers.get("Authorization", "")
        # Anthropic SDKs send the key as x-api-key instead of a bearer token.
        api_key = request.headers.get("x-api-key", "")

        if (
            path.startswith("/tavily/")
            and not auth
            and not api_key
            and request.method == "POST"
        ):
            try:
                body = await request.json()
                api_key = body.get("api_key", "") if isinstance(body, dict) else ""
                if not isinstance(api_key, str):
                    api_key = ""
            except (ValueError, TypeError):
                pass

        if act_as:
            token = ""
        elif auth.startswith("Bearer "):
            token = auth[7:]
        elif api_key:
            token = api_key
        else:
            return _auth_failed(
                request, "Missing or invalid Authorization header", count=False
            ), None

        if path == "/v1/realtime/calls" and request.method == "POST":
            from ..realtime_secrets import lookup

            if lookup(token) is not None:
                _auth_succeeded(request)
                _stamp_identity(request)
                return None, None

        # Static token auth (constant-time comparison). The single consumer
        # authenticates with the one configured bearer token.
        if auth_token and token and tokens_equal(token, auth_token):
            _auth_succeeded(request)
            with contextlib.suppress(Exception):
                request.state.principal = api_keys.ADMIN_TOKEN
            _stamp_identity(request)
            return None, None

        # Stored API keys: enabled / expiry, then quotas (429 + Retry-After).
        store = api_keys.get_store()
        if store.has_keys():
            try:
                principal = (
                    store.principal_by_id(act_as)
                    if act_as
                    else store.authenticate(token)
                )
                # an in-process loopback call rides the slot its outer request already
                # holds; a queued job's line has no outer request and is admitted
                internal = not act_as and tokens_equal(
                    request.headers.get(api_keys.INTERNAL_HEADER, ""),
                    api_keys.INTERNAL_SECRET,
                )
                if not internal:
                    store.admit(principal)
            except api_keys.AuthFailureError as exc:
                return _auth_failed(request, str(exc), exc.status), None
            except api_keys.QuotaExceededError as exc:
                return (
                    format_error_response(
                        request.url.path,
                        str(exc),
                        429,
                        code=f"{exc.which}_exceeded",
                        retry_after=exc.retry_after,
                    ),
                    None,
                )
            _auth_succeeded(request)
            request.state.principal = principal
            request.state.api_key_id = principal.key_id
            _stamp_identity(request, principal)
            if internal:
                return None, None
            return None, principal.key_id

        return _auth_failed(request, "Invalid or missing API key"), None
