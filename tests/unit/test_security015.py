"""CPU-only security regressions using mocked network and no server lifespan."""

import httpx
import pytest

from yunshu_engine import log_rotation, netguard
from yunshu_gateway.main import create_app


def test_mixed_wildcard_cors_disables_credentials(monkeypatch):
    monkeypatch.setenv("YUNSHU_CORS_ORIGINS", " https://client.example, * ")
    app = create_app()
    cors = next(m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware")
    assert cors.kwargs["allow_origins"] == ["https://client.example", "*"]
    assert cors.kwargs["allow_credentials"] is False


@pytest.mark.parametrize("cross_origin", [True, False])
async def test_download_headers_are_origin_bound(monkeypatch, tmp_path, cross_origin):
    async def resolve(url, **kw):
        p = netguard.parse_url(url)
        return netguard.Target(url, p.hostname, 443, "https", "93.184.216.34")

    monkeypatch.setattr(netguard, "resolve_target", resolve)
    seen = []

    def handle(req):
        seen.append(req)
        if len(seen) == 1:
            location = "https://other.example/end" if cross_origin else "/end"
            return httpx.Response(302, headers={"location": location})
        return httpx.Response(200, content=b"ok")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        await netguard.download_to_file(
            "https://source.example/start",
            str(tmp_path / "out"),
            max_bytes=100,
            timeout=2,
            client=client,
            headers={
                "Authorization": "Bearer secret",
                "X-Api-Key": "private",
                "Cookie": "sid=secret",
            },
        )
    assert seen[0].headers["authorization"] == "Bearer secret"
    for header in ("authorization", "x-api-key", "cookie"):
        assert (header in seen[1].headers) is (not cross_origin)


@pytest.mark.parametrize(
    "line",
    [
        "ERROR client_secret=ek_abcdefghijklmnopqrstuvwxyz1234567890",
        "ERROR /v1/realtime?token=ek_abcdefghijklmnopqrstuvwxyz1234567890",
        "ERROR openai-insecure-api-key.ek_abcdefghijklmnopqrstuvwxyz1234567890",
    ],
)
def test_ephemeral_secrets_redacted(line):
    assert "ek_abcdefghijklmnopqrstuvwxyz1234567890" not in log_rotation.redact(line)


async def test_transcription_secret_cannot_generate():
    from unittest.mock import AsyncMock, MagicMock

    from yunshu_gateway.routers.realtime import RealtimeSession

    session = RealtimeSession(MagicMock())
    session._secret_kind = "transcription"
    session.send_event = AsyncMock()
    await session._handle_response_create({"type": "response.create"})
    assert session._active_response is None
    error = session.send_event.call_args.args[0]
    assert error["error"]["code"] == "transcription_only"


def test_config_credentials_are_owner_only(tmp_path):
    from yunshu_engine import settings

    target = tmp_path / "config.toml"
    target.write_text("")
    target.chmod(0o644)
    settings.write_config_value("YUNSHU_AUTH_TOKEN", "private", target)
    assert target.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("repo_id", ["../model", "org/..", "./model"])
def test_pull_rejects_traversal_before_disk_access(monkeypatch, repo_id):
    from yunshu_cli import model

    monkeypatch.setattr(
        model, "weights_complete", lambda p: pytest.fail("unsafe disk access")
    )
    with pytest.raises(__import__("typer").Exit) as exc:
        model.pull(repo_id, "/unused", None, False)
    assert exc.value.exit_code == 2


def test_transcription_scope_is_applied_at_handshake(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from yunshu_gateway import realtime_secrets
    from yunshu_gateway.routers import realtime

    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "owner-token")
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "false")
    seen = []

    async def run(session):
        seen.append(session._secret_kind)

    monkeypatch.setattr(realtime.RealtimeSession, "run", run)
    secret = realtime_secrets.mint({}, "transcription", 60)
    app = FastAPI()
    app.include_router(realtime.router)
    try:
        with TestClient(app).websocket_connect(
            "/v1/realtime", headers={"authorization": f"Bearer {secret.value}"}
        ):
            pass
        assert seen == ["transcription"]
    finally:
        realtime_secrets.reset()
