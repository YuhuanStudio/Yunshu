"""The console against a default server: no YUNSHU_AUTH_TOKEN, auth not disabled.

Reads the console makes on every page (status, history, requests, models, cache tiers, downloads, local models)
follow the inference endpoints' access, so a user who never set a token sees live pages. The
management pages (keys, logs, service, ...) need the ``admin`` permission and answer 401, which the
console shows as its token prompt. Both sets are pinned here against the real app.
"""

from __future__ import annotations

import pytest

OPEN_READS = (
    "/v1/yunshu/status",
    "/v1/yunshu/history",
    "/v1/yunshu/requests/recent",
    "/v1/yunshu/requests/recent?limit=5&model=x",
    "/v1/requests",
    "/v1/models",
    "/v1/yunshu/cache/tiers",
    "/v1/yunshu/config",
    "/v1/yunshu/downloads",
    "/v1/yunshu/models/local",
)
NEEDS_TOKEN = (
    "/v1/yunshu/keys",
    "/v1/yunshu/logs",
    "/v1/yunshu/service",
    "/v1/yunshu/cors",
)


@pytest.fixture
def default_server(client, monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "false")
    return client


@pytest.mark.parametrize("path", OPEN_READS)
def test_console_reads_work_without_a_token(default_server, path):
    r = default_server.get(path)
    assert r.status_code == 200, (path, r.status_code, r.text[:200])


@pytest.mark.parametrize("path", NEEDS_TOKEN)
def test_management_reads_ask_for_a_token(default_server, path):
    assert default_server.get(path).status_code == 401


def test_bad_recent_limit_is_a_400(default_server):
    assert default_server.get("/v1/yunshu/requests/recent?limit=0").status_code == 400
    assert default_server.get("/v1/yunshu/requests/recent?limit=513").status_code == 400


def test_history_endpoints_read_without_a_token(default_server):
    assert default_server.get("/v1/yunshu/metrics/history").status_code == 200
    assert default_server.get("/v1/yunshu/requests/history").status_code == 200
    assert (
        default_server.get("/v1/yunshu/metrics/history?since=5&until=2").status_code
        == 400
    )
