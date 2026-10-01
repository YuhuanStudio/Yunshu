"""R16: request_size_limit must reject a malformed Content-Length, not fail open."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from yunshu_gateway.main import create_app


@pytest.mark.parametrize("value", ["abc", "-5", "1e3", "12 34", ""])
def test_malformed_content_length_rejected(value):
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            content=b"{}",
            headers={"content-length": value, "content-type": "application/json"},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_content_length"


def test_malformed_content_length_anthropic_shape():
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/messages",
            content=b"{}",
            headers={"content-length": "abc", "content-type": "application/json"},
        )
    assert resp.status_code == 400
    assert resp.json()["type"] == "error"


def test_valid_content_length_unaffected():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code in (200, 503)
