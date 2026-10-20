"""CLI chat must send the configured auth token on every gateway call."""

from __future__ import annotations

import httpx

from yunshu_cli import chat


class _Resp:
    status_code = 500
    text = "x"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_text(self):
        return iter(())


def test_chat_calls_carry_auth_header(monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "tok")
    seen = []

    def post(url, **kw):
        seen.append(kw.get("headers"))
        return _Resp()

    def stream(method, url, **kw):
        seen.append(kw.get("headers"))
        return _Resp()

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "stream", stream)
    chat.HISTORY.append({"role": "user", "content": "hi"})
    chat._send_non_stream("http://x", {})
    chat.HISTORY.append({"role": "user", "content": "hi"})
    chat._send_stream("http://x", {})
    assert seen == [{"Authorization": "Bearer tok"}] * 2
