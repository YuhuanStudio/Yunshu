"""Benchmark CLI: HTTP failures are not completions; per-request TPOT is per row."""

from __future__ import annotations

import httpx
import pytest
import typer

from yunshu_cli import benchmark


class _Client:
    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kw):
        return httpx.Response(500, json={"error": "boom"})


def test_throughput_http_errors_are_not_completed(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    with pytest.raises(typer.Exit):
        benchmark.bench_throughput(
            url="http://x", model=None, concurrency=2, num_requests=4, max_tokens=8
        )


def test_per_request_tpot_divides_by_rows():
    # 4 rows x 10 tokens each in 1s: each row sees 0.1s/token
    assert benchmark._per_request_tpot(1.0, 40, 4) == pytest.approx(0.1)
    assert benchmark._per_request_tpot(1.0, 0, 4) == 0
