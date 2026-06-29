"""MetricsMiddleware is the OUTERMOST middleware (runs before auth), so the
request path is attacker-controlled. _normalize_endpoint passes short unknown paths
through raw, so an unauthenticated attacker looping GET /a, /b, … would grow the
per-endpoint metrics dicts without bound (OOM). Cap the distinct-endpoint count."""

from __future__ import annotations

from yunshu_gateway.middleware.metrics import _MAX_DISTINCT_ENDPOINTS, _Metrics


def test_distinct_endpoint_cardinality_is_bounded():
    m = _Metrics()
    # Simulate an attacker hammering thousands of distinct short paths.
    for i in range(_MAX_DISTINCT_ENDPOINTS * 4):
        m.record_request(
            endpoint=f"/attack-{i}", method="GET", status=404, latency=0.001
        )
    # The per-endpoint dicts must not have grown unbounded — overflow paths collapse
    # into a single bucket once the cap is hit.
    assert len(m.latency_total_count) <= _MAX_DISTINCT_ENDPOINTS + 1
    assert "/{other}" in m.latency_total_count  # overflow bucket exists
    # request_count keys are method:endpoint:status — also bounded (endpoint is capped).
    assert len(m.request_count) <= (_MAX_DISTINCT_ENDPOINTS + 1) * 2  # generous bound


def test_known_endpoints_still_tracked_individually():
    m = _Metrics()
    m.record_request("/v1/chat/completions", "POST", 200, 0.5)
    m.record_request("/v1/chat/completions", "POST", 200, 0.7)
    m.record_request("/v1/models", "GET", 200, 0.01)
    assert m.latency_total_count["/v1/chat/completions"] == 2
    assert m.latency_total_count["/v1/models"] == 1
    assert "/{other}" not in m.latency_total_count  # under the cap, no overflow
