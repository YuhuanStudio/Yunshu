"""a grounded metrics hunt found the subsystem well-hardened, with one MEDIUM:
the Prometheus exporter's request_total counter received the RAW normalized endpoint, not
the /{other}-bucketed one that _Metrics.record_request computes past its distinct-endpoint
cap. So an endpoint flood (reachable on auth-disabled deployments) churned the exporter's
256-series cap → evicted series reset to 0 → spurious counter-resets corrupting
rate(yunshu_request_total). record_request now RETURNS the bucketed endpoint and the
middleware feeds it to the exporter + aggregator. Plus a LOW: _esc_prom now escapes \\r.
"""
from __future__ import annotations

from yunshu_gateway.middleware import (
    metrics as M,  # noqa: N812  # intentional short module alias
)


def test_record_request_returns_bucketed_endpoint():
    m = M._Metrics()
    # fill past the distinct-endpoint cap with unique short paths
    cap = M._MAX_DISTINCT_ENDPOINTS
    for i in range(cap + 5):
        ret = m.record_request(endpoint=f"/p{i}", method="GET", status=200, latency=0.01)
        # the return value is what the exporter should use
        assert ret is not None
    # a brand-new endpoint past the cap must come back bucketed as /{other}
    ret = m.record_request(endpoint="/brand-new-path", method="GET", status=200, latency=0.01)
    assert ret == "/{other}", f"expected /{{other}} past cap, got {ret!r}"
    # an already-seen endpoint is returned as itself
    assert m.record_request(endpoint="/p0", method="GET", status=200, latency=0.01) == "/p0"


def test_esc_prom_escapes_carriage_return():
    out = M._esc_prom('a\rb\nc"d\\e')
    assert "\r" not in out and "\\r" in out
    assert "\\n" in out and '\\"' in out and "\\\\" in out
