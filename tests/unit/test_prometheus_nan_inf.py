"""(deferred-LOW from the metrics hunt): the Prometheus exporter rendered a
NaN/Inf metric value as lowercase nan/inf/-inf via a bare f-string, which strict
OpenMetrics scrapers reject (the canonical tokens are NaN / +Inf / -Inf). Production value
sources are guarded against non-finite values, so this is defense-in-depth at the
exposition layer. _fmt_value now emits the canonical tokens, applied at the gauge / counter
/ histogram-sum format sites.
"""

from __future__ import annotations

from yunshu_gateway.middleware.prometheus_exporter import _fmt_value, _Gauge, _Histogram


def test_fmt_value_canonical_tokens():
    assert _fmt_value(float("nan")) == "NaN"
    assert _fmt_value(float("inf")) == "+Inf"
    assert _fmt_value(float("-inf")) == "-Inf"
    # finite values unchanged
    assert _fmt_value(1.5) == "1.5"
    assert _fmt_value(42) == "42"


def test_gauge_format_emits_canonical_not_lowercase():
    g = _Gauge("test_gauge", "help")
    g.set(float("inf"), {"k": "v"})
    out = g.format()
    assert "+Inf" in out and " inf" not in out and " nan" not in out


def test_histogram_sum_canonical():
    h = _Histogram("test_hist", "help", buckets=[0.5, 1.0])
    h.observe(float("inf"), {"k": "v"})
    out = h.format()
    # the _sum line must use the canonical +Inf, not lowercase inf
    sum_line = [ln for ln in out.splitlines() if "_sum" in ln][0]
    assert "+Inf" in sum_line and " inf" not in sum_line
