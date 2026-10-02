"""The APC snapshot reaches GET /metrics.

Two things once kept it out: the gauge names were not registered (the population
failed silently), and MetricsMiddleware serves /metrics before the router does, so
the router's population never ran. Both are covered by going through the app.
"""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from yunshu_gateway.middleware.metrics import MetricsMiddleware
from yunshu_gateway.middleware.prometheus_exporter import (
    PrometheusMetrics,
    reset_prometheus_metrics,
)
from yunshu_gateway.routers import monitoring


def _fake_engine():
    snap = {"resident_bytes": 3 * 2**30, "memory_max_bytes": 8 * 2**30, "entries": 5}
    snap.update(lookups_hit=7, disk_hits=2)
    return SimpleNamespace(apc_snapshot=lambda: snap)


def _patch_engine(monkeypatch):
    engine = _fake_engine()
    monkeypatch.setattr(
        "yunshu_gateway.engine.get_engine", lambda: engine, raising=False
    )
    monkeypatch.setattr(
        "yunshu_gateway.engine.get_model_manager", lambda: None, raising=False
    )


def _resident_line(text):
    return next(
        x for x in text.splitlines() if x.startswith("yunshu_apc_resident_bytes{")
    )


def test_apc_snapshot_populates_metrics(monkeypatch):
    _patch_engine(monkeypatch)
    pm = PrometheusMetrics()
    monitoring._populate_apc_metrics(pm)
    text = pm.generate()
    assert float(_resident_line(text).rsplit(" ", 1)[1]) == 3 * 2**30
    assert "yunshu_apc_entries" in text
    assert "yunshu_apc_lookups_hit_total" in text


def test_warm_and_storage_tiers_reach_metrics(monkeypatch):
    snap = {
        "warm_bytes": 123,
        "warm_hits": 4,
        "warm_ratio": 1.4,
        "tier_hits": {"ram": 3, "warm": 1, "ssd": 2, "none": 1},
        "storage_tiers": [
            {"name": "internal", "used_bytes": 10, "cap_bytes": 20, "entries": 2,
             "read_bps": 9e9, "hits": 2, "available": True},
            {"name": "nas", "used_bytes": 5, "cap_bytes": 50, "entries": 1,
             "read_bps": 1e8, "hits": 0, "available": False, "invalidated": 1},
        ],
    }  # fmt: skip
    engine = SimpleNamespace(apc_snapshot=lambda: snap)
    monkeypatch.setattr(
        "yunshu_gateway.engine.get_engine", lambda: engine, raising=False
    )
    monkeypatch.setattr(
        "yunshu_gateway.engine.get_model_manager", lambda: None, raising=False
    )
    pm = PrometheusMetrics()
    monitoring._populate_apc_metrics(pm)
    text = pm.generate()
    assert "yunshu_apc_warm_bytes" in text and "yunshu_apc_warm_hits_total" in text
    assert "yunshu_apc_tier_lookups_total{" in text and 'tier="warm"' in text
    line = next(
        x
        for x in text.splitlines()
        if x.startswith("yunshu_apc_storage_tier_available{") and 'tier="nas"' in x
    )
    assert float(line.rsplit(" ", 1)[1]) == 0.0
    assert "yunshu_apc_storage_tier_read_bytes_per_second{" in text


def test_metrics_endpoint_serves_apc_gauges(monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "1")
    _patch_engine(monkeypatch)
    reset_prometheus_metrics()
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)
    body = TestClient(app).get("/metrics").text
    assert float(_resident_line(body).rsplit(" ", 1)[1]) == 3 * 2**30
    reset_prometheus_metrics()
