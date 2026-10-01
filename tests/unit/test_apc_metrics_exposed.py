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


def test_metrics_endpoint_serves_apc_gauges(monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "1")
    _patch_engine(monkeypatch)
    reset_prometheus_metrics()
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)
    body = TestClient(app).get("/metrics").text
    assert float(_resident_line(body).rsplit(" ", 1)[1]) == 3 * 2**30
    reset_prometheus_metrics()
