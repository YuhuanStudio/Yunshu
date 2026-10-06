"""/metrics must be valid Prometheus text exposition and follow the naming rules vLLM's metrics follow
(vllm/v1/metrics/loggers.py): one HELP/TYPE per family, a counter's samples end in _total, units in names."""

from __future__ import annotations

import re

from prometheus_client.parser import text_string_to_metric_families

from .wire_harness import install


def _scrape(monkeypatch):
    c, _ = install(monkeypatch)
    c.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        },
    )
    r = c.get("/metrics")
    assert r.status_code == 200
    return r.text


def test_exposition_parses_and_families_are_unique(monkeypatch):
    text = _scrape(monkeypatch)
    types = re.findall(r"^# TYPE (\S+) (\S+)", text, re.M)
    names = [t[0] for t in types]
    assert len(names) == len(set(names)), [n for n in names if names.count(n) > 1]
    helps = re.findall(r"^# HELP (\S+)", text, re.M)
    assert len(helps) == len(set(helps))
    assert list(text_string_to_metric_families(text))


def test_counters_end_in_total(monkeypatch):
    text = _scrape(monkeypatch)
    bad = [
        n
        for n, t in re.findall(r"^# TYPE (\S+) (\S+)", text, re.M)
        if t == "counter" and not n.endswith(("_total", "_created"))
    ]
    assert not bad, f"counter families without _total: {bad}"
