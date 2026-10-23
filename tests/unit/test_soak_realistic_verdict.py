"""Deployment soak rejects short output and retained allocator memory at idle."""

import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "deploy_soak",
    Path(__file__).resolve().parents[2] / "scripts/research/soak_realistic.py",
)
soak = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(soak)


def ledger(apc=2, active=22, pool=0, footprint=24):
    return {
        "owners": [{"kind": "apc_ram", "bytes": apc * 2**30}],
        "mlx": {"active_bytes": active * 2**30, "cache_bytes": pool * 2**30},
        "process": {"footprint_bytes": footprint * 2**30},
    }


def test_idle_apc_growth_is_intended_but_pool_must_return():
    assert soak.idle_verdict(ledger(), ledger(apc=6, active=26, footprint=28)) == []
    assert "pool" in soak.idle_verdict(ledger(), ledger(pool=4))[0]


def test_missing_counters_and_unattributed_growth_fail():
    assert soak.idle_verdict({}, {})
    assert soak.idle_verdict(ledger(), ledger(active=26, footprint=28))


def test_short_or_empty_soak_never_passes():
    assert soak.run_verdict([], 3600, 3600, ledger(), ledger())
    assert soak.run_verdict([{"ok": True}], 10, 3600, ledger(), ledger())
    assert soak.run_verdict([{"ok": False}], 3600, 3600, ledger(), ledger())
    assert not soak.run_verdict([{"ok": True}], 3600, 3600, ledger(), ledger())


@pytest.mark.parametrize(
    "done,finish,failed",
    [(False, "stop", True), (True, None, True), (True, "stop", False)],
)
def test_client_rejects_eof_without_done(monkeypatch, done, finish, failed):
    event = {"choices": [{"delta": {"content": "AMBER"}, "finish_reason": finish}]}

    class Response(io.BytesIO):
        status = 200

    class Connection:
        def __init__(self, *a, **kw):
            pass

        def request(self, *a):
            pass

        def getresponse(self):
            return Response(
                (
                    "data: "
                    + json.dumps(event)
                    + "\n\n"
                    + ("data: [DONE]\n\n" if done else "")
                ).encode()
            )

        def close(self):
            pass

    monkeypatch.setattr(soak.http.client, "HTTPConnection", Connection)
    result = soak.Client("http://localhost:18990", "test", 10).chat([], 16)
    assert result["content"] == "AMBER"
    assert bool(result.get("error")) == failed


def test_paused_queue_time_is_not_soak_time(monkeypatch):
    monkeypatch.setattr(soak, "pause_intervals", lambda: [(0, 15), (30, 40), (90, 200)])
    assert soak.active_seconds(10, 100) == 65
    assert soak.run_verdict([{"ok": True}], float("nan"), 3600, ledger(), ledger())


def test_segment_pilot_uses_existing_lifecycle_and_always_stops_own_server(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "deploy_segment",
        Path(__file__).resolve().parents[2] / "scripts/research/deploy27b_segment.py",
    )
    segment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(segment)
    killed = []
    calls = []
    for name in ("AGENTIC_YUNSHU_SRC", "HF_HOME", "TMPDIR"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(segment.servers, "SERVER_HOME", tmp_path / "before")
    server = SimpleNamespace(
        proc=SimpleNamespace(pid=123),
        model_id="tiny",
        url="http://x",
        kill=lambda: killed.append(123),
    )
    monkeypatch.setattr(segment.servers, "start_server", lambda *a: (server, [18990]))
    monkeypatch.setattr(
        segment, "memory_ledger", lambda url: calls.append("ledger") or ledger()
    )
    monkeypatch.setattr(segment.time, "sleep", lambda n: None)
    monkeypatch.setattr(
        segment.Client,
        "chat",
        lambda *a, **k: calls.append("chat") or {"done": True, "content": "READY"},
    )
    output = tmp_path / "pilot.jsonl"
    monkeypatch.setattr(
        segment.sys,
        "argv",
        ["segment", "--pilot", "--model", "tiny", "--output", str(output)],
    )
    assert segment.main() == 0
    assert json.loads(output.read_text())["complete"] is True
    assert killed == [123]
    assert calls[0] == "chat"  # warm before measuring the resident baseline


def test_http_error_does_not_record_echoed_prompt(monkeypatch):
    from types import SimpleNamespace

    response = SimpleNamespace(status=400, read=lambda: b'{"input":"secret-prompt"}')
    conn = SimpleNamespace(
        request=lambda *a: None, getresponse=lambda: response, close=lambda: None
    )
    monkeypatch.setattr(soak.http.client, "HTTPConnection", lambda *a, **kw: conn)
    result = soak.Client("http://localhost:18990", "test", 10).chat([], 16)
    assert result["error"] == "HTTP 400"
    assert "secret-prompt" not in json.dumps(result)


def test_lost_sampler_or_missing_wired_data_fails(monkeypatch):
    monkeypatch.setattr(soak, "pause_intervals", lambda: [])

    def sample(t):
        return {
            "kind": "mem",
            "t": t,
            "footprint_gib": 22,
            "ledger": ledger(),
            "process_tree": {
                "processes": [{"wired_bytes": 10, "physical_footprint_bytes": 20}]
            },
        }

    assert soak.sample_verdict([sample(0), sample(60), sample(120)], 0, 180) == []
    assert soak.sample_verdict([sample(0)], 0, 180)
    assert soak.sample_verdict([], 0, 180)
    bad = sample(0)
    bad["process_tree"]["processes"][0]["wired_bytes"] = None
    assert soak.sample_verdict([bad], 0, 60)


def test_minute_slope_and_unknown_data():
    rows = [
        {"kind": "mem", "t": 1000 + i * 60, "footprint_gib": 20 + i / 30}
        for i in range(61)
    ]
    assert soak.footprint_slope_gib_h(rows) == pytest.approx(2)
    assert soak.footprint_slope_gib_h(rows[:1]) is None
    rows[4]["footprint_gib"] = None
    assert soak.footprint_slope_gib_h(rows) is None


def test_draft_window_is_intended_cache_but_unknown_growth_still_fails():
    before, after = ledger(), ledger(active=22.2, footprint=24.2)
    after["owners"].append({"kind": "apc_draft_window", "bytes": 0.2 * 2**30})
    assert not soak.idle_verdict(before, after)
    after["process"]["footprint_bytes"] += 0.6 * 2**30
    assert soak.idle_verdict(before, after)


def test_grammar_token_tables_explain_footprint_growth_only_once():
    before, after = ledger(), ledger(active=22.0, footprint=24.8)
    assert soak.idle_verdict(before, after)  # 0.8 GiB native growth, unexplained
    after["native_owners"] = [{"kind": "grammar_tokenizer", "bytes": 0.45 * 2**30}]
    assert not soak.idle_verdict(before, after)
    after["process"]["footprint_bytes"] += 0.6 * 2**30
    assert soak.idle_verdict(before, after)
