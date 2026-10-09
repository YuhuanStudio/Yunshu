"""Probe arguments/verdict CPU preflight before any 27B slot."""

import importlib.util
from pathlib import Path

import pytest


def probe():
    path = Path(__file__).resolve().parents[2] / "scripts/research/telemetry_probe.py"
    spec = importlib.util.spec_from_file_location("telemetry_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_dry_run(tmp_path):
    module = probe()
    module.main(
        ["--out", str(tmp_path / "result.json"), "--model", "fake", "--dry-run"]
    )
    assert '"complete": "dry-run"' in (tmp_path / "result.json").read_text()


def test_probe_verdict_fixture():
    module = probe()
    sample = {
        "telemetry": {
            "state": "ok",
            "watts": {"gpu": 30},
            "gpu": {"frequency_mhz": 900},
            "temperature": {"die_max_c": 60},
        }
    }
    result = {"energy": {"decode": {"joules": 20}}}
    assert module.validate([sample], result)["gpu_peak_watts"] == 30
    with pytest.raises(RuntimeError, match="missing telemetry"):
        module.validate([sample], {})


def test_tfbench_retains_efficiency_and_yv_reports_it():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify.analyze import speed_compare

    row = {
        "part": "decode",
        "phase": "cold",
        "ctx": 1024,
        "kind": "prose",
        "dec_tps": 80,
        "ttft_s": 1,
        "joules_per_token": 0.4,
        "gpu_watts_mean": 32,
    }
    result = speed_compare([[row]] * 3, [[row]] * 3)
    cell = next(c for c in result["cells"] if c["metric"] == "decode_tps")
    assert cell["efficiency"]["cand"] == {"joules_per_token": 0.4, "gpu_watts_mean": 32}


def test_yv_telemetry_stage_is_registered_and_fail_closed(tmp_path):
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify import stages, suites

    assert suites.parse_suite("telemetry")["stages"] == ["telemetry"]
    seen = []
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        '{"complete": true, "summary": {"gpu_peak_watts": 30, "gpu_mhz_max": 900, "die_max_c": 60}}\n'
    )

    def run(cells):
        seen.extend(cells)
        assert cells[0].validate(evidence) == (True, "")
        evidence.write_text('{"complete": false}\n')
        assert cells[0].validate(evidence)[0] is False
        return {"cand": SimpleNamespace(ok=False, reason="fake failure", evidence=None)}

    ctx = SimpleNamespace(
        cand=SimpleNamespace(path=tmp_path),
        run=SimpleNamespace(path=tmp_path, append=lambda *a: None),
        exe=SimpleNamespace(run_cells=run, jobs=[]),
        py="python",
        model="fake",
        big=False,
        mem_gb=14,
    )
    result = stages.STAGE_FUNCS["telemetry"](ctx)
    assert result.passed is False and result.reasons == ["fake failure"]
    assert seen[0].quiet is False
    assert seen[0].argv[seen[0].argv.index("--draft") + 1] == "off"


def test_probe_retains_closing_interval_after_fast_request(tmp_path, monkeypatch):
    import io
    import json
    import sys
    from types import SimpleNamespace

    module = probe()
    clock = [0.0]
    submitted = [None]
    killed = []

    class Server:
        def __init__(self, *args):
            self.model, self.url, self.log = (
                "fake",
                "http://fake",
                tmp_path / "server.log",
            )
            self.log.write_text("fake")

        def kill(self):
            killed.append(True)

    class Future:
        def done(self):
            return clock[0] - submitted[0] >= 0.2

        def result(self):
            return {"energy": {"decode": {"joules": 1}}}

    class Pool:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, *args):
            submitted[0] = clock[0]
            return Future()

    def response(*args, **kwargs):
        active = 1 <= clock[0] - submitted[0] < 2
        host = {
            "telemetry": {
                "state": "ok",
                "watts": {"gpu": 30 if active else 0},
                "gpu": {"frequency_mhz": 900 if active else None},
                "temperature": {"die_max_c": 60},
            }
        }
        return io.BytesIO(json.dumps(host).encode())

    monkeypatch.setitem(
        sys.modules,
        "tfbench",
        SimpleNamespace(
            Srv=Server, send=lambda *a: None, engaged_spec_mode=lambda *a: "off"
        ),
    )
    # Keep the fake clock/pool/HTTP local to this probe. The stdlib modules are
    # shared with background workers from unrelated tests; mutating their functions
    # can advance this clock/reset submitted while the probe is still sampling.
    import time as real_time

    real_sleep = real_time.sleep
    real_perf_counter = real_time.perf_counter
    monkeypatch.setattr(
        module,
        "concurrent",
        SimpleNamespace(futures=SimpleNamespace(ThreadPoolExecutor=Pool)),
    )
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            perf_counter=lambda: clock[0],
        ),
    )
    monkeypatch.setattr(
        module,
        "urllib",
        SimpleNamespace(
            request=SimpleNamespace(
                Request=module.urllib.request.Request, urlopen=response
            )
        ),
    )
    assert real_time.sleep is real_sleep and real_time.perf_counter is real_perf_counter
    out = tmp_path / "result.json"
    module.main(["--out", str(out), "--model", "fake", "--tokens", "32"])
    result = json.loads(out.read_text())
    assert result["complete"] is True and result["summary"]["gpu_peak_watts"] == 30
    assert killed == [True]


def test_yv_tiny_pilot_precedes_27b_and_uses_small_budget(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify import stages, suites

    seen = []

    def run(cells):
        seen.extend(cells)
        return {"cand": SimpleNamespace(ok=False, reason="fake failure", evidence=None)}

    monkeypatch.setattr(stages, "_finish", lambda ctx, result: result)
    ctx = SimpleNamespace(
        cand=SimpleNamespace(path=tmp_path),
        run=SimpleNamespace(path=tmp_path),
        exe=SimpleNamespace(run_cells=run),
        py="python",
        model="27B",
        big=True,
        mem_gb=60,
    )
    result = stages.STAGE_FUNCS["telemetry-tiny"](ctx)
    assert result.name == "telemetry-tiny"
    assert seen[0].mem_gb == 14 and not seen[0].quiet
    assert seen[0].timeout_min == 10
    assert "Qwen3.5-0.8B" in seen[0].argv[seen[0].argv.index("--model") + 1]
    assert seen[0].argv[seen[0].argv.index("--draft") + 1] == "off"
    assert ctx.model == "27B"
    ladder = suites.parse_suite("telemetry,telemetry-tiny,speed")["stages"]
    assert ladder == ["telemetry-tiny", "telemetry", "speed"]


def test_overhead_suite_tracks_long_cells_without_raising_priority():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from verify import suites

    policy = suites.parse_suite("telemetry-overhead")
    assert policy["ctx"] == [1024, 32768]
    assert policy["reps"] == 3 and policy["speed_tol_pct"] == 0
    assert policy["split_cells"] is True and policy["kinds"] == ["prose"]
    assert policy["stages"][:3] == ["preflight", "telemetry-tiny", "telemetry"]


def test_no_model_diagnostic_extracts_nested_tables_and_dry_run(tmp_path):
    import struct

    path = (
        Path(__file__).resolve().parents[2] / "scripts/research/telemetry_channels.py"
    )
    spec = importlib.util.spec_from_file_location("telemetry_channels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nodes = [
        {
            "IORegistryEntryName": "pmgr",
            "IORegistryEntryChildren": [
                {
                    "IORegistryEntryName": "pmgr-child",
                    "voltage-states9": struct.pack("<II", 900000000, 1),
                }
            ],
        }
    ]
    assert module.pmgr_tables(nodes) == [
        {"name": "pmgr-child", "pairs": [(900000000, 1)]}
    ]
    out = tmp_path / "diagnostic.json"
    module.main(["--out", str(out), "--dry-run"])
    assert '"dry-run"' in out.read_text()


def test_host_telemetry_is_opt_in():
    """The 27B A/B showed a small follow-up TTFT cost with the sampler on."""
    from yunshu_engine import settings

    assert settings.get("YUNSHU_TELEMETRY") == "off"
