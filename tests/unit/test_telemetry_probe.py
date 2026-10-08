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
