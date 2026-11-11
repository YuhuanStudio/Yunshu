"""Release gate speed rows for Flash-Next and the generic layer (Gemma 4 E2B) against stock mlx-vlm."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_stock_speed", REPO / "scripts/release/check_stock_speed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_stock_speed"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_case_ratio_is_median_over_median():
    m = _load()
    assert m.case_ratio([10, 12, 11], [20, 20, 22]) == 11 / 20
    assert m.case_ratio([0.0, 0.0], [20]) is None
    assert m.case_ratio([10], []) is None


def test_verdict_fails_a_slow_case_and_errors_without_rates():
    m = _load()
    ok = [{"ratio": 1.01}, {"ratio": 0.99}]
    assert m.verdict(ok, 0.90, 0.97) == "PASS"
    assert (
        m.verdict([{"ratio": 1.2}, {"ratio": 0.85}], 0.90, 0.97) == "FAIL"
    )  # one case under the floor
    assert (
        m.verdict([{"ratio": 0.95}, {"ratio": 0.96}], 0.90, 0.97) == "FAIL"
    )  # geomean under the floor
    assert m.verdict([{"ratio": 1.0}, {"ratio": None}], 0.90, 0.97) == "ERROR"
    assert m.verdict([], 0.90, 0.97) == "ERROR"


def test_stop_kills_the_whole_process_group():
    m = _load()
    proc = subprocess.Popen(["sh", "-c", "sleep 60 & wait"], start_new_session=True)
    m.stop(proc)
    assert proc.poll() is not None


def test_main_reports_could_not_measure_as_nonzero(tmp_path, monkeypatch, capsys):
    m = _load()
    monkeypatch.setattr(
        m, "measure", lambda a: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    rc = m.main(
        [
            "--url",
            "u",
            "--model",
            "m",
            "--served-name",
            "n",
            "--output",
            str(tmp_path / "o.json"),
        ]
    )
    assert rc == 2 and '"ERROR"' in capsys.readouterr().out


def test_gate_has_flashnext_and_generic_speed_rows():
    gate = (REPO / "scripts/release/gate.sh").read_text()
    assert '"flashnext|M_FLASHNEXT|' in gate and '"gemma4-e2b|M_GEMMA_E2B|' in gate
    assert re.search(r"flashnext\) family_speed \$label \$mpath identity stock", gate)
    assert re.search(r"gemma4-e2b\) family_speed \$label \$mpath stock", gate)
    assert "Flash-Next-oQ2.5e-mtp" in gate and "gemma-4-e2b" in gate
    assert "check_stock_speed.py" in gate
