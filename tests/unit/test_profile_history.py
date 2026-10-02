"""profile_history: variance estimates from run logs; missing data is reported as missing."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "profile_history", ROOT / "scripts/research/serving/profile_history.py"
)
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_pooled_cv_recovers_known_sigma(tmp_path):
    rng = random.Random(1)
    rows = [{"kind": "meta", "block": 6}]
    for task in ("a", "b", "c", "d"):
        base = rng.uniform(40, 90)
        for rep in range(8):
            rows.append(
                {
                    "kind": "run",
                    "task": task,
                    "rep": rep,
                    "decode_tps": base * rng.gauss(1, 0.02),
                }
            )
    _write(tmp_path / "x.jsonl", rows)
    rep = ph.build_report(tmp_path, tmp_path / "none.md")
    same = rep["metrics"]["decode_tps"]["same_prompt"]
    assert 0.012 < same["pooled_cv"] < 0.03
    assert same["df"] == 28
    assert (
        same["n_per_arm_10pct"] <= 2
        and same["n_per_arm_3pct"] > same["n_per_arm_10pct"]
    )
    diff = rep["metrics"]["decode_tps"]["different_prompts"]
    assert "missing" in diff  # one configuration group is not enough to estimate it


def test_meta_rows_separate_configurations(tmp_path):
    rows = []
    for block, tps in ((4, 50.0), (8, 100.0)):
        rows.append({"kind": "meta", "block": block})
        rows += [
            {"kind": "run", "pp": 1024, "decode_tps": tps + d}
            for d in (0, 0.5, -0.5, 0.2, -0.2, 0.1)
        ]
    _write(tmp_path / "x.jsonl", rows)
    rep = ph.build_report(tmp_path, tmp_path / "none.md")
    # configs not mixed: CV is tiny, not ~33%
    assert rep["metrics"]["decode_tps"]["same_prompt"]["pooled_cv"] < 0.01


def test_insufficient_data_is_reported_missing(tmp_path):
    _write(tmp_path / "x.jsonl", [{"kind": "run", "pp": 1, "decode_tps": 50.0}])
    rep = ph.build_report(tmp_path, tmp_path / "none.md")
    assert "missing" in rep["metrics"]["decode_tps"]["same_prompt"]
    assert "missing" in rep["metrics"]["ttft_ms"]["different_prompts"]
    assert "missing" in rep["perf_trend_ranges"]


def test_perf_trend_ranges_parsed(tmp_path):
    md = tmp_path / "p.md"
    md.write_text("default 92.0–93.7 tok/s, driver 85.1-88.4 tok/s\n")
    rows = ph.perf_trend_ranges(md)
    assert len(rows) == 2 and rows[0]["lo"] == 92.0


def test_n_per_arm_formula():
    assert ph.n_per_arm(0.15, 0.10) == 36
    assert ph.n_per_arm(0.0, 0.1) is None
    assert SimpleNamespace  # keep import used
