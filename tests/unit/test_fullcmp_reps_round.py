"""fullcmp_reps: a 2-rep cell whose per-round time differs by more than 1.5x between reps is
queued for rep 3, even when its metric gaps are within threshold."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "fullcmp_reps_round_t", ROOT / "scripts" / "research" / "fullcmp_reps.py"
)
reps = importlib.util.module_from_spec(_SPEC)
sys.modules["fullcmp_reps_round_t"] = reps
_SPEC.loader.exec_module(reps)

TH = {"decode_tps": 3.0, "cold_ttft": 5.0, "peak_gib": 5.0}


def _rep_rows(decode_ms, rounds=10):
    return [
        {"part": "session", "group": "g1"},
        {
            "part": "decode",
            "phase": "cold",
            "ctx": 1024,
            "kind": "code",
            "dec_tps": 100.0,
            "ttft_s": 1.0,
            "xy": {"decode_ms": decode_ms, "speculative": {"rounds": rounds}},
        },
    ]


def _write(root: Path, rep: int, rows):
    path = root / "yunshu" / f"cell-r{rep}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_round_time_gap_over_1_5x_sends_a_two_rep_cell_to_rep3(tmp_path):
    _write(tmp_path, 0, _rep_rows(1000.0))  # 100 ms per round
    _write(tmp_path, 1, _rep_rows(2000.0))  # 200 ms per round, same tok/s
    out = list(reps.scan(tmp_path, TH))
    assert len(out) == 1
    _eng, cell, _group, rec = out[0]
    assert rec["status"] == "need_rep3"
    assert rec["exceeding"] == ["round_time"]
    assert rec["rep"] == reps.REP3_INDEX


def test_round_time_within_1_5x_stays_ok(tmp_path):
    _write(tmp_path, 0, _rep_rows(1000.0))
    _write(tmp_path, 1, _rep_rows(1400.0))  # 1.4x: under the 1.5x rule
    _eng, _cell, _group, rec = next(iter(reps.scan(tmp_path, TH)))
    assert rec["status"] == "ok"


def test_round_time_check_needs_round_counts(tmp_path):
    plain = [
        {"part": "session", "group": "g1"},
        dict(_rep_rows(1.0)[1], xy={"decode_ms": 1.0}),
    ]
    plain2 = [plain[0], dict(plain[1], xy={"decode_ms": 9000.0})]
    _write(tmp_path, 0, plain)
    _write(tmp_path, 1, plain2)
    _eng, _cell, _group, rec = next(iter(reps.scan(tmp_path, TH)))
    assert rec["status"] == "ok"
