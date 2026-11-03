"""CPU-only tests for the full-comparison audit (scripts/research/fullcmp_audit.py).

A synthetic run (2 engines x 2 prompts x 3 reps x cold/warm/turn2) must pass every guard;
each guard then has one synthetic bad row set that must fail with that guard's name.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_audit as fa  # noqa: E402

PY = "/tree/.venv/bin/python"
MODEL = "Qwen3.8-27B"
ENG = ("yunshu", "mlx-serve")
PROMPTS = ("p0", "p1")
REPS = 3
W = "a" * 64
B = "b" * 64


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def make_data():
    """Return (manifest, files): files maps 'engine/<cell>-rN.jsonl' to its rows."""
    pins = {
        "yunshu": {
            "version": "main",
            "release_date": "2026-11-02",
            "git_sha": "a1b2c3d",
        },
        "mlx-serve": {
            "version": "v26.10.3",
            "release_date": "2026-10-20",
            "binary_sha256": B,
        },
    }
    groups, cells = {}, []
    for p in PROMPTS:
        gid = f"{MODEL}|1024-code-{p}"
        groups[gid] = {"order": list(ENG)}
        for eng in ENG:
            cells.append(
                {
                    "engine": eng,
                    "model": MODEL,
                    "column": f"1024-code-{p}",
                    "cell": f"q27_decode-1024-code-{p}",
                    "group": gid,
                    "ctx": 1024,
                    "kind": "code",
                    "prompt_id": p,
                    "max_tokens": 512,
                    "phases": ["cold", "warm", "turn2"],
                    "expected_records": 3,
                    "spec_mode": "dflash",
                    "weight_group": "q27-4bit",
                }
            )
    manifest = {
        "run_id": "fullcmp-test",
        "reps": REPS,
        "pins": pins,
        "yunshu": {"git_sha": "a1b2c3d", "python": PY},
        "controls": {
            "aa_delta_pct": 0.4,
            "aa_noise_pct": 1.5,
            "injected_delay_detected": True,
        },
        "groups": groups,
        "cells": cells,
    }
    files = {}
    slot = 0
    for rep in range(REPS):
        for p in PROMPTS:
            gid = f"{MODEL}|1024-code-{p}"
            cell = f"q27_decode-1024-code-{p}"
            for eng in fa.rotate(list(ENG), rep):
                start = 1000.0 + slot * 100
                slot += 1
                yun = eng == "yunshu"
                head = {
                    "part": "session",
                    "run_id": "fullcmp-test",
                    "engine": eng,
                    "model": MODEL,
                    "cell": cell,
                    "group": gid,
                    "rep": rep,
                    "version": pins[eng]["version"],
                    "release_date": pins[eng]["release_date"],
                    "git_sha": "a1b2c3d" if yun else "3f9e0c",
                    "binary_sha256": None if yun else B,
                    "cmd": [PY, "-m", "yunshu_cli", "serve"]
                    if yun
                    else ["/bin/mlx-serve", "--serve"],
                    "python": PY if yun else None,
                    "requested_spec_mode": "dflash",
                    "spec_mode_expected": "dflash",
                    "engaged_spec_mode": "dflash",
                    "engaged_log_line": "speculative: dflash drafter engaged",
                    "weights": [
                        {
                            "file": f"{eng}.safetensors",
                            "sha256": W,
                            "bits": 4,
                            "group_size": 64,
                        }
                    ],
                    "fresh_process": True,
                    "home": f"/Volumes/P5Plus/yunshu-build/homes/{eng}",
                    "job_id": f"job-{slot}",
                    "job_start_s": start,
                    "job_end_s": start + 50,
                    "gpu_exclusive": True,
                    "contended": False,
                    "device": "m5",
                }
                rows = [head]
                for ph, base_t, cached, pt in (
                    ("cold", 1.2, 0, 1024),
                    ("warm", 0.3, 1024, 1024),
                    ("turn2", 0.5, 1100, 1100),
                ):
                    rows.append(
                        {
                            "part": "decode",
                            "phase": ph,
                            "ctx": 1024,
                            "kind": "code",
                            "cell": cell,
                            "prompt_file": f"code-1024-{p}.txt",
                            "harness": "tfbench-fullcmp-1",
                            "prompt_ids_sha": f"t2-{eng}-{p}"
                            if ph == "turn2"
                            else f"sha-{p}",
                            "pt": pt,
                            "max_tokens": 512,
                            "ct": 512,
                            "temperature": 0,
                            "stop": [],
                            "ttft_s": round(base_t + 0.1 * rep, 3),
                            "total_s": 10.0,
                            "dec_tps": 80.0 + rep,
                            "cached": cached,
                            "finish": "length",
                            "sha": _sha(f"{eng}{ph}{p}{rep}"),
                            "contended": False,
                        }
                    )
                files[f"{eng}/{cell}-r{rep}.jsonl"] = rows
    return manifest, files


def write_run(root: Path, manifest, files) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, rows in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    mpath = root / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    return mpath


def failing(results) -> set:
    return {r.name for r in results if r.violations}


def _rows(files, name_part, eng=None):
    return [
        rows
        for rel, rows in files.items()
        if name_part in rel and (eng is None or rel.startswith(eng + "/"))
    ]


# ---- bad row sets, one per guard -------------------------------------------
def bad_prompt_identity(m, f):
    _rows(f, "p0-r0", "mlx-serve")[0][1]["prompt_ids_sha"] = "sha-other"


def bad_provenance_identity(m, f):
    for rows in _rows(f, "p0", "mlx-serve"):
        for r in rows[1:]:
            r["harness"] = "tfbench-other"


def bad_pinned_versions(m, f):
    for rows in _rows(f, "", "yunshu"):
        rows[0]["run_id"] = "fullcmp-OLD"


def bad_manifest_completeness(m, f):
    del f["mlx-serve/q27_decode-1024-code-p1-r2.jsonl"]


def bad_record_count(m, f):
    rows = f["yunshu/q27_decode-1024-code-p0-r0.jsonl"]
    del rows[-1]


def bad_engaged_mode(m, f):
    for rows in _rows(f, "", "mlx-serve"):
        rows[0]["engaged_spec_mode"] = "mtp"
        rows[0]["engaged_log_line"] = "speculative: mtp head engaged"


def bad_weight_identity(m, f):
    for rows in _rows(f, "", "mlx-serve"):
        rows[0]["weights"][0]["sha256"] = "c" * 64


def bad_cold_cache(m, f):
    f["yunshu/q27_decode-1024-code-p0-r1.jsonl"][1]["cached"] = 512


def bad_gpu_exclusive_rotation(m, f):
    f["yunshu/q27_decode-1024-code-p0-r1.jsonl"][0]["contended"] = True


def bad_output_sha(m, f):
    f["yunshu/q27_decode-1024-code-p0-r0.jsonl"][1]["sha"] = ""


def bad_measured_only(m, f):
    row = f["yunshu/q27_decode-1024-code-p1-r0.jsonl"][2]
    row["error"] = "incomplete stream"
    del row["dec_tps"]


def bad_fairness_controls(m, f):
    m["controls"]["injected_delay_detected"] = False


BAD_CASES = [
    ("prompt_identity", bad_prompt_identity),
    ("provenance_identity", bad_provenance_identity),
    ("pinned_versions", bad_pinned_versions),
    ("manifest_completeness", bad_manifest_completeness),
    ("record_count", bad_record_count),
    ("engaged_mode", bad_engaged_mode),
    ("weight_identity", bad_weight_identity),
    ("cold_cache", bad_cold_cache),
    ("gpu_exclusive_rotation", bad_gpu_exclusive_rotation),
    ("output_sha", bad_output_sha),
    ("measured_only", bad_measured_only),
    ("fairness_controls", bad_fairness_controls),
]


def _run(tmp_path, manifest, files):
    mpath = write_run(tmp_path / "run", manifest, files)
    return fa.run_guards(tmp_path / "run", mpath)


def test_good_run_passes_all_twelve_guards(tmp_path):
    manifest, files = make_data()
    _, results = _run(tmp_path, manifest, files)
    assert [r.name for r in results] == list(fa.GUARDS)
    assert len(fa.GUARDS) == 12
    assert failing(results) == set()
    assert all(r.checked > 0 for r in results)


@pytest.mark.parametrize("guard,mutate", BAD_CASES, ids=[g for g, _ in BAD_CASES])
def test_each_guard_fails_on_its_bad_rows(tmp_path, guard, mutate):
    manifest, files = make_data()
    mutate(manifest, files)
    _, results = _run(tmp_path, manifest, files)
    assert guard in failing(results), [r.violations for r in results if r.violations]


def test_truncated_stream_is_a_measured_only_error(tmp_path):
    manifest, files = make_data()
    mpath = write_run(tmp_path / "run", manifest, files)
    path = tmp_path / "run" / "mlx-serve" / "q27_decode-1024-code-p0-r0.jsonl"
    path.write_text(path.read_text()[:-40])  # cut the last row mid-line
    _, results = fa.run_guards(tmp_path / "run", mpath)
    measured = next(r for r in results if r.name == "measured_only")
    assert any("unparseable" in v for v in measured.violations)


def test_unplanned_file_fails_manifest_completeness(tmp_path):
    manifest, files = make_data()
    files["mlx-serve/q27_decode-1024-code-p9-r0.jsonl"] = files[
        "mlx-serve/q27_decode-1024-code-p0-r0.jsonl"
    ]
    _, results = _run(tmp_path, manifest, files)
    assert "manifest_completeness" in failing(results)


def test_cli_exit_codes_name_the_guard(tmp_path, capsys):
    manifest, files = make_data()
    mpath = write_run(tmp_path / "good", manifest, files)
    assert fa.main(["--root", str(tmp_path / "good"), "--manifest", str(mpath)]) == 0
    bad_m, bad_f = make_data()
    bad_output_sha(bad_m, bad_f)
    mpath = write_run(tmp_path / "bad", bad_m, bad_f)
    assert fa.main(["--root", str(tmp_path / "bad"), "--manifest", str(mpath)]) == 1
    assert "output_sha: FAIL" in capsys.readouterr().out


# ---- --recompute ------------------------------------------------------------
GOOD_TABLE = """\
### Qwen3.8-27B — decode_tps
| engine | 1024-code-p0 | 1024-code-p1 |
|---|---|---|
| yunshu | **81.0 ± 2.0** | 81.0 ± 2.0 |
| mlx-serve | 81.0 ± 2.0 | 81.0 ± 2.0 |

### Qwen3.8-27B — cold_ttft
| engine | 1024-code-p0 | 1024-code-p1 |
|---|---|---|
| yunshu | [1.3 ± 0.2](yunshu/raw.jsonl) | 1.3 ± 0.2 |
| mlx-serve | 1.3 ± 0.2 | 1.3 ± 0.2 |

### Derived — Yunshu ratio to best
| engine | 1024-code-p0 |
|---|---|
| yunshu | 1.00x |
"""


def _recompute(tmp_path, table, mutate=None):
    manifest, files = make_data()
    if mutate:
        mutate(manifest, files)
    mpath = write_run(tmp_path / "run", manifest, files)
    run, results = fa.run_guards(tmp_path / "run", mpath)
    checked, viol = fa.recompute_table(run, table, fa.violations_by_cell(results))
    return checked, viol


def test_recompute_matches_good_table(tmp_path):
    checked, viol = _recompute(tmp_path, GOOD_TABLE)
    assert checked == 8 and viol == []


def test_recompute_flags_number_that_differs_from_raw(tmp_path):
    table = GOOD_TABLE.replace(
        "| 81.0 ± 2.0 | 81.0 ± 2.0 |\n\n### Qwen3.8-27B — cold",
        "| 99.0 ± 2.0 | 81.0 ± 2.0 |\n\n### Qwen3.8-27B — cold",
        1,
    )
    _, viol = _recompute(tmp_path, table)
    assert any(v.startswith("recompute:") and "table 99.0" in v for v in viol)


def test_recompute_accepts_invalid_only_when_a_guard_fails(tmp_path):
    table = GOOD_TABLE.replace(
        "| 81.0 ± 2.0 | 81.0 ± 2.0 |\n\n### Qwen3.8-27B — cold",
        "| invalid: prompt mismatch | 81.0 ± 2.0 |\n\n### Qwen3.8-27B — cold",
        1,
    )
    _, viol = _recompute(tmp_path, table)
    assert any("marked invalid but every guard passes" in v for v in viol)


def test_recompute_number_without_source_is_measured_only(tmp_path):
    table = GOOD_TABLE.replace(
        "| mlx-serve | 1.3 ± 0.2 | 1.3 ± 0.2 |",
        "| mlx-serve | 1.3 ± 0.2 | 1.3 ± 0.2 |\n| ghost | 5.0 ± 0.0 | 5.0 ± 0.0 |",
    )
    _, viol = _recompute(tmp_path, table)
    assert any(v.startswith("measured_only") and "ghost" in v for v in viol)


def test_schema_lists_fields_bench_engines_does_not_emit_yet():
    missing = fa.schema_fields_not_emitted()
    for field in ("prompt_ids_sha", "max_tokens", "temperature", "harness", "error"):
        assert field in missing["request"]
    for field in (
        "weights",
        "binary_sha256",
        "release_date",
        "job_start_s",
        "fresh_process",
        "engaged_log_line",
        "home",
    ):
        assert field in missing["header"]
    assert "ttft_s" not in missing["request"] and "sha" not in missing["request"]
