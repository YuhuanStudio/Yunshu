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


def synthetic_telemetry(swap_used=1000, throttled=False, n=20):
    """Synthetic telemetry series (NOT captured output): header, samples, footer."""
    thermal = {"cpu_speed_limit_pct": 100, "throttled": throttled}
    rows = [
        {
            "part": "telemetry_header",
            "pid": 1,
            "port": 1,
            "interval_ms": 100,
            "probes": {},
            "thermal_start": thermal,
        }
    ]
    for i in range(n):
        rows.append(
            {
                "part": "sample",
                "t": 1000.0 + i * 0.1,
                "proc": {"rss_bytes": 10**9 + i, "cpu_pct": 50.0},
                "swap": {"used_bytes": swap_used},
                "gpu": {"device_util_pct": 90},
                "thermal": thermal,
            }
        )
    rows.append({"part": "telemetry_footer", "n_samples": n, "thermal_end": thermal})
    return rows


def telemetry_file(cell_path: Path) -> Path:
    return fa.telemetry_path(cell_path)


def write_run(root: Path, manifest, files, telemetry=True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, rows in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        if telemetry:
            telemetry_file(path).write_text(
                "".join(json.dumps(r) + "\n" for r in synthetic_telemetry())
            )
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


def test_good_run_passes_all_guards(tmp_path):
    manifest, files = make_data()
    _, results = _run(tmp_path, manifest, files)
    assert [r.name for r in results] == list(fa.GUARDS)
    assert len(fa.GUARDS) == 13
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


# ---- telemetry guard (guard 13) ---------------------------------------------
def _tel_path(tmp_path, rel="mlx-serve/q27_decode-1024-code-p0-r0.jsonl"):
    return telemetry_file(tmp_path / "run" / rel)


def _tel_results(tmp_path, mutate_file=None, remove=False, telemetry_required=True):
    manifest, files = make_data()
    manifest["telemetry_required"] = telemetry_required
    mpath = write_run(tmp_path / "run", manifest, files)
    if remove:
        _tel_path(tmp_path).unlink()
    if mutate_file is not None:
        p = _tel_path(tmp_path)
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        mutate_file(rows)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _, results = fa.run_guards(tmp_path / "run", mpath)
    return next(r for r in results if r.name == "telemetry")


def test_telemetry_guard_passes_on_synthetic_good_series(tmp_path):
    res = _tel_results(tmp_path)
    assert res.violations == []
    assert res.checked > 0


def test_telemetry_missing_file_is_flagged(tmp_path):
    res = _tel_results(tmp_path, remove=True)
    assert any("missing telemetry" in v for v in res.violations)


def test_telemetry_missing_allowed_when_not_required(tmp_path):
    res = _tel_results(tmp_path, remove=True, telemetry_required=False)
    assert not any("missing telemetry" in v for v in res.violations)


def test_telemetry_thermal_throttle_in_a_sample_is_flagged(tmp_path):
    def throttle(rows):
        for r in rows:
            if r.get("part") == "sample":
                r["thermal"] = {"cpu_speed_limit_pct": 70, "throttled": True}
                break

    res = _tel_results(tmp_path, mutate_file=throttle)
    assert any("thermal throttling flagged (sample)" in v for v in res.violations)


def test_telemetry_swap_growth_is_flagged(tmp_path):
    def grow(rows):
        samples = [r for r in rows if r.get("part") == "sample"]
        samples[-1]["swap"] = {"used_bytes": 5 * 2**20}

    res = _tel_results(tmp_path, mutate_file=grow)
    assert any("swap grew" in v for v in res.violations)


def test_telemetry_no_samples_is_flagged(tmp_path):
    res = _tel_results(
        tmp_path,
        mutate_file=lambda rows: rows.__setitem__(
            slice(None), [r for r in rows if r.get("part") != "sample"]
        ),
    )
    assert any("no samples" in v for v in res.violations)


# ---- derived efficiency (labeled DERIVED) ------------------------------------
FIX = Path(__file__).resolve().parent / "fixtures" / "fullcmp_telemetry"


def test_arch_from_real_27b_config_matches_table():
    arch = fa.arch_from_config(FIX / "qwen38_27b_config.json")
    m = fa.MODELS["q27-4bit"]
    assert arch["n_layers"] == 64
    assert arch["full_attn_layers"] == m["full_attn_layers"] == 16
    assert arch["linear_layers"] == m["linear_layers"] == 48
    assert arch["kv_heads"] == m["kv_heads"] == 4
    assert arch["head_dim"] == m["head_dim"] == 256
    assert arch["n_heads"] == m["n_heads"] == 24


def test_arch_from_real_flashnext_config_matches_table():
    arch = fa.arch_from_config(FIX / "qwen38_flashnext_config.json")
    m = fa.MODELS["flashnext"]
    assert arch["n_layers"] == m["n_layers"] == 48
    assert arch["full_attn_layers"] == m["full_attn_layers"] == 12
    assert arch["kv_heads"] == m["kv_heads"] == 2


def test_kv_and_gdn_bytes_for_27b():
    m = fa.MODELS["q27-4bit"]
    # 16 layers * 2 (K,V) * 4 kv heads * 256 * 2 bytes = 65536 B per context token (64 KiB)
    assert fa.kv_bytes_per_token(m, 1) == 65536
    # 48 linear layers * 48 heads * 128 * 128 * 4 bytes (fp32 state)
    assert fa.gdn_state_bytes(m) == 48 * 48 * 128 * 128 * 4


PEAKS = {
    "decode_gbs": {
        "max": 420.0,
        "median": 390.0,
        "stable": 400.0,
        "stable_stat": "top_half_median",
        "spread_pct": 4.0,
        "n": 3,
    },
    "bf16_tflops": {"max": 20.0, "median": 19.0, "spread_pct": 3.0, "n": 3},
    "read_gbs": {"max": 500.0, "median": 480.0, "spread_pct": 2.0, "n": 3},
}


def _bytes_round(model_id, pt, ct):
    m = fa.MODELS[model_id]
    ctx = pt + ct / 2.0
    return m["weight_bytes"] + fa.kv_bytes_per_token(m, ctx) + fa.gdn_state_bytes(m)


def test_derived_mbu_uses_stable_gemv_peak_not_spec_or_max():
    d = fa.derived_decode("q27-4bit", PEAKS, tok_s=40.0, pt=1024, ct=512)
    expect_bytes = _bytes_round("q27-4bit", 1024, 512)
    assert d["label"] == "DERIVED"
    assert d["tokens_per_round"] == 1.0  # plain AR: one token per round
    assert d["bytes_per_round"] == pytest.approx(expect_bytes)
    assert d["mbu"] == pytest.approx(expect_bytes * 40.0 / (400.0 * 1e9))
    assert d["peak_decode_gbs"] == 400.0
    assert d["peak_decode_stat"] == "top_half_median"
    # secondary column: same bytes against the spec-sheet bandwidth, labeled as such
    assert d["mbu_vs_spec"] == pytest.approx(
        expect_bytes * 40.0 / (fa.SPEC_SHEET["M5 Max"]["bw_gbs"] * 1e9)
    )


def test_ar_decode_30_tok_s_mbu_is_tok_s_times_bytes_over_peak():
    d = fa.derived_decode(
        "q27-4bit", PEAKS, tok_s=30.0, pt=1024, ct=512, speculative=False
    )
    # formula only: 30 tok/s x ~16 GB/token exceeds the 400 GB/s synthetic peak, so no bound here
    assert d["mbu"] == pytest.approx(_bytes_round("q27-4bit", 1024, 512) * 30.0 / 400e9)
    assert d["mbu"] > 0


def test_spec_decode_80_tok_s_at_3_7_tok_per_round_is_plausible():
    # 80 tok/s with 3.7 tokens per round = 21.6 rounds/s; weights are read once per round
    d = fa.derived_decode(
        "q27-4bit",
        PEAKS,
        tok_s=80.0,
        pt=1024,
        ct=512,
        speculative=True,
        tokens_per_round=3.7,
        rows_per_round=4,
    )
    assert d["rounds_per_s"] == pytest.approx(80.0 / 3.7)
    assert d["mbu"] == pytest.approx(
        _bytes_round("q27-4bit", 1024, 512) * (80.0 / 3.7) / 400e9
    )
    assert 0 < d["mbu"] < 1
    # the old per-token formula would have been > 1 at this peak: that was the bug
    naive = _bytes_round("q27-4bit", 1024, 512) * 80.0 / 400e9
    assert naive > 1
    # MFU: 4 verified rows per round, measured from the record
    flops_round = 4 * (
        2 * fa.MODELS["q27-4bit"]["active_params"]
        + fa.attn_flops_per_token(fa.MODELS["q27-4bit"], 1024 + 256)
    )
    assert d["mfu"] == pytest.approx(flops_round * (80.0 / 3.7) / 20e12)
    assert d["rows_per_round_src"] == "record"


def test_spec_decode_without_tokens_per_round_is_unavailable_not_a_number():
    d = fa.derived_decode(
        "q27-4bit", PEAKS, tok_s=80.0, pt=1024, ct=512, speculative=True
    )
    assert d["mbu"] is None and d["mbu_vs_spec"] is None
    assert "unavailable (tokens_per_round missing)" in d["reason"]
    assert d["mfu"] is None


def test_spec_mfu_rows_fallback_is_labeled_lower_bound():
    d = fa.derived_decode(
        "q27-4bit",
        PEAKS,
        tok_s=80.0,
        pt=1024,
        ct=512,
        speculative=True,
        tokens_per_round=4.0,
    )
    assert d["rows_per_round_src"] == "lower bound (tokens_per_round)"
    assert d["rows_per_round"] == 4.0
    assert d["mbu"] == pytest.approx(_bytes_round("q27-4bit", 1024, 512) * 20.0 / 400e9)


def test_is_speculative_maps_engaged_modes():
    assert fa.is_speculative("off") is False
    assert fa.is_speculative("") is False
    assert fa.is_speculative("mtp") is True
    assert fa.is_speculative("dflash") is True
    assert fa.is_speculative(None) is None


def test_spec_flops_unsourced_mfu_vs_spec_is_none_with_reason():
    d = fa.derived_decode("q27-4bit", PEAKS, tok_s=40.0, pt=1024, ct=512)
    assert d["mfu"] is not None and d["mfu"] >= 0
    assert d["mfu_vs_spec"] is None
    assert "spec-sheet FLOP/s unsourced" in d["reason"]


def test_derived_mfu_uses_measured_bf16_peak():
    d = fa.derived_prefill("q27-4bit", PEAKS, pt=4096, ttft_s=2.0)
    flops_tok = 2 * fa.MODELS["q27-4bit"]["active_params"] + fa.attn_flops_per_token(
        fa.MODELS["q27-4bit"], 2048
    )
    assert d["mfu"] == pytest.approx(flops_tok * 2048.0 / (20.0 * 1e12))
    assert d["prefill_tok_s"] == pytest.approx(2048.0)


def test_derived_flashnext_is_none_with_reason():
    d = fa.derived_decode("flashnext", PEAKS, tok_s=60.0, pt=1024, ct=512)
    assert d["mbu"] is None and d["mfu"] is None
    assert "weight bytes per token not in repo" in d["reason"]


def test_load_measured_peaks_fails_closed_when_missing(tmp_path):
    with pytest.raises(fa.MeasuredPeaksError, match="missing"):
        fa.load_measured_peaks(tmp_path / "nope.json")


def test_load_measured_peaks_rejects_invalid_and_incomplete(tmp_path):
    import probe_peak as pp

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(fa.MeasuredPeaksError, match="invalid"):
        fa.load_measured_peaks(bad)
    rep = pp.make_report(
        [pp.make_result("bw_read_sum_bf16", "GB/s", [1.0, 2.0, 3.0])],
        mlx_version="0.0.0",
        chip="Apple M5 Max",
        date="2026-11-03",
    )
    f = tmp_path / "partial.json"
    f.write_text(json.dumps(rep))
    with pytest.raises(fa.MeasuredPeaksError, match="lacks"):
        fa.load_measured_peaks(f)


def test_load_measured_peaks_reads_valid_file(tmp_path):
    import probe_peak as pp

    res = [
        pp.make_robust_result(
            "qgemv_w4g64_M1", "GB/s", [300.0, 320.0, 310.0, 330.0, 305.0, 315.0]
        ),
        pp.make_result("matmul_prefill27b_bf16", "TFLOP/s", [18.0, 19.0, 20.0]),
        pp.make_result("bw_read_sum_bf16", "GB/s", [500.0, 510.0, 505.0]),
    ]
    rep = pp.make_report(
        res, mlx_version="0.0.0", chip="Apple M5 Max", date="2026-11-03"
    )
    f = tmp_path / "peak.json"
    f.write_text(json.dumps(rep))
    got = fa.load_measured_peaks(f)
    assert got["decode_gbs"]["max"] == 330.0
    # the MBU denominator is the stable statistic (top-half median), and the JSON names it
    assert got["decode_gbs"]["stable_stat"] == "top_half_median"
    assert got["decode_gbs"]["stable"] == pytest.approx(
        pp.statistics.median([315.0, 320.0, 330.0])
    )
    assert got["bf16_tflops"]["max"] == 20.0
    assert got["read_gbs"]["max"] == 510.0


def test_load_measured_peaks_refuses_file_without_stable_statistic(tmp_path):
    import probe_peak as pp

    rep = pp.make_report(
        [
            pp.make_result("qgemv_w4g64_M1", "GB/s", [300.0, 320.0, 310.0]),
            pp.make_result("matmul_prefill27b_bf16", "TFLOP/s", [18.0, 19.0, 20.0]),
            pp.make_result("bw_read_sum_bf16", "GB/s", [500.0, 510.0, 505.0]),
        ],
        mlx_version="0.0.0",
        chip="Apple M5 Max",
        date="2026-11-03",
    )
    f = tmp_path / "old.json"
    f.write_text(json.dumps(rep))
    with pytest.raises(fa.MeasuredPeaksError, match="stable"):
        fa.load_measured_peaks(f)


def test_derived_rows_only_for_known_models_with_measured_inputs(tmp_path):
    manifest, files = make_data()
    mpath = write_run(tmp_path / "run", manifest, files)
    run = fa.load_run(tmp_path / "run", json.loads(mpath.read_text()))
    # synthetic model id "Qwen3.8-27B" is not in MODELS: nothing is derived for it
    assert fa.derived_rows(run, PEAKS) == []
    for cf in run.files:
        cf.cell["model"] = "q27-4bit"
    # the synthetic cells are speculative (dflash) and carry no tokens_per_round: decode is unavailable
    rows = fa.derived_rows(run, PEAKS)
    dec = [r for r in rows if r["metric"] == "decode"]
    assert dec and all(r["mbu"] is None for r in dec)
    assert all("unavailable (tokens_per_round missing)" in r["reason"] for r in dec)
    # prefill is unaffected and still uses the measured bf16 peak
    pre = [r for r in rows if r["metric"] == "prefill"]
    assert pre and all(r["mfu"] is not None for r in pre)
    # with the record carrying tokens_per_round, decode becomes a number again
    for cf in run.files:
        for row in cf.rows:
            row["tokens_per_round"] = 3.0
    dec = [r for r in fa.derived_rows(run, PEAKS) if r["metric"] == "decode"]
    assert dec and all(r["mbu"] is not None and r["mbu"] > 0 for r in dec)
    assert all(r["label"] == "DERIVED" for r in dec)
