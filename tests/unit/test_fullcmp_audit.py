"""CPU-only tests for the full-comparison audit (scripts/research/fullcmp_audit.py).

A synthetic run (2 engines x 2 prompts x 3 reps x cold/warm/turn2) must pass every guard;
each guard then has one synthetic bad row set that must fail with that guard's name.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_audit as fa  # noqa: E402
import fullcmp_engaged as fe  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402

PY = "/tree/.venv/bin/python"
MODEL = "Qwen3.8-27B"
ENG = ("yunshu", "mlx-serve-26.11.1")
PROMPTS = ("p0", "p1")
REPS = 3
W = "a" * 64
B = "b" * 64


def quiet_record(total=0.5, holders=(), limit=150.0, single=20.0):
    """A quiet_pre / quiet_post snapshot as fullcmp_quiet.check() records it (`limit` = total)."""
    hs = [dict(h) for h in holders]
    violators = [h for h in hs if h["pct"] >= single and h["name"] not in ("WindowServer", "kernel_task", "mds_stores")]
    return {
        "ok": total < limit and not violators,
        "max_single_pct": single,
        "max_total_pct": limit,
        "total_foreign_pct": total,
        "holders": hs,
        "violators": violators,
        "top": [],
        "load_avg": {"start": 2.0, "end": 2.0, "mean": 2.0},
        "window_s": 10.0,
        "samples": 11,
        "at": 1.0,
    }


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def attr_fixture(eng, row):
    """Layer-1 attribution fields for one decode row: every REQUIRED field the engine can expose
    is a value, the rest are its explicit unavailable reasons (fullcmp_attribution schema)."""
    import fullcmp_attribution as fa_attr

    spec = fa_attr.engine_attr(eng)
    t = 1000.0 + row["ttft_s"]
    values = {
        "t_receive": t - row["ttft_s"],
        "t_template": t - 0.05,
        "t_tokenize": t - 0.04,
        "t_admit": t - 0.03,
        "t_cache_lookup": t - 0.02,
        "t_prefill_chunks": [[t - 0.02, t, row["pt"]]],
        "t_first_token": t,
        "t_rounds": [t + 0.1],
        "t_last_token": t + row["total_s"] - 0.1,
        "t_response_end": t + row["total_s"],
        "rounds": 128,
        "tokens_per_round": [4],
        "rows_per_round": [1],
        "cache_hit_tokens": row["cached"],
        "chunk_sizes": [row["pt"]],
        "host_gaps_ms": [0.1],
        "engine_log_path": f"/tmp/x/{eng}.log",
    }
    attr = {k: v for k, v in values.items() if k not in spec.unavailable}
    return {"attr": attr, "attr_unavailable": dict(spec.unavailable)}


def make_data():
    """Return (manifest, files): files maps 'engine/<cell>-rN.jsonl' to its rows."""
    pins = {
        "yunshu": {
            "version": "main",
            "release_date": "2026-11-02",
            "git_sha": "a1b2c3d",
            "binary_sha256": B,
        },
        "mlx-serve-26.11.1": {
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
                    "binary_sha256": B,
                    "cmd": [PY, "-m", "yunshu_cli", "serve"]
                    if yun
                    else ["/bin/mlx-serve", "--serve"],
                    "python": PY if yun else None,
                    "observe": False if yun else None,
                    "yunshu_observe": None,
                    "profile": False,
                    "requested_spec_mode": "dflash",
                    "spec_mode_expected": "dflash",
                    "engaged_spec_mode": "dflash",
                    "engaged_log_line": "VLM batch runner: model draft=dflash"
                    if yun
                    else FIXTURE_LOG_LINE,
                    "engaged_evidence_source": None if yun else "slice",
                    "server_log_ready_offset": 0,
                    "weights": [
                        {
                            "file": f"{eng}.safetensors",
                            "sha256": W,
                            "bits": 4,
                            "group_size": 64,
                        }
                    ],
                    "fresh_process": True,
                    "harness_commit": "a" * 40,
                    "harness_dirty": False,
                    "home": f"/Volumes/P5Plus/yunshu-build/homes/{eng}/job-{slot}",
                    "job_id": f"job-{slot}",
                    "job_start_s": start,
                    "job_end_s": start + 50,
                    "gpu_exclusive": True,
                    "contended": False,
                    "device": "m5",
                    "quiet_pre": quiet_record(),
                    "quiet_post": quiet_record(),
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
                            "prompt_file_sha256": _sha(f"code-1024-{p}.txt"),
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
                for r_ in rows[1:]:
                    r_.update(attr_fixture(eng, r_))
                # tfbench writes part_done after the decode part completes
                rows.append({"part": "part_done", "which": "decode", "complete": True})
                # per-request acceptance (engaged evidence): every decode row drafted and accepted
                for r in rows:
                    if r.get("part") == "decode":
                        r["attr"]["accepted"] = 3
                        r["attr"]["drafted"] = 4
                        r["attr"]["rounds"] = 2
                if eng != "yunshu":  # attribution per measured row, as tfbench writes it (part=attr)
                    for k, r in enumerate([r for r in rows if r.get("part") == "decode"]):
                        rows.append({"part": "attr", "row": k, "accepted": 3, "drafted": 4, "rounds": 2,
                                     "tokens_per_round": round(r.get("ct", 1) / 2, 3)
                                     if isinstance(r.get("ct"), int) else "unavailable (no count)",
                                     "matched_line": FIXTURE_LOG_LINE})
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


@pytest.fixture(autouse=True)
def _prompt_table(monkeypatch):
    """Every fixture prompt file's sha is what its rows record (the real table lists corpus files only)."""

    class AnyTable(dict):
        def get(self, name, default=None):
            return _sha(name)

    monkeypatch.setattr(fa, "load_prompt_file_table", lambda: AnyTable())


FIXTURE_LOG_LINE = "speculative: dflash drafter engaged"


@pytest.fixture(autouse=True)
def _engaged_table(monkeypatch):
    """Engaged evidence for the fixture engine: the fixture's dflash line in its log slice (per
    request) and positive per-request acceptance (fullcmp_engaged; the real table is separate)."""
    entry = {
        "evidence": [
            {
                "kind": "log_regex",
                "pattern_or_path": FIXTURE_LOG_LINE,
                "source_ref": "fixture",
                "real_example": FIXTURE_LOG_LINE,
                "negative_example": "speculative: mtp head engaged",
                "per_request": True,
                "startup_sufficient": False,
            },
            {
                "kind": "acceptance_stat",
                "pattern_or_path": "attr.accepted",
                "source_ref": "fixture",
                "real_example": "attr.accepted=3",
                "negative_example": "attr.accepted=0",
                "per_request": True,
                "startup_sufficient": False,
            },
        ],
        "fallback": [],
    }
    table = {"_meta": {"fixture": True}, "engines": {"mlx-serve-26.11.1": {"modes": {"dflash": entry}}}}
    monkeypatch.setattr(fe, "load_table", lambda: table)


def write_run(root: Path, manifest, files, telemetry=True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, rows in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        # the cell's engine-log slice (tfbench writes <cell>-rN.server.log beside each cell file)
        line = (
            "VLM batch runner: model draft=dflash"
            if rel.startswith("yunshu/")
            else FIXTURE_LOG_LINE
        )
        if any(isinstance(r, dict) and r.get("_no_engine_log_evidence") for r in rows):
            line = "[round-profile] engine=fixture round=1 tokens=4"
        fa.server_log_slice_path(path).write_text(
            line + "\n[round-profile] engine=fixture round=1 drafts=4 accepted=3 tokens=4\n"
        )
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
    _rows(f, "p0-r0", "mlx-serve-26.11.1")[0][1]["prompt_ids_sha"] = "sha-other"


def bad_provenance_identity(m, f):
    for rows in _rows(f, "p0", "mlx-serve-26.11.1"):
        for r in rows[1:]:
            r["harness"] = "tfbench-other"


def bad_pinned_versions(m, f):
    for rows in _rows(f, "", "yunshu"):
        rows[0]["run_id"] = "fullcmp-OLD"


def bad_manifest_completeness(m, f):
    del f["mlx-serve-26.11.1/q27_decode-1024-code-p1-r2.jsonl"]


def bad_record_count(m, f):
    rows = f["yunshu/q27_decode-1024-code-p0-r0.jsonl"]
    del rows[-1]


def bad_engaged_mode(m, f):
    for rows in _rows(f, "", "mlx-serve-26.11.1"):
        rows[0]["engaged_spec_mode"] = "mtp"
        rows[0]["engaged_log_line"] = "speculative: mtp head engaged"


def bad_engaged_no_evidence(m, f):
    """Header says engaged == requested, but the engine log slice has no engaged line at all and the
    per-request acceptance is zero: the request was copied into the header (fail closed).
    write_run reads the marker and writes an empty engine-log slice for that file."""
    for rows in _rows(f, "", "mlx-serve-26.11.1"):
        rows[0]["_no_engine_log_evidence"] = True
        rows[0]["engaged_spec_mode"] = "dflash"
        rows[0]["engaged_log_line"] = FIXTURE_LOG_LINE
        for r in rows:
            if r.get("part") == "decode":
                r.setdefault("attr", {})["accepted"] = 0


def bad_weight_identity(m, f):
    for rows in _rows(f, "", "mlx-serve-26.11.1"):
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


def bad_attribution_fields(m, f):
    # A decode row with no attr at all (a layer-1 field MISSING) fails the cell.
    rows = f[next(k for k in f if k.startswith("yunshu/"))]
    rows[1].pop("attr")


def bad_quiet_machine(m, f):
    head = f["yunshu/q27_decode-1024-code-p0-r0.jsonl"][0]
    head["quiet_post"] = quiet_record(
        total=43.0, holders=[{"pid": 700, "name": "stray", "pct": 43.0}]
    )


def bad_yunshu_binary(m, f):
    # a yunshu header launched from a different python than the pinned one must fail the cell
    for rows in _rows(f, "", "yunshu"):
        rows[0]["binary_sha256"] = "c" * 64


def bad_observe_on_timed(m, f):
    # a timed Yunshu rep that ran observe on (YUNSHU_OBSERVE=1) must fail the cell
    for rows in _rows(f, "", "yunshu"):
        rows[0]["observe"] = True
        rows[0]["yunshu_observe"] = "1"


def bad_observe_field_missing(m, f):
    for rows in _rows(f, "", "yunshu"):
        del rows[0]["observe"]


def bad_profile_flag_on_timed(m, f):
    # a timed rep carrying profile=true (TFB_PROFILE leaked) fails: profile true needs observe on
    for rows in _rows(f, "", "yunshu"):
        rows[0]["profile"] = True


BAD_CASES = [
    ("observe_state", bad_observe_on_timed),
    ("observe_state", bad_observe_field_missing),
    ("observe_state", bad_profile_flag_on_timed),
    ("pinned_versions", bad_yunshu_binary),
    ("prompt_identity", bad_prompt_identity),
    ("provenance_identity", bad_provenance_identity),
    ("pinned_versions", bad_pinned_versions),
    ("manifest_completeness", bad_manifest_completeness),
    ("record_count", bad_record_count),
    ("engaged_mode", bad_engaged_mode),
    ("engaged_mode", bad_engaged_no_evidence),
    ("weight_identity", bad_weight_identity),
    ("cold_cache", bad_cold_cache),
    ("gpu_exclusive_rotation", bad_gpu_exclusive_rotation),
    ("output_sha", bad_output_sha),
    ("measured_only", bad_measured_only),
    ("fairness_controls", bad_fairness_controls),
    ("quiet_machine", bad_quiet_machine),
    ("attribution_fields", bad_attribution_fields),
]


def _run(tmp_path, manifest, files):
    mpath = write_run(tmp_path / "run", manifest, files)
    return fa.run_guards(tmp_path / "run", mpath)


def test_good_run_passes_all_guards(tmp_path):
    manifest, files = make_data()
    _, results = _run(tmp_path, manifest, files)
    assert [r.name for r in results] == list(fa.GUARDS)
    assert len(fa.GUARDS) == len(fa.GUARD_CHECKS) and "mixed_same_prompt" in fa.GUARDS
    assert failing(results) == set()
    # family_validity and behaviour pass with zero checks here, and only because the manifest has
    # no family cell (no tool/json/cjk/thinking/image cell) and no behaviour section
    assert not any(
        fm.is_family_cell(c["kind"], c.get("thinking"), c.get("media")) for c in manifest["cells"]
    )
    assert not manifest.get("behaviour")
    vacuous = {r.name for r in results if r.checked == 0}
    assert vacuous == {"family_validity", "behaviour"}
    assert all(r.checked > 0 for r in results if r.name not in vacuous)


def test_family_validity_with_no_family_rows_fails_when_the_manifest_has_family_cells(tmp_path):
    manifest, files = make_data()
    for c in manifest["cells"]:
        c["kind"] = "tool-chat"  # the manifest now plans family cells ...
    _, results = _run(tmp_path, manifest, {})  # ... and no cell file (no rows) was written
    assert "family_validity" in failing(results)


@pytest.mark.parametrize("guard,mutate", BAD_CASES, ids=[g for g, _ in BAD_CASES])
def test_each_guard_fails_on_its_bad_rows(tmp_path, guard, mutate):
    manifest, files = make_data()
    mutate(manifest, files)
    _, results = _run(tmp_path, manifest, files)
    assert guard in failing(results), [r.violations for r in results if r.violations]


def test_truncated_stream_is_a_measured_only_error(tmp_path):
    manifest, files = make_data()
    mpath = write_run(tmp_path / "run", manifest, files)
    path = tmp_path / "run" / "mlx-serve-26.11.1" / "q27_decode-1024-code-p0-r0.jsonl"
    path.write_text(path.read_text()[:-40])  # cut the last row mid-line
    _, results = fa.run_guards(tmp_path / "run", mpath)
    measured = next(r for r in results if r.name == "measured_only")
    assert any("unparseable" in v for v in measured.violations)


def test_unplanned_file_fails_manifest_completeness(tmp_path):
    manifest, files = make_data()
    files["mlx-serve-26.11.1/q27_decode-1024-code-p9-r0.jsonl"] = files[
        "mlx-serve-26.11.1/q27_decode-1024-code-p0-r0.jsonl"
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
| mlx-serve-26.11.1 | 81.0 ± 2.0 | 81.0 ± 2.0 |

### Qwen3.8-27B — cold_ttft
| engine | 1024-code-p0 | 1024-code-p1 |
|---|---|---|
| yunshu | [1.3 ± 0.2](yunshu/raw.jsonl) | 1.3 ± 0.2 |
| mlx-serve-26.11.1 | 1.3 ± 0.2 | 1.3 ± 0.2 |

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
        "| mlx-serve-26.11.1 | 1.3 ± 0.2 | 1.3 ± 0.2 |",
        "| mlx-serve-26.11.1 | 1.3 ± 0.2 | 1.3 ± 0.2 |\n| ghost | 5.0 ± 0.0 | 5.0 ± 0.0 |",
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
def _tel_path(tmp_path, rel="mlx-serve-26.11.1/q27_decode-1024-code-p0-r0.jsonl"):
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


# --- guards 3/4/5 regressions (smoke findings) ------------------------------


def _cf(cell, rep, header, rows=()):
    return fa.CellFile(cell, rep, Path("x"), header, list(rows))


def test_pinned_versions_maps_yunshu_main_to_the_yunshu_pin():
    """A cell folder / harness id 'yunshu-main' is the manifest 'yunshu' pin, not a binary."""
    sha = "0f2960047855546a9f13b2631b680352b1c57cfe"
    manifest = {
        "run_id": "r1",
        "pins": {
            "yunshu": {
                "version": "0.1.7.dev0",
                "release_date": "2026-11-03",
                "git_sha": sha,
                "binary_sha256": B,
            }
        },
        "yunshu": {"git_sha": sha, "python": PY},
    }
    head = {
        "run_id": "r1",
        "version": "0.1.7.dev0",
        "release_date": "2026-11-03",
        "git_sha": sha,
        "python": PY,
        "cmd": [PY, "-m", "yunshu_cli", "serve"],
        "binary_sha256": B,
    }
    cell = {"cell": "c", "engine": "yunshu-main", "group": "g"}
    run = fa.Run(manifest, [_cf(cell, 0, head)], [], [], [])
    assert fa.check_pinned_versions(run).violations == []
    bad = dict(head, git_sha="deadbeef")
    run = fa.Run(manifest, [_cf(cell, 0, bad)], [], [], [])
    assert any("pinned tree" in v for v in fa.check_pinned_versions(run).violations)


def test_manifest_engine_inverts_the_registry():
    assert fa.manifest_engine("yunshu-main") == "yunshu"
    assert fa.manifest_engine("yunshu") == "yunshu"
    assert fa.manifest_engine("tf-1.2.1") == "tf-1.2.1"


def test_rotation_dedupes_two_cells_of_one_engine():
    """Two cells of one engine run back to back: that is one visit, not two in the interleave."""
    order = ["yunshu", "mlx-serve-26.11.1"]
    manifest = {"groups": {"g": {"order": order}}}

    def hdr(s, e):
        return {
            "gpu_exclusive": True,
            "contended": False,
            "job_start_s": s,
            "job_end_s": e,
        }

    c1 = {"cell": "c1", "engine": "yunshu", "group": "g"}
    c2 = {"cell": "c2", "engine": "yunshu", "group": "g"}
    c3 = {"cell": "c3", "engine": "mlx-serve-26.11.1", "group": "g"}
    files = [
        _cf(c1, 0, hdr(0, 10)),
        _cf(c2, 0, hdr(10, 20)),
        _cf(c3, 0, hdr(20, 30)),
    ]
    run = fa.Run(manifest, files, [], [], [])
    assert fa.check_gpu_exclusive_rotation(run).violations == []


def test_home_is_unique_per_server_job_not_per_engine():
    def cf(cell, job, home):
        head = {"job_id": job, "home": home, "run_id": "r"}
        return _cf(
            {"cell": cell, "engine": "yunshu", "group": "g"},
            0,
            head,
            [{"temperature": 0}],
        )

    manifest = {
        "controls": {
            "aa_delta_pct": 0.0,
            "aa_noise_pct": 1.0,
            "injected_delay_detected": True,
        }
    }
    # two cells of one server (same job, same home) plus a second server of the same engine
    ok = [cf("a", "j1", "/h/1"), cf("b", "j1", "/h/1"), cf("c", "j2", "/h/2")]
    assert fa.check_fairness_controls(fa.Run(manifest, ok, [], [], [])).violations == []
    shared = [cf("a", "j1", "/h/1"), cf("c", "j2", "/h/1")]
    res = fa.check_fairness_controls(fa.Run(manifest, shared, [], [], []))
    assert any("shared" in v for v in res.violations)
    real = [cf("a", "j1", fa.os.path.expanduser("~"))]
    res = fa.check_fairness_controls(fa.Run(manifest, real, [], [], []))
    assert any("real HOME" in v for v in res.violations)


# --- conc cells (tfbench part=conc) are audited -------------------------------


def _conc_record(trial=0, n=2, ct=16, drop=None):
    rec = {
        "part": "conc",
        "n": n,
        "trial": trial,
        # schema (tfbench request_records): one entry per request of the record
        "requests": [{"rid": i, "pt": 1024, "ct": ct, "cached": 0, "ttft_s": 0.5, "dec_tps": 30.0,
                      "spec": {"accepted": 3, "drafted": 4, "rounds": 2}} for i in range(n)],
        "ttfts": [0.5 + i for i in range(n)],
        "per_req_dec": [30.0] * n,
        "total_s": [3.0] * n,
        "cts": [ct] * n,
        "pts": [1034] * n,
        "cached": [0] * n,
        "shas": ["a" * 16] * n,
        "finish": ["length"] * n,
        "prompt_ids_sha": ["c" * 16] * n,
        "max_tokens": 16,
        "temperature": 0,
        "stop": [],
        "harness": "tfbench/fullcmp-1",
        "ctx": 1024,
        "kind": "mixed",
        "prompt_file": "concurrent_prompt",
    }
    for k in drop or ():
        rec.pop(k)
    return rec


def test_conc_record_expands_to_one_row_per_request():
    rows = fa.conc_request_rows(_conc_record(n=3))
    assert len(rows) == 3
    assert all(r["part"] == "conc" and r["phase"] == "conc" for r in rows)
    assert [r["ttft_s"] for r in rows] == [0.5, 1.5, 2.5]
    assert all(r["finish"] == "length" and r["ct"] == 16 for r in rows)


def test_conc_request_missing_a_list_entry_is_not_silently_counted():
    rec = _conc_record(n=2)
    rec["finish"] = ["length"]  # second request has no finish reason recorded
    rows = fa.conc_request_rows(rec)
    cell = {
        "cell": "c",
        "engine": "yunshu",
        "group": "g",
        "expected_records": 2,
        "phases": ["conc"],
        "max_tokens": 16,
    }
    run = fa.Run({"groups": {}}, [_cf(cell, 0, {"part": "session"}, rows)], [], [], [])
    res = fa.check_record_count(run)
    assert any("incomplete stream" in v for v in res.violations)


def _b1_decode_row(phase, ct=512):
    return {
        "part": "decode",
        "phase": phase,
        "ttft_s": 0.5,
        "total_s": 5.0,
        "ct": ct,
        "pt": 1024,
        "cached": 0,
        "finish": "length",
    }


def _b1_cell(name, phases=("cold", "warm", "fu")):
    return {
        "cell": name,
        "engine": "yunshu",
        "group": "B1.64K.prose.c1",
        "expected_records": len(phases),
        "phases": list(phases),
        "max_tokens": 512,
        "ctx": 65536,
        "kind": "prose",
        "thinking": "off",
        "media": "none",
        "reps": 1,
    }


def test_truncated_cell_without_part_done_is_a_violation_naming_the_cell(tmp_path):
    # the 1103-220358 shape: the 64K prose cell wrote its cold row, then the job was killed
    # (no warm/turn2 rows, no part_done); the 8K cell never ran (no file at all)
    trunc = _b1_cell("B1.core.yunshu.64K.c1.prose.o512")
    missing8k = _b1_cell("B1.core.yunshu.8K.c1.code.o512")
    missing8k["ctx"], missing8k["kind"] = 8192, "code"
    manifest = {"reps": 1, "cells": [trunc, missing8k]}
    d = tmp_path / "root" / "yunshu"
    d.mkdir(parents=True)
    lines = [
        {"part": "session", "run_id": "r", "job_start_s": 1.0},
        _b1_decode_row("cold"),
    ]
    (d / "B1.core.yunshu.64K.c1.prose.o512-r0.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines)
    )
    run = fa.load_run(tmp_path / "root", manifest)
    assert run.files[0].truncated == ("decode",)
    rec = fa.check_record_count(run)
    assert any(
        "B1.core.yunshu.64K.c1.prose.o512" in v and "truncated" in v and "part_done" in v
        for v in rec.violations
    )
    assert any("record" in v or "request rows" in v for v in rec.violations)
    comp = fa.check_manifest_completeness(run)
    assert any(
        "B1.core.yunshu.8K.c1.code.o512" in v and "missing" in v for v in comp.violations
    )


def test_complete_cell_with_part_done_has_no_truncation(tmp_path):
    cell = _b1_cell("B1.core.yunshu.1K.c1.code.o512", phases=("cold",))
    manifest = {"reps": 1, "cells": [cell]}
    d = tmp_path / "root" / "yunshu"
    d.mkdir(parents=True)
    lines = [
        {"part": "session", "run_id": "r"},
        _b1_decode_row("cold"),
        {"part": "part_done", "which": "decode", "complete": True},
    ]
    (d / "B1.core.yunshu.1K.c1.code.o512-r0.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines)
    )
    run = fa.load_run(tmp_path / "root", manifest)
    assert run.files[0].truncated == ()
    assert not any("truncated" in v for v in fa.check_record_count(run).violations)


def test_conc_cell_is_read_from_disk_and_counts_its_requests(tmp_path):
    cell = {
        "cell": "q27_conc-1024-code",
        "engine": "yunshu",
        "group": "g",
        "expected_records": 4,
        "phases": ["conc"],
        "max_tokens": 16,
        "ctx": 1024,
        "kind": "mixed",
    }
    manifest = {"reps": 1, "cells": [cell]}
    d = tmp_path / "root" / "yunshu"
    d.mkdir(parents=True)
    head = {"part": "session", "run_id": "r"}
    # tfbench writes part_done after the part; a conc part without it would be truncated
    done = {"part": "part_done", "which": "conc", "complete": True}
    lines = [head, _conc_record(trial=0, n=2), _conc_record(trial=1, n=2), done]
    (d / "q27_conc-1024-code-r0.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines)
    )
    run = fa.load_run(tmp_path / "root", manifest)
    assert run.missing == [] and len(run.files) == 1
    assert len(run.files[0].rows) == 4
    assert fa.check_record_count(run).violations == []


# ---------------------------------------------------------------------------
# quiet_machine (guard 14) and round_time_outlier (guard 15): direct, on fake Run objects
# ---------------------------------------------------------------------------


def _fake_run(files):
    return fa.Run(manifest={}, files=files, missing=[], extra=[], parse_errors=[])


def _qcell(cell, rep, pre, post, rows=()):
    head = {"part": "session", "quiet_pre": pre, "quiet_post": post}
    return fa.CellFile({"cell": cell}, rep, Path("x"), head, list(rows))


def test_quiet_machine_passes_when_both_snapshots_are_below_limit():
    run = _fake_run([_qcell("c1", 0, quiet_record(3.0), quiet_record(0.2))])
    res = fa.check_quiet_machine(run)
    assert res.violations == [] and res.checked == 1


def test_quiet_machine_flags_post_snapshot_over_limit_and_names_holder():
    hog = [{"pid": 700, "name": "stray", "pct": 43.0}]
    run = _fake_run([_qcell("c1", 1, quiet_record(1.0), quiet_record(43.0, hog))])
    res = fa.check_quiet_machine(run)
    assert len(res.violations) == 1
    assert "quiet_post" in res.violations[0] and "stray" in res.violations[0]


def test_quiet_machine_system_noise_exempts_only_windowserver_and_kernel_task():
    import fullcmp_quiet as fq

    assert fq.SYSTEM_NOISE == frozenset({"WindowServer", "kernel_task"})
    for name in ("mds_stores", "mds"):  # Spotlight is never exempt
        holder = [{"pid": 700, "name": name, "pct": 43.0}]
        run = _fake_run([_qcell("c1", 0, quiet_record(1.0), quiet_record(43.0, holder))])
        assert fa.check_quiet_machine(run).violations, name
    ws = [{"pid": 634, "name": "WindowServer", "pct": 43.0}]
    run = _fake_run([_qcell("c1", 0, quiet_record(1.0), quiet_record(43.0, ws))])
    assert fa.check_quiet_machine(run).violations == []


def test_wire_1107_kapitsa_mixed_quiet_pre_is_flagged():
    # the real record: quiet_pre of kapitsa-0.2.26 SMOKE mixed v0 (wire-1107) had mds at 48.66%,
    # total 112.78%, ok false; the cell ran anyway. Under the audit this must fail.
    holders = [{"pid": 590, "name": "mds", "pct": 48.66}, {"pid": 85506, "name": "node", "pct": 12.62}]
    pre = quiet_record(48.66, holders)
    pre["total_foreign_pct"] = 112.78
    pre["ok"] = False
    run = _fake_run([_qcell("kapitsa-mixed-v0", 0, pre, quiet_record(1.0))])
    res = fa.check_quiet_machine(run)
    assert res.violations and "quiet_pre" in res.violations[0] and "mds" in res.violations[0]


def test_quiet_machine_uses_the_limit_recorded_in_the_snapshot():
    # A 30% reading against a recorded 40% limit passes; the same reading at 25% fails.
    ok = _fake_run([_qcell("c1", 0, quiet_record(30.0, limit=40.0), quiet_record(0.0))])
    bad = _fake_run([_qcell("c1", 0, quiet_record(30.0, limit=25.0), quiet_record(0.0))])
    assert fa.check_quiet_machine(ok).violations == []
    assert fa.check_quiet_machine(bad).violations


def test_quiet_machine_host_load_is_not_a_violation_for_a_stub_engine_run():
    # Regression (mixed e2e flake, 2026-11-03): the e2e tests run beside other pytest/agent
    # processes, so the real host snapshot showed a 438% foreign holder ("bfs") and the stub run
    # failed. A stub_engine run has no measurement to protect: load is not checked there.
    hog = [{"pid": 66302, "name": "bfs", "pct": 438.17}]
    run = _fake_run([_qcell("c1", 0, quiet_record(1.0), quiet_record(492.19, hog))])
    run.manifest = {"stub_engine": True}
    assert fa.check_quiet_machine(run).violations == []


def test_quiet_machine_stub_run_still_fails_closed_without_a_snapshot():
    run = _fake_run([_qcell("c1", 0, None, quiet_record(0.0))])
    run.manifest = {"stub_engine": True}
    assert any("no quiet_pre" in v for v in fa.check_quiet_machine(run).violations)


def test_quiet_machine_real_run_still_flags_the_same_host_load():
    hog = [{"pid": 66302, "name": "bfs", "pct": 438.17}]
    run = _fake_run([_qcell("c1", 0, quiet_record(1.0), quiet_record(492.19, hog))])
    run.manifest = {}
    res = fa.check_quiet_machine(run)
    assert res.violations and "bfs" in res.violations[0]


def test_quiet_machine_missing_snapshot_fails_closed():
    run = _fake_run([_qcell("c1", 0, None, quiet_record(0.0))])
    res = fa.check_quiet_machine(run)
    assert any("no quiet_pre snapshot" in v for v in res.violations)


def _round_rows(ms, rounds, ctx=1024, phase="cold", kind="code"):
    return {
        "part": "decode",
        "phase": phase,
        "ctx": ctx,
        "kind": kind,
        "xy": {"decode_ms": ms, "speculative": {"rounds": rounds}},
    }


def test_round_time_outlier_flags_rep_more_than_1_5x_the_median():
    runs = [
        _fake_run([_qcell("c1", 0, quiet_record(), quiet_record(), [_round_rows(1000, 10)])]),
        _fake_run([_qcell("c1", 1, quiet_record(), quiet_record(), [_round_rows(1010, 10)])]),
    ]
    cf0 = runs[0].files[0]
    cf1 = runs[1].files[0]
    # Three reps: r2 is 3x slower than the other two.
    cf2 = _qcell("c1", 2, quiet_record(), quiet_record(), [_round_rows(3000, 10)])
    run = _fake_run([cf0, cf1, cf2])
    res = fa.check_round_time_outlier(run)
    # Literal rule: each rep against the median of the OTHER reps. The 3x rep is flagged, and
    # the two normal reps are flagged too (their other-rep median includes the outlier).
    assert any("r2" in v and "2.99x" in v for v in res.violations)
    assert all("cold ctx1024 code" in v for v in res.violations)


def test_round_time_outlier_passes_within_1_5x():
    cells = [
        _qcell("c1", r, quiet_record(), quiet_record(), [_round_rows(ms, 10)])
        for r, ms in enumerate((1000, 1200, 1400))
    ]
    assert fa.check_round_time_outlier(_fake_run(cells)).violations == []


def test_round_time_outlier_ignores_requests_without_round_counts():
    plain = {"part": "decode", "phase": "cold", "ctx": 1024, "kind": "code", "xy": {"decode_ms": 1}}
    cells = [
        _qcell("c1", 0, quiet_record(), quiet_record(), [plain]),
        _qcell("c1", 1, quiet_record(), quiet_record(), [dict(plain, xy={"decode_ms": 9000})]),
    ]
    res = fa.check_round_time_outlier(_fake_run(cells))
    assert res.violations == []


def test_round_ms_is_decode_ms_over_speculative_rounds():
    assert fa.round_ms(_round_rows(500, 25)) == pytest.approx(20.0)
    assert fa.round_ms({"xy": {"decode_ms": 500}}) is None
    assert fa.round_ms({"xy": {"decode_ms": 500, "speculative": {"rounds": 0}}}) is None


def test_conc_rows_pass_cold_cache_with_recorded_cached_counts():
    # Smoke4 (2026-11-03): conc rows have phase "conc"; the guard rejected them as an unknown phase.
    cell = {"cell": "c", "engine": "yunshu", "group": "g", "phases": ["conc"]}
    rows = fa.conc_request_rows(_conc_record(n=2))
    run = fa.Run({"groups": {}}, [_cf(cell, 0, {"part": "session", "fresh_process": True}, rows)], [], [], [])
    res = fa.check_cold_cache(run)
    assert res.violations == []
    assert res.checked > 0


def test_conc_row_without_recorded_cached_count_fails_cold_cache():
    cell = {"cell": "c", "engine": "yunshu", "group": "g", "phases": ["conc"]}
    rows = fa.conc_request_rows(_conc_record(n=2))
    rows[1]["cached"] = None
    run = fa.Run({"groups": {}}, [_cf(cell, 0, {"part": "session", "fresh_process": True}, rows)], [], [], [])
    res = fa.check_cold_cache(run)
    assert any("cached count not recorded" in v for v in res.violations)


import fullcmp_attribution as fat  # noqa: E402

# --- attribution layers (lead ruling 2026-11-03): layer 1 in every timed row, layer 2 in the
# profile rep only, a cell with no profile rep is "profile rep missing" in full mode.


def _attr_run(tmp_path, rows, *, profile=None, full=False, engine="yunshu"):
    cell = {"cell": "q27_decode-1024-code", "engine": engine, "group": "g", "phases": ["decode"]}
    cf = _cf(cell, 0, {"part": "session"}, rows)
    # the cell file sits where the audit reads it, with its engine-log slice beside it
    cf.path = tmp_path / engine / "q27_decode-1024-code-r0.jsonl"
    fa.server_log_slice_path(cf.path).parent.mkdir(parents=True, exist_ok=True)
    fa.server_log_slice_path(cf.path).write_text("[round-profile] engine=yunshu round=1\n")
    root = tmp_path / "profile"
    if profile is not None:
        d = root / engine
        d.mkdir(parents=True, exist_ok=True)
        (d / "q27_decode-1024-code-prof.jsonl").write_text(
            "\n".join(json.dumps(x) for x in profile) + "\n"
        )
        fa.server_log_slice_path(d / "q27_decode-1024-code-prof.jsonl").write_text(
            "[round-profile] engine=yunshu round=1\n"
        )
    run = fa.Run({"groups": {}}, [cf], [], [], [])
    run.profile_root = root if (profile is not None or full) else None
    run.full = full
    return fa.check_attribution_fields(run)


def _full_layer1_row():
    attr = {f: 1.0 for f in list(fat.LAYER1_TIMESTAMPS) + list(fat.LAYER1_COUNTERS)}
    attr.update({"rounds": 3, "tokens_per_round": [2, 2, 1], "rows_per_round": [1, 1, 1],
                 "cache_hit_tokens": 0, "chunk_sizes": [512], "host_gaps_ms": [1.0],
                 "engine_log_path": "/x.log", "t_rounds": [1.0], "t_prefill_chunks": [[0, 1, 512]]})
    return {"part": "decode", "attr": attr, "attr_unavailable": {}}


def _profile_row(complete=True):
    stages = {"decode_round": {k: 1.0 for k in ("drafter", "plan", "copy_lookup", "verify_weights",
                                                 "verify_attention", "commit")},
              "prefill": {k: 1.0 for k in ("chunk_compute", "eval_wait", "apc_store", "drafter_context")},
              "ttft": {k: 1.0 for k in ("template", "lookup", "restore", "prefill", "drafter_rebuild", "first_round")}}
    row = {"part": "profile", "stages": stages, "host_sync_count": 4, "op_census": {"matmul": 10}}
    if not complete:
        del row["op_census"]
    return row


def test_timed_row_without_observe_only_fields_passes(tmp_path):
    # Observe-only fields (stage timers, host syncs, op census) are layer 2: never required here.
    res = _attr_run(tmp_path, [_full_layer1_row()])
    assert res.violations == []


def test_timed_row_lacking_a_layer1_field_fails(tmp_path):
    row = _full_layer1_row()
    row["attr"].pop("t_first_token")
    res = _attr_run(tmp_path, [row])
    assert any("t_first_token" in v for v in res.violations)


def test_profile_rep_lacking_layer2_fails(tmp_path):
    res = _attr_run(tmp_path, [_full_layer1_row()], profile=[_profile_row(complete=False)], full=True)
    assert any("profile rep lacks layer 2" in v and "op_census" in v for v in res.violations)


def test_profile_rep_with_layer2_passes(tmp_path):
    res = _attr_run(tmp_path, [_full_layer1_row()], profile=[_profile_row()], full=True)
    assert res.violations == []


def test_no_profile_rep_fails_in_full_mode(tmp_path):
    res = _attr_run(tmp_path, [_full_layer1_row()], full=True)
    assert any("profile rep missing" in v for v in res.violations)


def test_no_profile_rep_is_not_checked_outside_full_mode(tmp_path):
    res = _attr_run(tmp_path, [_full_layer1_row()])
    assert res.violations == []


def test_conc_rows_are_covered_by_layer1(tmp_path):
    # Conc rows carry the fields their record has; the rest must be explicit reasons, not MISSING.
    rows = fa.conc_request_rows(_conc_record(n=2))
    res = _attr_run(tmp_path, rows)
    assert res.violations == [], res.violations


def test_conc_row_with_unexplained_gap_fails(tmp_path):
    rows = fa.conc_request_rows(_conc_record(n=2))
    rows[0]["attr_unavailable"].pop("t_admit")
    rows[0]["attr"].pop("t_admit", None)
    res = _attr_run(tmp_path, rows)
    assert any("t_admit" in v for v in res.violations)


# --- usage_fields (lead ruling 2026-11-03): no None counts on decode rows, conc aggregates -----


def _usage_run(rows):
    cell = {"cell": "c", "engine": "yunshu", "group": "g", "phases": ["decode"]}
    return fa.Run({"groups": {}}, [_cf(cell, 0, {"part": "session"}, rows)], [], [], [])


def _decode_row(**kw):
    row = {"part": "decode", "phase": "warm", "ct": 64, "pt": 1034, "cached": 1024}
    row.update(kw)
    return row


def test_usage_none_completion_count_fails():
    res = fa.check_usage_fields(_usage_run([_decode_row(ct=None)]))
    assert any("ct=None" in v for v in res.violations)


def test_usage_none_prompt_count_fails():
    res = fa.check_usage_fields(_usage_run([_decode_row(pt=None)]))
    assert any("pt=None" in v for v in res.violations)


def test_usage_cached_none_without_reason_fails():
    res = fa.check_usage_fields(_usage_run([_decode_row(cached=None)]))
    assert any("no cached_unavailable reason" in v for v in res.violations)


def test_usage_cached_none_with_reason_passes_usage_but_cold_cache_fails_it():
    row = _decode_row(cached=None, cached_unavailable="usage has no cached count", phase="cold")
    assert fa.check_usage_fields(_usage_run([row])).violations == []
    cell = {"cell": "c", "engine": "yunshu", "group": "g", "phases": ["decode"]}
    run = fa.Run({"groups": {}}, [_cf(cell, 0, {"part": "session", "fresh_process": True}, [row])], [], [], [])
    assert any("must be server-reported 0" in v for v in fa.check_cold_cache(run).violations)


def _conc_with_aggregates(decs=(10.0, 20.0), **override):
    rec = _conc_record(n=len(decs))
    rec["per_req_dec"] = list(decs)
    rec.update(
        {
            "wall_s": 2.0,
            "total_tokens": 32,
            "agg_tps": 16.0,
            "dec_min": min(decs),
            "dec_max": max(decs),
            "dec_median": statistics.median(decs),
            "fair_max_min": round(max(decs) / min(decs), 3),
        }
    )
    rec.update(override)
    return fa.conc_request_rows(rec)


def test_conc_aggregates_present_and_consistent_pass():
    assert fa.check_usage_fields(_usage_run(_conc_with_aggregates())).violations == []


def test_conc_missing_aggregate_fails():
    rows = _conc_with_aggregates(dec_median=None)
    res = fa.check_usage_fields(_usage_run(rows))
    assert any("dec_median is missing or None" in v for v in res.violations)


def test_conc_aggregate_not_matching_its_requests_fails():
    rows = _conc_with_aggregates(dec_max=99.0)
    res = fa.check_usage_fields(_usage_run(rows))
    assert any("dec_min/dec_max differ" in v for v in res.violations)


def test_conc_agg_tps_must_equal_total_over_wall():
    rows = _conc_with_aggregates(agg_tps=40.0)
    res = fa.check_usage_fields(_usage_run(rows))
    assert any("agg_tps" in v for v in res.violations)


def _rot_cf(engine, rep, start, group="g"):
    import types

    header = {"job_start_s": start, "job_end_s": start + 5.0, "gpu_exclusive": True, "contended": False}
    return types.SimpleNamespace(header=header, cell={"cell": f"{engine}-c", "group": group, "engine": engine},
                                 rep=rep, rows=[], path=None)


def _rot_run(order, observed_engines, untimed_engines=(), rep=0):
    files = [_rot_cf(e, rep, 100.0 + 10 * i) for i, e in enumerate(observed_engines)]
    files += [_untimed_cf(e, rep) for e in untimed_engines]
    manifest = {"groups": {"g": {"order": list(order)}}}
    return fa.Run(manifest=manifest, files=files, missing=[], extra=[], parse_errors=[])


def _untimed_cf(engine, rep):
    import types

    return types.SimpleNamespace(header={"gpu_exclusive": None}, cell={"cell": f"{engine}-c", "group": "g",
                                                                       "engine": engine}, rep=rep, rows=[], path=None)


# the real wire-1106c observed interleave (failed engines absent) against the planned order
PLANNED_1106 = ["yunshu", "tf-1.2.1", "tf-1.1.8", "splash-1.5.0", "kapitsa-0.2.26", "mlx-serve-26.11.1",
                "llama-cpp-b12061"]
OBSERVED_1106 = ["yunshu", "tf-1.2.1", "tf-1.1.8", "splash-1.5.0", "mlx-serve-26.11.1", "llama-cpp-b12061"]


def test_rotation_compares_the_completed_subsequence_in_planned_order():
    """Failing-before (wire-1106c): the observed interleave (kapitsa failed, absent) was compared with the full
    planned rotation. The completed engines follow the planned relative order: no violation."""
    res = fa.check_gpu_exclusive_rotation(_rot_run(PLANNED_1106, OBSERVED_1106, untimed_engines=["kapitsa-0.2.26"]))
    assert res.violations == [], res.violations
    assert any("kapitsa-0.2.26" in n and "failed job" in n for n in res.notes)


def test_rotation_swapped_completed_engines_is_still_a_violation():
    swapped = ["tf-1.2.1", "yunshu", "tf-1.1.8", "splash-1.5.0", "mlx-serve-26.11.1", "llama-cpp-b12061"]
    res = fa.check_gpu_exclusive_rotation(_rot_run(PLANNED_1106, swapped))
    assert res.violations and "planned order of the completed engines" in res.violations[0]


# --- cold_cache for concurrent and mixed requests (wire-1107 records, values copied in) -------


def _mixed_rows(cached):
    return [{"part": "mixed", "phase": "mixed", "cached": c, "ct": 16} for c in cached]


def test_wire_1107_yunshu_mixed_trial1_fails_the_cold_cache_guard():
    # wire-1107y yunshu SMOKE mixed 8K c4 v0: trial 1 cached [8192, 0, 0, 0]
    run = _fake_run([_cf({"cell": "SMOKE.smoke-mixed.yunshu.8K.c4.code.o64.mixed.th-off.none.greedy.v0"},
                         0, {"part": "session", "fresh_process": True}, _mixed_rows([0, 0, 0, 0, 8192, 0, 0, 0]))])
    res = fa.check_cold_cache(run)
    assert any("mixed" in v and "cached=8192" in v for v in res.violations), res.violations


def test_wire_1107_splash_and_tf_121_mixed_trial1_fail_the_cold_cache_guard():
    # splash cached [8160, 0, 0, 0] and tf-1.2.1 cached [8187, 3, 3, 3] in trial 1
    for cell, cached in (
        ("SMOKE.smoke-mixed.splash-1.5.0.8K.c4.code.o64.mixed.th-off.none.greedy.v0", [8160, 0, 0, 0]),
        ("SMOKE.smoke-mixed.tf-1.2.1.8K.c4.code.o64.mixed.th-off.none.greedy.v0", [8187, 3, 3, 3]),
    ):
        run = _fake_run([_cf({"cell": cell}, 0, {"part": "session", "fresh_process": True}, _mixed_rows(cached))])
        assert fa.check_cold_cache(run).violations, cell


def test_missing_cached_fails_closed_for_mixed_and_conc_rows():
    rows = [{"part": "conc", "phase": "conc", "ct": 16}]  # no cached field at all
    run = _fake_run([_cf({"cell": "B2.conc.yunshu.8K.c4.code.o512.single.th-off.none.greedy"}, 0,
                         {"part": "session", "fresh_process": True}, rows)])
    assert fa.check_cold_cache(run).violations


def test_conc_same_shares_its_prompt_by_design():
    rows = _conc_same_rows([0, 512, 512, 512], [1024] * 4)
    run = _fake_run([_cf({"cell": "B2.conc-same.yunshu.8K.c4.code.o512.single.th-off.none.greedy"}, 0,
                         {"part": "session", "fresh_process": True}, rows)])
    assert fa.check_cold_cache(run).violations == []


def _conc_same_rows(cached, pts):
    return [
        {"part": "conc", "phase": "conc", "cached": c, "pt": p, "req_index": i, "ct": 16}
        for i, (c, p) in enumerate(zip(cached, pts))
    ]


def test_conc_same_trial_with_every_request_cached_fails():
    # the trial's first arrival must be a miss; if all n requests hit, the prompt was primed earlier
    rows = _conc_same_rows([512, 512, 512, 512], [1024] * 4)
    run = _fake_run([_cf({"cell": "B2.conc-same.yunshu.8K.c4.code.o512.single.th-off.none.greedy"}, 0,
                         {"part": "session", "fresh_process": True}, rows)])
    assert any("first arrival must be a miss" in v for v in fa.check_cold_cache(run).violations)


def test_conc_same_request_zero_cached_passes_when_another_request_is_the_miss():
    # server order is arbitrary: index 0 may be processed after a sibling that was the first arrival
    rows = _conc_same_rows([512, 0, 512, 512], [1024] * 4)
    run = _fake_run([_cf({"cell": "B2.conc-same.yunshu.8K.c4.code.o512.single.th-off.none.greedy"}, 0,
                         {"part": "session", "fresh_process": True}, rows)])
    assert fa.check_cold_cache(run).violations == []


def test_conc_same_later_requests_may_hit_the_shared_prefix_only():
    rows = _conc_same_rows([0, 512, 512, 512], [1024] * 4)
    run = _fake_run([_cf({"cell": "B2.conc-same.yunshu.8K.c4.code.o512.single.th-off.none.greedy"}, 0,
                         {"part": "session", "fresh_process": True}, rows)])
    assert fa.check_cold_cache(run).violations == []
    over = _conc_same_rows([0, 2048, 0, 0], [1024] * 4)
    run2 = _fake_run([_cf({"cell": "B2.conc-same.yunshu.8K.c4.code.o512.single.th-off.none.greedy"}, 0,
                          {"part": "session", "fresh_process": True}, over)])
    assert any("> prompt tokens" in v for v in fa.check_cold_cache(run2).violations)


def test_conc_distinct_any_cached_request_fails():
    rows = _conc_same_rows([0, 0, 512, 0], [1024] * 4)
    rows = [{**r, "phase": "conc"} for r in rows]
    run = _fake_run([_cf({"cell": "B2.conc.yunshu.8K.c4.code.o512.single.th-off.none.greedy"}, 0,
                         {"part": "session", "fresh_process": True}, rows)])
    assert any("must be server-reported 0" in v for v in fa.check_cold_cache(run).violations)


def test_yunshu_aliases_never_take_the_competitor_branch():
    """wire-1108 B1 (2026-11-04): the audit crashed (UnboundLocalError 'ready') because yunshu-pb2 failed an
    `engine != "yunshu"` test and entered the competitor-only concurrent check. Every Yunshu alias must be
    routed by fe.is_yunshu, never by string equality."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "scripts" / "research" / "fullcmp_audit.py"
    text = src.read_text()
    assert not re.search(r'engine\s*(!=|==)\s*"yunshu"', text), "use fe.is_yunshu(engine), not string equality"


def test_quiet_flagged_rerun_keeps_its_original_slot_in_the_rotation_order():
    """Failing-before (wire-1108 B1): omlx-0.7.3 was quiet-flagged and its -qretry1 rerun ran last, so the
    observed interleave put omlx at the end. The superseded attempt (slot 1793763668) keeps the slot."""
    import types

    order = ["omlx-0.7.3", "tf-1.2.1", "splash-1.5.0"]
    files = [_rot_cf("tf-1.2.1", 0, 1793763730.0), _rot_cf("splash-1.5.0", 0, 1793763800.0)]
    omlx = _rot_cf("omlx-0.7.3", 0, 1793765093.0)  # the rerun's own, later, timestamps
    omlx.slot = (1793763668.0, 1793763728.0)  # the superseded first attempt
    files.insert(0, omlx)
    manifest = {"groups": {"g": {"order": order}}}
    run = fa.Run(manifest=manifest, files=files, missing=[], extra=[], parse_errors=[])
    assert fa.check_gpu_exclusive_rotation(run).violations == []
    omlx.slot = None  # without the superseded slot the rerun breaks the observed order
    res = fa.check_gpu_exclusive_rotation(run)
    assert res.violations and "planned order" in res.violations[0], res.violations
    assert types  # keep the import local


def test_superseded_slot_is_read_from_the_superseded_file_header(tmp_path):
    cell = tmp_path / "c-r0.jsonl"
    cell.write_text('{"part": "session", "job_start_s": 9.0, "job_end_s": 10.0}\n')
    assert fa._superseded_slot(cell) is None
    (tmp_path / "c-r0.superseded-1.jsonl").write_text(
        '{"part": "session", "job_start_s": 1.0, "job_end_s": 2.0}\n'
    )
    (tmp_path / "c-r0.superseded-1.telemetry.jsonl").write_text('{"x": 1}\n')
    assert fa._superseded_slot(cell) == (1.0, 2.0)
