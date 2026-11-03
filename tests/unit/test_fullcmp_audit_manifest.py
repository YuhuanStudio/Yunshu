"""CPU-only synthetic tests for the audit manifest builder and the per-cell rep rule.

Covers: manifest engine ids -> harness ids or stubs; cell/pin/record fields built from synthetic
census/pin/manifest rows; control verdicts (A/A pass, injected delay and flip detected, a broken
control refused); the audit's per-cell rep count (ok -> 2, need_rep3 -> 3, pending -> violation).
No real run directory is read: every input is written under tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import bench_engines as be  # noqa: E402
import fullcmp_audit as fa  # noqa: E402
import fullcmp_audit_manifest as fam  # noqa: E402
import fullcmp_reps as fr  # noqa: E402

MANIFEST_IDS = (
    "yunshu",
    "tf-1.2.1",
    "tf-1.1.8",
    "splash-1.5.0",
    "omlx-0.7.3",
    "omlx-0.7.3rc1",
    "mlx-serve-26.11.1",
    "mlx-lm-server",
    "mlx-vlm-server",
    "llama-cpp-b12061",
    "lmstudio",
    "kapitsa-0.2.26",
    "ollama",
    "best-competitor (TBD)",
)
H = "a" * 64


def _census():
    tf = {
        "latest_date": "2026-11-01T23:52:33Z",
        "sha256": {"1.2.1 tarball": H, "1.1.8 tarball": H},
    }
    return {
        "_meta": {
            "models": {
                "qwen3.8-27b-4bit-jundot": {
                    "sha256": {"model-00001-of-00004.safetensors": H}
                },
                "qwen3.8-27b-4bit-kapitsa": {
                    "sha256": {"model-00001-of-00003.safetensors": H}
                },
            }
        },
        "yunshu": {"latest_date": "2026-11-03 (commit 0f2960047)"},
        "tensorfold": tf,
        "splash": {
            "latest_date": "2026-10-28T20:15:32Z",
            "sha256": {"1.5.0 tarball": H},
        },
        "omlx": {
            "latest_date": "2026-10-31T16:42:55Z",
            "latest_rc_date": "2026-10-27T09:22:10Z",
            "sha256": {"0.7.3 wheel cp313": H, "0.7.3rc1 wheel cp313": H},
        },
        "mlx-serve": {
            "latest_date": "2026-11-01T19:34:34Z",
            "sha256": {"26.11.1 binary mlx-serve": H},
        },
        "mlx_lm_server": {
            "latest_date": "2026-10-16T02:49:34Z",
            "sha256": {"mlx_lm-0.32.2-py3-none-any.whl (PyPI digest)": H},
        },
        "kapitsa": {
            "latest_date": "2026-11-01T22:32:02Z",
            "sha256": {"0.2.26 binary kapitsa": H},
        },
    }


PIN = {
    "sha": "0f2960047855546a9f13b2631b680352b1c57cfe",
    "version": "0.1.7.dev0",
    "python_executable": "/venv/bin/python",
}
THRESH = {"decode_tps": 3.03, "cold_ttft": 5.0}


def _row(
    eng,
    cid,
    ctx=1024,
    kind="code",
    conc=1,
    phases=("cold", "warm", "fu"),
    batch="B1",
    status="unverified",
    **kw,
):
    r = {
        "batch": batch,
        "conc": conc,
        "ctx": f"{ctx // 1024}K",
        "ctx_tokens": ctx,
        "engine": eng,
        "id": cid,
        "media": "none",
        "model": "q27-4bit",
        "out_len": 512,
        "phases": list(phases),
        "prompt_kind": kind,
        "reps": 2,
        "sampling": "greedy",
        "status": status,
        "table": "decode",
        "thinking": "off",
        "turns": "single",
        "reason": "",
    }
    r.update(kw)
    return r


def _verdict(overall, failed_stage=None, stages=()):
    return {"overall": overall, "failed_stage": failed_stage, "stages": list(stages)}


def _speed_stage(deltas):
    cells = [
        {"metric": m, "delta_pct": d, "noise_pct": 0.1, "ctx": 1024, "kind": "code"}
        for m, d in deltas
    ]
    return {
        "name": "speed",
        "status": "PASS",
        "reasons": [],
        "numbers": {"cells": cells},
    }


def _verify_tree(tmp_path, aa_delta=0.3, slow_fail=True, flip_fail=True):
    base = tmp_path / "verify"
    specs = {
        fam.CONTROL_AA: _verdict(
            "PASS",
            None,
            [_speed_stage([("decode_tps", aa_delta), ("followup_decode_tps", -0.5)])],
        ),
        fam.CONTROL_SLOW: _verdict(
            "FAIL" if slow_fail else "PASS",
            "speed" if slow_fail else None,
            [
                {
                    "name": "speed",
                    "status": "FAIL" if slow_fail else "PASS",
                    "reasons": ["code@1024 decode_tps: 139 -> 89 (-36%)"]
                    if slow_fail
                    else [],
                    "numbers": {},
                }
            ],
        ),
        fam.CONTROL_FLIP: _verdict(
            "FAIL" if flip_fail else "PASS",
            "identity" if flip_fail else None,
            [
                {
                    "name": "identity",
                    "status": "FAIL" if flip_fail else "PASS",
                    "reasons": ["base != cand: digest differs"] if flip_fail else [],
                    "numbers": {},
                }
            ],
        ),
    }
    for name, v in specs.items():
        d = base / name
        d.mkdir(parents=True)
        (d / "verdict.json").write_text(json.dumps(v))
    thr = tmp_path / "rep_thresholds.json"
    thr.write_text(json.dumps({"thresholds_pct": THRESH}))
    return base, thr


# ---- engine ids ---------------------------------------------------------------------------------


def test_every_manifest_engine_maps_to_a_harness_id_or_a_stub():
    for mid in MANIFEST_IDS:
        mapped = mid in be.MANIFEST_ENGINES
        stub = mid in be.UNSUPPORTED_ENGINES
        assert mapped != stub, f"{mid} must be exactly one of mapped / stub"
        if mapped:
            assert be.tfbench_engine(mid) in be.ENGINES
        else:
            assert be.stub_reason(mid).startswith("unsupported (harness: ")


def test_yunshu_maps_to_pinned_tree_engine_not_legacy_id():
    assert be.tfbench_engine("yunshu") == "yunshu-main"
    assert "yunshu" in be.LEGACY and be.is_new_engine("yunshu-main")


def test_stub_engine_has_no_tfbench_id():
    with pytest.raises(KeyError):
        be.tfbench_engine("llama-cpp-b12061")


def test_new_engines_launch_their_census_binary():
    for mid, binary in (
        ("tf-1.2.1", "tensorfold-1.2.1"),
        ("tf-1.1.8", "tensorfold-1.1.8"),
        ("splash-1.5.0", "splash-1.5.0"),
        ("omlx-0.7.3", "venvs/omlx-0.7.3/"),
        ("omlx-0.7.3rc1", "venvs/omlx-0.7.3rc1/"),
        ("mlx-serve-26.11.1", "mlxserve-bin/26.11.1"),
        ("mlx-lm-server", "mlx-lm-0.32.2"),
        ("kapitsa-0.2.26", "kapitsa-0.2.26"),
    ):
        launch = be.build_launch(mid, 18999, "/nonexistent/home", {})
        assert binary in launch.cmd[0], (mid, launch.cmd[0])
        assert "/Volumes/P5Plus/yunshu-build/competitors/" in launch.cmd[0]


def test_unverified_spec_mode_fails_closed():
    # no log evidence can make an unverified mode engaged: check_engaged always refuses
    for mid in ("mlx-serve-26.11.1", "kapitsa-0.2.26"):
        eid = mid
        assert be.ENGINES[eid].expected_mode == "unverified"
        with pytest.raises(RuntimeError):
            be.check_engaged(eid, be.detect_mode(eid, "any log line draft mtp dflash"))


def test_tensorfold_1_2_1_log_is_detected_as_dflash():
    assert be.detect_mode("tf-1.2.1", "drafts: DFlash2 (/x)") == "dflash"


# ---- manifest build ----------------------------------------------------------------------------


def test_build_manifest_cells_pins_and_phases():
    rows = [
        _row("yunshu", "B1.core.yunshu.1K.c1.code"),
        _row("omlx-0.7.3", "B1.core.omlx-0.7.3.1K.c1.code"),
        _row("tf-1.1.8", "B1.core.tf-1.1.8.1K.c1.code", phases=("cold", "warm")),
        _row("tf-1.1.8", "B1.conc.tf-1.1.8.1K.c4.code", conc=4),
        _row(
            "kapitsa-0.2.26", "B1.img.kapitsa-0.2.26.1K", media="image", reason="image"
        ),
        _row(
            "llama-cpp-b12061",
            "B1.core.llama.1K",
            status="unsupported",
            reason="no recipe",
        ),
        _row("mlx-vlm-server", "B1.core.vlm.1K"),
    ]
    m = fam.build_manifest(
        rows, _census(), PIN, THRESH, {"aa_delta_pct": 0.3}, {}, ("B1",)
    )
    by = {(c["engine"], c["cell"]): c for c in m["cells"]}
    y = by[("yunshu", "B1.core.yunshu.1K.c1.code")]
    assert y["phases"] == ["cold", "warm", "turn2"] and y["expected_records"] == 3
    assert (
        y["spec_mode"] == "dflash" and y["rep_decision"] == "pending" and y["reps"] == 2
    )
    assert y["column"] == "1K/code/c1/o512" and y["max_tokens"] == 512
    assert y["weight_group"] == "qwen3.8-27b-4bit-jundot"
    assert by[("tf-1.1.8", "B1.core.tf-1.1.8.1K.c1.code")]["expected_records"] == 2
    assert all(c["engine"] != "kapitsa-0.2.26" for c in m["cells"])
    assert any(
        u["engine"] == "mlx-vlm-server"
        and u["reason"].startswith("unsupported (harness: ")
        for u in m["unsupported"]
    )
    assert any(u["engine"] == "llama-cpp-b12061" for u in m["unsupported"])
    assert any(n["cell"] == "B1.img.kapitsa-0.2.26.1K" for n in m["not_in_harness"])
    # conc cells are audited: one conc record per trial, n requests per record
    conc4 = by[("tf-1.1.8", "B1.conc.tf-1.1.8.1K.c4.code")]
    assert conc4["phases"] == ["conc"] and conc4["expected_records"] == 4 * 2
    assert not any(n["cell"].startswith("B1.conc") for n in m["not_in_audit"])
    assert m["pins"]["omlx-0.7.3"]["version"] == "0.7.3"
    assert m["pins"]["tf-1.1.8"]["release_date"] is None
    assert any(
        u["engine"] == "tf-1.1.8" and "release date" in u["reason"]
        for u in m["unresolved"]
    )
    assert m["groups"]["B1.1K.code.c1"]["order"] == ["yunshu", "omlx-0.7.3", "tf-1.1.8"]


def test_duplicate_cell_stops_the_build():
    row = _row("yunshu", "B1.dup")
    with pytest.raises(SystemExit):
        fam.build_manifest([row, dict(row)], _census(), PIN, THRESH, {}, {}, ("B1",))


def test_rep_decision_per_cell_reaches_the_audit_rule():
    rows = [_row("yunshu", "c-ok"), _row("yunshu", "c-r3"), _row("yunshu", "c-pend")]
    dec = {("yunshu", "c-ok"): ("ok", False), ("yunshu", "c-r3"): ("need_rep3", True)}
    m = fam.build_manifest(rows, _census(), PIN, THRESH, {}, dec, ("B1",))
    by = {c["cell"]: c for c in m["cells"]}
    assert by["c-ok"]["rep_decision"] == "ok" and by["c-ok"]["need_rep3"] is False
    assert by["c-r3"]["need_rep3"] is True and by["c-pend"]["rep_decision"] == "pending"


# ---- controls -----------------------------------------------------------------------------------


def test_controls_pass_and_detections_recorded(tmp_path):
    base, thr = _verify_tree(tmp_path)
    controls, evidence = fam.load_controls(base, thr)
    assert controls["aa_delta_pct"] == pytest.approx(0.5)
    assert controls["aa_noise_pct"] == 3.03
    assert (
        controls["injected_delay_detected"] is True
        and controls["injected_flip_detected"] is True
    )
    assert set(evidence) == {fam.CONTROL_AA, fam.CONTROL_SLOW, fam.CONTROL_FLIP}
    assert len(evidence[fam.CONTROL_AA]["verdict_sha256"]) == 64


def test_aa_outside_noise_is_refused(tmp_path):
    base, thr = _verify_tree(tmp_path, aa_delta=5.0)
    with pytest.raises(fam.ControlError):
        fam.load_controls(base, thr)


def test_undetected_delay_or_flip_is_refused(tmp_path):
    base, thr = _verify_tree(tmp_path, slow_fail=False)
    with pytest.raises(fam.ControlError, match="injected-delay"):
        fam.load_controls(base, thr)
    base2, thr2 = _verify_tree(tmp_path / "b", flip_fail=False)
    with pytest.raises(fam.ControlError, match="injected-flip"):
        fam.load_controls(base2, thr2)


# ---- audit: per-cell rep rule -------------------------------------------------------------------


def _audit_manifest(cells):
    return {
        "run_id": "t",
        "reps": 2,
        "pins": {
            "yunshu": {"version": "v", "release_date": "2026-11-03", "git_sha": "x"}
        },
        "yunshu": {"git_sha": "x", "python": "p"},
        "controls": {},
        "groups": {},
        "cells": cells,
    }


def _cell(cid, **kw):
    c = {
        "engine": "yunshu",
        "cell": cid,
        "phases": ["cold"],
        "expected_records": 1,
        "max_tokens": 4,
    }
    c.update(kw)
    return c


def test_audit_expected_reps_per_cell():
    m = _audit_manifest(
        [
            _cell("a", rep_decision="ok"),
            _cell("b", rep_decision="need_rep3"),
            _cell("c"),
        ]
    )
    assert fa.expected_reps(m["cells"][0], m) == 2
    assert fa.expected_reps(m["cells"][1], m) == 3
    assert (
        fa.expected_reps(m["cells"][2], m) == 2
    )  # no decision recorded: manifest reps


def test_audit_load_run_expects_rep3_only_for_flagged_cells(tmp_path):
    m = _audit_manifest(
        [_cell("a", rep_decision="ok"), _cell("b", rep_decision="need_rep3")]
    )
    d = tmp_path / "yunshu"
    d.mkdir()
    for name in ("a-r0.jsonl", "a-r1.jsonl", "b-r0.jsonl", "b-r1.jsonl"):
        (d / name).write_text("")
    run = fa.load_run(tmp_path, m)
    missing = sorted(p.name for _, _, p in run.missing)
    assert missing == ["b-r2.jsonl"]
    res = fa.check_manifest_completeness(run)
    missing_msgs = [v for v in res.violations if "missing" in v]
    assert len(missing_msgs) == 1 and missing_msgs[0].startswith(
        "manifest_completeness: b: r2 missing"
    )


def test_audit_pending_decision_is_a_violation(tmp_path):
    m = _audit_manifest([_cell("p", rep_decision="pending")])
    run = fa.load_run(tmp_path, m)
    res = fa.check_manifest_completeness(run)
    assert any("rep decision pending" in v for v in res.violations)


def test_rep_decisions_from_raw_results(tmp_path):
    def rows(dec_tps, ttft):
        out = [{"part": "session", "group": "g"}]
        for phase in ("cold", "warm", "turn2"):
            out.append(
                {
                    "part": "decode",
                    "phase": phase,
                    "dec_tps": dec_tps,
                    "ttft_s": ttft,
                    "finish": "length",
                }
            )
        return out

    d = tmp_path / "yunshu"
    d.mkdir()
    lines = lambda rs: "\n".join(json.dumps(r) for r in rs) + "\n"  # noqa: E731
    (d / "ok-cell-r0.jsonl").write_text(lines(rows(100.0, 1.0)))
    (d / "ok-cell-r1.jsonl").write_text(lines(rows(100.5, 1.0)))
    (d / "gap-cell-r0.jsonl").write_text(lines(rows(100.0, 1.0)))
    (d / "gap-cell-r1.jsonl").write_text(lines(rows(110.0, 1.0)))
    (d / "one-cell-r0.jsonl").write_text(lines(rows(100.0, 1.0)))
    got = fam.rep_decisions(tmp_path, {"decode_tps": 3.0, "cold_ttft": 5.0})
    assert got[("yunshu", "ok-cell")] == ("ok", False)
    assert got[("yunshu", "gap-cell")] == ("need_rep3", True)
    assert ("yunshu", "one-cell") not in got
    assert fr.rel_gap_pct([100.0, 110.0]) == pytest.approx(10.0)
