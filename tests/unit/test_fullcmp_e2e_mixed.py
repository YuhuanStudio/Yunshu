"""E2E mixed-load and same-prompt conc cells through the REAL tfbench path (no GPU, no model).

Same chain as test_fullcmp_e2e_session.py: manifest rows -> fullcmp_run plan and wrapper -> real
tfbench (real Srv launch, TFB_STUB_SERVER replaces only the engine process) -> real telemetry ->
real fullcmp_audit with zero violations. Engines: yunshu and tf-1.2.1.

The mixed cell sends one 128K-class request and seven 1K requests together. The stub variant
(fullcmp_stub_mixed) counts a long prompt's tokens from its length, so no tokenizer runs for it.
The negative test restarts on a wiped HOME (TFB_STUB_WIPE_HOME_ON_RESTART, test-only hook in
tfbench.restart_server): the restart row gets cached 0 and the audit must fail.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "research"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "scripts" / "dev"))
sys.path.insert(0, str(ROOT / "tests" / "unit"))

import fullcmp_audit as fa  # noqa: E402
import fullcmp_audit_manifest as fam  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_run as fr  # noqa: E402
import fullcmp_stub_session as sess  # noqa: E402

from .test_fullcmp_e2e import _run_audit  # noqa: E402
from .test_fullcmp_e2e_coverage import cell_type_key  # noqa: E402
from .test_fullcmp_e2e_session import (  # noqa: E402
    _audit_clean,
    _decode,
    _records,
    _rows,
    _run_cell_rows,
)

ENGINES_UNDER_TEST = ("yunshu", "tf-1.2.1")
MIXED_STUB = "fullcmp_stub_mixed"
SHORT_N = 7


@pytest.fixture(params=ENGINES_UNDER_TEST)
def engine(request):
    return request.param


def _single_variant_rows(engine, specs):
    """Runnable rows at prompt variant 0 only (the mixed cell's long prompt has no smoke variant
    file on disk); one rep each, the expected count carried on the row (smoke-style)."""
    rows = []
    for sp in specs:
        row = fm._cell_row(engine, fm.SMOKE_BATCH, sp)
        row["id"] = f"{row['id']}.v0"
        row["reps"] = 1
        row["prompt_variant"] = 0
        row["est_gpu_min"] = 0.0
        rows.append(row)
    return rows


def _mixed_spec(out=64):
    return fm.spec(
        sweep="mixed",
        ctx="128K",
        conc=SHORT_N + 1,
        kind="code",
        out=out,
        turns="mixed",
        phases=("mixed",),
    )


def _same_prompt_row(n):
    return fm._cell_row(
        "yunshu",
        fm.SMOKE_BATCH,
        fm.spec(sweep="conc-same", ctx="8K", conc=n, phases=("cold",)),
    )


# coverage keys are cell_type_key()s of the manifest rows the tests run (test_fullcmp_e2e_coverage)
COVERED_CELL_TYPES = {
    cell_type_key(fm._cell_row("yunshu", fm.SMOKE_BATCH, _mixed_spec())): (
        "test_mixed_cell_long_and_short_requests_run_together_and_audit_is_clean"
    ),
    **{
        cell_type_key(
            _same_prompt_row(n)
        ): "test_same_prompt_conc_sends_one_prompt_n_times"
        for n in (2, 4, 8)
    },
}


def _mixed_records(raw, engine, row):
    recs = [r for r in _records(raw, engine, row["id"]) if r.get("part") == "mixed"]
    return recs


# ---------------------------------------------------------------- end to end
def _long_prompt_variant_dir(tmp: Path, monkeypatch) -> None:
    """The long prompt needs a smoke variant (the audit wants two prompts per kind and ctx). No
    variant file exists for 128K on disk, so the test writes one into tmp: variant 0's corpus text
    with a marker in front. tfbench reads TFB_PROMPT_VARIANT_DIR from its environment."""
    work = Path(os.environ.get("TFB_WORK", "/Volumes/P5Plus/yunshu-build/tfnew"))
    base = (work / "prompts" / "code-131072.txt").read_text()
    variants = tmp / "variants"
    variants.mkdir(parents=True, exist_ok=True)
    # the other smoke variants (the short requests' 32K sources) come along unchanged
    original = Path(
        os.environ.get(
            "TFB_PROMPT_VARIANT_DIR", "/Volumes/Micron/yunshu-fullcmp/smoke-prompts"
        )
    )
    for f in original.glob("*-v*.txt"):
        (variants / f.name).write_bytes(f.read_bytes())
    (variants / "code-131072-v1.txt").write_text(
        "Variant one of the long prompt.\n" + base
    )
    monkeypatch.setenv("TFB_PROMPT_VARIANT_DIR", str(variants))


def test_mixed_cell_long_and_short_requests_run_together_and_audit_is_clean(
    tmp_path, engine, monkeypatch
):
    _long_prompt_variant_dir(tmp_path / "prompts", monkeypatch)
    rows = _rows(engine, [_mixed_spec()])
    raw, plan, pin = _run_cell_rows(tmp_path, rows, stub_module=MIXED_STUB)
    for row in rows:
        recs = _mixed_records(raw, engine, row)
        assert len(recs) == fr.CONC_TRIALS, f"mixed records {len(recs)}"
        for rec in recs:
            assert rec["n"] == SHORT_N + 1
            assert rec["roles"] == ["long"] + ["short"] * SHORT_N
            for key in ("ttfts", "per_req_dec", "cts", "pts", "cached", "shas"):
                assert len(rec[key]) == rec["n"], key
            assert all(c == row["out_len"] for c in rec["cts"]), rec["cts"]
            assert all(d > 0 for d in rec["per_req_dec"]) and all(
                t > 0 for t in rec["ttfts"]
            )
            # the long request carries the long prompt; the shorts are 1K-class
            assert rec["pts"][0] > 10 * max(rec["pts"][1:]), rec["pts"]
            assert rec["ctx"] == 131072 and rec["short_ctx"] == 1024
            # fairness summary is over the short requests only
            shorts = rec["per_req_dec"][1:]
            assert rec["dec_min"] == min(shorts) and rec["dec_max"] == max(shorts)
            assert rec["dec_min"] <= rec["dec_median"] <= rec["dec_max"]
            assert rec["fair_max_min"] == round(max(shorts) / min(shorts), 3)
            assert rec["total_tokens"] == sum(rec["cts"])
            assert rec["agg_tps"] > 0 and rec["wall_s"] > 0
            assert rec["long_ttft_s"] == rec["ttfts"][0]
    _audit_clean(raw, rows, plan, pin, tmp_path)


def test_same_prompt_conc_sends_one_prompt_n_times(tmp_path, engine):
    specs = [
        fm.spec(
            sweep="conc-same", ctx="1K", kind="code", out=64, conc=n, phases=("cold",)
        )
        for n in (2, 4, 8)
    ]
    rows = _rows(engine, specs)
    raw, plan, pin = _run_cell_rows(tmp_path, rows)
    for row in rows:
        recs = [r for r in _records(raw, engine, row["id"]) if r.get("part") == "conc"]
        assert len(recs) == fr.CONC_TRIALS
        for rec in recs:
            assert rec["same_prompt"] is True
            assert len(set(rec["shas"])) == 1, "one prompt digest for every stream"
            assert len(set(rec["prompt_ids_sha"])) == 1
            assert rec["n"] == row["conc"] and len(rec["shas"]) == row["conc"]
    _audit_clean(raw, rows, plan, pin, tmp_path)


def test_restart_that_wipes_home_loses_the_disk_tier_and_audit_fails(
    tmp_path, engine, monkeypatch
):
    """Negative case for the restart cell: the relaunch runs on a wiped HOME, so the restart row
    reports cached 0 and the audit (check_cold_cache, restart rule) must flag it."""
    monkeypatch.setenv("TFB_STUB_WIPE_HOME_ON_RESTART", "1")
    sp = fm.spec(
        sweep="restart",
        ctx="1K",
        kind="code",
        out=64,
        turns="restart",
        phases=("cold", "restart"),
    )
    rows = _rows(engine, [sp])
    raw, plan, pin = _run_cell_rows(tmp_path, rows)
    for row in rows:
        cold, restart = _decode(raw, engine, row)
        assert restart["cached"] == 0, "a wiped HOME has no disk tier to hit"
    manifest = fr.audit_manifest(rows, [fm.SMOKE_BATCH], pin, plan.engine_order)
    manifest["stub_engine"] = True
    res = _run_audit(raw, manifest, tmp_path)
    assert res.returncode != 0, res.stdout[-2000:]
    assert "restart: cached=0" in res.stdout, res.stdout[-2000:]


# ---------------------------------------------------------------- fast checks (no run)
def test_harness_part_maps_mixed_restart_and_same_prompt_rows():
    mixed = _single_variant_rows("yunshu", [_mixed_spec()])[0]
    part, extra, reason = fr.harness_part(mixed)
    assert reason is None and part == "mixed"
    assert extra[extra.index("--mixed-long-ctx") + 1] == "131072"
    assert extra[extra.index("--mixed-short-n") + 1] == str(SHORT_N)
    assert extra[extra.index("--mixed-short-ctx") + 1] == "1024"
    assert extra[extra.index("--conc-tokens") + 1] == "64"

    restart = _single_variant_rows(
        "yunshu",
        [
            fm.spec(
                sweep="restart",
                ctx="32K",
                turns="restart",
                phases=("cold", "restart"),
            )
        ],
    )[0]
    part, extra, reason = fr.harness_part(restart)
    assert reason is None and part == "decode"
    assert extra[extra.index("--phases") + 1] == "cold,restart"

    same = _single_variant_rows(
        "yunshu",
        [fm.spec(sweep="conc-same", ctx="8K", conc=4, phases=("cold",))],
    )[0]
    part, extra, reason = fr.harness_part(same)
    assert reason is None and part == "conc" and "--conc-same" in extra


def test_mixed_and_restart_phases_are_named_for_the_audit():
    assert fam.PHASE_NAMES["mixed"] == "mixed"
    assert fam.PHASE_NAMES["restart"] == "restart"
    row = _single_variant_rows("yunshu", [_mixed_spec()])[0]
    assert fr.expected_records(row) == fr.row_reps(row) * (SHORT_N + 1) * fr.CONC_TRIALS


def test_smoke_audit_manifest_names_conc_and_mixed_cells_by_their_record_part():
    """Regression (mixed e2e): a conc cell planned as cold failed record_count ('conc' recorded,
    'cold' planned) on every conc-same row; concurrent cells plan the record part itself."""
    rows = [
        _single_variant_rows(
            "yunshu",
            [fm.spec(sweep="conc-same", ctx="8K", conc=4, phases=("cold",))],
        )[0],
        _single_variant_rows("yunshu", [_mixed_spec()])[0],
        _single_variant_rows(
            "yunshu", [fm.spec(sweep="smoke-core", ctx="1K", kind="code", out=64)]
        )[0],
    ]
    pin = json.loads(fr.PIN_PATH.read_text())
    cells = fr.audit_manifest(rows, [fm.SMOKE_BATCH], pin, ["yunshu"])["cells"]
    by_sweep = {c["cell"].split(".")[1]: c for c in cells}
    assert by_sweep["conc-same"]["phases"] == ["conc"]
    assert by_sweep["mixed"]["phases"] == ["mixed"]
    assert by_sweep["smoke-core"]["phases"] == ["cold", "warm", "turn2"]


def test_tfbench_phase_selection_accepts_restart_and_keeps_the_default():
    import tfbench as tb

    assert tb.parse_phases("cold,restart") == {"cold", "restart"}
    assert "restart" not in tb.ALL_PHASES, (
        "the default decode run must not add a restart phase"
    )
    with pytest.raises(SystemExit):
        tb.parse_phases("cold,nope")


def test_restart_cell_is_unsupported_for_an_engine_without_a_disk_tier():
    """An engine whose census says no disk prefix tier gets 'unsupported', never a fake run."""
    sp = fm.spec(
        sweep="restart", ctx="32K", turns="restart", phases=("cold", "restart")
    )
    no_disk = [
        eng
        for eng, e in fm.ENGINES.items()
        if e.get("supported") and e.get("caps", {}).get("disk_cache") is False
    ]
    assert no_disk, "the census marks at least one engine without a disk tier"
    status, reason = fm.cell_status(no_disk[0], sp)
    assert status == fm.UNS and "disk" in reason


def test_mixed_guard_flags_two_long_requests_in_one_record():
    rows = []
    for role in ("long", "long", "short"):
        rows.append(
            {
                "part": "mixed",
                "phase": "mixed",
                "role": role,
                "n": 3,
                "trial": 0,
                "cached": 0,
            }
        )
    run = fa.Run(
        manifest={"reps": 1},
        files=[
            fa.CellFile({"cell": "c1", "engine": "yunshu"}, 0, Path("x"), None, rows)
        ],
        missing=[],
        extra=[],
        parse_errors=[],
    )
    res = fa.check_mixed_same_prompt(run)
    assert res.violations and "long" in res.violations[0]


def test_usage_shapes_are_read_from_the_table_not_hard_coded(tmp_path):
    """The stub's usage shape comes from fullcmp_usage_shapes.json (a GPU smoke updates it). A table
    that says 'unavailable' yields a usage object with no cache count, and tfbench reads None."""
    import tfbench as tb

    table = tmp_path / "shapes.json"
    table.write_text(json.dumps({"_doc": "note", "yunshu": "unavailable"}))
    shapes = sess.load_usage_shapes(table)
    assert shapes == {"yunshu": "unavailable"}
    usage = sess.usage_object(shapes["yunshu"], 100, 8, 40)
    assert "prompt_tokens_details" not in usage
    assert tb.cached_tokens(usage) is None
    on_disk = sess.load_usage_shapes()
    assert set(on_disk) >= set(ENGINES_UNDER_TEST)
    assert on_disk == sess.USAGE_SHAPE


def test_stub_counts_long_prompts_without_a_tokenizer():
    import fullcmp_stub_mixed as mx

    assert mx.count_long_prompt_tokens("x" * 524288) == 131072
    assert mx.count_long_prompt_tokens("") == 1
