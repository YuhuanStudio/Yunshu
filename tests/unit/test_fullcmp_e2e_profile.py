"""Layer-2 profile rep and B4 behaviour checks through the stub chain (no GPU, no real server).

Profile (A): the smoke job chain runs the timed job, then the -prof job, which runs the real tfbench
path against fullcmp_stub_profile (YUNSHU_DEBUG_TREE_PROFILE=1 in the stub's env). fullcmp_profile
appends the part=profile row; the audit with --profile-root --full must accept it CLEAN, and must
reject a missing profile rep, a missing layer-2 field, and profile rows in table numbers.

Behaviour (B): fullcmp_behaviour.py against the in-process stub: the full variant passes every
check; the base stub fails the tool, JSON and logprob checks and answers the two API routes with
404 (unsupported). The audit's behaviour guard accepts the expected statuses and rejects the rest.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "research"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "scripts" / "dev"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullcmp_attribution as fat  # noqa: E402
import fullcmp_audit as fa  # noqa: E402
import fullcmp_behaviour as fb  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_profile as fp  # noqa: E402
import fullcmp_reps as freps  # noqa: E402
import fullcmp_run as fr  # noqa: E402
import fullcmp_stub_profile as stub_profile  # noqa: E402
import fullcmp_stub_server as stub_base  # noqa: E402
from fullcmp_pin_io import stub_pin_io  # noqa: E402
from test_fullcmp_e2e_coverage import cell_type_key, profile_type_key  # noqa: E402

STUB_DIR = Path(__file__).resolve().parent
FAKE_JOB_ID = "2026-110301-01-fullcmp-SMOKE"
PROFILE_STUB_ENV = {"TFB_STUB_SERVER": str(STUB_DIR), "TFB_STUB_SCRIPT": "fullcmp_stub_profile.py"}


@pytest.fixture(autouse=True)
def _pin_io(monkeypatch):
    stub_pin_io(monkeypatch)


# Smoke rows (yunshu, 1K code, prompt variants 0 and 1): one timed type, one profile type.
SMOKE_TYPE = cell_type_key(fm.build_smoke_rows()[0])
SMOKE_PROFILE_TYPE = profile_type_key(fm.build_smoke_rows()[0])
BEHAVIOUR_TYPES = {
    f"part=behaviour;check={c}": "test_behaviour_full_stub_passes_every_check"
    for c in fb.CHECKS
}
def _real_spec(pairs, sweep, **match):
    """The manifest's own spec of this sweep (the type itself, not a copy of its shape)."""
    for _batch, sp in pairs:
        if sp["sweep"] == sweep and all(sp[k] == v for k, v in match.items()):
            return dict(sp)
    raise AssertionError(f"manifest has no {sweep} spec {match}")


# The profile-rep job of every manifest shape that has a timed row: the shape's own spec, one
# prompt variant, one rep. The timed twin of each is covered by its own module (session, content,
# mixed, the smoke chain); a rep key only belongs here.
PROFILE_SHAPES = {
    "conc c2": _real_spec(fm.b2_specs(), "conc", ctx="8K", conc=2),
    "conc c4": _real_spec(fm.b2_specs(), "conc", ctx="8K", conc=4),
    "conc c8": _real_spec(fm.b2_specs(), "conc", ctx="8K", conc=8),
    "conc-same c2": _real_spec(fm.b2_specs(), "conc-same", conc=2),
    "conc-same c4": _real_spec(fm.b2_specs(), "conc-same", conc=4),
    "conc-same c8": _real_spec(fm.b2_specs(), "conc-same", conc=8),
    "smoke-conc code": fm.spec(sweep="smoke-conc", ctx="1K", kind="code", out=64, conc=2, phases=("conc",)),
    "smoke-conc prose": fm.spec(sweep="smoke-conc", ctx="1K", kind="prose", out=64, conc=2, phases=("conc",)),
    "mixed": _real_spec(fm.b2_specs(), "mixed", conc=8),
    "core prose": _real_spec(fm.b1_specs(), "core", ctx="1K", kind="prose"),
    "smoke-core prose": fm.spec(sweep="smoke-core", ctx="1K", kind="prose", out=64, phases=("cold", "warm", "turn2")),
    "core code": _real_spec(fm.b1_specs(), "core", ctx="1K", kind="code"),
    "ctx code": _real_spec(fm.b2_specs(), "ctx", ctx="4K"),
    "out code": _real_spec(fm.b2_specs(), "out", ctx="1K", out=128),
    "idle5": _real_spec(fm.b2_specs(), "idle5", ctx="8K"),
    "restart": _real_spec(fm.b2_specs(), "restart", ctx="32K"),
    "image img1": _real_spec(fm.b2_specs(), "image", media="img1"),
    "image img4": _real_spec(fm.b2_specs(), "image", media="img4"),
    "kind cjk": _real_spec(fm.b2_specs(), "kind", kind="cjk"),
    "kind json-schema": _real_spec(fm.b2_specs(), "kind", kind="json-schema"),
    "kind tool-chat": _real_spec(fm.b2_specs(), "kind", kind="tool-chat"),
    "thinking": _real_spec(fm.b2_specs(), "thinking"),
    "genconfig": _real_spec(fm.b2_specs(), "sampling", ctx="1K", sampling="genconfig"),
    "smoke-profile": fm.spec(sweep="smoke-profile", ctx="1K", kind="code", out=64, phases=("cold",)),
}


def _profile_row(sp):
    row = fm._cell_row("yunshu", fm.SMOKE_BATCH, sp)
    row["id"] = f"{row['id']}.v0"
    row["reps"] = 1
    row["prompt_variant"] = 0
    row["est_gpu_min"] = 0.0
    return row


COVERED_CELL_TYPES = {
    SMOKE_PROFILE_TYPE: "test_profile_chain_audit_full_is_clean",
    **{
        profile_type_key(_profile_row(sp)): f"test_profile_rep_job_runs_for_each_shape[{name}]"
        for name, sp in PROFILE_SHAPES.items()
    },
    **BEHAVIOUR_TYPES,
}


@pytest.mark.parametrize("shape", list(PROFILE_SHAPES))
def test_profile_rep_job_runs_for_each_shape(tmp_path, shape):
    """The layer-2 profile rep of one manifest shape: the timed job, then the -prof job, both against
    the profile stub. The profile record lands under profile/ with no error."""
    row = _profile_row(PROFILE_SHAPES[shape])
    fr.configure(tmp_path / "raw", tmp_path / "work")
    pin = json.loads(fr.PIN_PATH.read_text())
    plan = fr.plan_batch([row], fm.SMOKE_BATCH, pin, profile_rep=True)
    assert not plan.skipped, [why for _r, why in plan.skipped]
    timed = [j for j in plan.jobs if not j.label.endswith("-prof")]
    prof = [j for j in plan.jobs if j.label.endswith("-prof")]
    assert len(timed) == 1 and len(prof) == 1, [j.label for j in plan.jobs]
    _run_wrapper(timed[0], PROFILE_STUB_ENV)
    _run_wrapper(prof[0], PROFILE_STUB_ENV)
    out = tmp_path / "raw" / "profile" / "yunshu" / f"{row['id']}-prof.jsonl"
    assert out.exists(), out
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    profs = [r for r in rows if r.get("part") == "profile"]
    assert len(profs) == 1 and not profs[0].get("error"), profs[:1]


# ---------------------------------------------------------------- A: profile rep through the chain


def _smoke_rows():
    rows = fm.build_smoke_rows()
    keep = [r for r in rows if r["sweep"] == "smoke-core" and r["prompt_kind"] == "code"]
    assert {r["prompt_variant"] for r in keep} == {0, 1}
    return keep


def _run_wrapper(job, extra_env):
    wrapper = fr.write_wrapper(job)
    env = dict(os.environ, GPUQ_JOB_ID=FAKE_JOB_ID, **extra_env)
    proc = subprocess.run(
        ["/bin/sh", str(wrapper)], env=env, capture_output=True, text=True, timeout=900, cwd=str(ROOT)
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """Timed job, then the -prof job, both against the profile stub. The profile job's stub runs with
    the engine's profiling switch on (the switch is in the job's own env assignments)."""
    work = tmp_path_factory.mktemp("fullcmp-profile")
    raw = work / "raw"
    fr.configure(raw, work / "work")
    pin = json.loads(fr.PIN_PATH.read_text())
    rows = _smoke_rows()
    plan = fr.plan_batch(rows, fm.SMOKE_BATCH, pin, profile_rep=True)
    timed = [j for j in plan.jobs if not j.label.endswith("-prof")]
    prof = [j for j in plan.jobs if j.label.endswith("-prof")]
    assert len(timed) == 1 and len(prof) == 1, [j.label for j in plan.jobs]
    _run_wrapper(timed[0], PROFILE_STUB_ENV)
    _run_wrapper(prof[0], PROFILE_STUB_ENV)
    manifest = fr.audit_manifest(rows, [fm.SMOKE_BATCH], pin, plan.engine_order)
    manifest["stub_engine"] = True  # test only: the engine process is the stub
    return {"work": work, "raw": raw, "rows": rows, "pin": pin, "manifest": manifest, "plan": plan}


def _audit(root, manifest, work, profile_root=None, full=False):
    mpath = work / f"audit-{abs(hash((str(root), str(profile_root), full)))}.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    argv = [fr.HARNESS_PY, str(SCRIPTS / "fullcmp_audit.py"), "--root", str(root), "--manifest", str(mpath)]
    if profile_root is not None:
        argv += ["--profile-root", str(profile_root)]
    if full:
        argv += ["--full"]
    return subprocess.run(argv, capture_output=True, text=True, timeout=600, cwd=str(ROOT))


def test_profile_job_is_the_extra_job_after_the_timed_one(chain):
    jobs = chain["plan"].jobs
    prof = next(j for j in jobs if j.label.endswith("-prof"))
    assert prof.flags == ["profile"]
    assert "YUNSHU_DEBUG_TREE_PROFILE=1" in prof.script
    assert "/profile/yunshu/" in prof.script
    assert prof.script.index("fullcmp_profile.py") > prof.script.index("tfbench.py")


def test_profile_rows_land_under_profile_and_carry_the_layer2_fields(chain):
    prof_dir = chain["raw"] / "profile" / "yunshu"
    files = sorted(prof_dir.glob("*-prof.jsonl"))
    assert len(files) == len(chain["rows"]), [f.name for f in files]
    for f in files:
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        prof = [r for r in rows if r.get("part") == "profile"]
        assert len(prof) == 1, f.name
        assert "error" not in prof[0] or not prof[0]["error"], prof[0]
        assert fat.check_profile("yunshu", prof[0]) and not fat.missing(fat.check_profile("yunshu", prof[0]))
        assert prof[0]["host_sync_count"] == 3 and prof[0]["op_census"] == {"matmul": 120, "rope": 40}
    # the timed rows never went under profile/, and the profile rows never beside them
    assert not list((chain["raw"] / "yunshu").glob("*-prof.jsonl"))


def test_profile_chain_audit_full_is_clean(chain):
    """Layer 1 on the timed rows, layer 2 on the profile rows, every cell (full mode). Zero violations."""
    res = _audit(chain["raw"], chain["manifest"], chain["work"], chain["raw"] / "profile", full=True)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-3000:]
    assert "0 violations" in res.stdout, res.stdout[-2000:]


def _copy_tree(src, dst):
    shutil.copytree(src, dst)
    return dst


def test_profile_rep_missing_is_a_violation(chain, tmp_path):
    prof = _copy_tree(chain["raw"] / "profile", tmp_path / "profile")
    victim = next((prof / "yunshu").glob("*-prof.jsonl"))
    victim.unlink()
    res = _audit(chain["raw"], chain["manifest"], chain["work"], prof, full=True)
    assert res.returncode != 0
    assert "profile rep missing" in res.stdout, res.stdout[-2000:]


def test_profile_row_missing_is_a_violation(chain, tmp_path):
    prof = _copy_tree(chain["raw"] / "profile", tmp_path / "profile")
    victim = next((prof / "yunshu").glob("*-prof.jsonl"))
    kept = [x for x in victim.read_text().splitlines() if json.loads(x).get("part") != "profile"]
    victim.write_text("\n".join(kept) + "\n")
    res = _audit(chain["raw"], chain["manifest"], chain["work"], prof, full=True)
    assert res.returncode != 0
    assert "no part=profile row" in res.stdout, res.stdout[-2000:]


def test_profile_without_layer2_fields_fails_closed(chain, tmp_path):
    """The shape the real engine prints today: the three profile lines only. The layer-2 fields that
    the lines do not carry must be reported MISSING, never invented."""
    prof = _copy_tree(chain["raw"] / "profile", tmp_path / "profile")
    for victim in (prof / "yunshu").glob("*-prof.jsonl"):
        out = []
        for line in victim.read_text().splitlines():
            row = json.loads(line)
            if row.get("part") == "profile":
                row.pop("host_sync_count", None)
                row.pop("op_census", None)
                row["stages"] = {"decode_round": row["stages"]["decode_round"]}
            out.append(json.dumps(row))
        victim.write_text("\n".join(out) + "\n")
    res = _audit(chain["raw"], chain["manifest"], chain["work"], prof, full=True)
    assert res.returncode != 0
    assert "profile rep lacks layer 2" in res.stdout
    assert "host_sync_count" in res.stdout and "op_census" in res.stdout
    assert "stages.decode_round.plan" in res.stdout


def test_real_engine_profile_lines_alone_do_not_satisfy_layer2():
    """Today's engine prints only dflash_fast.py's three profile lines (plus no layer-2 contract line).
    Parsed alone they must leave the layer-2 fields MISSING (the --full gate then fails closed)."""
    lines = "\n".join(stub_profile.PROFILE_LINES)
    row = {"stages": fp.parse_yunshu_stages(lines)}
    row.update(fp.parse_yunshu_layer2(lines))
    missing = fat.missing(fat.check_profile("yunshu", row))
    assert "host_sync_count" in missing and "op_census" in missing
    assert "stages.decode_round.plan" in missing


def test_profile_rows_are_excluded_from_table_numbers(chain, tmp_path):
    """The table-number scan (fullcmp_reps.load_results) reads <engine>/<cell>-rN.jsonl only: a profile
    row must not become a rep, a cell or a group entry even when a profile file sits beside it."""
    results = _copy_tree(chain["raw"], tmp_path / "results")
    # a profile file misplaced beside the timed cells, with a profile row and a rep-like name
    stray = results / "yunshu" / "stray-prof.jsonl"
    stray.write_text(json.dumps({"part": "profile", "stages": {}}) + "\n")
    cells = freps.load_results(results)
    engines_cells = {cell for (_eng, cell) in cells}
    timed_ids = {r["id"] for r in chain["rows"]}
    assert engines_cells == timed_ids, engines_cells ^ timed_ids
    for key, entry in cells.items():
        for rep, rows in entry["reps"].items():
            assert all(r.get("part") != "profile" for r in rows), key
    # the profile tree is never scanned as a table engine either
    assert all(eng != "profile" for (eng, _c) in cells)


def test_audit_extra_scan_ignores_profile_and_behaviour_dirs(chain):
    """The audit's unplanned-file scan must not count profile/ or behaviour/ as unplanned output."""
    res = _audit(chain["raw"], chain["manifest"], chain["work"], chain["raw"] / "profile", full=True)
    assert "unplanned file" not in res.stdout


# ---------------------------------------------------------------- B: behaviour checks (B4)


def _serve(stub_module, tmp_path, engine="yunshu"):
    server, port, _state = stub_module.start(engine, "stub-model", tmp_path / "server.log")
    return server, f"http://127.0.0.1:{port}"


@pytest.fixture
def full_stub(tmp_path):
    server, url = _serve(stub_profile, tmp_path)
    yield url
    server.shutdown()


@pytest.fixture
def base_stub(tmp_path):
    server, port, _state = stub_base.start("yunshu", "stub-model", tmp_path / "server.log")
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _checks(out_path):
    rows = [json.loads(x) for x in Path(out_path).read_text().splitlines()]
    return rows[0], {r["check"]: r for r in rows[1:] if r.get("part") == "behaviour"}


def test_behaviour_full_stub_passes_every_check(full_stub, tmp_path):
    out = tmp_path / "behaviour" / "yunshu.jsonl"
    rc = fb.run("yunshu", full_stub, "stub-model", out, "RUN-1")
    head, got = _checks(out)
    assert head["part"] == "behaviour_session" and head["run_id"] == "RUN-1"
    assert set(got) == set(fb.CHECKS)
    bad = {k: (v["status"], v["reason"]) for k, v in got.items() if v["status"] != "pass"}
    assert rc == 0 and not bad, bad


def test_behaviour_base_stub_fails_tool_json_logprobs_and_reports_missing_routes(base_stub, tmp_path):
    out = tmp_path / "behaviour" / "yunshu.jsonl"
    rc = fb.run("yunshu", base_stub, "stub-model", out, "RUN-2")
    _head, got = _checks(out)
    assert got["stop_sequences"]["status"] == "pass"
    assert got["cancel_latency"]["status"] == "pass"
    assert got["tool_validity"]["status"] == "fail"
    assert got["json_validity"]["status"] == "fail"
    assert got["logprobs"]["status"] == "fail"
    assert got["anthropic_api"]["status"] == "unsupported"
    assert got["responses_api"]["status"] == "unsupported"
    assert rc == 1


def test_behaviour_audit_guard_accepts_expected_statuses_and_rejects_others(full_stub, base_stub, tmp_path):
    """Expectations: pass for every check on the full stub; on the base stub the routes are expected
    unsupported and the tool/JSON/logprob checks are expected pass (so they must be violations)."""
    expect_all_pass = {c: "pass" for c in fb.CHECKS}
    root_ok = tmp_path / "ok"
    fb.run("yunshu", full_stub, "stub-model", root_ok / "behaviour" / "yunshu.jsonl", "R")
    man = {"cells": [], "behaviour": {"yunshu": expect_all_pass}}
    run = fa.load_run(root_ok, man)
    assert fa.check_behaviour(run).violations == []

    root_bad = tmp_path / "bad"
    fb.run("yunshu", base_stub, "stub-model", root_bad / "behaviour" / "yunshu.jsonl", "R")
    man_bad = {"cells": [], "behaviour": {"yunshu": expect_all_pass}}
    vio = fa.check_behaviour(fa.load_run(root_bad, man_bad)).violations
    joined = "\n".join(vio)
    for check in ("tool_validity", "json_validity", "logprobs", "anthropic_api", "responses_api"):
        assert f"yunshu.{check}" in joined, joined
    # the same stub with the census saying the routes are unsupported: exactly the route checks pass
    man_census = {"cells": [], "behaviour": {"yunshu": dict(expect_all_pass, anthropic_api="unsupported", responses_api="unsupported")}}
    vio2 = "\n".join(fa.check_behaviour(fa.load_run(root_bad, man_census)).violations)
    assert "anthropic_api" not in vio2 and "responses_api" not in vio2


def test_behaviour_audit_missing_and_unplanned_files_fail_closed(full_stub, tmp_path):
    root = tmp_path / "none"
    (root / "behaviour").mkdir(parents=True)
    man = {"cells": [], "behaviour": {"yunshu": {"logprobs": "pass"}}}
    vio = fa.check_behaviour(fa.load_run(root, man)).violations
    assert any("behaviour file missing" in v for v in vio)
    fb.run("tf-1.2.1", full_stub, "stub-model", root / "behaviour" / "tf-1.2.1.jsonl", "R", ("logprobs",))
    vio = fa.check_behaviour(fa.load_run(root, man)).violations
    assert any("unplanned behaviour file" in v for v in vio)


def test_behaviour_checker_exit_and_unknown_check_are_strict(full_stub, tmp_path):
    out = tmp_path / "b.jsonl"
    rc = fb.main(["--engine", "yunshu", "--url", full_stub, "--model", "stub-model", "--out", str(out),
                  "--checks", "stop_sequences"])
    assert rc == 0
    with pytest.raises(SystemExit):
        fb.main(["--engine", "yunshu", "--url", full_stub, "--model", "stub-model", "--out", str(out),
                 "--checks", "nope"])


def test_behaviour_expectations_follow_the_manifest_census():
    """fullcmp_run.behaviour_expectations: unverified rows expect pass, unsupported rows expect unsupported."""
    rows = [
        {"table": "behaviour", "batch": "B4", "engine": "e1", "check": "logprobs", "status": "unverified"},
        {"table": "behaviour", "batch": "B4", "engine": "e1", "check": "anthropic_api", "status": "unsupported"},
        {"table": "decode", "batch": "B1", "engine": "e1", "check": None, "status": "unverified"},
    ]
    assert fr.behaviour_expectations(rows, ["B4"]) == {"e1": {"logprobs": "pass", "anthropic_api": "unsupported"}}
