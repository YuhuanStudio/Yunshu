"""Coverage check for the full-comparison harness: every cell type in the manifests has a test.

A cell type is the canonical key of one manifest row (cell_type_key). Types come from the smoke
manifest (fullcmp_manifest.build_smoke_rows) and from the real manifest
(compsync/fullcmp-1103/manifest.jsonl, every status, batches B1/B2/B4). Each row gives a timed
type and, for decode/conc rows, a profile type (the --profile-rep job of the same cell).

Covering: each test module declares COVERED_CELL_TYPES = {key: "test id"}. The check imports the
modules listed in COVERAGE_MODULES and FAILS when a module is absent or cannot be imported (no
importorskip: a missing module hides nothing), when a type has no covering entry, when a covered
key matches no manifest type (a typo), and when a type listed in XFAIL_CELL_TYPES is covered
(remove the stale entry). Types in XFAIL_CELL_TYPES need a reason.

No GPU, no server. Pure manifest and registry reads.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "research"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fullcmp_manifest as fm  # noqa: E402

COVERAGE_MODULES = (
    "test_fullcmp_e2e",
    "test_fullcmp_e2e_session",
    "test_fullcmp_e2e_content",
    "test_fullcmp_e2e_profile",
    "test_fullcmp_e2e_coverage",
    "test_fullcmp_e2e_mixed",
)
REAL_BATCHES = ("B1", "B2", "B4")
# Types that may stay uncovered on purpose: key -> reason. Empty means the gate is strict.
XFAIL_CELL_TYPES: dict[str, str] = {}

# Phase names of the manifest and of the smoke rows: smoke calls the follow-up "turn2".
PHASE_NAMES = {"turn2": "fu"}


def cell_type_key(row: dict) -> str:
    """Canonical cell type of one manifest row. Dimensions: part, kind (sweep:prompt kind), phase,
    conc, turns, cache_state, thinking, media, sampling, profile. ctx and the model are not
    dimensions: a 1K and a 32K cell of the same shape are one type."""
    if row.get("table") == "behaviour":
        return f"part=behaviour;check={row['check']}"
    conc = int(row.get("conc") or 1)
    phases = [PHASE_NAMES.get(p, p) for p in (row.get("phases") or [])]
    if "warm" in phases:
        cache = "prefix-hit"
    elif "restart" in phases:
        cache = "restart-disk"
    else:
        cache = "cold"
    fields = {
        "part": "conc" if conc > 1 else "decode",
        "kind": f"{row.get('sweep')}:{row.get('prompt_kind')}",
        "phase": "+".join(phases) or "-",
        "conc": conc,
        "turns": row.get("turns") or "single",
        "cache_state": cache,
        "thinking": row.get("thinking") or "off",
        "media": row.get("media") or "none",
        "sampling": row.get("sampling") or "greedy",
        "profile": "timed",
    }
    return ";".join(f"{k}={v}" for k, v in fields.items())


def profile_type_key(row: dict) -> str:
    """The layer-2 profile rep of the same cell (fullcmp_run --profile-rep)."""
    return cell_type_key(row).replace("profile=timed", "profile=rep")


def _real_manifest_all_rows() -> list[dict]:
    path = Path(fm.DEFAULT_OUT)
    if not path.is_file():
        raise AssertionError(
            f"real manifest missing: {path} (cell types cannot be enumerated)"
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("batch") in REAL_BATCHES]


def real_manifest_rows() -> list[dict]:
    """The rows this batch would run: status unverified (the run tests use these)."""
    return [r for r in _real_manifest_all_rows() if r.get("status") == "unverified"]


def manifest_cell_types() -> dict[str, str]:
    """{type key: example row id} over the smoke and the real manifests. Every manifest row counts
    as a cell type, whatever its status: a 'cut' row (dropped from this batch for budget, e.g. the image
    cells) or an 'unsupported' one is still a cell of the manifest, and its type needs a covering test."""
    rows = list(fm.build_smoke_rows()) + _real_manifest_all_rows()
    types: dict[str, str] = {}
    for r in rows:
        types.setdefault(cell_type_key(r), r["id"])
        if r.get("table") != "behaviour":
            types.setdefault(profile_type_key(r), r["id"])
    return types


def collect_covered() -> tuple[dict[str, str], list[str]]:
    """({type key: test id}, problems). A module that is absent or fails to import is a problem."""
    covered: dict[str, str] = {}
    problems: list[str] = []
    for name in COVERAGE_MODULES:
        try:
            mod = importlib.import_module(
                f"{__package__}.{name}" if __package__ else name
            )
        except Exception as exc:  # noqa: BLE001 - any failure to load is a coverage failure
            problems.append(f"{name}: not importable ({type(exc).__name__}: {exc})")
            continue
        table = getattr(mod, "COVERED_CELL_TYPES", None)
        if not isinstance(table, dict):
            problems.append(f"{name}: no COVERED_CELL_TYPES dict")
            continue
        for key, test_id in table.items():
            if key in covered:
                problems.append(
                    f"{key}: covered twice ({covered[key]} and {name}::{test_id})"
                )
            covered[key] = f"{name}::{test_id}"
    return covered, problems


def coverage_report() -> dict:
    types = manifest_cell_types()
    covered, problems = collect_covered()
    uncovered = sorted(
        k for k in types if k not in covered and k not in XFAIL_CELL_TYPES
    )
    stale_xfail = sorted(k for k in XFAIL_CELL_TYPES if k in covered)
    unknown = sorted(
        k for k in list(covered) + list(XFAIL_CELL_TYPES) if k not in types
    )
    return {
        "types": types,
        "covered": covered,
        "problems": problems,
        "uncovered": uncovered,
        "stale_xfail": stale_xfail,
        "unknown": unknown,
    }


def test_every_manifest_cell_type_has_a_covering_test():
    rep = coverage_report()
    msgs = []
    if rep["problems"]:
        msgs.append("modules: " + "; ".join(rep["problems"]))
    if rep["uncovered"]:
        msgs.append(
            f"{len(rep['uncovered'])} uncovered cell types:\n  "
            + "\n  ".join(rep["uncovered"])
        )
    if rep["stale_xfail"]:
        msgs.append(
            "XFAIL types that are covered now (remove them): "
            + "; ".join(rep["stale_xfail"])
        )
    if rep["unknown"]:
        msgs.append(
            "covered or XFAIL keys matching no manifest type: "
            + "; ".join(rep["unknown"])
        )
    assert not msgs, "\n".join(msgs)


def test_taxonomy_enumerates_the_manifest_dimensions():
    """The taxonomy must see the real B2 sampling, thinking, media, kind and idle/restart/mixed shapes,
    not only the smoke row, so the coverage check cannot pass on an empty or short enumeration."""
    types = manifest_cell_types()
    text = "\n".join(types)
    for needle in (
        "sampling=genconfig",
        "thinking=on",
        "media=img1",
        "kind=kind:cjk",
        "kind=kind:json-schema",
        "kind=kind:tool-chat",
        "cache_state=restart-disk",
        "conc=8",
        "phase=cold+fu_idle",
        "part=behaviour;check=cancel_latency",
        "profile=rep",
    ):
        assert needle in text, f"taxonomy lacks {needle}"
    assert len(types) >= 40


def test_xfail_types_are_real_and_carry_a_reason():
    for key, reason in XFAIL_CELL_TYPES.items():
        assert isinstance(reason, str) and reason.strip(), key


# ---------------------------------------------------------------- genconfig sampling (tfbench)
# The model's generation_config (temperature 1.0, top_p 0.95, top_k 20 for Qwen3.8-27B) is sent
# explicitly at conc 1 (B2 sampling rows, cold phase, 1K and 32K code). The stub run below checks the
# request carries those values and the rows record them.

GENCONFIG_TYPE = cell_type_key(
    {
        "table": "decode",
        "sweep": "sampling",
        "prompt_kind": "code",
        "conc": 1,
        "phases": ["cold"],
        "turns": "single",
        "thinking": "off",
        "media": "none",
        "sampling": "genconfig",
    }
)
COVERED_CELL_TYPES = {
    GENCONFIG_TYPE: "test_genconfig_sampling_stub_run",
}


def test_genconfig_reads_the_model_sampler_and_fails_closed(tmp_path):
    import tfbench as tb

    with pytest.raises(SystemExit, match="cannot read"):
        tb.genconfig_sampling(tmp_path)
    (tmp_path / "generation_config.json").write_text(json.dumps({"top_k": 20}))
    with pytest.raises(SystemExit, match="no temperature/top_p"):
        tb.genconfig_sampling(tmp_path)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "eos_token_id": 1})
    )
    assert tb.genconfig_sampling(tmp_path) == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    }


def test_genconfig_sampling_stub_run(tmp_path):
    """tfbench --sampling genconfig against the stub: the request carries the model's sampler and the
    rows record it. Runs the exact command line the runner builds for a real genconfig row."""
    import fullcmp_run as fr
    import tfbench as tb

    stub_dir = Path(__file__).resolve().parent
    fr.configure(tmp_path / "raw", tmp_path / "work")
    pin = json.loads(fr.PIN_PATH.read_text())
    rows = [r for r in real_manifest_rows() if r.get("sampling") == "genconfig"]
    assert rows, "no genconfig rows in the real manifest"
    row = next(r for r in rows if r["ctx_tokens"] == 1024)
    row = dict(row, id=row["id"] + ".stubrun", out_len=16)
    argv = fr.cell_command(row, 0, pin)
    assert argv is not None and "--sampling" in argv
    assert argv[argv.index("--sampling") + 1] == "genconfig"
    env = dict(
        os.environ,
        GPUQ_JOB_ID="2026-110301-01-fullcmp-SMOKE",
        TFB_STUB_SERVER=str(stub_dir),
    )
    # the job wrapper makes the raw directory before tfbench appends to it; this runs argv directly
    Path(fr.raw_path(row["engine"], row["id"], 0)).parent.mkdir(
        parents=True, exist_ok=True
    )
    proc = subprocess.run(
        argv, env=env, capture_output=True, text=True, timeout=600, cwd=str(ROOT)
    )
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-3000:]
    out = Path(fr.raw_path(row["engine"], row["id"], 0))
    decs = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    decs = [r for r in decs if r.get("part") == "decode"]
    assert decs, "no decode rows"
    want = tb.genconfig_sampling(tb.M)
    for r in decs:
        assert r["temperature"] == want["temperature"] == 1.0
        assert r["top_p"] == want["top_p"] == 0.95
        assert r["top_k"] == want["top_k"] == 20


def test_log_usage_sample_records_the_raw_usage_chunk_through_the_stub(tmp_path):
    """tfbench --log-usage-sample (the GPU smoke's usage evidence) writes the first request's raw usage
    chunk into part_done.usage_sample. Runs the runner's real command line plus the flag, on the stub."""
    import fullcmp_run as fr

    stub_dir = Path(__file__).resolve().parent
    fr.configure(tmp_path / "raw", tmp_path / "work")
    pin = json.loads(fr.PIN_PATH.read_text())
    row = next(r for r in real_manifest_rows() if r.get("ctx_tokens") == 1024)
    row = dict(row, id=row["id"] + ".usagesample", out_len=16)
    argv = fr.cell_command(row, 0, pin) + ["--log-usage-sample"]
    env = dict(
        os.environ,
        GPUQ_JOB_ID="2026-110301-01-fullcmp-SMOKE",
        TFB_STUB_SERVER=str(stub_dir),
    )
    Path(fr.raw_path(row["engine"], row["id"], 0)).parent.mkdir(
        parents=True, exist_ok=True
    )
    proc = subprocess.run(
        argv, env=env, capture_output=True, text=True, timeout=600, cwd=str(ROOT)
    )
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-3000:]
    out = Path(fr.raw_path(row["engine"], row["id"], 0))
    # finalize_session writes the end-of-cell fields (usage_sample included) into the session header
    head = json.loads(out.read_text().splitlines()[0])
    assert head.get("part") == "session", head.keys()
    sample = head.get("usage_sample")
    assert isinstance(sample, dict) and sample, (
        f"usage_sample not recorded: {sorted(head)}"
    )
    assert "usage" in sample, sorted(sample)
