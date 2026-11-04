"""CPU end-to-end of the B2 request-family cells (tool-chat, cjk, json-schema, thinking on, images 1 and
4) through the same chain as test_fullcmp_e2e.py: smoke-style rows -> fullcmp_run builds the job argv
and wrapper -> the wrapper runs real tfbench against the request-family stub (TFB_STUB_SCRIPT) with
its telemetry -> real fullcmp_audit (guard family_validity included): zero violations.

No GPU, no engine: the stub answers each family the way a conforming engine does (tool_calls, schema
JSON, reasoning_content split, image tokens). Negative cases make one reply invalid and check the row
is flagged and the audit fails.

COVERED_CELL_TYPES maps each covered family to the test that proves it end to end.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "research"
UNIT = ROOT / "tests" / "unit"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "scripts" / "dev"))
sys.path.insert(0, str(UNIT))

import fullcmp_audit as fa  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_run as fr  # noqa: E402
import fullcmp_stub_content as stub  # noqa: E402
import tfbench as tb  # noqa: E402
from .test_fullcmp_e2e_coverage import cell_type_key  # noqa: E402

CHAIN_ENGINES = ("yunshu", "tf-1.2.1")
FAMILY_SWEEPS = ("kind", "thinking", "image")

_FAMILY_TEST = {
    "tool-chat": "test_tool_chat_cell_replies_with_valid_tool_calls",
    "json-schema": "test_json_schema_cell_replies_conformant_json",
    "cjk": "test_cjk_cell_replies_cjk_text",
    "thinking": "test_thinking_cell_splits_reasoning_from_content",
    "image": "test_image_cells_are_accepted_and_report_image_tokens",
}


def _family_cover():
    """cell_type_key of every family spec the chain runs (the B2 kind, thinking and image sweeps),
    mapped to the test that validates its replies. Keys come from the manifest's own specs."""
    out = {}
    for _batch, sp in fm.b2_specs():
        if sp["sweep"] not in FAMILY_SWEEPS:
            continue
        if sp["sweep"] == "kind" and sp["kind"] not in fm.FAMILY_KINDS:
            continue
        name = sp["kind"] if sp["sweep"] == "kind" else sp["sweep"]
        out[cell_type_key(fm._cell_row("yunshu", fm.SMOKE_BATCH, sp))] = _FAMILY_TEST[name]
    return out


COVERED_CELL_TYPES = _family_cover()


def _family_rows(engine):
    """Two prompt variants of every family cell at 8K (images at 1K), as the smoke matrix does.
    tf-1.2.1 gets no image rows here: a text-only 400 would kill the job (see the strict xfail)."""
    rows = []
    for variant in (0, 1):
        for batch, sp in fm.b2_specs():
            if sp["sweep"] not in FAMILY_SWEEPS:
                continue
            if sp["sweep"] == "kind" and sp["kind"] not in fm.FAMILY_KINDS:
                continue
            if sp["media"] != "none" and engine != "yunshu":
                continue
            row = fm._cell_row(engine, fm.SMOKE_BATCH, sp)
            row["id"] = f"{row['id']}.v{variant}"
            row["reps"] = 1
            row["prompt_variant"] = variant
            row["est_gpu_min"] = 0.0
            rows.append(row)
    return rows


def _family_of(row):
    if row["prompt_kind"] in fm.FAMILY_KINDS:
        return row["prompt_kind"]
    if row["thinking"] == "on":
        return "thinking"
    return "images"


def _pin():
    return json.loads(fr.PIN_PATH.read_text())


def _run_chain(engine, work):
    """Wrapper per planned job, exactly as test_fullcmp_e2e.good_run does, with the family stub."""
    fr.configure(work / "raw", work / "work")
    pin = _pin()
    rows = _family_rows(engine)
    plan = fr.plan_batch(rows, fm.SMOKE_BATCH, pin)
    assert not plan.skipped, [(r["id"], why) for r, why in plan.skipped]
    assert plan.jobs, "the family rows must plan at least one job"
    for n, job in enumerate(plan.jobs, 1):
        # one gpuq job id per job, as gpuq assigns them (the fairness guard counts HOME per job)
        env = dict(os.environ)
        env["GPUQ_JOB_ID"] = f"2026-110301-{n:02d}-fullcmp-SMOKE"
        env["TFB_STUB_SERVER"] = str(UNIT)
        env["TFB_STUB_SCRIPT"] = "fullcmp_stub_content.py"
        wrapper = fr.write_wrapper(job)
        proc = subprocess.run(
            ["/bin/sh", str(wrapper)],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(ROOT),
        )
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    return {"rows": rows, "pin": pin, "plan": plan, "work": work, "raw": work / "raw"}


@pytest.fixture(scope="module", params=CHAIN_ENGINES)
def chain(request, tmp_path_factory):
    return _run_chain(request.param, tmp_path_factory.mktemp(f"fam-{request.param}"))


def _cell_path(chain, row):
    return chain["raw"] / row["engine"] / f"{row['id']}-r0.jsonl"


def _decode_rows(chain, row):
    """The decode rows tfbench wrote for one planned family row (header line skipped)."""
    recs = [json.loads(x) for x in _cell_path(chain, row).read_text().splitlines()[1:]]
    return [r for r in recs if r.get("part") == "decode"]


def _rows_of(chain, family):
    """Planned rows of one family: 'images' is every image cell, the others by family name."""
    if family == "images":
        return [r for r in chain["rows"] if r["media"] != "none"]
    return [r for r in chain["rows"] if _family_of(r) == family]


def _audit(chain):
    manifest = fr.audit_manifest(chain["rows"], [fm.SMOKE_BATCH], chain["pin"], chain["plan"].engine_order)
    manifest["stub_engine"] = True  # test only: the engine process is the family stub
    mpath = chain["work"] / "family-audit-manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    return subprocess.run(
        [fr.HARNESS_PY, str(SCRIPTS / "fullcmp_audit.py"), "--root", str(chain["raw"]), "--manifest", str(mpath)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
    )


def test_family_chain_audit_is_clean(chain):
    res = _audit(chain)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    assert "family_validity" in res.stdout
    assert " 0 violations" in res.stdout, res.stdout[-2000:]


def test_tool_chat_cell_replies_with_valid_tool_calls(chain):
    for row in _rows_of(chain, "tool-chat"):
        rows = _decode_rows(chain, row)
        assert rows
        for r in rows:
            assert r["family"] == "tool-chat" and r["valid"] is True, r.get("invalid_reason")
            assert r["tool_calls"], "tool_calls must be recorded"
            for c in r["tool_calls"]:
                assert c["name"] in {"read_file", "run_shell"}
                assert isinstance(json.loads(c["arguments"]), dict)


def test_json_schema_cell_replies_conformant_json(chain):
    for row in _rows_of(chain, "json-schema"):
        for r in _decode_rows(chain, row):
            assert r["family"] == "json-schema" and r["valid"] is True, r.get("invalid_reason")
            obj = json.loads(r["text"])
            assert set(obj) == {"title", "items"}


def test_cjk_cell_replies_cjk_text(chain):
    for row in _rows_of(chain, "cjk"):
        for r in _decode_rows(chain, row):
            assert r["family"] == "cjk" and r["valid"] is True, r.get("invalid_reason")
            assert r["text"].strip()


def test_thinking_cell_splits_reasoning_from_content(chain):
    for row in _rows_of(chain, "thinking"):
        for r in _decode_rows(chain, row):
            assert r["thinking"] == "on"
            assert r["reasoning_chars"] > 0
            assert r["valid"] is True, r.get("invalid_reason")
            assert "<think>" not in r["text"]


def test_image_cells_are_accepted_and_report_image_tokens(chain):
    if chain["rows"][0]["engine"] != "yunshu":
        pytest.xfail("tf-1.2.1: no image rows in its chain; see test_images_on_tf_1_2_1_are_valid")
    assert _rows_of(chain, "images")
    for row in _rows_of(chain, "images"):
        for r in _decode_rows(chain, row):
            assert r["image_parts"] == fm.IMAGE_COUNT[row["media"]]
            assert r["image_tokens"] == stub.IMAGE_TOKENS * r["image_parts"]
            assert r["valid"] is True, r.get("invalid_reason")


# ------------------------------------------------------- all other engines: no chain yet
def _other_engine_params():
    params = []
    for eng in fm.ENGINE_IDS:
        if eng in CHAIN_ENGINES:
            continue
        params.append(
            pytest.param(
                eng,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=f"{eng}: no family e2e chain (the wrapper pin and stub run only for "
                    f"{', '.join(CHAIN_ENGINES)}); census support for these cells is not confirmed",
                ),
            )
        )
    return params


@pytest.mark.parametrize("engine", _other_engine_params())
def test_other_engines_run_the_family_chain(engine, tmp_path):
    # Strict xfail: no family chain is built for this engine, so the assertion fails by design.
    assert engine in CHAIN_ENGINES, f"no family chain for {engine}"
    _run_chain(engine, tmp_path)


# ------------------------------------------------ images on a text-only engine (tf-1.2.1)
@pytest.mark.xfail(
    strict=True,
    reason="tf-1.2.1: image input is not confirmed in the census; the stub engine rejects image_url "
    "parts with 400 as a text-only server does, so the image cell cannot be valid",
)
def test_images_on_tf_1_2_1_are_valid(tmp_path):
    srv, port, _ = stub.start("tf-1.2.1", "stub-model", tmp_path / "e.log")
    try:
        s = types.SimpleNamespace(url=f"http://127.0.0.1:{port}", model="stub-model", log=tmp_path / "e.log")
        body = tb.family_apply(tb.req("stub-model", "hello", 8), "code", 0, "off", "img1")
        r = tb.send(s.url, body)  # raises HTTP 400 from the text-only stub
        assert not tb.family_problems("code", r, "off", "img1")
    finally:
        srv.shutdown()


# ------------------------------------------------ negative cases: a bad reply is flagged
@pytest.fixture
def family_server(tmp_path):
    servers = []

    def make(faults):
        srv, port, _ = stub.start("yunshu-main", "stub-model", tmp_path / f"n{len(servers)}.log", faults=faults)
        servers.append(srv)
        return types.SimpleNamespace(
            url=f"http://127.0.0.1:{port}", model="stub-model", log=tmp_path / "n.log", engine="yunshu-main"
        )

    yield make
    for srv in servers:
        srv.shutdown()


def _part_decode_rows(s, tmp_path, kind, *, thinking="off", media="none", ctx=1024):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / f"{kind}-{thinking}-{media}.jsonl"
    argv = [
        "--engine", "yunshu-main", "--part", "decode", "--rep", "0", "--out", str(out),
        "--only-ctx", str(ctx), "--only-kind", kind, "--decode-tokens", "64", "--phases", "cold",
        "--thinking", thinking, "--media", media,
    ]
    a = tb.parse_args(argv)
    with out.open("w") as fh:
        tb.part_decode(s, fh, a)
    return [json.loads(x) for x in out.read_text().splitlines()]


@pytest.fixture
def no_gpu_contention(monkeypatch):
    monkeypatch.setattr(tb, "was_contended", lambda: False)


def test_invalid_tool_json_flags_the_row(family_server, tmp_path, no_gpu_contention):
    rows = _part_decode_rows(family_server({"bad_tool_json"}), tmp_path, "tool-chat")
    assert rows and rows[0]["valid"] is False
    assert "arguments are not JSON" in rows[0]["invalid_reason"]


def test_nonconformant_json_output_flags_the_row(family_server, tmp_path, no_gpu_contention):
    cut = _part_decode_rows(family_server({"bad_json"}), tmp_path / "cut", "json-schema")
    assert cut[0]["valid"] is False and "not JSON" in cut[0]["invalid_reason"]
    wrong = _part_decode_rows(family_server({"bad_schema"}), tmp_path / "type", "json-schema")
    assert wrong[0]["valid"] is False and "schema" in wrong[0]["invalid_reason"]


def test_missing_reasoning_split_flags_the_row(family_server, tmp_path, no_gpu_contention):
    rows = _part_decode_rows(family_server({"no_reasoning"}), tmp_path, "code", thinking="on")
    assert rows[0]["valid"] is False
    assert "no reasoning split" in rows[0]["invalid_reason"]


def test_a_flagged_row_fails_the_audit(chain, tmp_path):
    """Mutation on the good chain output: one tool-chat row flagged invalid -> audit fails closed."""
    if chain["rows"][0]["engine"] != "yunshu":
        pytest.skip("mutation runs once, on the yunshu chain")
    root = tmp_path / "raw"
    shutil.copytree(chain["raw"], root)
    row = _rows_of(chain, "tool-chat")[0]
    path = root / _cell_path(chain, row).relative_to(chain["raw"])
    lines = path.read_text().splitlines()
    for i in range(1, len(lines)):
        rec = json.loads(lines[i])
        if rec.get("part") == "decode":
            rec["valid"] = False
            rec["invalid_reason"] = "mutated: arguments are not JSON"
        lines[i] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    manifest = fr.audit_manifest(chain["rows"], [fm.SMOKE_BATCH], chain["pin"], chain["plan"].engine_order)
    manifest["stub_engine"] = True
    mpath = tmp_path / "audit-manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    res = subprocess.run(
        [fr.HARNESS_PY, str(SCRIPTS / "fullcmp_audit.py"), "--root", str(root), "--manifest", str(mpath)],
        capture_output=True, text=True, timeout=600, cwd=str(ROOT),
    )
    assert res.returncode != 0
    assert "family reply invalid" in res.stdout
