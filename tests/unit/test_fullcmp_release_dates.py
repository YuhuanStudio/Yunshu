"""Every fullcmp engine id resolves a release date from census.json, and the runner refuses otherwise.

Regression for the 2026-11-04 bug: omlx-0.7.3rc1 failed on provenance ("no release date for omlx
0.7.3rc1: set TFB_RELEASE_DATE") after a smoke launch. The census lookup is the same code tfbench's
header uses: fullcmp_provenance.census_key_for (id -> census key) + bench_engines.engine_version
(version) + fullcmp_provenance.release_date (census date, no TFB_RELEASE_DATE).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "research"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_engines  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_provenance as fp  # noqa: E402
from fullcmp_pin_io import stub_pin_io  # noqa: E402

# manifest ids that never run: no release date is needed for them
NOT_MEASURED = {"ollama", "lmstudio", fm.PLACEHOLDER}
MEASURED_IDS = [e for e in fm.ENGINES if e not in NOT_MEASURED]


def _census_or_skip() -> dict:
    if not Path(fp.CENSUS_PATH).is_file():
        pytest.skip(f"census not present on this machine: {fp.CENSUS_PATH}")
    return fp.load_json(fp.CENSUS_PATH)


def test_manifest_engine_ids_are_the_twelve_measured_engines():
    assert MEASURED_IDS == [
        "yunshu-pb2",
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
        "kapitsa-0.2.26",
    ]


@pytest.mark.parametrize("engine", MEASURED_IDS)
def test_engine_id_resolves_release_date_through_tfbench_path(engine):
    census = _census_or_skip()
    harness = bench_engines.tfbench_engine(engine)  # --engine id tfbench receives (yunshu-main)
    version, _sha = bench_engines.engine_version(harness, str(ROOT / "python"))
    assert version != "unknown", f"{engine}: no version"
    date = fp.engine_release_date(census, harness, version, {})  # tfbench's header call
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), f"{engine}: {date!r}"


def test_release_date_source_is_recorded_for_every_engine_version():
    census = _census_or_skip()
    missing = []
    for engine in MEASURED_IDS:
        harness = bench_engines.tfbench_engine(engine)
        version, _sha = bench_engines.engine_version(harness, str(ROOT / "python"))
        key = fp.census_key_for(harness)
        entry = census.get(key) or {}
        if version not in (entry.get("release_date_source") or {}):
            missing.append(f"{engine} ({key}:{version})")
    assert not missing, f"no release_date_source for: {missing}"


def test_missing_release_date_names_the_engine_in_the_gate():
    """Failing before the fix: with omlx's census dates gone, the gate must name omlx-0.7.3rc1."""
    census = copy.deepcopy(_census_or_skip())
    census["omlx"].pop("release_dates", None)
    census["omlx"].pop("latest_rc_date", None)
    pin = {"worktree": "/fake/tree", "python_executable": "/fake/venv/bin/python"}
    model = fp.census_models(census)["qwen3.8-27b-4bit-jundot"]["path"]
    problems = fp.dry_run_problems(["omlx-0.7.3rc1"], census, pin, model, environ={})
    assert any(
        p.startswith("omlx-0.7.3rc1:") and "no release date" in p for p in problems
    ), problems


def test_env_release_date_does_not_stand_in_for_the_census():
    """A cell's job env never carries TFB_RELEASE_DATE, so the gate must not accept one from the
    operator's shell: a census gap is refused even when the shell exports the date."""
    census = copy.deepcopy(_census_or_skip())
    census["omlx"].pop("release_dates", None)
    census["omlx"].pop("latest_rc_date", None)
    pin = {"worktree": "/fake/tree", "python_executable": "/fake/venv/bin/python"}
    model = fp.census_models(census)["qwen3.8-27b-4bit-jundot"]["path"]
    problems = fp.dry_run_problems(
        ["omlx-0.7.3rc1"],
        census,
        pin,
        model,
        environ={"TFB_RELEASE_DATE": "2026-10-27"},
    )
    assert any(p.startswith("omlx-0.7.3rc1:") for p in problems), problems


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "fullcmp_run", SCRIPTS / "fullcmp_run.py"
    )
    run = importlib.util.module_from_spec(spec)
    sys.modules["fullcmp_run"] = run
    spec.loader.exec_module(run)
    return run


def test_runner_dry_run_refuses_an_engine_without_release_date(
    tmp_path, monkeypatch, capsys
):
    from test_fullcmp_run import PIN, row  # the runner test module's cell rows and pin

    stub_pin_io(monkeypatch)
    run = _load_runner()
    census = copy.deepcopy(_census_or_skip())
    census["omlx"].pop("release_dates", None)
    census["omlx"].pop("latest_rc_date", None)
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps(census))
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(PIN))
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(row("omlx-0.7.3rc1", "B1.rc1")) + "\n")

    monkeypatch.setattr(run, "PIN_PATH", pin_path)
    monkeypatch.setattr(run, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(run, "PREFLIGHT_DIR", tmp_path / "preflight")
    monkeypatch.setattr(run.fp, "CENSUS_PATH", census_path)
    monkeypatch.setattr(run, "prompt_corpus_problems", lambda rows: [])
    import fullcmp_engine_tables as fet

    # the engine-table pin check runs its own subprocesses; this test is about the release date
    monkeypatch.setattr(fet, "check_all", lambda ids, ctx=None: [])

    def boom(*_a, **_k):
        raise AssertionError("refused run must not reach gpuq or subprocess")

    monkeypatch.setattr(run, "submit_all", boom)
    monkeypatch.setattr(run.subprocess, "run", boom)

    rc = run.main(["--batch", "B1", "--manifest", str(manifest), "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "refusing: provenance: omlx-0.7.3rc1:" in err
    assert not (tmp_path / "audit").exists()
