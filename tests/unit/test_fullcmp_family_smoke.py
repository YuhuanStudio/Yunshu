"""wire-1108: one smallest real B1/B2 cell per family slot per supporting engine, taken from the real manifest."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fullcmp_manifest", ROOT / "scripts/research/fullcmp_manifest.py"
)
fm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fm)

REAL = fm.REAL_MANIFEST


@pytest.fixture(scope="module")
def real_rows():
    if not Path(REAL).exists():
        pytest.skip("real manifest not present on this host")
    return [json.loads(line) for line in Path(REAL).read_text().splitlines() if line.strip()]


def _key(r):
    return json.dumps(r, sort_keys=True)


def test_every_b1_b2_sweep_is_covered_for_each_engine_that_supports_it(real_rows):
    chosen = fm.family_smoke_rows(real_rows)
    chosen_keys = {(r["engine"], r["id"]) for r in chosen}
    sweeps_by_engine = {}
    for r in real_rows:
        if r["batch"] in ("B1", "B2"):
            sweeps_by_engine.setdefault(r["engine"], set()).add(r["sweep"])
    all_sweeps = {s for ss in sweeps_by_engine.values() for s in ss}
    for eng, sweeps in sweeps_by_engine.items():
        for sweep in sweeps:
            picked = [
                c for c in chosen
                if c["engine"] == eng and c["sweep"] == sweep
            ]
            assert picked, f"{eng} supports {sweep} but no smoke row was selected"
    # every B1/B2 sweep in the real manifest is represented by at least one selected row
    assert {c["sweep"] for c in chosen} == all_sweeps
    assert len(chosen_keys) == len(chosen)


def test_each_engine_gets_each_row_at_most_once(real_rows):
    chosen = fm.family_smoke_rows(real_rows)
    ids = [(r["engine"], r["id"]) for r in chosen]
    assert len(ids) == len(set(ids))


def test_selected_rows_are_the_real_manifest_rows_unchanged(real_rows):
    chosen = fm.family_smoke_rows(real_rows)
    real_by_key = {_key(r) for r in real_rows}
    for r in chosen:
        assert _key(r) in real_by_key, f"row {r.get('id')} is not a real manifest row"


def test_family_smoke_writes_the_selected_rows(tmp_path, real_rows):
    out = tmp_path / "wire_1108_manifest.jsonl"
    assert fm.main(["--family-smoke", "--out", str(out)]) == 0
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert written == fm.family_smoke_rows(real_rows)
