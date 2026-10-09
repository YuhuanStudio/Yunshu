"""CPU fixtures ensure snapshot failures remain unknown and stop one engine."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

import bench_snapshot as bs  # noqa: E402
from scripts.verify import snapshot, stages, suites  # noqa: E402


def test_snapshot_is_an_explicit_suite_only():
    assert suites.parse_suite("snapshot")["stages"] == ["snapshot"]
    assert "snapshot" not in suites.parse_suite("full")["stages"]
    assert stages.STAGE_FUNCS["snapshot"] is snapshot.stage_snapshot


def test_snapshot_stops_failed_engine_and_records_all_other_evidence(
    monkeypatch, tmp_path
):
    jobs = bs.plan_pilots(["yunshu-new", "mlxlm"], tmp_path, {})
    later = bs.plan_cells(["yunshu-new", "mlxlm"], 1, tmp_path, {})[:2]
    monkeypatch.setattr(bs, "plan_pilots", lambda *a: jobs)
    monkeypatch.setattr(bs, "plan_cells", lambda *a: later)
    monkeypatch.setattr(stages, "_finish", lambda ctx, result: result)
    seen = []

    def run(cells):
        cell = cells[0]
        seen.append(cell.key)
        return {
            cell.key: SimpleNamespace(
                ok=cell.key != "yunshu-new-smoke-pilot",
                reason="fixture failed",
                evidence=tmp_path / cell.key,
            )
        }

    ctx = SimpleNamespace(
        cand=SimpleNamespace(path=Path(__file__).resolve().parents[2], key="candidate"),
        base=SimpleNamespace(path=tmp_path, key="release", commit="release"),
        env={},
        suite={"snapshot_agents": False},
        run=SimpleNamespace(path=tmp_path),
        exe=SimpleNamespace(run_cells=run),
    )
    result = snapshot.stage_snapshot(ctx)
    assert not result.passed
    assert result.numbers["failed_engines"] == ["yunshu-new"]
    assert result.numbers["planned_jobs"] == 4
    assert result.numbers["complete_jobs"] == 2
    assert "yunshu-new-d1k-r0" not in seen
    assert "mlxlm-d1k-r0" in seen


def test_select_jobs_by_cell_names():
    import pytest

    select_jobs = snapshot.select_jobs
    out = Path("/x")
    env = {"SNAPSHOT_CELLS": "splash-d32k-r4,llamacpp-d1k-r1"}
    jobs = select_jobs(bs, list(bs.ENGINE_ORDER), out, {}, env)
    assert [j.name for j in jobs] == ["splash-d32k-r4", "llamacpp-d1k-r1"]
    assert jobs[0].rep == 4 and all(j.stage != "pilot" for j in jobs)
    with pytest.raises(ValueError):
        select_jobs(bs, ["splash"], out, {}, {"SNAPSHOT_CELLS": "llamacpp-d1k-r0"})
    full = select_jobs(bs, ["splash"], out, {}, {})
    assert full[0].stage == "pilot" and len(full) > 20
