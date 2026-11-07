"""fullcmp_provenance.resolve_job_id: GPUQ_JOB_ID first; else TFB_JOB_LABEL resolves to exactly one
running job in $GPUQ_DIR/jobs. Fake jobs dir only; no gpuq, no GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import fullcmp_provenance as fp  # noqa: E402


def _job(jobs: Path, name: str, jid: str, label: str, state: str):
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / f"{name}.json").write_text(
        json.dumps({"id": jid, "label": label, "state": state})
    )


@pytest.fixture
def jobs(tmp_path):
    d = tmp_path / "gpuq" / "jobs"
    _job(d, "0001", "1103-1-aa", "fullcmp-B1-yunshu-q27-r0", "done")
    _job(
        d, "0002", "1103-2-bb", "fullcmp-B1-yunshu-q27-r0", "failed"
    )  # same label, not running
    _job(d, "0003", "1103-3-cc", "fullcmp-B1-omlx-q27-r0", "running")
    _job(d, "0004", "1103-4-dd", "fullcmp-B1-yunshu-q27-r1", "running")
    return d


def env(jobs_root, **extra):
    base = {"TFB_RUN_ID": "run1", "TFB_GROUP": "g1", "GPUQ_DIR": str(jobs_root.parent)}
    base.update(extra)
    return base


def test_gpuq_job_id_wins_when_exported(jobs):
    e = env(jobs, GPUQ_JOB_ID="1103-9-zz", TFB_JOB_LABEL="fullcmp-B1-omlx-q27-r0")
    assert fp.resolve_job_id(e) == "1103-9-zz"


def test_label_resolves_the_one_running_job(jobs):
    e = env(jobs, TFB_JOB_LABEL="fullcmp-B1-omlx-q27-r0")
    assert fp.resolve_job_id(e) == "1103-3-cc"


def test_label_matching_only_finished_jobs_fails_closed(jobs):
    e = env(jobs, TFB_JOB_LABEL="fullcmp-B1-yunshu-q27-r0")
    # one "done" and one "failed" job carry it: neither is running, and the running one is not
    # this label, so there is nothing to attribute the cell to
    with pytest.raises(fp.ProvenanceError, match="no running gpuq job labelled"):
        fp.resolve_job_id(e)


def test_two_running_jobs_with_one_label_fail_closed(jobs):
    _job(jobs, "0005", "1103-5-ee", "fullcmp-B1-omlx-q27-r0", "running")
    with pytest.raises(
        fp.ProvenanceError, match="2 running gpuq jobs labelled"
    ) as info:
        fp.resolve_job_id(env(jobs, TFB_JOB_LABEL="fullcmp-B1-omlx-q27-r0"))
    assert "1103-3-cc" in str(info.value) and "1103-5-ee" in str(info.value)


def test_no_label_and_no_job_id_names_gpuq_job_id(jobs):
    with pytest.raises(fp.ProvenanceError, match="GPUQ_JOB_ID"):
        fp.resolve_job_id(env(jobs))


def test_label_without_gpuq_dir_names_the_missing_variable(jobs):
    e = {"TFB_RUN_ID": "r", "TFB_GROUP": "g", "TFB_JOB_LABEL": "x"}
    with pytest.raises(fp.ProvenanceError, match="GPUQ_DIR"):
        fp.resolve_job_id(e)


def test_unreadable_job_files_are_skipped(jobs):
    (jobs / "0006.json").write_text("{not json")
    assert (
        fp.resolve_job_id(env(jobs, TFB_JOB_LABEL="fullcmp-B1-omlx-q27-r0"))
        == "1103-3-cc"
    )


def test_explicit_jobs_dir_is_used(jobs, tmp_path):
    e = {"TFB_RUN_ID": "r", "TFB_GROUP": "g", "TFB_JOB_LABEL": "fullcmp-B1-omlx-q27-r0"}
    assert fp.resolve_job_id(e, jobs_dir=jobs) == "1103-3-cc"


def test_check_required_env_returns_the_resolved_id(jobs):
    e = env(jobs, TFB_JOB_LABEL="fullcmp-B1-omlx-q27-r0")
    assert fp.check_required_env(e) == "1103-3-cc"
    assert fp.check_required_env(dict(e, GPUQ_JOB_ID="1103-9-zz")) == "1103-9-zz"


def test_check_required_env_still_names_missing_run_identifiers(jobs):
    with pytest.raises(fp.ProvenanceError, match="TFB_RUN_ID"):
        fp.check_required_env({"TFB_GROUP": "g", "GPUQ_JOB_ID": "x"})
