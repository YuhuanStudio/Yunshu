"""Fail-closed CPU checks for publishing fresh comparison cells."""

import json

from scripts.research import bench016_report as r


def test_three_distinct_reps_required_and_range_is_unrounded():
    sample = dict.fromkeys(r.METRICS, 1.001)
    assert r.summarize({0: sample, 1: sample})["status"] == "incomplete"
    result = r.summarize({0: sample, 1: dict.fromkeys(r.METRICS, 1.003), 2: sample})
    assert result["metrics"]["decode_tps"] == dict(median=1.001, min=1.001, max=1.003)


def test_receipt_is_rechecked_and_first_valid_attempt_wins(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    state = {}
    for attempt, contended in [(1, True), (2, False), (3, False)]:
        jid = str(attempt)
        (jobs / f"{jid}.json").write_text(
            json.dumps(dict(rc=0, contended=contended, state="done"))
        )
        state[f"engine-L32768-r0-try{attempt}"] = dict(id=jid, out=jid, errors=[])
    monkeypatch.setattr(r.b.bs, "read_rows", lambda path: [int(str(path))])
    monkeypatch.setattr(
        r.b,
        "verdict",
        lambda job, rows, receipt: ["contended"] if receipt["contended"] else [],
    )
    monkeypatch.setattr(r, "metrics", lambda rows: dict.fromkeys(r.METRICS, rows[0]))
    result = r.collect(dict(order=["engine"], yunshu_sha="abc"), state, tmp_path)
    assert not result["complete"]
    assert result["cells"][0]["reps"] == [0]
    assert list(result["rejected"]) == ["engine-L32768-r0-try1"]
    assert "incomplete" in r.markdown(result)
