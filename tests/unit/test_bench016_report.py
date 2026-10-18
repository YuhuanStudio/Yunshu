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


def test_memory_rejection_retries_without_accepting_or_blocking_engine():
    from scripts.research.bench016 import retryable_measurement

    log = "\n".join(
        [
            '{"memory_window":{"errors":["pageouts changed during capture"]}}',
            "gpuq: failed rc=2",
        ]
    )
    assert retryable_measurement({"rc": 2, "contended": False}, log)
    assert retryable_measurement({"rc": 0, "contended": True}, "")
    assert not retryable_measurement({"rc": 1, "contended": False}, log)
    assert not retryable_measurement({"rc": 2, "contended": False}, "")
    assert not retryable_measurement(
        {"rc": 2, "contended": False}, '{"memory_window":{"errors":[]}}'
    )
    assert not retryable_measurement(
        {"rc": 2, "contended": False}, '{"memory_window":null}'
    )


def test_unreported_cache_counter_is_not_a_fabricated_zero():
    rows = [
        dict(part="decode", phase=p, ttft_s=1, dec_tps=2, cached=None)
        for p in ("cold", "warm", "turn2")
    ]
    rows.append(dict(part="memory", peak_gib=3))
    value = r.metrics(rows)
    assert value["cached_tokens"] is None
    result = r.summarize(dict.fromkeys(range(3), value))
    assert result["status"] == "qualified"
    assert result["metrics"]["cached_tokens"] is None
    assert result["metrics"]["decode_tps"]["median"] == 2
    assert "not reported" in r.markdown(
        dict(cells=[dict(engine="e", depth=32768, **result)])
    )
