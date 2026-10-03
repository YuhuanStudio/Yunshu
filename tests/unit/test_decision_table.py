import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "research"))
import decision_table as dt  # noqa: E402


def _arm(path, rows, tps, sha="a", *, complete=True, contended=False):
    recs = [
        {
            "part": "decode",
            "ctx": 1,
            "kind": "k",
            "phase": "warm",
            "dec_tps": tps,
            "sha": sha,
        }
    ]
    if complete:
        recs.append({"complete": True, "mode": "mtp", "contended": contended})
    path.write_text("\n".join(json.dumps(r) for r in recs))


def test_copy_table_flags_digest_changes_and_rejects_bad_arms(tmp_path):
    stem = str(tmp_path / "s")
    _arm(tmp_path / "s-w0-r0.jsonl", 0, 50.0)
    _arm(tmp_path / "s-w16-r0.jsonl", 16, 70.0, sha="b")
    _arm(tmp_path / "s-w8-r0.jsonl", 8, 60.0, complete=False)
    _arm(tmp_path / "s-w12-r0.jsonl", 12, 61.0, contended=True)
    lines, rejected = dt.copy_table([stem])
    assert "16: 70.0 (1) DIGEST DIFFERS" in lines[1]
    assert "8:" not in lines[1] and "12:" not in lines[1]
    assert len(rejected) == 2


def test_requires_a_final_successful_record_and_proved_mode():
    assert dt.verdict([{"complete": True, "mode": "mtp"}, {"partial": 1}], "mtp")
    assert dt.verdict([{"complete": True}], "mtp")
    assert dt.verdict(
        [{"complete": True, "mode": ["Speculative decoding: dflash"]}], "mtp"
    )
    assert (
        dt.verdict(
            [
                {
                    "complete": True,
                    "mode": ["Speculative decoding: mtp (checkpoint MTP head)"],
                }
            ],
            "mtp",
        )
        is None
    )
    assert dt.verdict([{"parity": False}, {"complete": True}], None)


def test_conv_arms_are_never_pooled_into_one_median(tmp_path):
    output = tmp_path / "draft.jsonl"
    rows = [
        dict(
            context=1024,
            task="code",
            mode="dflash",
            bits=8,
            context_fused=False,
            selector=False,
            compiled_conv=compiled,
            tps=rate,
            parity=True,
        )
        for compiled, rate in [(False, 100), (True, 110)]
    ]
    rows.append({"complete": True, "parity": True})
    output.write_text("\n".join(json.dumps(row) for row in rows))
    lines, rejected = dt.draft_table([str(output)])
    assert not rejected
    assert len(lines) == 3
    assert "100.0 (1)" in "\n".join(lines)
    assert "110.0 (1)" in "\n".join(lines)


def test_copy_cost_and_fixed_widths_are_distinct_and_parent_failures_reject(tmp_path):
    stem = str(tmp_path / "s")
    _arm(tmp_path / "s-w16-r0.jsonl", 16, 80.0)
    _arm(tmp_path / "s-w16-r0-cost.jsonl", 16, 85.0)
    lines, rejected = dt.copy_table([stem])
    assert not rejected
    assert "16: 80.0 (1)" in lines[1]
    assert "16-cost: 85.0 (1)" in lines[1]
    Path(stem + ".jsonl").write_text(
        json.dumps({"complete": True, "success": False, "contended": True})
    )
    lines, rejected = dt.copy_table([stem])
    assert len(lines) == 1 and len(rejected) == 1


def test_reclassified_queue_evidence_overrides_only_old_contention():
    job = dict(state="done", rc=0, contended=False, reclassified="new threshold")
    rows = [
        {"tps": 70, "contended": True},
        {"complete": True, "success": False, "contended": True},
    ]
    assert dt.verdict(rows, None, job) is None
    assert dt.verdict(rows, None)
    for bad in [
        dict(job, state="failed", rc=1),
        dict(job, contended=True),
        dict(job, reclassified=None),
        {},
    ]:
        assert dt.verdict(rows, None, bad)
    assert dt.verdict([{"rc": 1}, *rows], None, job)
    assert dt.verdict([{"error": "skipped"}, *rows], None, job)
    assert dt.verdict([dict(rows[-1], reason="admission skipped")], None, job)
    assert dt.verdict([{"complete": True, "success": False}], None, job)
    assert dt.verdict([{"parity": False}, *rows], None, job)
    assert dt.verdict(rows[:-1], None, job)


def test_job_evidence_requires_exact_unambiguous_declared_output(tmp_path):
    job = dict(
        outputs=["/out/x.jsonl"],
        state="done",
        rc=0,
        contended=False,
        reclassified="new threshold",
    )
    (tmp_path / "one.json").write_text(json.dumps(job))
    assert dt.job_evidence(tmp_path) == {"/out/x.jsonl": job}
    (tmp_path / "two.json").write_text(json.dumps(job))
    assert dt.job_evidence(tmp_path) == {"/out/x.jsonl": {}}
