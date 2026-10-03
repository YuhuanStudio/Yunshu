"""Fail-closed summary of complete, quiet full-shape DFlash comparisons."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(records, receipt, *, baseline="main"):
    if receipt.get("rc") != 0 or receipt.get("state") != "done":
        raise ValueError("job must finish with rc 0")
    if not receipt.get("quiet") or receipt.get("contended"):
        raise ValueError("quiet clean job required for timing")
    if (
        not records
        or records[-1].get("complete") is not True
        or records[-1].get("success") is not True
    ):
        raise ValueError("final successful complete record required")
    snapshots = [r for r in records if r.get("part") == "snapshot"]
    if len(snapshots) != 1 or snapshots[0].get("mode") != "dflash":
        raise ValueError("one engaged DFlash snapshot required")
    snapshot = snapshots[0]
    if snapshot.get("barrier") or snapshot.get("capture_proposals"):
        raise ValueError("barrier diagnostics are not production timing")
    for key in ("arms", "contexts", "tasks", "reps", "python_sha256", "harness_sha256"):
        if not snapshot.get(key):
            raise ValueError(f"missing snapshot {key}")
    if snapshot["reps"] < 3:
        raise ValueError("at least three reps required")
    refs = {}
    groups = defaultdict(list)
    seen = set()
    for row in records:
        if row.get("part") == "reference":
            key = (row["context"], row["task"])
            if key in refs:
                raise ValueError("duplicate reference")
            refs[key] = row["sha"]
        if row.get("part") != "result":
            continue
        key = (row["context"], row["task"], row["arm"], row["rep"])
        if key in seen:
            raise ValueError("duplicate cell")
        seen.add(key)
        if row.get("parity") is not True or row["sha"] != refs.get(key[:2]):
            raise ValueError("serial/spec digest mismatch or missing reference")
        for metric in ("tps", "round_ms", "commits_per_round"):
            if not math.isfinite(row[metric]) or row[metric] <= 0:
                raise ValueError(f"invalid {metric}")
        if sum(r["committed"] for r in row["rounds"]) != row["tokens"] - 1:
            raise ValueError("round trace does not cover emitted budget")
        groups[key[:3]].append(row)
    expected = {
        (ctx, task, arm, rep)
        for ctx in snapshot["contexts"]
        for task in snapshot["tasks"]
        for arm in snapshot["arms"]
        for rep in range(snapshot["reps"])
    }
    if seen != expected:
        raise ValueError("incomplete or unexpected matrix")
    result = []
    for (ctx, task, arm), rows in sorted(groups.items()):
        base = groups.get((ctx, task, baseline))
        if not base:
            raise ValueError(f"{baseline} baseline required")
        tps = statistics.median(r["tps"] for r in rows)
        base_tps = statistics.median(r["tps"] for r in base)
        result.append(
            dict(
                context=ctx,
                task=task,
                arm=arm,
                tps=tps,
                gain_pct=100 * (tps / base_tps - 1),
                round_ms=statistics.median(r["round_ms"] for r in rows),
                commits_per_round=statistics.median(
                    r["commits_per_round"] for r in rows
                ),
                reps=len(rows),
                parity=True,
            )
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("--baseline", default="main")
    a = p.parse_args()
    print(
        json.dumps(
            summarize(
                [json.loads(s) for s in a.results.read_text().splitlines()],
                json.loads(a.receipt.read_text()),
                baseline=a.baseline,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
