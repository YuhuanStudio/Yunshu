"""Causal I1 sibling allocation on recorded MTP rounds (CPU only).

Only the first rejected position's target token is valid for a new sibling.
Later target rows followed a rejected draft and cannot label a branch tail.
Each matching sibling therefore gets at most ONE additional committed token;
this script neither invents continuation acceptance nor claims serving speed.
"""

import argparse
import json
import math
from pathlib import Path


def allocate(row, spare):
    options = []
    reach = 1.0
    for position, (ids, logp) in enumerate(zip(row["alts"], row["lp"], strict=True)):
        chosen = row["drafted"][position]
        for token, score in zip(ids, logp, strict=True):
            if token != chosen and reach > 0:
                options.append((reach * math.exp(score), position, token))
        # The trace's full-head top-1 sometimes differs from the fast draft
        # readout. Exclude the ACTUAL chosen token; do not spend a spare row
        # duplicating it or price reach using another token's probability.
        reach *= math.exp(logp[ids.index(chosen)]) if chosen in ids else 0.0
    return sorted(options, reverse=True)[:spare]


def replay(records, spare):
    committed = sum(row["acc"] + 1 for row in records)
    eligible = hits = 0
    predicted = 0.0
    for index, row in enumerate(records):
        if row["copy"] or "alts" not in row:
            continue
        # Exclude short end windows and each request's final logged round.
        # The log lacks a remaining-budget/stop field, so extra final tokens
        # are not established even if a top-2 target happens to match.
        if row["bs"] < 6 or index == len(records) - 1:
            continue
        last = row["tgt"][row["acc"]]
        if records[index + 1]["b"] != last:
            continue
        selected = allocate(row, spare)
        predicted += sum(score for score, _, _ in selected)
        eligible += 1
        rejected = row["acc"]
        if rejected >= len(row["drafted"]):
            continue
        hits += any(pos == rejected and token == last for _, pos, token in selected)
    return {
        "rounds": len(records),
        "eligible_model_rounds": eligible,
        "spare_rows": spare,
        "committed_proxy": committed,
        "extra_committed_lower_bound": hits,
        "token_gain_pct": 100 * hits / committed,
        "break_even_cost_pct": 100 * hits / committed,
        "head_predicted_extra": predicted,
        "top1_differs_from_draft": sum(
            token != row["alts"][i][0]
            for row in records
            if "alts" in row
            for i, token in enumerate(row["drafted"])
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", type=Path, nargs="+")
    ap.add_argument("--spare", type=int, nargs="+", default=[2, 4, 6, 9, 15])
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as out:
        for path in a.traces:
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for spare in a.spare:
                row = dict(trace=str(path), **replay(records, spare))
                out.write(json.dumps(row) + "\n")
                print(json.dumps(row))
        out.write(
            json.dumps({"complete": True, "measurement": "CPU acceptance bound"}) + "\n"
        )


if __name__ == "__main__":
    main()
