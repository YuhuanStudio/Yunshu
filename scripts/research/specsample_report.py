"""Summarize specsample results.jsonl: mode x sampling x ctx -> decode tok/s, acceptance, tokens per round."""

import json
import sys
from collections import defaultdict
from statistics import mean

rows = [json.loads(line) for line in open(sys.argv[1])]
g = defaultdict(list)
for r in rows:
    g[(r["mode"], r["ctx"], r["sampling"])].append(r)
print(
    "| mode | ctx | sampling | n | tok/s (reps) | accept rate | tokens/round | completion |"
)
print("|---|---|---|---|---|---|---|---|")
for (m, c, s), rs in sorted(g.items()):
    tps = [x["decode_tok_s"] for x in rs]
    sp = [x["spec"] for x in rs if x.get("spec")]
    acc = mean(x["acceptance_rate"] for x in sp) if sp else None
    tpr = (
        mean(
            x["completion_tokens"] / x["spec"]["rounds"]
            for x in rs
            if x.get("spec") and x["spec"].get("rounds")
        )
        if sp
        else None
    )
    print(
        f"| {m} | {c} | {s} | {len(rs)} | {mean(tps):.1f} ({', '.join(f'{t:.1f}' for t in tps)}) "
        f"| {'-' if acc is None else f'{acc:.3f}'} | {'-' if tpr is None else f'{tpr:.2f}'} "
        f"| {mean(x['completion_tokens'] for x in rs):.0f} |"
    )
