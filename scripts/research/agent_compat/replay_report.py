"""Compare two census replays (baseline vs current Yunshu) request by request.

python replay_report.py <baseline replay.json> <current replay.json>
"""

from __future__ import annotations

import json
import sys


def load(p):
    with open(p) as f:
        return {(r["session"], r["i"]): r for r in json.load(f)}


def verdict(r):
    if r is None:
        return "-"
    st = r.get("status")
    if st != 200:
        body = (r.get("body") or "")[:110].replace("\n", " ")
        return f"{st} {body}"
    ev = r.get("events") or []
    if ev:
        ok = (ev[0] in ("message_start", "response.created") or ev[0] is None) and any(
            e in ("message_stop", "response.completed", "response.incomplete")
            for e in ev[-3:]
        )
        return "200 stream ok" if ok else f"200 stream ODD {ev[:2]}..{ev[-2:]}"
    return "200"


def main():
    base, cur = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(base) | set(cur))
    print("| session[i] | request | baseline | current |")
    print("|---|---|---|---|")
    for k in keys:
        b, c = base.get(k), cur.get(k)
        r = c or b
        vb, vc = verdict(b), verdict(c)
        mark = "" if vb == vc else " **changed**"
        print(
            f"| {k[0]}[{k[1]}] | {r['method']} {r['path'].split('?')[0]} | {vb} | {vc}{mark} |"
        )


if __name__ == "__main__":
    main()
