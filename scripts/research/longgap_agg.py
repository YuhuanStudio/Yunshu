"""Aggregate longgap tfbench jsonl (arm vs TF, long cells). Fails closed on incomplete files.

  longgap_agg.py DIR [BASE_ARM]
Files are <arm>-<ctx><kind>-r<rep>.jsonl. Per (arm, ctx, kind): median cold/warm/follow-up TTFT and decode
tok/s over reps. GAP rows: decode = arm tok/s / TF tok/s; TTFT = TF / arm (>100% means the arm is faster).
"""

import glob
import json
from pathlib import Path
import re
import statistics as st
import sys
from collections import defaultdict


def label(path):
    name = path.rsplit("/", 1)[-1]
    return re.sub(r"-\d+(code|prose)-r\d+\.jsonl$", "", name)


def load(paths):
    rows, bad = defaultdict(list), []
    for f in paths:
        recs = [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]
        if not any(r["part"] == "part_done" and r.get("complete") for r in recs):
            bad.append(f)
            continue
        for r in recs:
            if r["part"] == "decode":
                rows[(label(f), r["ctx"], r["kind"])].append(r)
    return rows, bad


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 3) if xs else None


def cell(rs):
    def g(ph, k):
        return med([r[k] for r in rs if r["phase"] == ph])

    return dict(
        cold=g("cold", "ttft_s"),
        warm=g("warm", "ttft_s"),
        fu=g("turn2", "ttft_s"),
        dec=med([r["dec_tps"] for r in rs if r["phase"] in ("cold", "warm")]),
        n=len([r for r in rs if r["phase"] == "cold"]),
        shas=sorted({r["sha"] for r in rs if r["phase"] in ("cold", "warm")}),
    )


def pct(a, b):
    return "n/a" if not a or not b else f"{100 * a / b:.0f}%"


def main(argv):
    rows, bad = load(sorted(glob.glob(argv[0] + "/*.jsonl")))
    if bad:
        print("INCOMPLETE:", bad)
    print("cell | arm | n | cold s | warm s | followup s | decode tok/s | digests")
    for c, k in sorted({(c, k) for _, c, k in rows}):
        out = {}
        for e in sorted({e for e, cc, kk in rows if (cc, kk) == (c, k)}):
            v = out[e] = cell(rows[(e, c, k)])
            print(
                f"{c // 1024}K {k} | {e} | {v['n']} | {v['cold']} | {v['warm']} | {v['fu']} | {v['dec']} | {len(v['shas'])}"
            )
        t = out.get("tf-new")
        for e, y in out.items():
            if t and e != "tf-new":
                print(
                    f"  GAP {e} {c // 1024}K {k}: decode {pct(y['dec'], t['dec'])}, cold {pct(t['cold'], y['cold'])}, "
                    f"warm {pct(t['warm'], y['warm'])}, followup {pct(t['fu'], y['fu'])}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
