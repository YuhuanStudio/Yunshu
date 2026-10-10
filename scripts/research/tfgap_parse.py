"""Split a TensorFold round log into requests (position resets) and summarise rounds/keep/ms; pair with jsonl phases."""

import json, sys, statistics as st, collections, ast, re

D = "/Volumes/P5Plus/yunshu-build/decode16/tf-instrumented/run"


def reqs(path):
    out, cur, last = [], [], None
    for l in open(path):
        p = l.split()
        if len(p) != 5:
            continue
        pos, kind, rows, keep, ms = int(p[0]), p[1], int(p[2]), int(p[3]), float(p[4])
        if last is not None and pos < last:
            out.append(cur)
            cur = []
        cur.append((pos, kind, rows, keep, ms))
        last = pos
    if cur:
        out.append(cur)
    return [r for r in out if len(r) > 5]


def phases(path):
    return [d for d in map(json.loads, open(path)) if d.get("part") == "decode"]


for cell in sys.argv[1:]:
    tfp = phases(f"{D}/tf-new-{cell}.jsonl")
    rs = reqs(f"{D}/tf-new-{cell}.rounds")
    print(f"== {cell}: TF requests in log={len(rs)} phases={[p['phase'] for p in tfp]}")
    for ph, r in zip(tfp, rs):
        kinds = collections.Counter(k for _, k, *_ in r)
        keep = sum(x[3] for x in r)
        ms = sum(x[4] for x in r)
        print(
            f"  TF {ph['phase']:7} tps={ph['dec_tps']:6.1f} rounds={len(r):4} commit/rd={keep / len(r):5.2f} ms/rd={ms / len(r):6.2f} "
            f"med_ms={st.median(x[4] for x in r):6.2f} rows/rd={st.mean(x[2] for x in r):5.1f} sum_ms={ms / 1e3:6.2f}s wall_dec={ph['total_s'] - ph['ttft_s']:6.2f}s kinds={dict(kinds)}"
        )
    try:
        for ph in phases(f"{D}/yunshu-{cell}.jsonl"):
            xy = str(ph.get("xy") or "")
            m = {
                k: re.search(rf"'{k}': ([0-9.]+)", xy)
                for k in ("drafted", "accepted", "rounds")
            }
            m = {k: float(v.group(1)) if v else None for k, v in m.items()}
            dec = ph["total_s"] - ph["ttft_s"]
            rd = m["rounds"] or 0
            print(
                f"  YS {ph['phase']:7} tps={ph['dec_tps']:6.1f} rounds={rd:6.0f} commit/rd={ph['ct'] / rd if rd else 0:5.2f} ms/rd={dec * 1e3 / rd if rd else 0:6.2f} drafted/rd={(m['drafted'] or 0) / rd if rd else 0:5.1f} xy={xy[:160]}"
            )
    except FileNotFoundError:
        print("  YS missing")
