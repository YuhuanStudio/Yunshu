"""Summarize hostpath_probe.py outputs: median per (arm, load) across reps.

python scripts/research/hostpath_report.py 'DIR/m-*.jsonl'
"""

import glob
import json
import statistics
import sys
from collections import defaultdict


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def rows_of(paths):
    for p in paths:
        if p.endswith(".mon.jsonl"):
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if "arm" in r:
                    yield p, r


def main(argv):
    paths = sorted(p for a in argv for p in glob.glob(a))
    cells = defaultdict(lambda: defaultdict(list))
    for _p, r in rows_of(paths):
        key = (r["arm"], r["load"])
        k = r.get("kind")
        if k in ("short", "long8k_cold", "long8k_warm"):
            cells[key][k + "_ttft"].append(r["ttft_ms"])
            cells[key][k + "_head"].append(r["head_ms"])
        elif k == "conc":
            cells[key][f"c{r['n']}_agg"].append(r["agg_tps"])
            cells[key][f"c{r['n']}_stream"].extend(r["per_stream_tps"])
    cols = sorted({c for v in cells.values() for c in v})
    print("arm load | " + " | ".join(cols))
    for key in sorted(cells):
        print(*key, "|", " | ".join(str(med(cells[key].get(c, []))) for c in cols))


if __name__ == "__main__":
    main(sys.argv[1:])
