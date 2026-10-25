"""Print per-scenario per-row decode tok/s, TTFT and memory of a conc_probe jsonl, then slice stats of its log."""

import json
import subprocess
import sys

jsonl, log = sys.argv[1], sys.argv[2]
for line in open(jsonl):
    d = json.loads(line)
    if "scenario" not in d:
        continue
    mem = d.get("memory") or {}
    rows = [(r["name"], r["decode_tps"], r["ttft"]) for r in d["requests"]]
    print(
        d["scenario"],
        rows,
        {k: v for k, v in mem.items() if "peak" in k or "active" in k},
    )
stats = __file__.replace("cell_report", "slice_stats")
print(
    subprocess.run([sys.executable, stats, log], capture_output=True, text=True).stdout
)
