"""Stand-in for paired_eval.py run: PAIRED_OUT/mmlu_pro/<arm>.jsonl, FAKE_CHUNK items per invocation."""

import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("cmd")
ap.add_argument("--bench")
ap.add_argument("--mmlu-n", type=int)
ap.add_argument("--arm")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--model")
ap.add_argument("--model-name")
ap.add_argument("--budget-min")
ap.add_argument("--concurrency")
ap.add_argument("--port")
a = ap.parse_args()
env = dict(kv.split("=", 1) for kv in a.env)
p = Path(os.environ["PAIRED_OUT"]) / a.bench / f"{a.arm}.jsonl"
p.parent.mkdir(parents=True, exist_ok=True)
done = (
    {json.loads(x)["id"] for x in p.read_text().splitlines() if x.strip()}
    if p.exists()
    else set()
)
chunk = int(os.environ.get("FAKE_CHUNK", "1000"))
worse = int(env.get("FAKE_WORSE", "0"))
n = 0
with p.open("a") as f:
    for i in range(a.mmlu_n):
        qid = f"q{i}"
        if qid in done:
            continue
        if n >= chunk:
            break
        correct = i % 3 != 0 and not (i < worse * 3 and i % 3 == 1)
        f.write(json.dumps({"kind": "q", "id": qid, "correct": correct}) + "\n")
        n += 1
