"""Three interleaved quiet HTTP runs per checkpoint arm and context, through gpuq."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--arms", nargs="+", default=["legacy", "sync", "deferred", "reserved", "tf"]
    )
    ap.add_argument("--ctx", nargs="+", type=int, default=[8192, 32768])
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    orders = [a.arms, list(reversed(a.arms)), a.arms[1:] + a.arms[:1]]
    with a.out.open("w") as output:
        for rep, order in enumerate(orders):
            for ctx in a.ctx if rep % 2 == 0 else reversed(a.ctx):
                for arm in order:
                    # gpuq checks foreign CPU; load average additionally rejects
                    # stale saturation from preceding jobs / frontend builds.
                    deadline = time.monotonic() + 600
                    while os.getloadavg()[0] > 3 and time.monotonic() < deadline:
                        print(f"quiet wait: load={os.getloadavg()[0]:.2f}", flush=True)
                        time.sleep(10)
                    result_path = a.out.with_name(
                        f"{a.out.stem}-{ctx}-{arm}-r{rep}.jsonl"
                    )
                    cmd = [
                        sys.executable,
                        str(Path(__file__).with_name("probe_checkpoint_http.py")),
                        "--mode",
                        arm,
                        "--ctx",
                        str(ctx),
                        "--rep",
                        str(rep),
                        "--out",
                        str(result_path),
                    ]
                    print(f"arm start: {ctx} {arm} r{rep}", flush=True)
                    rc = subprocess.call(cmd) if os.getloadavg()[0] <= 3 else 3
                    records = (
                        [
                            json.loads(line)
                            for line in result_path.read_text().splitlines()
                        ]
                        if result_path.exists()
                        else []
                    )
                    complete = bool(records and records[-1].get("phase") == "complete")
                    clean = all(
                        not r.get("contended") and r.get("load_1m", 0) <= 3
                        for r in records
                    )
                    ok = rc == 0 and complete and clean
                    record = dict(
                        ctx=ctx,
                        arm=arm,
                        rep=rep,
                        rc=rc,
                        complete=complete,
                        quiet=clean,
                        success=ok,
                        result=str(result_path),
                    )
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    print(json.dumps(record), flush=True)
                    if not ok:
                        failures.append(record)
        output.write(
            json.dumps(
                dict(phase="complete", success=not failures, failures=len(failures))
            )
            + "\n"
        )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
