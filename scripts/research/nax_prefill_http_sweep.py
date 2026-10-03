"""One gpuq job, balanced cold HTTP sessions, independent arm receipts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arms",
        nargs="+",
        choices=(
            "base",
            "planes",
            "stock",
            "tile128",
            "lane64",
            "lane128",
            "narrow",
            "combo",
            "production",
        ),
        required=True,
    )
    p.add_argument("--contexts", nargs="+", type=int, default=[8192, 32768])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        print(json.dumps(vars(a), default=str))
        return
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with a.out.open("w") as out:
        for ctx in a.contexts:
            for rep in range(a.reps):
                order = a.arms[rep % len(a.arms) :] + a.arms[: rep % len(a.arms)]
                for arm in order:
                    target = a.out.with_name(f"{a.out.stem}-{ctx}-r{rep}-{arm}.jsonl")
                    cmd = [
                        sys.executable,
                        str(Path(__file__).with_name("nax_prefill_http.py")),
                        "--arm",
                        arm,
                        "--ctx",
                        str(ctx),
                        "--rep",
                        str(rep),
                        "--out",
                        str(target),
                    ]
                    rc = subprocess.call(cmd)
                    rows = (
                        [json.loads(line) for line in target.read_text().splitlines()]
                        if target.exists()
                        else []
                    )
                    success = (
                        rc == 0
                        and bool(rows)
                        and rows[-1].get("phase") == "complete"
                        and rows[-1].get("success") is True
                    )
                    row = dict(
                        ctx=ctx,
                        rep=rep,
                        arm=arm,
                        rc=rc,
                        success=success,
                        result=str(target),
                    )
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(json.dumps(row), flush=True)
                    if not success:
                        failures.append(row)
        out.write(
            json.dumps(dict(phase="complete", success=not failures, failures=failures))
            + "\n"
        )
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
