"""Run BASE then candidate concurrency probes in one gpuq reservation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_pair(out: Path, commands: dict) -> None:
    with out.open("w") as combined:
        for arm in ("base", "cand"):
            evidence = out.with_name(f"{out.stem}.{arm}.jsonl")
            argv = [arg.replace("{arm_out}", str(evidence)) for arg in commands[arm]]
            # The probe appends; a leftover file from an earlier attempt at the same path
            # would double the rounds (0,1,2,0,1,2) and fail an otherwise complete run.
            evidence.unlink(missing_ok=True)
            subprocess.run(argv, check=True)
            rows = [
                json.loads(line) for line in evidence.read_text().splitlines() if line
            ]
            trials = [row for row in rows if row.get("part") == "conc32"]
            if not rows or rows[-1].get("complete") is not True:
                raise RuntimeError(f"{arm}: missing final complete record")
            if len(trials) != 3 or [row.get("trial") for row in trials] != [0, 1, 2]:
                raise RuntimeError(f"{arm}: incomplete concurrency rounds")
            if any(row.get("cts") != [1024, 1024] for row in trials):
                raise RuntimeError(f"{arm}: short concurrency output")
            for row in trials:
                combined.write(json.dumps(dict(row, arm=arm)) + "\n")
            combined.flush()
        combined.write(json.dumps({"complete": True}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commands", required=True)
    args = parser.parse_args()
    run_pair(args.out, json.loads(args.commands))


if __name__ == "__main__":
    main()
