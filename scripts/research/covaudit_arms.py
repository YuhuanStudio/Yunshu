"""Run one covaudit_session.py command on several source trees in one job (same model, same flags).

    python covaudit_arms.py --arm main=/path/python --arm v013=/path2/python -- run --model M --out DIR/x.jsonl ...

Each arm gets `--src` and an `--out` with the arm name inserted before the suffix. Exit nonzero when
the arms disagree on pass/fail or any arm errors (fail closed); the per-arm verdicts are printed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def arm_out(out: str, name: str) -> str:
    p = Path(out)
    return str(p.with_name(f"{p.stem}-{name}{p.suffix}"))


def verdict(rc: int) -> str:
    return {0: "PASS", 1: "FAIL"}.get(rc, f"ERROR({rc})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    rest = [x for x in a.rest if x != "--"]
    i = rest.index("--out")
    res = {}
    for arm in a.arm:
        name, src = arm.split("=", 1)
        cmd = list(rest)
        cmd[i + 1] = arm_out(rest[i + 1], name)
        res[name] = subprocess.run(
            [sys.executable, str(HERE / "covaudit_session.py"), *cmd, "--src", src]
        ).returncode
    for n, rc in res.items():
        print(f"ARM {n}: {verdict(rc)}")
    same = len({verdict(rc) for rc in res.values()}) == 1
    print("ARMS", "AGREE" if same else "DISAGREE")
    return 0 if same and all(rc in (0, 1) for rc in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
