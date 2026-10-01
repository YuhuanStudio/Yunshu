#!/usr/bin/env python3
"""Type-check gate: mypy on python/ must not produce errors beyond the baseline.

The baseline (scripts/dev/mypy_baseline.txt) is the existing type debt, keyed by
(file, error code, message) without line numbers so unrelated edits do not
churn it. Counts per key may only go down. New keys, or more occurrences of an
old key, fail the gate. Use ``--update`` to rewrite the baseline after paying
debt down (never to absorb new errors).

Run through ``uv run`` so local ``just lint`` and CI use the same mypy.
"""

from __future__ import annotations

import collections
import pathlib
import re
import subprocess
import sys

BASELINE = pathlib.Path(__file__).with_name("mypy_baseline.txt")
_LINE = re.compile(
    r"^(?P<file>[^:]+):\d+(?::\d+)?: error: (?P<msg>.*?)(?:  \[(?P<code>[\w-]+)\])?$"
)


# Messages can cite other lines ("already defined on line 10158"); those shift with
# unrelated edits just like the error's own line, so they are not part of the key.
_LINE_REF = re.compile(r"\bline \d+")


def _key(file: str, code: str | None, msg: str) -> str:
    return f"{file}\t{code or '-'}\t{_LINE_REF.sub('line N', msg)}"


def collect(output: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for line in output.splitlines():
        m = _LINE.match(line)
        if m:
            counts[_key(m["file"], m["code"], m["msg"])] += 1
    return counts


def load_baseline() -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    if BASELINE.exists():
        for line in BASELINE.read_text().splitlines():
            n, _, key = line.partition("\t")
            if n.isdigit():
                counts[_key(*key.split("\t", 2))] += int(n)
    return counts


def main(argv: list[str]) -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "python/"], capture_output=True, text=True
    )
    if "error:" not in proc.stdout and proc.returncode not in (0, 1):
        sys.stderr.write(proc.stdout + proc.stderr)
        return 2
    current = collect(proc.stdout)
    if "--update" in argv:
        BASELINE.write_text("".join(f"{n}\t{k}\n" for k, n in sorted(current.items())))
        print(f"baseline updated: {sum(current.values())} errors")
        return 0
    base = load_baseline()
    new = {k: n - base.get(k, 0) for k, n in current.items() if n > base.get(k, 0)}
    fixed = sum(max(0, n - current.get(k, 0)) for k, n in base.items())
    if new:
        print(f"mypy: {sum(new.values())} NEW error(s) beyond the baseline:")
        for k, n in sorted(new.items()):
            f, code, msg = k.split("\t", 2)
            print(f"  {f}: {msg} [{code}] (+{n})")
        return 1
    print(
        f"mypy: no new errors ({sum(current.values())} baseline debt, "
        f"{fixed} fewer than baseline; run --update to ratchet)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
