"""Summarize `slice t= dt=` DEBUG lines of a server log: per (rows, decode_only) step time."""

import re
import statistics
import sys
from collections import defaultdict

PAT = re.compile(
    r"slice t=(\S+) dt=(\S+) spec=(\S+) decode_only=(\S+) rows=(\d+) emitted=(\d+) uncached_waiting=(\d+)"
)


def parse(lines):
    out = defaultdict(list)
    for line in lines:
        m = PAT.search(line)
        if m:
            _, dt, spec, only, rows, emitted, waiting = m.groups()
            out[(int(rows), only == "True", int(waiting) > 0)].append(
                (float(dt), int(emitted))
            )
    return out


def main(path):
    for key, v in sorted(parse(open(path, errors="replace")).items()):
        dts = [d for d, _ in v]
        em = sum(e for _, e in v)
        print(
            f"rows={key[0]} decode_only={key[1]} waiting={key[2]} n={len(v)} "
            f"median_ms={statistics.median(dts) * 1000:.1f} mean_ms={statistics.mean(dts) * 1000:.1f} "
            f"tok_per_slice={em / len(v):.2f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
