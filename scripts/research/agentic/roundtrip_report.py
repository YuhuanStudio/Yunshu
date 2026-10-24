"""Round-trip token match per client from YUNSHU_DEBUG_STREAM_CAPTURE files (CPU only).

    roundtrip_report.py claude=<capture.jsonl>[,<more>] codex=... opencode=...

Each row's ``round_trip`` (yunshu_engine.roundtrip_capture) says how many of the earlier response's generated
tokens the next prompt reproduced in place. Prints, per client: pairs, pairs whose whole generation was
reproduced, and reusable = sum(matched_gen) / sum(prev_gen_len) over matched pairs.
"""

from __future__ import annotations

import json
import sys


def summarize(paths: list[str]) -> dict:
    pairs = whole = matched = gen = rows = 0
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            rows += 1
            rt = json.loads(line).get("round_trip")
            if not rt:
                continue
            pairs += 1
            whole += bool(rt.get("whole_gen_reproduced"))
            matched += rt.get("matched_gen", 0)
            gen += rt.get("prev_gen_len", 0)
    return {
        "rows": rows,
        "pairs": pairs,
        "whole": whole,
        "matched_gen": matched,
        "prev_gen": gen,
        "reusable": (matched / gen) if gen else None,
    }


def main(argv: list[str]) -> int:
    for arg in argv:
        name, _, files = arg.partition("=")
        s = summarize(files.split(","))
        r = "n/a" if s["reusable"] is None else f"{100 * s['reusable']:.1f}%"
        print(
            f"| {name} | {s['pairs']} | {s['whole']} | "
            f"{s['matched_gen']}/{s['prev_gen']} = {r} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
