"""Shared-head analysis across sessions of the same agent (CPU only).

    cross_session.py --model M --group NAME=GLOB [--group NAME2=GLOB2 ...]

Each GLOB lists one session's request bodies in order. For every session pair of a group:
the longest common prefix of their FIRST requests (what a new session could reuse from an
earlier one: system prompt, tool list) and where they diverge.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_prefix import lcp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--session", action="append", required=True, help="AGENT:LABEL=GLOB"
    )
    a = ap.parse_args()
    logging.disable(logging.CRITICAL)
    from render_bodies import Renderer, load_body

    r = Renderer(a.model)
    tok = r.engine._tokenizer
    sessions: dict[str, list[tuple[str, list[int]]]] = {}
    for spec in a.session:
        key, pattern = spec.split("=", 1)
        agent, label = key.split(":", 1)
        files = [f for f in sorted(glob.glob(pattern)) if Path(f).stat().st_size > 0]
        ids = [r.render(load_body(Path(f)))["ids"] for f in files]
        sessions.setdefault(agent, []).append((label, ids))
    for agent, ss in sessions.items():
        print(f"== {agent}")
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                for ai, ia in enumerate(ss[i][1][:3]):
                    for bi, ib in enumerate(ss[j][1][:3]):
                        n = lcp(ia, ib)
                        if n < 100:
                            continue
                        print(
                            f"  {ss[i][0]}[{ai}] ({len(ia)}) vs {ss[j][0]}[{bi}] ({len(ib)}): "
                            f"common {n} tokens  diverge: {tok.decode(ia[max(0, n - 6) : n + 8])!r} | "
                            f"{tok.decode(ib[max(0, n - 6) : n + 8])!r}"
                        )


if __name__ == "__main__":
    main()
