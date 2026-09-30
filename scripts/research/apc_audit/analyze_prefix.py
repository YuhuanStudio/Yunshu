"""Per-request prefix-reuse analysis of captured agent sessions (CPU only).

For every request of a session: render the prompt with the server's real router + adapter +
chat template + tokenizer, then compare with every earlier request of the session:

  lcp_prev   longest common token prefix with the previous request
  lcp_best   longest common prefix with ANY earlier request (best possible reuse if a state
             existed at that exact token)
  sim        what the APC checkpoint rules would reuse (final checkpoint at len-1, one
             interval-aligned checkpoint, 2-entry LRU, exact-prefix lookup only)
  gen_ckpt   what a checkpoint taken when the previous request FINISHED generating would
             reuse (prompt + generated tokens, assuming the re-rendered assistant turn is
             token-identical to what was generated)
  actual     cached tokens the server reported (optional, --actual JSON list)

    analyze_prefix.py --model $M --name opencode --bodies 'DIR/*-req.json' [--actual 0,1200,..]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INTERVAL = 2048
ENTRIES = 2


def lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    # chunked compare keeps this fast for 100K-token prompts
    step = 4096
    while i + step <= n and a[i : i + step] == b[i : i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return i


class ApcSim:
    """The exact-checkpoint rules of mlx_vlm's APC for a hybrid (GDN) model."""

    def __init__(self, entries: int = ENTRIES, interval: int = INTERVAL):
        self.entries = entries
        self.interval = interval
        self.lru: OrderedDict[tuple, list[int]] = OrderedDict()

    def lookup(self, ids: list[int]) -> int:
        best, key = 0, None
        for k, stored in self.lru.items():
            L = len(stored)
            if best < L < len(ids) and ids[:L] == stored:
                best, key = L, k
        if key is not None:
            self.lru.move_to_end(key)
        return best

    def store(self, ids: list[int], prefix: int) -> None:
        final = len(ids) - 1
        lengths = {final}
        if self.entries > 1 and self.interval > 0:
            last = ((final - 1) // self.interval) * self.interval
            first = max(self.interval, last - (self.entries - 2) * self.interval)
            for b in range(first, last + 1, self.interval):
                if 16 <= b < final:
                    lengths.add(b)
        for n in sorted(lengths):
            if n <= prefix:  # already covered by the restored state
                continue
            self.lru[tuple(ids[:n])] = ids[:n]
            self.lru.move_to_end(tuple(ids[:n]))
            while len(self.lru) > self.entries:
                self.lru.popitem(last=False)


def analyze(renders: list[list[int]], im_end: int, actual: list[int] | None = None):
    sim = ApcSim()
    rows = []
    for i, ids in enumerate(renders):
        prev = renders[i - 1] if i else []
        lp = lcp(ids, prev) if i else 0
        best_j, best = -1, 0
        for j in range(i):
            v = lcp(ids, renders[j])
            if v > best:
                best_j, best = j, v
        hit = sim.lookup(ids)
        sim.store(ids, hit)
        # Assistant output re-rendered inside this request, right after the previous prompt.
        gen_end = 0
        if i and lp >= len(prev) - 1:
            try:
                k = ids.index(im_end, len(prev))
                gen_end = k  # state after prompt + generated tokens (im_end not fed)
            except ValueError:
                gen_end = 0
        rows.append(
            dict(
                i=i,
                n=len(ids),
                lcp_prev=lp,
                lcp_best=best,
                best_j=best_j,
                sim=hit,
                gen_ckpt=max(gen_end, best),
                gen_len=(gen_end - len(prev)) if gen_end else 0,
                actual=(actual[i] if actual and i < len(actual) else None),
            )
        )
    return rows


def cached_from_response(body_file: Path) -> int:
    """Cached tokens the server reported for a request: max over its response stream."""
    resp = body_file.with_name(body_file.name.replace("-req.json", "-resp.txt"))
    if not resp.exists():
        return 0
    found = re.findall(
        r'"(?:cached_tokens|cache_read_input_tokens)"\s*:\s*(\d+)', resp.read_text()
    )
    return max((int(x) for x in found), default=0)


def divergence_context(tok, a: list[int], b: list[int], at: int, w: int = 12):
    def dec(x):
        return tok.decode(x[max(0, at - w) : at + w])

    return dec(a), dec(b)


def summarize(rows):
    n = sum(r["n"] for r in rows)
    first = rows[0]["n"] if rows else 0
    tot = lambda k: sum((r[k] or 0) for r in rows)  # noqa: E731
    out = dict(
        requests=len(rows),
        prompt_tokens=n,
        first_cold=first,
        ceiling_prompt=tot("lcp_best") / n if n else 0,
        ceiling_with_gen=tot("gen_ckpt") / n if n else 0,
        sim=tot("sim") / n if n else 0,
    )
    if all(r["actual"] is not None for r in rows):
        out["actual"] = tot("actual") / n if n else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--bodies",
        required=True,
        action="append",
        help="glob of request bodies, in order (repeat for several sessions served "
        "one after the other by the same server)",
    )
    ap.add_argument(
        "--actual", help="comma list of the server's cached tokens per request"
    )
    ap.add_argument(
        "--actual-from",
        action="store_true",
        help="read cached tokens from the NNNN-resp.txt next to each body",
    )
    ap.add_argument("--show", action="store_true", help="print divergence contexts")
    ap.add_argument("--json", help="write rows here")
    a = ap.parse_args()
    logging.disable(logging.CRITICAL)
    from render_bodies import Renderer, load_body

    r = Renderer(a.model)
    tok = r.engine._tokenizer
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    files = [f for g in a.bodies for f in sorted(glob.glob(g))]
    files = [f for f in files if Path(f).stat().st_size > 0]
    # error captures (NNNN-error-req.json) carry no response to compare with
    files = [f for f in files if "-error-" not in f]
    renders = [r.render(load_body(Path(f)))["ids"] for f in files]
    actual = [int(x) for x in a.actual.split(",")] if a.actual else None
    if a.actual_from:
        actual = [cached_from_response(Path(f)) for f in files]
    rows = analyze(renders, im_end, actual)
    print(f"== {a.name}  ({len(files)} requests)")
    print("  i      n  lcp_prev  lcp_best(j)   sim  gen_ckpt  gen_len  actual")
    for x in rows:
        print(
            f"{x['i']:3d} {x['n']:6d} {x['lcp_prev']:8d} {x['lcp_best']:9d}({x['best_j']:2d}) "
            f"{x['sim']:6d} {x['gen_ckpt']:9d} {x['gen_len']:8d} {x['actual'] if x['actual'] is not None else '-':>7}"
        )
        if a.show and x["i"]:
            p = renders[x["i"] - 1]
            c = renders[x["i"]]
            if x["lcp_prev"] < min(len(p), len(c)):
                s1, s2 = divergence_context(tok, p, c, x["lcp_prev"])
                print(f"      prev: {s1!r}\n      this: {s2!r}")
    s = summarize(rows)
    print(
        "  ratios: "
        + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()
        )
    )
    if a.json:
        Path(a.json).write_text(json.dumps(dict(name=a.name, rows=rows, summary=s)))


if __name__ == "__main__":
    main()
