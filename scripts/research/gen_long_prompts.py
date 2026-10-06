"""Generate the 65536-token prompts the same way /Volumes/P5Plus/yunshu-build/tfnew/gen_prompts.py
made the 1K/8K/32K ones (same corpus order, same ask, same salt formula, binary-searched to the
token budget). CPU only (tokenizer). Usage: gen_long_prompts.py [ctx ...] (default 65536)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tfbench as t  # noqa: E402

ASK = {
    "prose": "\n\n---\nWrite a long, detailed essay in your own words that explains the main ideas of the material above. Do not use lists. Keep writing at length.",
    "code": "\n\n---\nWrite a thorough code review of the code above: for every module quote the key functions in full, then propose a rewritten version of each. Keep writing at length.",
}


def build(kind, ctx, salt, count):
    files = t.corpus(kind)
    pieces, i, tot = [], salt * 7, 0
    while tot < ctx + 2000:
        f = files[i % len(files)]
        i += 1
        pieces.append(f"\n### {f.name}\n{f.read_text(errors='ignore')}")
        tot += count(pieces[-1])
    full = "".join(pieces)
    lo, hi = 0, len(full)
    while hi - lo > 8:
        mid = (lo + hi) // 2
        if count(full[:mid] + ASK[kind]) <= ctx:
            lo = mid
        else:
            hi = mid
    return full[:lo] + ASK[kind]


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(t.M)

    def count(s):
        return len(tok(s).input_ids)

    for ctx in [int(x) for x in sys.argv[1:]] or [65536]:
        for kind in ("prose", "code"):
            dst = t.WORK / "prompts" / f"{kind}-{ctx}.txt"
            if dst.exists():
                print("exists", dst)
                continue
            s = build(kind, ctx, ctx // 1024 + (3 if kind == "code" else 0), count)
            dst.write_text(s)
            print(kind, ctx, count(s), len(s), flush=True)


if __name__ == "__main__":
    main()
