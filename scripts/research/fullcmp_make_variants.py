#!/usr/bin/env python3
"""Smoke-only prompt variants: copy-and-vary of the real corpus prompts (fullcmp smoke manifest).

For every real prompt <kind>-<ctx>.txt (kinds code/prose, ctx 1K and 32K: the 32K file is the
concurrent-request source) this writes <kind>-<ctx>-v1.txt into the smoke prompt dir. A variant is:
  1. identifier renaming (code only): ~half of the identifiers that occur at least twice get a seeded
     suffix, consistently across the file;
  2. a distinct leading task sentence;
  3. a seeded cut of the body from the front, at a word boundary, so the token count stays close to
     the original (the search keeps the count within 1% of the original).

Token check: the model tokenizer (tokenizer.json next to the checkpoint) when it is readable; the
variant then must be within 1% in tokens, else the run fails. Without it, the check is by characters
within 1% and every file is marked "unverified" in the manifest. Nothing here reads or writes the real
corpus. A sha256 manifest (manifest.json) lists every written file.

Usage (CPU only, no model load):
  uv run python scripts/research/fullcmp_make_variants.py
  uv run python scripts/research/fullcmp_make_variants.py --src DIR --out DIR --tokenizer FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import keyword
import os
import random
import re
import sys
from pathlib import Path

DEFAULT_SRC = (
    Path(os.environ.get("TFB_WORK", "/Volumes/P5Plus/yunshu-build/tfnew")) / "prompts"
)
DEFAULT_OUT = Path("/Volumes/Micron/yunshu-fullcmp/smoke-prompts")
DEFAULT_TOKENIZER = Path(
    "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp/tokenizer.json"
)
KINDS = ("code", "prose")
CTXS = ("1024", "32768")
VARIANTS = (1,)
TOL = 0.01
BASE_SEED = 20261103
WS = re.compile(r"\s")
IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
STOP = set(keyword.kwlist) | {
    "self",
    "None",
    "True",
    "False",
    "print",
    "return",
    "import",
    "class",
}


def task_sentence(variant: int, kind: str) -> str:
    what = "code" if kind == "code" else "prose"
    return (
        f"Variant {variant} task: read the {what} excerpt below, state its main purpose in one "
        "sentence, then continue the analysis in your own words.\n\n"
    )


def rename_identifiers(text: str, rng: random.Random, variant: int) -> str:
    """Seeded consistent renaming of about half of the repeated identifiers (code)."""
    counts: dict[str, int] = {}
    for m in IDENT.finditer(text):
        counts[m.group(0)] = counts.get(m.group(0), 0) + 1
    mapping = {}
    for name in sorted(counts):  # sorted: the seeded draw order is stable across runs
        if counts[name] >= 2 and name not in STOP and rng.random() < 0.5:
            mapping[name] = f"{name}_v{variant}{rng.randrange(16**3):03x}"
    if not mapping:
        return text
    return IDENT.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


def _word_cut(text: str, k: int) -> str:
    """text[k:] starting after the first whitespace at or after k (k=0 keeps the text)."""
    if k <= 0:
        return text
    m = WS.search(text, k)
    return text[m.end() :] if m else ""


def make_variant(
    src_text: str, kind: str, variant: int, rng: random.Random, count
) -> str:
    """Copy-and-vary. count(s) is the length measure (tokens or characters)."""
    body = rename_identifiers(src_text, rng, variant) if kind == "code" else src_text
    head = task_sentence(variant, kind)
    target = count(src_text)
    # Largest cut k in [0, len] whose variant is still at or above the target (counts grow as k falls).
    lo, hi = 0, len(body)
    best = _word_cut(body, 0)
    for _ in range(40):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        cand = head + _word_cut(body, mid)
        if count(cand) > target:
            lo = mid + 1
        else:
            hi = mid - 1
            best = _word_cut(body, mid)
    return head + best


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(src: Path, out: Path, tokenizer: Path | None) -> dict:
    tok = None
    if tokenizer is not None and tokenizer.exists():
        from tokenizers import Tokenizer  # noqa: PLC0415 (optional, CPU)

        tok = Tokenizer.from_file(str(tokenizer))

        def count(s: str) -> int:
            return len(tok.encode(s, add_special_tokens=False).ids)

        mode = "tokens"
    else:

        def count(s: str) -> int:
            return len(s)

        mode = "chars"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for kind in KINDS:
        for ctx in CTXS:
            src_path = src / f"{kind}-{ctx}.txt"
            src_text = src_path.read_text()
            for variant in VARIANTS:
                rng = random.Random(f"{BASE_SEED}:{kind}:{ctx}:{variant}")
                text = make_variant(src_text, kind, variant, rng, count)
                orig, got = count(src_text), count(text)
                delta = abs(got - orig) / max(orig, 1)
                if delta > TOL:
                    raise SystemExit(
                        f"{kind}-{ctx}-v{variant}: {mode} {got} vs original {orig} "
                        f"({delta:.2%} > {TOL:.0%}); not written"
                    )
                dst = out / f"{kind}-{ctx}-v{variant}.txt"
                dst.write_text(text)
                files.append(
                    {
                        "file": dst.name,
                        "sha256": sha256(dst),
                        "bytes": dst.stat().st_size,
                        "src": src_path.name,
                        "src_sha256": sha256(src_path),
                        "check": mode if mode == "tokens" else "unverified",
                        "length_src": orig,
                        "length_variant": got,
                        "delta_pct": round(100 * delta, 4),
                    }
                )
    manifest = {
        "check_mode": mode,
        "tokenizer": str(tokenizer) if mode == "tokens" else None,
        "tolerance_pct": 100 * TOL,
        "files": files,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = ap.parse_args(argv)
    m = run(args.src, args.out, args.tokenizer)
    sys.stdout.write(
        f"wrote {len(m['files'])} variant files to {args.out} (check: {m['check_mode']})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
