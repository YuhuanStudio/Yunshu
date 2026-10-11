"""N-prompt decode corpus for the snapshot: averages out greedy-divergence content.

A single prompt per cell lets one engine's greedy path land on copy-heavy or repetitive text
and read as a large speed gap (the 32K code "gap" to TensorFold was content). Decode tok/s
cells therefore run N distinct, exactly tokenized prompts per (ctx, kind) and report the mean,
the median and the paired per-prompt ratio against a reference engine. TTFT cells stay on the
single immutable prompt: prefill time does not depend on what the model would say.

CPU only (tokenizer files, no MLX).
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

CORPUS_N = 10
CORPUS_CTXS = (1024, 8192, 32768)
KINDS = ("prose", "code")
SALT0, SALT_STEP = 300, 11
REPO = Path("/Users/yuhuan/Documents/YuhuanStudio/Yunshu")
PATTERNS = {
    "prose": [
        "reference/vllm/docs/**/*.md",
        "reference/sglang/docs/**/*.md",
        "reference/llama.cpp/docs/**/*.md",
        "docs/**/*.md",
    ],
    "code": [
        "reference/mlx-lm/mlx_lm/**/*.py",
        "reference/mlx-vlm/mlx_vlm/**/*.py",
        "python/**/*.py",
    ],
}


def variant_name(kind: str, ctx: int, index: int) -> str:
    return f"{kind}-{ctx}-v{index:02d}"


def corpus_files(
    directory: Path, ctx: int, kinds=KINDS, n: int = CORPUS_N
) -> list[Path]:
    return [
        directory / f"{variant_name(k, ctx, i)}.txt" for k in kinds for i in range(n)
    ]


def source_files(kind: str, root: Path = REPO) -> list[Path]:
    import random

    files: list[Path] = []
    for pattern in PATTERNS[kind]:
        files += sorted(root.glob(pattern))
    files = [
        f
        for f in files
        if f.is_file() and 2000 < f.stat().st_size < 200000 and "research" not in str(f)
    ]
    random.Random(11).shuffle(files)
    return files


def build_source(
    kind: str, ctx: int, index: int, count_tokens, root: Path = REPO
) -> str:
    """Concatenated files, starting at a variant-specific offset, long enough for ``ctx`` tokens."""
    files = source_files(kind, root)
    pieces: list[str] = []
    i = (SALT0 + index * SALT_STEP) * 7
    total = 0
    while total < ctx + 2000:
        f = files[i % len(files)]
        i += 1
        pieces.append(f"\n### {f.name}\n{f.read_text(errors='ignore')}")
        total += count_tokens(pieces[-1])
    return "".join(pieces)


def generate(
    source: Path, destination: Path, n: int = CORPUS_N, ctxs=CORPUS_CTXS
) -> dict:
    """Write ``n`` exact-token prompts per (kind, ctx). ``source`` holds the immutable single
    prompts whose instruction suffix (``\\n\\n---\\n`` onward) every variant reuses."""
    from snapshot_prompts import assert_chat_prompt, exact_chat_prompt, token_count
    from tfbench import LONG_ASK

    destination.mkdir(parents=True, exist_ok=True)
    records = []
    seen: set[str] = set()
    for ctx in ctxs:
        for kind in KINDS:
            original = (source / f"{kind}-{ctx}.txt").read_text()
            split = original.rfind("\n\n---\n")
            if split < 0:
                raise ValueError("missing corpus instruction boundary")
            ask = original[split:]
            for index in range(n):
                body = build_source(kind, ctx, index, token_count)
                text = exact_chat_prompt(body, ctx, ask + LONG_ASK)
                count = assert_chat_prompt(text, ctx)
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in seen:
                    raise ValueError(f"duplicate corpus prompt {kind} {ctx} {index}")
                seen.add(digest)
                name = variant_name(kind, ctx, index)
                (destination / f"{name}.txt").write_text(text)
                records.append(
                    {
                        "name": name,
                        "kind": kind,
                        "ctx": ctx,
                        "reference_prompt_tokens": count,
                        "sha256": digest,
                    }
                )
    manifest = {"complete": True, "n": n, "prompts": records}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize(
    per_engine: dict[str, dict[str, float]], reference: str = "yunshu-new"
) -> dict:
    """Per engine: mean, median, n; and the paired ratio against ``reference`` on common prompts
    (mean of per-prompt ratios, plus how many prompts the engine matches or beats it on)."""
    out: dict[str, dict] = {}
    ref = per_engine.get(reference, {})
    for engine, values in per_engine.items():
        tps = list(values.values())
        row = {
            "n": len(tps),
            "mean": statistics.fmean(tps) if tps else None,
            "median": statistics.median(tps) if tps else None,
        }
        common = [p for p in values if p in ref and ref[p] > 0]
        if common and engine != reference:
            row["ratio_vs_ref"] = statistics.fmean(values[p] / ref[p] for p in common)
            row["wins_vs_ref"] = sum(values[p] >= ref[p] for p in common)
            row["paired_n"] = len(common)
        out[engine] = row
    return out


def render(
    groups: dict[tuple[int, str], dict[str, dict[str, float]]],
    engines,
    reference="yunshu-new",
) -> str:
    lines = [
        "",
        f"## Decode corpus (N={CORPUS_N} distinct prompts per cell; mean / median tok/s; ratio is the mean "
        f"per-prompt speed relative to {reference})",
        "",
        "| cell | " + " | ".join(engines) + " |",
        "|---|" + "---|" * len(engines),
    ]
    for (ctx, kind), per_engine in sorted(groups.items()):
        stats = summarize(per_engine, reference)
        cells = []
        for e in engines:
            row = stats.get(e)
            if not row or row["mean"] is None:
                cells.append("n/a")
                continue
            text = f"{row['mean']:.1f} / {row['median']:.1f} (n={row['n']})"
            if "ratio_vs_ref" in row:
                text += f"; x{row['ratio_vs_ref']:.3f} vs ref ({row['wins_vs_ref']}/{row['paired_n']} >=)"
            cells.append(text)
        lines.append(f"| {kind}-{ctx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
