#!/usr/bin/env python3
"""CPU-only annotation cache A/B using real three-client schemas and token masks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import time
from pathlib import Path

from bench_grammar_compile import captured_schemas
from yunshu_engine import grammar_compile as gc
from yunshu_engine.grammar_constraint import (
    LlgJsonSchemaConstraint,
    validate_llg_json_schema,
)


def benchmark(cases, tok, repeats):
    import llguidance.numpy as lnp
    import numpy as np

    strip = gc.annotation_free_source
    rows = []
    try:
        for case in cases:
            original = case["schema"]
            variant = copy.deepcopy(original)
            variant["description"] = (
                "Independent documentation for the same tool arguments."
            )
            before, after = [], []
            positions, equal = 0, True
            for rep in range(repeats):
                pair = []
                for optimized, times in [(False, before), (True, after)]:
                    gc.ARTIFACTS.clear()
                    gc.annotation_free_source = (
                        strip if optimized else lambda source: source
                    )
                    validate_llg_json_schema(original)
                    LlgJsonSchemaConstraint(original, tok)
                    start = time.perf_counter_ns()
                    validate_llg_json_schema(variant)
                    constraint = LlgJsonSchemaConstraint(variant, tok)
                    times.append((time.perf_counter_ns() - start) / 1e6)
                    pair.append(constraint)
                if rep < 3:
                    rng = random.Random(rep)
                    for _ in range(32):
                        masks = []
                        for constraint in pair:
                            mask = np.zeros((1, constraint._words), dtype=np.int32)
                            lnp.fill_next_token_bitmask(constraint._matcher, mask, 0)
                            masks.append(mask)
                        same = np.array_equal(*masks)
                        equal &= same
                        positions += 1
                        if not same or pair[0]._matcher.is_stopped():
                            break
                        allowed = np.flatnonzero(
                            np.unpackbits(masks[0].view(np.uint8), bitorder="little")[
                                : pair[0]._llt.vocab_size
                            ]
                        ).tolist()
                        if not allowed:
                            break
                        tid = rng.choice(allowed)
                        if pair[0]._matcher.consume_token(tid) != pair[
                            1
                        ]._matcher.consume_token(tid):
                            equal = False
                            break
            rows.append(
                {
                    "client": case["client"],
                    "tool": case["tool"],
                    "annotation_variant_ms_before": statistics.median(before),
                    "annotation_variant_ms_after": statistics.median(after),
                    "mask_positions": positions,
                    "mask_equal": equal,
                }
            )
    finally:
        gc.annotation_free_source = strip
        gc.ARTIFACTS.clear()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    if args.repeats < 3:
        ap.error("at least three interleaved repetitions required")
    from transformers import AutoTokenizer
    from yunshu_engine.tool_call_grammar import llg_tokenizer

    cases, corpus = captured_schemas(args.captures, 6)
    if len(cases) != 18:
        raise ValueError("six real schemas from each of the three clients required")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    llg_tokenizer(tok, len(tok))
    load_start = os.getloadavg()
    rows = benchmark(cases, tok, args.repeats)
    result = {
        "cpu_only": True,
        "complete": all(row["mask_equal"] for row in rows),
        "corpus": corpus,
        "rows": rows,
        "load_start": load_start,
        "load_end": os.getloadavg(),
        "median_before_ms": statistics.median(
            row["annotation_variant_ms_before"] for row in rows
        ),
        "median_after_ms": statistics.median(
            row["annotation_variant_ms_after"] for row in rows
        ),
        "mask_positions": sum(row["mask_positions"] for row in rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "corpus")}))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
