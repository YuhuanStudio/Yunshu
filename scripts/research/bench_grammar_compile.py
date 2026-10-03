#!/usr/bin/env python3
"""CPU-only C7: compile cost for real client tool parameter schemas.

Example: PYTHONPATH=python uv run --no-sync python scripts/research/bench_grammar_compile.py
--captures docs/research/runs --tokenizer /path/to/tokenizer --out /tmp/grammar.json
No model is loaded and no MLX import or inference is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from yunshu_engine.grammar_compile import ARTIFACTS
from yunshu_engine.grammar_constraint import (
    LlgJsonSchemaConstraint,
    validate_llg_json_schema,
)


def captured_schemas(root: Path, limit: int):
    selected = {}
    counts = {client: 0 for client in ("claude", "codex", "opencode")}
    unique = {client: set() for client in counts}
    for path in sorted(root.rglob("requests.jsonl")):
        session = path.parent.name
        client = next(
            (
                c
                for prefix, c in (
                    ("cc_", "claude"),
                    ("cx_", "codex"),
                    ("oc_", "opencode"),
                )
                if session.startswith(prefix)
            ),
            None,
        )
        if client is None:
            continue
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            body = rec.get("body") or {}
            for tool in body.get("tools") or []:
                tool = tool.get("function", tool)
                schema = tool.get("input_schema", tool.get("parameters"))
                if not isinstance(schema, dict):
                    continue
                key = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
                counts[client] += 1
                unique[client].add(key)
                if (client, key) not in selected and sum(
                    c == client for c, _ in selected
                ) < limit:
                    selected[client, key] = {
                        "client": client,
                        "tool": tool.get("name"),
                        "capture": str(path),
                        "schema": schema,
                    }
    return list(selected.values()), {
        c: {"occurrences": counts[c], "unique_schemas": len(unique[c])} for c in counts
    }


def mask_digest(constraint, tokenizer):
    # The initial legal token IDs must be identical with a fresh vs copied parser.
    ids = constraint.get_allowed_tokens(tokenizer, [])
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def benchmark(cases, tokenizer, repeats):
    rows = []
    for case in cases:
        schema = case["schema"]
        cold, warm, digests = [], [], []
        try:
            for _ in range(repeats):
                ARTIFACTS.clear()
                start = time.perf_counter_ns()
                validate_llg_json_schema(schema)
                fresh = LlgJsonSchemaConstraint(schema, tokenizer)
                cold.append((time.perf_counter_ns() - start) / 1e6)
                start = time.perf_counter_ns()
                validate_llg_json_schema(schema)
                reused = LlgJsonSchemaConstraint(schema, tokenizer)
                warm.append((time.perf_counter_ns() - start) / 1e6)
                digests.append(
                    mask_digest(fresh, tokenizer) == mask_digest(reused, tokenizer)
                )
            rows.append(
                {
                    **{k: v for k, v in case.items() if k != "schema"},
                    "cold_ms": cold,
                    "cached_ms": warm,
                    "cold_median_ms": statistics.median(cold),
                    "cached_median_ms": statistics.median(warm),
                    "mask_equal": all(digests),
                }
            )
        except Exception as exc:
            rows.append(
                {**{k: v for k, v in case.items() if k != "schema"}, "error": str(exc)}
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--schemas-per-client", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.repeats < 3 or a.schemas_per_client < 1:
        ap.error("repeats >= 3 and schemas-per-client >= 1 required")
    cases, corpus = captured_schemas(a.captures, a.schemas_per_client)
    if not cases or any(corpus[c]["occurrences"] == 0 for c in corpus):
        raise ValueError("captures must contain schemas from all three clients")
    result = {
        "corpus": corpus,
        "cases": len(cases),
        "tokenizer": str(a.tokenizer),
        "load_start": os.getloadavg(),
        "cpu_only": True,
    }
    if not a.dry_run:
        from transformers import AutoTokenizer

        from yunshu_engine.tool_call_grammar import llg_tokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        llg_tokenizer(tok, len(tok))  # tokenizer setup is not grammar compilation
        result["rows"] = benchmark(cases, tok, a.repeats)
        result["load_end"] = os.getloadavg()
        result["complete"] = all(
            row.get("mask_equal", False) or "error" in row for row in result["rows"]
        )
    else:
        result["complete"] = "dry-run"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
