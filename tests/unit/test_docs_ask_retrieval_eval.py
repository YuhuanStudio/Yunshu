"""Labeled retrieval eval for the console docs Ask AI over the real docs (3 locales)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from yunshu_console.docs_ask import ChunkIndex

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
EVAL = json.loads((ROOT / "tests/fixtures/docsask/eval.json").read_text())
LOCALES = ("en", "zh-TW", "zh-CN")
DUMP = (
    'import {buildChunks} from "./scripts/docs-chunks.mjs";'
    "console.log(JSON.stringify(Object.fromEntries("
    + json.dumps(list(LOCALES))
    + ".map(l=>[l,buildChunks(l).chunks]))))"
)


@pytest.fixture(scope="module")
def indexes():
    if shutil.which("node") is None or not (FRONTEND / "node_modules").exists():
        pytest.skip("node + frontend/node_modules needed to build the docs chunks")
    out = subprocess.run(
        ["node", "--input-type=module", "-e", DUMP],
        cwd=FRONTEND,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    return {k: ChunkIndex(v) for k, v in json.loads(out).items()}


def evaluate(indexes) -> dict[str, dict[str, float]]:
    rows: dict[str, list[tuple[bool, float]]] = {}
    for item in EVAL:
        hits = indexes[item["l"]].search(item["q"], 8)
        slugs = [h["slug"] for h in hits]
        rank = next((i + 1 for i, s in enumerate(slugs) if s in item["p"]), 0)
        rows.setdefault(item["l"], []).append((0 < rank <= 3, 1 / rank if rank else 0))
        rows.setdefault("all", []).append(rows[item["l"]][-1])
    return {
        k: {
            "recall@3": sum(r for r, _ in v) / len(v),
            "mrr": sum(m for _, m in v) / len(v),
            "n": len(v),
        }
        for k, v in rows.items()
    }


def test_retrieval_quality(indexes, capsys):
    result = evaluate(indexes)
    with capsys.disabled():
        print("\ndocs-ask retrieval eval:", json.dumps(result))
    assert result["all"]["n"] >= 30
    # Baseline before the tokenizer/rerank fix: recall@3 0.727, MRR 0.663
    # (en .611/.597, zh-TW .857/.762, zh-CN .875/.724).
    assert result["all"]["recall@3"] >= 0.9
    assert result["all"]["mrr"] >= 0.88
    for locale in LOCALES:
        assert result[locale]["recall@3"] >= 0.85
