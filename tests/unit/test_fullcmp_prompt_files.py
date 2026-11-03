"""Every (kind, ctx) the fullcmp matrix runs has the prompt file tfbench reads.

tfbench loads WORK/prompts/<base>-<ctx_tokens>.txt, where WORK is TFB_WORK. Family kinds
(tool-chat, cjk, json-schema) build on the prose corpus file (tfbench FAMILY_KINDS_LOCAL), so their
base is prose. The real manifest (when present) and the generated matrix rows are both checked.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fullcmp_manifest", ROOT / "scripts/research/fullcmp_manifest.py"
)
fm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fm)

REAL_MANIFEST = Path(
    os.environ.get(
        "FULLCMP_MANIFEST",
        "/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/manifest.jsonl",
    )
)
PROMPT_DIR = (
    Path(os.environ.get("TFB_WORK", "/Volumes/P5Plus/yunshu-build/snapshot014-b2")) / "prompts"
)


def _base_kind(kind: str) -> str:
    return "prose" if kind in fm.FAMILY_KINDS else kind


def _pairs_from_rows(rows):
    pairs = set()
    for r in rows:
        if "ctx_tokens" not in r or not r.get("prompt_kind"):
            continue  # behaviour / agent rows carry no decode prompt
        pairs.add((r["prompt_kind"], int(r["ctx_tokens"])))
    return pairs


def _missing(pairs):
    return sorted(
        f"{_base_kind(k)}-{ctx}.txt"
        for k, ctx in pairs
        if not (PROMPT_DIR / f"{_base_kind(k)}-{ctx}.txt").is_file()
    )


@pytest.mark.skipif(not REAL_MANIFEST.is_file(), reason="real fullcmp manifest not present")
def test_real_manifest_every_kind_ctx_has_a_prompt_file():
    rows = [json.loads(line) for line in REAL_MANIFEST.read_text().splitlines() if line.strip()]
    pairs = _pairs_from_rows(rows)
    assert pairs, "real manifest has no decode prompt pairs"
    if not PROMPT_DIR.is_dir():
        pytest.skip(f"prompt dir {PROMPT_DIR} not present")
    missing = _missing(pairs)
    assert not missing, f"real-manifest prompt files missing in {PROMPT_DIR}: {missing}"


def test_generated_matrix_every_kind_ctx_has_a_prompt_file():
    if not PROMPT_DIR.is_dir():
        pytest.skip(f"prompt dir {PROMPT_DIR} not present")
    pairs = _pairs_from_rows(fm.build_rows()) | _pairs_from_rows(fm.build_smoke_rows())
    assert pairs
    missing = _missing(pairs)
    assert not missing, f"generated-matrix prompt files missing in {PROMPT_DIR}: {missing}"
