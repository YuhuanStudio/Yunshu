"""Every (kind, ctx, variant) the fullcmp driver can run has its prompt file and a matching sha256 row.

The check is the runner's own gate, fullcmp_run.prompt_corpus_problems, which the driver calls before any
job is written. Variant 0 reads TFB_WORK/prompts (family kinds read the prose file); variant N reads the
smoke variant dir and needs a row in the prompt sha table. The real manifest (when present) and every
generated matrix (real, smoke, smoke-conc) are all checked.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
SPEC = importlib.util.spec_from_file_location(
    "fullcmp_run", ROOT / "scripts" / "research" / "fullcmp_run.py"
)
run = importlib.util.module_from_spec(SPEC)
sys.modules["fullcmp_run"] = run
SPEC.loader.exec_module(run)
fm = run.fm

REAL_MANIFEST = Path(
    "/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/manifest.jsonl"
)
CORPUS_OK = Path(run.TFB_WORK_DIR, "prompts").is_dir()
VARIANTS_OK = run.PROMPT_VARIANT_DIR.is_dir()


def _real_rows():
    return [json.loads(x) for x in REAL_MANIFEST.read_text().splitlines() if x.strip()]


@pytest.mark.skipif(not REAL_MANIFEST.is_file(), reason="real fullcmp manifest not present")
def test_real_manifest_prompt_files_and_sha_rows_present():
    if not CORPUS_OK:
        pytest.skip("corpus dir not present")
    problems = run.prompt_corpus_problems(_real_rows())
    assert not problems, "\n".join(problems)


@pytest.mark.skipif(not (CORPUS_OK and VARIANTS_OK), reason="prompt dirs not present")
def test_generated_matrices_prompt_files_and_sha_rows_present():
    rows = [*fm.build_rows()[0], *fm.build_smoke_rows(), *fm.build_smoke_conc_rows()]
    assert any(r.get("prompt_variant") for r in rows), "no variant rows generated"
    problems = run.prompt_corpus_problems(rows)
    assert not problems, "\n".join(problems)


def test_missing_variant_file_or_sha_row_is_refused_and_named(tmp_path):
    """Negative case: a variant row with no file, and one whose sha row is absent, are both refused."""
    corpus = tmp_path / "corpus"
    (corpus / "prompts").mkdir(parents=True)
    (corpus / "prompts" / "code-8192.txt").write_text("corpus")
    variants = tmp_path / "variants"
    variants.mkdir()
    (variants / "code-1024-v1.txt").write_text("v1")
    table = {"code-1024-v1.txt": hashlib.sha256(b"v1").hexdigest()}
    rows = [
        {"prompt_kind": "code", "ctx_tokens": 8192, "prompt_variant": 1, "id": "m"},
        {"prompt_kind": "code", "ctx_tokens": 1024, "prompt_variant": 1, "id": "n"},
    ]
    # code-8192-v1.txt absent in variant dir -> refused, named
    problems = run.prompt_corpus_problems(
        rows, work_dir=str(corpus), variant_dir=variants, sha_table=table
    )
    assert any("kind=code ctx=8192 variant=1" in p for p in problems)
    # the 1K variant file exists and matches its table row -> no problem for it
    assert not any("ctx=1024" in p for p in problems)
    # file present but no table row -> refused, named
    (variants / "code-8192-v1.txt").write_text("x")
    problems = run.prompt_corpus_problems(
        rows[:1], work_dir=str(corpus), variant_dir=variants, sha_table={}
    )
    assert any("no sha256 row" in p and "variant=1" in p for p in problems)


@pytest.mark.skipif(not REAL_MANIFEST.is_file(), reason="real fullcmp manifest not present")
@pytest.mark.xfail(
    strict=True,
    reason="real matrix plans one prompt (variant 0) per (model, ctx, kind); output_sha needs >= 2",
)
def test_real_manifest_groups_have_two_prompt_ids_for_output_sha_guard():
    """output_sha (fullcmp_audit.check_output_sha) needs >= 2 distinct prompt ids per (model, ctx, kind).
    Strict xfail: it turns into a failure once the real matrix plans a second prompt per group."""
    groups: dict[tuple, set] = {}
    for r in _real_rows():
        if "ctx_tokens" not in r or not r.get("prompt_kind"):
            continue
        key = (r.get("model"), r["ctx"], r["prompt_kind"])
        groups.setdefault(key, set()).add(f"{r['prompt_kind']}-{r['ctx_tokens']}-v{run.prompt_variant(r)}")
    short = sorted(k for k, ids in groups.items() if len(ids) < 2)
    assert not short, f"groups with < 2 prompt ids: {short}"
