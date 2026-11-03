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
def test_real_manifest_one_prompt_per_group_passes_the_narrowed_output_sha_rule():
    """The narrowed output_sha rule (fullcmp_audit.check_output_sha): one planned prompt per (model, ctx, kind) is
    fine when no variant is planned; the real matrix therefore has no output_sha violation from prompt counts."""
    import sys
    import types

    import fullcmp_audit as fa

    cells = []
    seen = set()
    for r in _real_rows():
        if "ctx_tokens" not in r or not r.get("prompt_kind"):
            continue
        cid = r.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        cells.append({"cell": cid, "id": cid, "model": r.get("model"), "ctx": r["ctx"], "kind": r["prompt_kind"],
                      "prompt_id": f"{r['prompt_kind']}-{r['ctx_tokens']}-v{run.prompt_variant(r)}"})
    res = fa.check_output_sha(types.SimpleNamespace(files=[], manifest={"cells": cells}))
    assert res.violations == [], res.violations[:3]


def _sha_run(cells_rows):
    import types

    files = []
    for engine, prompt_id, sha in cells_rows:
        files.append(types.SimpleNamespace(
            cell={"cell": f"{engine}-c", "engine": engine, "model": "m", "ctx": "8K", "kind": "code",
                  "prompt_id": prompt_id},
            rows=[{"sha": "a" * 16, "prompt_file_sha256": sha}], rep=0, path=None, header=None))
    cells = [{"cell": f"{e}-c", "id": f"{e}-c", "model": "m", "ctx": "8K", "kind": "code", "prompt_id": p}
             for e, p, _ in cells_rows]
    import fullcmp_audit as fa

    return fa.check_output_sha(types.SimpleNamespace(files=files, manifest={"cells": cells}))


def test_group_with_differing_prompt_sha_across_engines_fails():
    res = _sha_run([("yunshu", "code-8192-v0", "1" * 64), ("tf-1.2.1", "code-8192-v0", "2" * 64)])
    assert any("differs across engines" in v for v in res.violations)


def test_group_with_one_sha_across_engines_passes():
    res = _sha_run([("yunshu", "code-8192-v0", "1" * 64), ("tf-1.2.1", "code-8192-v0", "1" * 64)])
    assert not any("differs across engines" in v for v in res.violations)


def test_planned_variant_group_with_one_prompt_fails():
    import types

    import fullcmp_audit as fa

    cells = [{"cell": "c0", "id": "x.v0", "model": "m", "ctx": "1K", "kind": "code", "prompt_id": "code-1024-v0"},
             {"cell": "c1", "id": "x.v1", "model": "m", "ctx": "1K", "kind": "code", "prompt_id": "code-1024-v0"}]
    res = fa.check_output_sha(types.SimpleNamespace(files=[], manifest={"cells": cells}))
    assert any("variants planned" in v for v in res.violations)
