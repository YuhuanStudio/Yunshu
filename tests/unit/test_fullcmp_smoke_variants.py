"""Smoke prompt variants: the audit's >= 2 prompts per (model, ctx, kind) guard can pass on the smoke
matrix, the real matrix is byte-identical, and tfbench loads variant N from the smoke-only directory.

CPU only: no gpuq call, no GPU, no server. Manifest rows are built by fullcmp_manifest; the driver and
tfbench are imported as modules.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_manifest as fm  # noqa: E402
import fullcmp_run as fr  # noqa: E402
import tfbench  # noqa: E402

# sha256 of the full real manifest (build_rows + apply_cuts, JSONL as main writes it). Re-pinned for
# the conc-same cells (39 new rows: same-prompt concurrency, B2); the est_gpu_min of the other rows
# moves by float rounding. Re-pinned 11-03 after the regenerated manifest matched it byte for byte.
# Re-pinned 11-03 again: the cost model now uses the measured cell timing (fullcmp_cell_timing.json:
# per-cell server start + memory idle, warm decode), so every est_gpu_min moved on purpose.
# Re-pinned 2026-11-04: yunshu-pb2 rows added (same config as yunshu; release date = commit 4f99270cae40),
# and the cut step mirrors yunshu's decisions onto pb2 (CUT_MIRROR)
# Re-pinned 2026-11-04: tf-1.2.1 image cells are unsupported (its measured refusal, wire-1108 B2), no longer cut
# Re-pinned 2026-11-04: restart cells of tf-1.2.1, splash and llama.cpp are unsupported (no disk tier, evidence cited)
# Re-pinned 2026-11-04 (launch review): the documented disk tiers are on (splash, mlx-serve, mlx-vlm APC, kapitsa,
# oMLX, yunshu), tf-1.1.8 images and restart, llama.cpp restart are unsupported with cited evidence
# Re-pinned 2026-11-04: llama.cpp image cells are unsupported in this run (mmproj not provisioned, follow-up)
# Re-pinned 2026-11-04: oMLX 0.7.4.dev1 added as an EXTRA engine (rows mirror omlx-0.7.3, hours outside the plan total).
# The bytes of every other engine's rows are unchanged: REAL_MANIFEST_SHA_12 is the previous pin (the manifest the real run uses).
# Re-pinned 2026-11-07: mlx-lm-server thinking-on cells are unsupported (census reasoning_split false, audit fixes 5b2e17d50)
# and kapitsa-0.2.31 (pinned re-run of kapitsa-0.2.26, an EXTRA engine: the other engines' bytes did not move with it).
REAL_MANIFEST_SHA = "d7db4b5a6d9ac02b06ef4c20ac5526b8b4895a7d3af03cdd065877b1d4ccb854"
REAL_MANIFEST_SHA_12 = (
    "17ff25d5690b62c1030c70b06bff53ea7cd28f8e7f77ce537afbfd27c75b73d3"
)


def _real_manifest_bytes() -> bytes:
    rows, _meta = fm.build_rows()
    fm.apply_cuts(rows, out=io.StringIO())
    return b"".join(
        (json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n").encode()
        for r in rows
    )


def test_the_extra_engine_leaves_every_other_rows_bytes_unchanged():
    """The rows of the engines the real run uses are byte-identical to the manifest before the extra engine existed (its
    hours are outside the plan total, its cut decisions mirror omlx-0.7.3): adding it changes no other cell."""
    rows, _meta = fm.build_rows()
    fm.apply_cuts(rows, out=io.StringIO())
    data = b"".join(
        (json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n").encode()
        for r in rows
        if r["engine"] not in fm.EXTRA_ENGINES
    )
    assert hashlib.sha256(data).hexdigest() == REAL_MANIFEST_SHA_12


def test_the_extra_engine_rows_mirror_omlx_073_statuses_and_cells():
    rows, _meta = fm.build_rows()
    fm.apply_cuts(rows, out=io.StringIO())
    base = {
        r["id"].replace("omlx-0.7.3", "@"): r
        for r in rows
        if r["engine"] == "omlx-0.7.3"
    }
    extra = {
        r["id"].replace("omlx-0.7.4.dev1", "@"): r
        for r in rows
        if r["engine"] == "omlx-0.7.4.dev1"
    }
    assert extra and set(extra) == set(base)
    for k, r in extra.items():
        assert r["status"] == base[k]["status"], (
            k
        )  # the same cut / unsupported decisions
        assert (
            r["batch"] == base[k]["batch"]
            and r["est_gpu_min"] == base[k]["est_gpu_min"]
        )


def test_real_manifest_bytes_unchanged():
    data = _real_manifest_bytes()
    assert hashlib.sha256(data).hexdigest() == REAL_MANIFEST_SHA


def test_real_rows_carry_no_variant_and_keep_prompt_id():
    rows, _meta = fm.build_rows()
    for r in rows:
        assert "prompt_variant" not in r
    for r in rows:
        if r["batch"] in ("B1", "B2") and fr.harness_part(r)[0] is not None:
            assert fr.prompt_variant(r) == 0
            assert fr.prompt_id(r) == f"{r['prompt_kind']}-{r['ctx_tokens']}"
            assert "--prompt-variant" not in fr.harness_part(r)[1]


def test_smoke_rows_are_yunshu_1k_two_variants():
    rows = fm.build_smoke_rows()
    assert rows, "smoke manifest is empty"
    assert {r["batch"] for r in rows} == {fm.SMOKE_BATCH}
    assert {r["engine"] for r in rows} == {fm.SMOKE_ENGINE}
    assert {r["ctx"] for r in rows} == {"1K"}
    assert {r["prompt_variant"] for r in rows} == set(fm.SMOKE_VARIANTS) == {0, 1}
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert all(r["id"].endswith(f".v{r['prompt_variant']}") for r in rows)
    # 1K code/prose cold+warm+turn2, conc 2 code/prose, one profile rep: 5 specs per variant
    per_variant = [r for r in rows if r["prompt_variant"] == 0]
    assert len(per_variant) == 5
    assert all(r["est_gpu_min"] == 0.0 for r in rows)


def test_smoke_each_ctx_kind_has_two_prompt_ids():
    """The audit guard (fullcmp_audit.check_output_sha): >= 2 distinct prompt_id per (model, ctx, kind)."""
    prompts = defaultdict(set)
    for r in fm.build_smoke_rows():
        if fr.harness_part(r)[0] is None:
            continue
        prompts[(r["model"], r["ctx_tokens"], r["prompt_kind"])].add(fr.prompt_id(r))
    assert prompts
    for key, ids in prompts.items():
        assert len(ids) >= 2, key


def test_prompt_id_suffix_only_for_nonzero_variant():
    base = {"prompt_kind": "code", "ctx_tokens": 1024}
    assert fr.prompt_id({**base}) == "code-1024"
    assert fr.prompt_id({**base, "prompt_variant": 0}) == "code-1024"
    assert fr.prompt_id({**base, "prompt_variant": 1}) == "code-1024-v1"


def test_harness_part_passes_variant_flag_only_when_nonzero():
    row = next(
        r for r in fm.build_smoke_rows() if r["prompt_variant"] == 1 and r["conc"] == 1
    )
    part, extra, _ = fr.harness_part(row)
    assert part == "decode"
    i = extra.index("--prompt-variant")
    assert extra[i : i + 2] == ["--prompt-variant", "1"]
    zero = {**row, "prompt_variant": 0}
    assert "--prompt-variant" not in fr.harness_part(zero)[1]


def test_smoke_conc_row_uses_conc_part_with_variant():
    row = next(
        r for r in fm.build_smoke_rows() if r["prompt_variant"] == 1 and r["conc"] == 2
    )
    part, extra, _ = fr.harness_part(row)
    assert part == "conc"
    i = extra.index("--prompt-variant")
    assert extra[i : i + 2] == ["--prompt-variant", "1"]


def test_expected_records_for_smoke_rows_come_from_the_manifest():
    rows = fm.build_smoke_rows()
    core = next(
        r for r in rows if r["sweep"] == "smoke-core" and r["prompt_variant"] == 0
    )
    conc = next(
        r for r in rows if r["sweep"] == "smoke-conc" and r["prompt_variant"] == 0
    )
    assert fr.expected_records(core) == 3  # cold, warm, turn2: one rep file
    assert fr.expected_records(conc) == 2 * fr.CONC_TRIALS
    # the real matrix gets its counts too (they were None: the record_count guard failed every cell)
    real, _ = fm.build_rows()
    assert all(
        isinstance(fr.expected_records(r), int)
        for r in real
        if r["batch"] in ("B1", "B2") and r["status"] == "unverified"
    )


def test_smoke_manifest_writes_to_its_own_path(tmp_path):
    out = tmp_path / "smoke.jsonl"
    assert fm.main(["--smoke", "--out", str(out)]) == 0
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == len(fm.build_smoke_rows())
    assert fm.SMOKE_OUT != fm.DEFAULT_OUT


def test_tfbench_variant_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(tfbench, "PROMPT_VARIANT_DIR", tmp_path)
    assert (
        tfbench.prompt_path("code-1024", 0)
        == tfbench.WORK / "prompts" / "code-1024.txt"
    )
    assert tfbench.prompt_path("code-1024", 1) == tmp_path / "code-1024-v1.txt"
    assert tfbench.decode_prompt_file("code", 1024, False, 0) == str(
        tfbench.WORK / "prompts" / "code-1024.txt"
    )
    assert tfbench.decode_prompt_file("code", 1024, False, 1) == str(
        tmp_path / "code-1024-v1.txt"
    )


def test_tfbench_decode_prompt_reads_variant_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tfbench, "PROMPT_VARIANT_DIR", tmp_path)
    monkeypatch.delenv("TFB_EXACT_PROMPTS", raising=False)
    (tmp_path / "prose-1024-v1.txt").write_text("VARIANT TEXT")
    assert tfbench.decode_prompt("prose", 1024, False, 1) == "VARIANT TEXT"


def test_tfbench_missing_variant_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(tfbench, "PROMPT_VARIANT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        tfbench.decode_prompt("prose", 1024, False, 1)


def test_tfbench_parse_args_variant_default_zero():
    base = ["--engine", "yunshu", "--part", "decode", "--out", "x.jsonl"]
    assert tfbench.parse_args(base).prompt_variant == 0
    assert tfbench.parse_args(base + ["--prompt-variant", "1"]).prompt_variant == 1
