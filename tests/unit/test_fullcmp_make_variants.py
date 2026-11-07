"""scripts/research/fullcmp_make_variants.py: copy-and-vary smoke prompts stay within 1% and are hashed.

CPU only, temp dirs only. The real corpus is never written (the tests point --src at a temp copy).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_make_variants as mv  # noqa: E402

CODE = (
    "def load_weights(path_arg):\n"
    "    weights_table = read_weights(path_arg)\n"
    "    for layer_index in range(len(weights_table)):\n"
    "        apply_layer(weights_table[layer_index], layer_index)\n"
    "    return weights_table\n"
) * 60
PROSE = (
    "The cache keeps prefix blocks so that a repeated request skips prefill work. "
) * 200


def _src_dir(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for kind, text in (("code", CODE), ("prose", PROSE)):
        for ctx in mv.CTXS:
            (src / f"{kind}-{ctx}.txt").write_text(text)
    return src


def test_variant_within_one_percent_by_chars():
    rng = random.Random(7)
    for src_text in (CODE, PROSE):
        for kind in ("code", "prose"):
            out = mv.make_variant(src_text, kind, 1, random.Random(rng.random()), len)
            assert out != src_text
            assert abs(len(out) - len(src_text)) / len(src_text) <= mv.TOL


def test_variant_leads_with_distinct_task_sentence():
    out = mv.make_variant(PROSE, "prose", 1, random.Random(1), len)
    assert out.startswith(mv.task_sentence(1, "prose"))
    assert "Variant 1 task" in out


def test_rename_is_consistent_across_occurrences():
    text = (
        "alpha_value = 1\nbeta_value = alpha_value + alpha_value\nprint(alpha_value)\n"
    )
    out = mv.rename_identifiers(text, random.Random(3), 1)
    for name in ("alpha_value", "beta_value"):
        src_count = len(re.findall(rf"\b{name}\b", text))
        kept = len(re.findall(rf"\b{name}\b", out))
        renamed = len(re.findall(rf"\b{name}_v1[0-9a-f]{{3}}\b", out))
        assert (kept, renamed) in ((src_count, 0), (0, src_count))


def test_keywords_are_never_renamed():
    text = "return return return import import import None None None\n"
    assert mv.rename_identifiers(text, random.Random(0), 1) == text


def test_run_writes_files_and_sha_manifest(tmp_path):
    src = _src_dir(tmp_path)
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    out = tmp_path / "out"
    m = mv.run(src, out, tokenizer=None)
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before, (
        "source was modified"
    )
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk == m
    assert len(m["files"]) == len(mv.KINDS) * len(mv.CTXS) * len(mv.VARIANTS)
    for f in m["files"]:
        data = (out / f["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == f["sha256"]
        assert f["delta_pct"] <= 100 * mv.TOL
        assert (
            f["src_sha256"] == hashlib.sha256((src / f["src"]).read_bytes()).hexdigest()
        )


def test_chars_mode_is_marked_unverified(tmp_path):
    m = mv.run(_src_dir(tmp_path), tmp_path / "out", tokenizer=None)
    assert m["check_mode"] == "chars"
    assert all(f["check"] == "unverified" for f in m["files"])


def test_tokens_mode_when_tokenizer_present(tmp_path):
    if not mv.DEFAULT_TOKENIZER.exists():
        pytest.skip("model tokenizer not on this machine")
    m = mv.run(_src_dir(tmp_path), tmp_path / "out", tokenizer=mv.DEFAULT_TOKENIZER)
    assert m["check_mode"] == "tokens"
    assert all(f["check"] == "tokens" for f in m["files"])
    assert all(f["delta_pct"] <= 100 * mv.TOL for f in m["files"])


def test_out_of_tolerance_fails_closed(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for kind in mv.KINDS:
        for ctx in mv.CTXS:
            (src / f"{kind}-{ctx}.txt").write_text(
                "x"
            )  # one word: no cut can keep 1 char
    with pytest.raises(SystemExit, match="not written"):
        mv.run(src, tmp_path / "out", tokenizer=None)
    assert not list((tmp_path / "out").glob("*.txt"))
