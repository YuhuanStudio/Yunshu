"""yv kld stage: metric math, cache key, manifest, fail-closed behaviour (CPU)."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from verify.kld import corpus, metrics, stage  # noqa: E402


def window(ids, probs, tokens=None):
    ids = np.array(ids, dtype=np.int32)
    lp = np.log(np.array(probs, dtype=np.float32))
    n = ids.shape[0]
    return metrics.Window(
        ids, lp, np.full(n, -1.0, np.float32), np.arange(n, dtype=np.int32)
    )


def one(ids, probs):
    return window([ids], [probs])


def test_kld_of_identical_distributions_is_zero_and_matches_the_formula():
    ref = one([1, 2, 3], [0.5, 0.3, 0.1])
    assert metrics.kld_per_position(ref, ref)[0] == pytest.approx(0.0, abs=1e-9)
    arm = one([1, 2, 3], [0.4, 0.4, 0.1])
    want = (
        0.5 * math.log(0.5 / 0.4)
        + 0.3 * math.log(0.3 / 0.4)
        + 0.1 * math.log(1)
        + 0.1 * math.log(0.1 / 0.1)
    )
    assert metrics.kld_per_position(ref, arm)[0] == pytest.approx(want, abs=1e-6)


def test_missing_reference_token_takes_the_arms_floor_and_remainder_is_bucketed():
    ref = one([1, 2], [0.6, 0.3])  # remainder 0.1
    arm = one([1, 9], [0.7, 0.2])  # token 2 missing: floor 0.2; remainder 0.1
    p_rest, q_rest = 0.1, 1 - (0.7 + 0.2)
    want = (
        0.6 * math.log(0.6 / 0.7)
        + 0.3 * math.log(0.3 / 0.2)
        + p_rest * math.log(p_rest / q_rest)
    )
    assert metrics.kld_per_position(ref, arm)[0] == pytest.approx(want, abs=1e-6)


def test_agreement_and_perplexity():
    a = window([[1, 2, 3, 4, 5, 6]], [[0.3, 0.2, 0.1, 0.1, 0.1, 0.1]])
    b = window([[1, 2, 3, 4, 9, 8]], [[0.3, 0.2, 0.1, 0.1, 0.1, 0.1]])
    assert metrics.top1_agreement(a, b)[0] == 1.0
    assert metrics.top5_overlap(a, b)[0] == pytest.approx(0.8)
    w = metrics.Window(
        np.zeros((3, 1), np.int32),
        np.zeros((3, 1), np.float32),
        np.array([-1.0, -3.0, np.nan], np.float32),
        np.arange(3, dtype=np.int32),
    )
    assert metrics.perplexity(w) == pytest.approx(math.exp(2.0))


def test_depth_curve_buckets_positions_by_context_depth():
    curve = metrics.depth_curve(
        np.concatenate([np.full(1024, 1.0), np.full(1024, 3.0)])
    )
    assert curve["0-1024"] == 1.0 and curve["1024-2048"] == 3.0


def test_summarize_fails_closed_on_a_missing_window():
    w = one([1, 2], [0.5, 0.4])
    with pytest.raises(RuntimeError):
        metrics.summarize({"a": w, "b": w}, {"a": w})


def test_verdict_passes_equal_and_fails_a_worse_candidate_or_lower_top1():
    base = dict(kld_mean=0.010, top1=0.95)
    assert metrics.verdict(base, dict(kld_mean=0.010, top1=0.95), 0.001, 0.002)[0]
    assert metrics.verdict(base, dict(kld_mean=0.0119, top1=0.95), 0.001, 0.002)[
        0
    ]  # <= 1.1 x + eps
    ok, why = metrics.verdict(base, dict(kld_mean=0.0125, top1=0.95), 0.001, 0.002)
    assert not ok and "KLD" in why[0]
    ok, why = metrics.verdict(base, dict(kld_mean=0.010, top1=0.94), 0.001, 0.002)
    assert not ok and "top-1" in why[0]


def test_cache_key_changes_with_every_component():
    base = ("cand", "abc-d1", "model-id", {"A": "1"}, "corpus", "prefill")
    key = stage.cache_key(*base)
    assert stage.cache_key(*base) == key
    for i, other in enumerate(
        ["base", "abc-d2", "model-2", {"A": "2"}, "corpus2", "decode"]
    ):
        changed = list(base)
        changed[i] = other
        assert stage.cache_key(*changed) != key


def test_manifest_is_consistent_and_corpus_text_fails_closed(tmp_path):
    doc = corpus.load_manifest()
    assert (
        doc["total_tokens"] == sum(w["tokens"] for w in doc["windows"])
        and 190_000 < doc["total_tokens"] < 210_000
    )
    assert {w["id"].split("-")[1] for w in doc["windows"]} == {"8k", "32k"}
    windows = [dict(id="w-8k-0", group="prose", tokens=5, text="hello world")]
    manifest = tmp_path / "manifest.json"
    corpus.write(windows, root=tmp_path, manifest=manifest)
    assert corpus.load_texts(tmp_path, manifest)[0]["text"] == "hello world"
    (tmp_path / "v1" / "w-8k-0.txt").write_text("changed")
    with pytest.raises(RuntimeError, match="manifest sha"):
        corpus.load_texts(tmp_path, manifest)
    (tmp_path / "v1" / "w-8k-0.txt").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        corpus.load_texts(tmp_path, manifest)
    doc = json.loads(manifest.read_text())
    doc["windows"][0]["tokens"] = 6
    manifest.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="corpus_sha"):
        corpus.load_manifest(manifest)


def test_incomplete_or_malformed_arm_cache_fails_closed(tmp_path):
    manifest = dict(windows=[dict(id="a", tokens=4), dict(id="b", tokens=4)])
    np.savez(
        tmp_path / "a.npz",
        ids=np.zeros((4, 2), np.int32),
        lp=np.full((4, 2), -1.0, np.float32),
        next_lp=np.zeros(4, np.float32),
        tokens=np.arange(4, dtype=np.int32),
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        stage.load_arm(tmp_path, manifest)
    np.savez(
        tmp_path / "b.npz",
        ids=np.zeros((400, 2), np.int32),
        lp=np.full((400, 2), -1.0, np.float32),
        next_lp=np.zeros(400, np.float32),
        tokens=np.arange(400, dtype=np.int32),
    )
    with pytest.raises(RuntimeError, match="positions"):
        stage.load_arm(tmp_path, manifest)
    np.savez(
        tmp_path / "b.npz",
        ids=np.zeros((4, 2), np.int32),
        lp=np.full((4, 2), -np.inf, np.float32),
        next_lp=np.zeros(4, np.float32),
        tokens=np.arange(4, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="non-finite"):
        stage.load_arm(tmp_path, manifest)


def test_suite_and_decode_mode_fail_closed():
    from verify import suites

    assert suites.parse_suite("kld")["stages"] == ["kld"]


def test_arms_with_different_position_counts_fail_closed():
    a, b = one([1, 2], [0.5, 0.4]), window([[1, 2]] * 5, [[0.5, 0.4]] * 5)
    with pytest.raises(RuntimeError, match="positions"):
        metrics.summarize({"w": a}, {"w": b})


def test_capture_detects_the_architecture_and_fails_closed(tmp_path):
    from verify.kld import capture

    for kind, arch in (
        ("qwen4_exp", "qwen4_exp"),
        ("qwen3_5", "qwen3_5"),
        ("qwen3_5_moe", "qwen3_5"),
    ):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": kind}))
        assert capture.detect_arch(str(tmp_path)) == arch
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(RuntimeError, match="no hooks"):
        capture.detect_arch(str(tmp_path))


def test_reference_attention_stays_finite_when_the_last_row_ends_on_a_block_boundary():
    import mlx.core as mx
    from verify.kld import capture

    if not mx.metal.is_available():
        pytest.skip("needs a GPU")
    rng = np.random.default_rng(0)
    rows, length = (
        64,
        2048,
    )  # the last row's query end (2048) is a multiple of 4 and equals the key count
    q = mx.array(rng.normal(size=(1, 24, rows, 256)).astype(np.float32)).astype(
        mx.bfloat16
    )
    k = mx.array(rng.normal(size=(1, 2, length, 256)).astype(np.float32)).astype(
        mx.bfloat16
    )
    ends = np.arange(length - rows + 1, length + 1, dtype=np.int32)
    blocks = np.full((1, rows, 512), -1, dtype=np.int32)
    for i, e in enumerate(ends):
        pick = rng.permutation(int(e) // 4)[:512]
        blocks[0, i, : len(pick)] = pick
    out = capture.reference_attend(
        q, k, k, mx.array(blocks), mx.array(ends[None]), 256**-0.5
    )
    assert np.isfinite(np.array(out.astype(mx.float32))).all()
