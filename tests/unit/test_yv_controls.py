"""yv positive/negative controls on synthetic cells in yv's real jsonl format.

A/A must pass, an injected slowdown must fail with a delta near the injected size, and a single
flipped token must fail identity and name its cell. Rows go through jsonl files and core.read_jsonl
exactly like a stage reads its evidence. Noise is drawn to match the rep-to-rep spread measured on
real verify runs (median abs deviation: TTFT ~0.45 %, decode tok/s ~1.2 %; tails ~3 %)."""

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from verify import analyze  # noqa: E402
from verify.core import read_jsonl  # noqa: E402

CTX = 8192
KIND = "prose"
PHASES = ("cold", "warm", "turn2")
CT = 2048
ROUNDS = 490  # 2048 tokens at ~4.2 tokens per 50 ms round
TTFT_SIGMA = 0.007  # lognormal rep-to-rep spread of TTFT
DECODE_SIGMA = 0.012  # lognormal rep-to-rep spread of per-round time
TOL = 2.0  # speed_tol_pct of the default speed stage
PLANNED = [(CTX, KIND)]
ID_PLANNED = [(CTX, KIND, p) for p in PHASES]


def _sha(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16]


def _tokens(seed):
    rng = random.Random(seed)
    return [rng.randrange(1000) for _ in range(CT)]


def _decode_row(phase, ttft_s, dec_tps, tokens, lps=None):
    return {
        "part": "decode",
        "ctx": CTX,
        "kind": KIND,
        "phase": phase,
        "ttft_s": round(ttft_s, 3),
        "dec_tps": round(dec_tps, 2),
        "ct": len(tokens),
        "pt": 8237,
        "cached": 0 if phase == "cold" else 8000,
        "finish": "length",
        "sha": _sha(tokens),
        "lp_tokens": None if lps is None else [str(t) for t in tokens],
        "lps": lps,
    }


def _rep(rng, ttft_base=1.0, round_extra_ms=0.0, ttft_extra_s=0.0):
    toks = _tokens(0)
    rows = []
    for phase in PHASES:
        ttft = ttft_base * rng.lognormvariate(0, TTFT_SIGMA) + ttft_extra_s
        round_s = (50.0 + round_extra_ms) / 1000.0 * rng.lognormvariate(0, DECODE_SIGMA)
        rows.append(_decode_row(phase, ttft, CT / (ROUNDS * round_s), toks))
    return rows


def _reps(tmp_path, tag, seed, n, **kw):
    out = []
    for i in range(n):
        rng = random.Random(f"{seed}-{i}")
        path = tmp_path / f"{tag}-r{i}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in _rep(rng, **kw)))
        out.append(read_jsonl(path))
    return out


def _metric(sp, name):
    return next(c for c in sp["cells"] if c["metric"] == name)


def _speed(base, cand):
    """The default stage: 3 paired reps, and on a regression 2 more reps judged robustly."""
    sp = analyze.speed_compare(base, cand, TOL, planned=PLANNED)
    if sp["regressions"]:
        return analyze.speed_compare(
            base, cand, TOL, robust=True, planned=PLANNED
        ), True
    return sp, False


def test_aa_same_distribution_passes(tmp_path):
    base = _reps(tmp_path, "base", "aa-b", 3)
    cand = _reps(tmp_path, "cand", "aa-c", 3)
    sp, _ = _speed(base, cand)
    assert sp["ok"], sp["regressions"]
    for c in sp["cells"]:
        assert c["verdict"] == "neutral", c
        assert c["limit_pct"] >= c["noise_pct"]


def test_injected_decode_slowdown_fails_with_delta_near_injection(tmp_path):
    # +3 ms per decode round on 50 ms rounds: 53/50 more time per token, tok/s -5.66 %
    base = _reps(tmp_path, "base", "sl-b", 3)
    cand = _reps(tmp_path, "slow", "sl-c", 3, round_extra_ms=3.0)
    sp, _ = _speed(base, cand)
    assert not sp["ok"]
    dec = _metric(sp, "decode_tps")
    assert dec["verdict"] == "regression", dec
    injected = 100.0 * (50.0 / 53.0 - 1.0)  # -5.66 % tok/s
    assert abs(dec["delta_pct"] - injected) <= 0.2 * abs(injected), dec


def test_injected_ttft_slowdown_fails_with_delta_near_injection(tmp_path):
    # +50 ms on a 1.0 s first token: +5 %. (On a 7.5 s TTFT the same 50 ms is 0.7 % and
    # below the 2 % tolerance, so it is not detectable there; see the report.)
    base = _reps(tmp_path, "base", "tt-b", 3)
    cand = _reps(tmp_path, "ttft", "tt-c", 3, ttft_extra_s=0.05)
    sp, _ = _speed(base, cand)
    assert not sp["ok"]
    t = _metric(sp, "cold_ttft_s")
    assert t["verdict"] == "regression", t
    assert abs(t["delta_pct"] - 5.0) <= 1.0, t


def _id_rows(tokens, flip_at=None):
    toks = list(tokens)
    if flip_at is not None:
        toks[flip_at] = (toks[flip_at] + 1) % 1000  # a different token id
    return [_decode_row(p, 1.0, 83.0, toks, lps=[-0.5] * len(toks)) for p in PHASES]


def test_identity_aa_passes(tmp_path):
    toks = _tokens(7)
    base = read_jsonl(_write(tmp_path / "idb.jsonl", _id_rows(toks)))
    cand = read_jsonl(_write(tmp_path / "idc.jsonl", _id_rows(toks)))
    r = analyze.compare_identity(base, cand, planned=ID_PLANNED, strict_logprobs=True)
    assert r["ok"], r["mismatches"]
    assert r["compared"] == 3


def test_identity_single_flipped_token_fails_and_names_cell(tmp_path):
    toks = _tokens(7)
    base = read_jsonl(_write(tmp_path / "idb.jsonl", _id_rows(toks)))
    cand_rows = _id_rows(toks)
    cand_rows[0] = _id_rows(toks, flip_at=20)[
        0
    ]  # token 20 flipped in the cold cell only
    cand = read_jsonl(_write(tmp_path / "idc.jsonl", cand_rows))
    r = analyze.compare_identity(base, cand, planned=ID_PLANNED)
    assert not r["ok"]
    bad = [m for m in r["mismatches"] if m.get("why") == "digest differs"]
    assert [m["cell"] for m in bad] == [[CTX, KIND, "cold"]]
    assert r["compared"] == 3


def _write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path
