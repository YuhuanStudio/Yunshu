import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from verify import analyze  # noqa: E402

K = (32768, "prose", "cold")


def _row(lps, sha="abc", toks=None):
    return {
        "part": "decode",
        "ctx": K[0],
        "kind": K[1],
        "phase": K[2],
        "sha": sha,
        "ct": len(lps),
        "finish": "length",
        "lp_tokens": toks or [f"t{i}" for i in range(len(lps))],
        "lps": lps,
    }


def _cmp(a, b, strict):
    return analyze.compare_identity(
        [a], [b], "spec_off", "cand", planned=[K], strict_logprobs=strict
    )


def test_spec_on_off_equal_everything_passes():
    r = _cmp(_row([-0.1, -0.2]), _row([-0.1, -0.2]), True)
    assert r["ok"] and r["max_abs_dlp"] == 0.0


def test_spec_on_off_equal_tokens_but_1e4_logprob_diff_fails():
    r = _cmp(_row([-0.1, -0.2]), _row([-0.1, -0.2001]), True)
    assert not r["ok"]
    assert r["mismatches"][0]["why"] == "logprobs differ"
    assert r["max_abs_dlp"] > 9e-5


def test_spec_on_off_missing_logprobs_fails_closed():
    a = _row([-0.1])
    a["lps"] = None
    assert not _cmp(a, _row([-0.1]), True)["ok"]


def test_base_vs_cand_logprob_diff_is_drift_not_failure():
    r = _cmp(_row([-0.1, -0.2]), _row([-0.1, -0.2001]), False)
    assert r["ok"] and r["mismatches"] == []
    assert r["drift"][0]["max_abs_dlp"] > 9e-5


def test_digest_difference_still_fails_without_strict():
    assert not _cmp(_row([-0.1]), _row([-0.1], sha="zzz"), False)["ok"]


def test_apc_hit_logprob_diff_fails():
    cold = _row([-0.1, -0.2])
    warm = dict(_row([-0.1, -0.2001]), phase="warm", cached=100)
    assert not analyze.check_apc([cold, warm])["ok"]
    warm = dict(_row([-0.1, -0.2]), phase="warm", cached=100)
    assert analyze.check_apc([cold, warm])["ok"]


def test_followup_drift_ok_lists_turn2_only():
    """--followup-drift-ok: the follow-up resumes from the reply's decode KV (user decision
    2026-11-02), so a base-vs-cand turn2 digest change is listed, not failed; cold / warm stay
    strict, and a turn2 of another length still fails."""

    def row(phase, sha, ct=2):
        r = _row([-0.1] * ct, sha=sha)
        r["phase"] = phase
        return r

    planned = [(32768, "prose", "cold"), (32768, "prose", "turn2")]
    a = [row("cold", "x"), row("turn2", "y")]
    ok = analyze.compare_identity(
        a,
        [row("cold", "x"), row("turn2", "z")],
        "base",
        "cand",
        planned=planned,
        followup_drift_ok=True,
    )
    assert ok["ok"] and ok["followup_drift"][0]["cell"] == [32768, "prose", "turn2"]
    strict = analyze.compare_identity(
        a, [row("cold", "x"), row("turn2", "z")], "base", "cand", planned=planned
    )
    assert not strict["ok"]
    cold = analyze.compare_identity(
        a,
        [row("cold", "w"), row("turn2", "z")],
        "base",
        "cand",
        planned=planned,
        followup_drift_ok=True,
    )
    assert not cold["ok"]
    short = analyze.compare_identity(
        a,
        [row("cold", "x"), row("turn2", "z", ct=1)],
        "base",
        "cand",
        planned=planned,
        followup_drift_ok=True,
    )
    assert not short["ok"]
