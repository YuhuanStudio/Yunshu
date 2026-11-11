"""yv ab --arith-change-ok: a deliberate arithmetic change is excused by identity only with both KLD stages."""

import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from verify import analyze, cli, runner, suites  # noqa: E402


def rows(sha_by_phase):
    return [
        {
            "part": "decode",
            "ctx": 1024,
            "kind": "code",
            "phase": p,
            "sha": s,
            "ct": 8,
            "pt": 100,
        }
        for p, s in sha_by_phase.items()
    ]


def test_digest_difference_fails_unless_arith_change_ok():
    base, cand = rows({"cold": "a", "warm": "a"}), rows({"cold": "b", "warm": "a"})
    strict = analyze.compare_identity(base, cand, "base", "cand")
    assert not strict["ok"] and strict["arith_changed"] == []
    ok = analyze.compare_identity(base, cand, "base", "cand", arith_change_ok=True)
    assert ok["ok"] and len(ok["arith_changed"]) == 1 and ok["mismatches"] == []


def test_spec_on_off_stays_strict_with_the_flag():
    """The spec-off comparison never receives arith_change_ok, and strict logprobs still apply."""
    on, off = rows({"cold": "a"}), rows({"cold": "b"})
    assert not analyze.compare_identity(
        off, on, "spec_off", "cand", strict_logprobs=True
    )["ok"]


def test_flag_requires_both_kld_stages_before_speed():
    cfg = suites.parse_suite("quick")
    out = runner.apply_overrides(cfg, Namespace(arith_change_ok=True))
    st = out["stages"]
    assert out["arith_change_ok"] and "kld" in st and "kld_decode" in st
    assert st.index("kld") < st.index("kld_decode") < st.index("speed")
    ident = runner.apply_overrides(
        suites.parse_suite("smoke,identity"), Namespace(arith_change_ok=True)
    )
    assert ident["stages"][-2:] == ["kld", "kld_decode"]
    assert "arith_change_ok" not in runner.apply_overrides(cfg, Namespace())


def test_cli_exposes_the_flag():
    a = cli.build_parser().parse_args(
        [
            "ab",
            "--base",
            "main",
            "--cand",
            "x",
            "--suite",
            "quick",
            "--label",
            "t",
            "--arith-change-ok",
        ]
    )
    assert a.arith_change_ok
