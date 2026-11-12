"""yv ab --spec-mode-change-ok: base and cand engage different spec modes on purpose."""

import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from verify import cli, runner, stages, suites  # noqa: E402


def sess(mode):
    return [{"part": "session", "engaged_spec_mode": mode}]


def test_modes_differ_fail_without_the_flag():
    assert stages.spec_mode_failures(sess("dflash"), sess("mtp"), {}, "speed")
    assert stages.spec_mode_failures(sess("off"), sess("mtp"), {}, "identity")


def test_modes_differ_pass_with_the_flag():
    cfg = {"spec_mode_change_ok": True}
    assert stages.spec_mode_failures(sess("dflash"), sess("mtp"), cfg, "identity") == []
    assert stages.spec_mode_failures(sess("off"), sess("mtp"), cfg, "speed") == []
    note = stages.spec_mode_note(sess("dflash"), sess("mtp"))
    assert note == "spec mode: base=dflash cand=mtp (expected)"


def test_equal_modes_fail_with_the_flag():
    reasons = stages.spec_mode_failures(
        sess("mtp"), sess("mtp"), {"spec_mode_change_ok": True}, "identity"
    )
    assert (
        reasons
        and "--spec-mode-change-ok given but base and cand both engaged 'mtp'"
        in reasons[0]
    )


def test_unknown_or_missing_mode_fails_with_the_flag():
    cfg = {"spec_mode_change_ok": True}
    for base, cand in (
        (sess(None), sess("mtp")),
        (sess("mtp"), sess(None)),
        (sess("mtp"), []),
        ([], sess("mtp")),
    ):
        reasons = stages.spec_mode_failures(base, cand, cfg, "identity")
        assert reasons and "missing or ambiguous" in reasons[0]
        assert stages.spec_mode_note(base, cand) is None


def test_expected_mode_still_enforced_with_the_flag():
    cfg = {"spec_mode_change_ok": True}
    reasons = stages.spec_mode_failures(
        sess("off"), sess("mtp"), cfg, "identity", {"base": "dflash", "cand": "mtp"}
    )
    assert reasons and "base engaged spec mode 'off', expected 'dflash'" in reasons[0]


def test_flag_forces_spec_on_vs_off_and_is_cli_visible():
    out = runner.apply_overrides(
        suites.parse_suite("identity"), Namespace(spec_mode_change_ok=True)
    )
    assert out["spec_mode_change_ok"] and out["spec_off"] is True
    assert "spec_mode_change_ok" not in runner.apply_overrides(
        suites.parse_suite("identity"), Namespace()
    )
    a = cli.build_parser().parse_args(
        [
            "ab",
            "--base",
            "main",
            "--cand",
            "x",
            "--suite",
            "identity",
            "--label",
            "t",
            "--spec-mode-change-ok",
        ]
    )
    assert a.spec_mode_change_ok
