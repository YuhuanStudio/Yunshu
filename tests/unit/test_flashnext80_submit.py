import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import flashnext80_submit as fs  # noqa: E402

SCRIPT = str(
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "flashnext80_footprint.py"
)


def cmd(*extra):
    return ["python", SCRIPT, "--tree", "/t", "--model", "/m", "--out", "/o", *extra]


def test_accepts_valid_and_rejects_bad_arguments():
    assert (
        fs.check_command(cmd("--env", "A=B", "--env", "C=D", "--require-ple")) is None
    )
    assert fs.check_command(cmd("--no-such-flag")) is not None
    # the zsh word-splitting bug: one argument holding two words
    assert fs.check_command(cmd("--env YUNSHU_VLM_DRAFT=off")) is not None


def test_split_and_refusal(monkeypatch, capsys):
    options, command = fs.split_command(["--label", "x", "--", "python", "a.py"])
    assert options == ["--label", "x"] and command == ["python", "a.py"]
    called = []
    monkeypatch.setattr(fs.subprocess, "call", lambda *a, **k: called.append(a) or 0)
    assert fs.main(["--label", "x", "--", *cmd("--bogus")]) == 2
    assert not called and "refusing" in capsys.readouterr().err
    assert fs.main(["--label", "x", "--", *cmd()]) == 0 and called
