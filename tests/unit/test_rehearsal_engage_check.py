import importlib.util
import json
import pathlib

spec = importlib.util.spec_from_file_location(
    "rec",
    pathlib.Path(__file__).parents[2] / "scripts/research/rehearsal_engage_check.py",
)
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)


def rows(cached, completed, text="t"):
    base = dict(part="decode", ctx=1024, kind="prose", text=text)
    out = [
        dict(base, phase="cold", pt=1000, ttft_s=1.0),
        dict(base, phase="warm", pt=1000, ttft_s=0.1),
        dict(
            base,
            phase="turn2",
            pt=1100,
            cached=cached,
            ttft_s=0.5,
            rehearsals_completed=completed,
        ),
    ]
    return out


def test_pass_when_engaged_and_identical():
    problems, _ = rec.check(rows(1000, None), rows(1090, 1))
    assert problems == []


def test_fails_when_not_engaged():
    problems, _ = rec.check(rows(1000, None), rows(1000, 0))
    assert any("completed rehearsals" in p for p in problems)
    assert any("cached" in p for p in problems)


def test_fails_on_output_difference():
    problems, _ = rec.check(rows(1000, None, "a"), rows(1090, 1, "b"))
    assert any("differs" in p for p in problems)


def test_incomplete_file_fails_closed(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text(json.dumps(dict(part="decode")) + "\n")
    g = tmp_path / "y.jsonl"
    g.write_text("")
    assert rec.main([str(f), str(g)]) == 1
