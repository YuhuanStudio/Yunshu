import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "rr", pathlib.Path(__file__).parents[2] / "scripts/research/rehearsal_retry.py"
)
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)


def test_judge_passes_a_full_hit_and_fails_a_partial_one():
    first = {"pt": 1000, "cached": 0}
    assert rr.judge(first, {"pt": 1000, "cached": 999}) == []
    bad = rr.judge(first, {"pt": 41375, "cached": 28159})
    assert bad and "68%" in bad[0]


def test_judge_fails_closed_without_token_counts():
    assert rr.judge({}, {"pt": 10, "cached": 10})
    assert rr.judge({"pt": 10}, {})
