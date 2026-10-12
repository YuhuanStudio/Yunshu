import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "pid", pathlib.Path(__file__).parents[2] / "scripts/research/preempt_identity.py"
)
pid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pid)


def test_verdict_passes_only_on_equal_chunks_and_real_evals():
    assert pid.verdict(["a", "b"], ["a", "b"], 32)["pass"]


def test_verdict_fails_on_a_difference():
    v = pid.verdict(["a", "b"], ["a", "x"], 32)
    assert not v["pass"] and "1" in v["problems"][0]


def test_verdict_fails_when_the_hook_never_evaluated():
    v = pid.verdict(["a"], ["a"], 0)
    assert not v["pass"] and "not engaged" in v["problems"][0]


def test_verdict_fails_closed_on_empty():
    assert not pid.verdict([], [], 5)["pass"]
