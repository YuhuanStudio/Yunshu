"""HumanEval check_answer must not pass an arbitrary long answer."""

from __future__ import annotations

from yunshu_cli.eval import HumanEvalBenchmark


def test_long_wrong_answer_is_not_correct():
    b = HumanEvalBenchmark()
    item = {"canonical_solution": "    return a + b\n"}
    assert not b.check_answer("def add(a, b):\n    return a - b  # wrong", item)


def test_canonical_match_is_correct():
    b = HumanEvalBenchmark()
    item = {"canonical_solution": "    return a + b\n"}
    assert b.check_answer("return a + b", item)
