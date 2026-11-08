"""Fail-closed CPU checks of the real-server oracle validator."""

import pytest
from scripts.research.t1_prompt_logprobs import compare


def test_reference_validator():
    scores = [
        None,
        {
            "token_id": 2,
            "logprob": -1.25,
            "top_logprobs": [{"token_id": 2, "logprob": -1.25}],
        },
    ]
    assert compare(scores, scores) == 0
    with pytest.raises(AssertionError):
        compare(scores[:-1], scores)
    wrong = [
        None,
        {"token_id": 2, "logprob": -1.0, "top_logprobs": scores[1]["top_logprobs"]},
    ]
    with pytest.raises(AssertionError):
        compare(wrong, scores)


def test_reference_tolerance_is_explicit_and_bounded():
    ref = [None, {"token_id": 2, "logprob": -1.0, "top_logprobs": []}]
    got = [None, {"token_id": 2, "logprob": -1.000001, "top_logprobs": []}]
    with pytest.raises(AssertionError):
        compare(got, ref)
    assert compare(got, ref, tolerance=1e-5) < 1e-5
    with pytest.raises(AssertionError):
        compare(got, ref, tolerance=1e-8)
