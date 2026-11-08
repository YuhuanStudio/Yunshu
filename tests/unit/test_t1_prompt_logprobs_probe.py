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
