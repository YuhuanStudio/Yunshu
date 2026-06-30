"""(HIGH): the /batch item `body` is a raw unvalidated dict, so every sampling
param except max_tokens/logit_bias was forwarded to the engine with NO range check —
bypassing the Field(ge/le) bounds the chat/completions endpoints enforce. An out-of-range
value (presence_penalty 1e18, min_p 5.0, xtc_threshold 0.8) produced degenerate/garbage
output for that item, and temperature<0 was silently coerced to greedy (a no-op, not a
422). New _validate_batch_sampling validates to the same bounds; a bad row raises ValueError
(the batch per-item error contract) so it fails cleanly instead of emitting junk.
"""

from __future__ import annotations

import pytest

from yunshu_gateway.routers.batch_inference import _validate_batch_sampling


@pytest.mark.parametrize(
    "body",
    [
        {"presence_penalty": 1e18},
        {"presence_penalty": -3.0},
        {"frequency_penalty": 5.0},
        {"min_p": 5.0},
        {"top_p": 1.5},
        {"temperature": -1.0},
        {"temperature": 3.0},
        {"xtc_threshold": 0.8},  # engine hard-requires [0, 0.5]
        {"xtc_probability": 2.0},
        {"repetition_penalty": 9.0},
        {"top_k": -1},
        {"n": 0},
        {"n": 2},  # batch is one completion per item; n>1 rejected
        {"seed": 2**63},
        {"temperature": float("nan")},
        {"min_p": float("inf")},
    ],
)
def test_out_of_range_params_rejected(body):
    with pytest.raises(ValueError):
        _validate_batch_sampling(body)


def test_valid_params_and_defaults_pass():
    # empty body → all defaults in-range
    _validate_batch_sampling({})
    # a fully-specified in-range body
    _validate_batch_sampling(
        {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.5,
            "presence_penalty": -0.5,
            "xtc_probability": 0.3,
            "xtc_threshold": 0.4,
            "n": 1,
            "seed": 12345,
        }
    )
