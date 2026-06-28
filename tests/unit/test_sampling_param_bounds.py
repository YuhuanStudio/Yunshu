"""a grounded sampling-param-validation hunt found the four request models
well-hardened, with two cross-layer/cross-endpoint inconsistencies fixed here:

- xtc_threshold: all 4 routers declared Field(le=1.0), but the engine hard-requires
  [0, 0.5] (raises ValueError deep in the GPU sampler) → a request with xtc_probability>0
  and xtc_threshold in (0.5, 1.0] passed Pydantic then 500'd instead of a clean 422.
  Tightened all 4 to le=0.5.
- completions `seed` had no 64-bit bound (chat.py:449 does) → an out-of-range seed could
  overflow the downstream numpy/Gumbel PRNG. Added parity bound.

(The /batch raw-dict path HIGH and image-cancel MEDIUM from the same hunt round are
separate waves.)
"""
from __future__ import annotations

import pydantic
import pytest

from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest
from yunshu_gateway.routers.chat import ChatCompletionRequest
from yunshu_gateway.routers.completions import CompletionRequest
from yunshu_gateway.routers.responses import ResponsesRequest


def test_xtc_threshold_rejected_above_half_all_endpoints():
    # > 0.5 must now be a validation error (was accepted → engine 500)
    with pytest.raises(pydantic.ValidationError):
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], xtc_threshold=0.8)
    with pytest.raises(pydantic.ValidationError):
        CompletionRequest(model="m", prompt="hi", xtc_threshold=0.8)
    with pytest.raises(pydantic.ValidationError):
        ResponsesRequest(model="m", input="hi", xtc_threshold=0.8)
    with pytest.raises(pydantic.ValidationError):
        AnthropicMessagesRequest(model="m", messages=[{"role": "user", "content": "hi"}],
                                 max_tokens=16, xtc_threshold=0.8)


def test_xtc_threshold_half_still_allowed():
    # exactly 0.5 is the engine's upper bound — must remain valid
    assert CompletionRequest(model="m", prompt="hi", xtc_threshold=0.5).xtc_threshold == 0.5


def test_completions_seed_64bit_bound():
    with pytest.raises(pydantic.ValidationError):
        CompletionRequest(model="m", prompt="hi", seed=2**63)
    with pytest.raises(pydantic.ValidationError):
        CompletionRequest(model="m", prompt="hi", seed=-(2**63) - 1)
    # in-range is fine
    assert CompletionRequest(model="m", prompt="hi", seed=12345).seed == 12345
