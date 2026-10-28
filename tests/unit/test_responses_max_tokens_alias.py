"""/v1/responses takes ``max_tokens`` as an alias of ``max_output_tokens`` (Rapid-MLX 3de1c6c4)
instead of dropping it into extra fields and running to the model's output limit."""

import pytest
from pydantic import ValidationError

from yunshu_gateway.routers.responses import ResponsesRequest


def _req(**kw):
    return ResponsesRequest(model="m", input="hi", **kw)


def test_max_tokens_bounds_the_reply():
    assert _req(max_tokens=33).max_output_tokens == 33


def test_equal_values_are_fine_and_conflicts_are_a_400():
    assert _req(max_tokens=20, max_output_tokens=20).max_output_tokens == 20
    with pytest.raises(ValidationError, match="conflicts with max_output_tokens"):
        _req(max_tokens=20, max_output_tokens=30)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_is_rejected(bad):
    with pytest.raises(ValidationError):
        _req(max_tokens=bad)
    with pytest.raises(ValidationError):
        _req(max_output_tokens=bad)
