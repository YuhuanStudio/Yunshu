"""apc_manager._sequence_hash (ids -> bytes once) equals mlx_vlm.apc._sequence_hash."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_vlm.apc import _sequence_hash as upstream

from yunshu_engine.apc_manager import _sequence_hash


@pytest.mark.parametrize(
    "ids",
    [
        [],
        [0],
        [1, 2, 3],
        list(range(5000)),
        tuple(range(70000, 70100)),
        [-1, 2**31 - 1, 7],
        [np.int64(5), np.int32(9), True],
    ],
)
@pytest.mark.parametrize("extra", [0, 12345, -7, 2**70])
@pytest.mark.parametrize("block", [16, 256])
def test_same_digest(ids, extra, block):
    assert _sequence_hash(ids, extra, block) == upstream(ids, extra, block)


def test_other_inputs_fall_back_to_upstream():
    arr = mx.array([4, 5, 6], dtype=mx.int32)
    assert _sequence_hash(arr.tolist(), 1, 16) == upstream([4, 5, 6], 1, 16)
    assert _sequence_hash(iter([4, 5, 6]), 1, 16) == upstream([4, 5, 6], 1, 16)
