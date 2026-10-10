import pytest

mx = pytest.importorskip("mlx.core")

from yunshu_engine.tree_verify import conv_tail  # noqa: E402


@pytest.mark.parametrize(
    "path", [[0], [0, 2], [0, 1, 4], [0, 3, 5, 9], [0, 1, 2, 3, 4, 5, 6, 7]]
)
def test_conv_tail_equals_the_concatenate_and_gather(path):
    mx.random.seed(len(path))
    prev = mx.random.normal((1, 3, 6))
    mixed = mx.random.normal((1, 16, 6))
    conv_idx = mx.array(([0, 1, 2] + [3 + r for r in path])[-3:], dtype=mx.int32)
    tail_idx = mx.array(path[-3:], dtype=mx.int32) if len(path) >= 3 else None
    want = mx.take(mx.concatenate([prev, mixed], axis=1)[0], conv_idx, axis=0)[None]
    assert mx.array_equal(conv_tail(prev, mixed, path, conv_idx, tail_idx), want).item()
