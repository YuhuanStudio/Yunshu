import numpy as np
from scripts.research.bench_draft_gemm import batch_inputs, mx


def test_batched_probe_cannot_collapse_to_one_identical_matmul():
    x = mx.zeros((1, 4, 64), dtype=mx.bfloat16)
    xs = batch_inputs(x, 4)
    assert len(xs) == 4
    # Distinct data, not just Python array wrappers of the same graph.
    values = [np.array(value.astype(mx.float32)) for value in xs]
    assert all(
        not np.array_equal(a, b) for i, a in enumerate(values) for b in values[i + 1 :]
    )
