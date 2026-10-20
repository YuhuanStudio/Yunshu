"""A marked QuantizedLinear keeps one decode row's bits at any shared-batch size.

sg8 takes at most 8 rows; above that the stock multi-row QMM reduces differently, so a
ninth client changed a surviving row's arithmetic. Runs in a subprocess because
``batch_invariant.install`` patches ``nn.QuantizedLinear`` for the process.
"""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("mlx.core")

_SCRIPT = textwrap.dedent(
    """
    import mlx.core as mx
    import mlx.nn as nn
    from yunshu_engine.kernels import batch_invariant as bi

    mx.random.seed(1)
    lin = nn.Linear(2048, 1024, bias=False)
    lin.weight = (mx.random.normal(lin.weight.shape) * 0.02).astype(mx.bfloat16)
    q = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
    q.scales = q.scales.astype(mx.bfloat16)
    q.biases = q.biases.astype(mx.bfloat16)

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = q

    bi.install(LM())
    bi.set_active(True)
    x = mx.random.normal((32, 1, 2048)).astype(mx.bfloat16)
    ref = q(x[:1])[0]
    bad = [b for b in (2, 8, 9, 16, 17, 32) if not mx.array_equal(q(x[:b])[0], ref).item()]
    print("BAD", bad)
    """
)


def test_decode_row_bits_do_not_depend_on_batch_size():
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "BAD []" in out.stdout, out.stdout
