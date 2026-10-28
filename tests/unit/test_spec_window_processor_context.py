"""Verify-window processors (mtp / dflash lanes, ProcessorWindow) see only the context tail the
penalties read: logits stay bit-identical to running every row on the full prompt history."""

import mlx.core as mx
import pytest

from yunshu_engine.constrained_spec import PENALTY_CONTEXT, apply_row_processors
from yunshu_engine.vlm_batch_runner import TokenMaskProcessor, build_penalty_processors


def _reference(processors, logits, context, drafted, calls):
    rows = []
    for j in range(logits.shape[1]):
        row = logits[:, j, :]
        hist = mx.array(list(context) + list(drafted[:j]), dtype=mx.int32)
        for proc in processors:
            if hasattr(proc, "pure_at"):
                row = proc.pure_at(calls + j + 1, row)
            elif hasattr(proc, "window_row"):
                row = proc.window_row([context[-1], *drafted[:j]], row)
            else:
                row = proc(hist, row)
        rows.append(row)
    return mx.stack(rows, axis=1)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("n_ctx", [5, PENALTY_CONTEXT, 300, 5000])
def test_tail_context_is_bit_identical(dtype, n_ctx):
    mx.random.seed(11)
    vocab = 97
    context = [int(t) for t in mx.random.randint(0, vocab, (n_ctx,)).tolist()]
    drafted = [context[-3], 5, 5]
    logits = mx.random.normal((1, 4, vocab)).astype(dtype)
    procs = build_penalty_processors(1.3, 0.4, 0.7, {3: 2.0, 9: -1.5})
    procs.append(TokenMaskProcessor(suppress=[1, 2], eos_ids=[4], min_tokens=6))
    got = apply_row_processors(procs, logits, context, drafted, 2)
    want = _reference(procs, logits, context, drafted, 2)
    assert mx.array_equal(got, want).item()


def test_unknown_processor_still_gets_full_history():
    seen = []

    def spy(tokens, logits):
        seen.append(tokens.size)
        return logits

    context = list(range(100))
    apply_row_processors([spy], mx.zeros((1, 2, 8)), context, [1], 0)
    assert seen == [100, 101]
