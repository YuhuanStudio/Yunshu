"""Logits processors inside a speculative window equal the serial step-by-step application."""

import mlx.core as mx
from mlx_lm.sample_utils import make_logits_processors

from yunshu_engine import constrained_spec as cs
from yunshu_engine.vlm_batch_runner import TokenMaskProcessor, spec_safe_processors


def _penalties():
    return list(
        make_logits_processors(
            logit_bias={3: 2.0, 9: -1.5},
            repetition_penalty=1.3,
            presence_penalty=0.7,
            frequency_penalty=0.4,
        )
    )


def _serial(processors, logits, context, drafted):
    """What the batch generator does: one step per input token, context grown by it."""
    rows = []
    for j in range(logits.shape[1]):
        history = mx.array(context + drafted[:j], dtype=mx.int32)
        row = logits[:, j, :]
        for p in processors:
            if hasattr(p, "process_last_token"):
                row = p.process_last_token(0, row)
            else:
                row = p(history, row)
        rows.append(row)
    return mx.stack(rows, axis=1)


def test_penalty_rows_match_serial_steps():
    mx.random.seed(3)
    logits = mx.random.normal((1, 5, 64)).astype(mx.bfloat16)
    context, drafted = [5, 6, 7, 7, 9], [7, 3, 3, 11]
    got = cs.apply_row_processors(_penalties(), logits, context, drafted, 0)
    want = _serial(_penalties(), logits, context, drafted)
    assert mx.array_equal(got, want).item()


def test_min_tokens_mask_counts_rows_like_serial_calls_and_advances():
    mx.random.seed(4)
    logits = mx.random.normal((1, 4, 32)).astype(mx.bfloat16)
    make = lambda: TokenMaskProcessor(eos_ids=[2], min_tokens=3)  # noqa: E731
    pure, serial = make(), make()
    got = cs.apply_row_processors([pure], logits, [1, 1], [4, 5, 6], 0)
    want = _serial([serial], logits, [1, 1], [4, 5, 6])
    assert mx.array_equal(got, want).item()
    assert pure._generated == 0  # applying a window never counts
    pure.advance(2)  # two tokens committed
    assert pure._generated == 2 and serial._generated == 4


def test_only_pure_processors_keep_the_speculative_lane():
    assert spec_safe_processors(_penalties())
    assert spec_safe_processors([TokenMaskProcessor(suppress=[1])])
    assert not spec_safe_processors([lambda tokens, logits: logits])

    class Stateful:
        def process_last_token(self, token, logits):
            return logits

    assert not spec_safe_processors([Stateful()])
