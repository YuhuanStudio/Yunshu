"""VLM frequency / presence penalty follow OpenAI: every GENERATED token counts, not mlx-lm's
window of the last 20 prompt + generated tokens. repetition_penalty keeps the mlx-lm window.

The speculative lane applies the penalty to verify rows with ``window_row`` / ``commit_tokens``;
those rows must equal the serial ``process_last_token`` steps bit for bit (spec on == spec off).
"""

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine.constrained_spec import (
    ProcessorWindow,
    SpecRequest,
    apply_row_processors,
)
from yunshu_engine.text_logit_processors import FrequencyPresencePenalty
from yunshu_engine.vlm_batch_runner import (
    TokenMaskProcessor,
    build_penalty_processors,
    spec_safe_processors,
)

VOCAB = 97


def _generated_penalty():
    from yunshu_engine.text_logit_processors import GeneratedTokenPenalty  # noqa: PLC0415

    return GeneratedTokenPenalty


def test_openai_semantics_through_the_generator_interface():
    """Drive the built processors the way the batch generator does (``process_last_token`` when
    present, else the full token history) and compare with a numpy count of ALL generated tokens."""
    prompt = [3, 4, 5]
    generated = [7, *range(20, 60), 7, 59]
    procs = build_penalty_processors(1.0, 0.5, 0.25, None)
    logits = mx.zeros((1, VOCAB))
    out = None
    for i, g in enumerate(generated):
        out = logits
        for p in procs:
            if hasattr(p, "process_last_token"):
                out = p.process_last_token(g, out)
            else:
                out = p(mx.array([*prompt, *generated[: i + 1]]), out)
    counts = np.zeros(VOCAB)
    for g in generated:
        counts[g] += 1
    want = -0.5 * counts - 0.25 * (counts > 0)
    np.testing.assert_allclose(np.array(out)[0], want, atol=1e-6)


def _logits(seed, rows, dtype):
    mx.random.seed(seed)
    return mx.random.normal((1, rows, VOCAB)).astype(dtype)


def _serial(proc, generated, logits):
    """The batch generator: one step per consumed token, each with that token's logits row."""
    return [
        proc.process_last_token(g, logits[:, i, :]) for i, g in enumerate(generated)
    ]


def test_every_generated_token_counts_not_the_last_twenty():
    proc = build_penalty_processors(1.0, 0.5, 0.25, None)
    assert len(proc) == 1 and isinstance(proc[0], _generated_penalty())
    p = proc[0]
    logits = mx.zeros((1, VOCAB))
    out = None
    # token 7 once, then 40 other tokens: mlx-lm's 20-token window would have forgotten it
    for t in [7, *range(20, 60)]:
        out = p.process_last_token(t, logits)
    row = np.array(out)[0]
    assert row[7] == pytest.approx(-(0.5 * 1 + 0.25))  # one count + presence
    assert row[59] == pytest.approx(-(0.5 * 1 + 0.25))
    assert row[3] == 0.0
    # a repeated token accumulates the frequency term
    out = p.process_last_token(59, logits)
    assert np.array(out)[0][59] == pytest.approx(-(0.5 * 2 + 0.25))


def test_prompt_tokens_are_not_counted():
    p = _generated_penalty()(1.0, 1.0)
    logits = mx.zeros((1, VOCAB))
    out = p(mx.array([1, 2, 3, 4]), logits)  # the prompt pass
    assert mx.array_equal(out, logits).item()
    out = p.process_last_token(9, logits)
    assert np.array(out)[0][1] == 0.0 and np.array(out)[0][9] == -2.0


def test_negative_penalties_boost_like_the_text_path():
    p = _generated_penalty()(-0.5, -1.0)
    out = p.process_last_token(4, mx.zeros((1, VOCAB)))
    assert np.array(out)[0][4] == pytest.approx(1.5)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_same_values_as_the_text_path_processor(dtype):
    gen = [5, 5, 9, 70, 5, 9, 1, 1, 1]
    logits = _logits(1, len(gen), dtype)
    mine = _generated_penalty()(0.6, 0.3)
    text = FrequencyPresencePenalty(0.6, 0.3, n_prompt=3)
    tokens = [11, 12, 13]  # prompt
    for i, g in enumerate(gen):
        got = mine.process_last_token(g, logits[:, i, :])
        tokens = [*tokens, g]
        want = text(mx.array(tokens), logits[:, i, :])
        np.testing.assert_allclose(
            np.array(got.astype(mx.float32)),
            np.array(want.astype(mx.float32)),
            rtol=1e-2 if dtype == mx.bfloat16 else 1e-6,
            atol=1e-2 if dtype == mx.bfloat16 else 1e-6,
        )


def test_repetition_penalty_stays_mlx_lm_windowed():
    procs = build_penalty_processors(1.3, 0.0, 0.0, None)
    assert len(procs) == 1
    assert procs[0].__module__.startswith("mlx_lm.sample_utils")
    mixed = build_penalty_processors(1.3, 0.4, 0.0, {3: 1.0})
    kinds = [type(p).__name__ for p in mixed]
    assert kinds[0] == "LogitBiasProcessor" and kinds[-1] == "GeneratedTokenPenalty"
    assert len(mixed) == 3
    assert build_penalty_processors(1.0, 0.0, 0.0, None) == []


def test_penalty_keeps_the_speculative_lane():
    assert spec_safe_processors(build_penalty_processors(1.3, 0.4, 0.7, {3: 2.0}))


def _windowed(processors, prompt, generated, logits, plan):
    """Replay ``generated`` as speculative windows. ``plan`` is a list of
    ``(window_size, wrong_draft_at)``; a wrong draft makes the window accept fewer tokens."""
    request = SpecRequest()
    request.context = list(prompt)
    request.processors = processors
    window = ProcessorWindow.for_request(request, generated[0])
    rows = []
    s = 0
    calls = 0
    for size, wrong in plan:
        size = min(size, len(generated) - s)
        drafted = list(generated[s + 1 : s + size])
        if wrong is not None and wrong < len(drafted):
            drafted[wrong] = (drafted[wrong] + 1) % VOCAB  # a rejected draft
        out = window.apply(logits[:, s : s + size, :], drafted, calls + 1)
        accepted = wrong if wrong is not None and wrong < len(drafted) else size - 1
        for j in range(accepted + 1):
            rows.append(out[:, j, :])
        tokens = list(generated[s + 1 : s + accepted + 2])
        window.commit(tokens)
        calls += accepted + 1
        s += accepted + 1
        if s >= len(generated):
            break
    return rows, s


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_verify_rows_equal_serial_steps_across_windows(dtype):
    prompt = [3, 4, 5, 6, 7, 3]
    generated = [8, 8, 40, 3, 8, 40, 41, 3, 8, 8, 9, 40, 3, 3, 8, 41, 42, 43, 3, 8, 9]
    logits = _logits(7, len(generated), dtype)

    def procs():
        return build_penalty_processors(1.3, 0.4, 0.7, {3: 2.0, 9: -1.5})

    # serial reference with the same processor objects the batch generator would call
    serial_procs = procs()
    want = []
    for i, g in enumerate(generated):
        row = logits[:, i, :]
        history = mx.array([*prompt, *generated[: i + 1]], dtype=mx.int32)
        for p in serial_procs:
            row = (
                p.process_last_token(g, row)
                if hasattr(p, "process_last_token")
                else p(history, row)
            )
        want.append(row)

    plan = [
        (4, None),
        (3, 1),
        (5, None),
        (2, 0),
        (6, 2),
        (4, None),
        (6, None),
        (6, None),
    ]
    got, upto = _windowed(procs(), prompt, generated, logits, plan)
    assert upto >= 12
    for i, row in enumerate(got):
        assert mx.array_equal(row, want[i]).item(), f"row {i}"


def test_window_rows_do_not_count_until_committed():
    p = _generated_penalty()(0.5, 0.5)
    logits = mx.zeros((1, 3, VOCAB))
    a = apply_row_processors([p], logits, [1, 2, 9], [4, 4], 0)
    b = apply_row_processors([p], logits, [1, 2, 9], [4, 4], 0)
    assert mx.array_equal(a, b).item()  # applying a window changes nothing
    assert p._cnt is None or float(mx.sum(p._cnt).item()) == 0.0


def test_mask_and_penalty_compose_in_a_window():
    prompt = [1, 2]
    generated = [6, 6, 7, 6]
    logits = _logits(2, len(generated), mx.bfloat16)

    def make():
        return [
            _generated_penalty()(0.4, 0.2),
            TokenMaskProcessor(suppress=[5], eos_ids=[0], min_tokens=2),
        ]

    serial = make()
    want = []
    for i, g in enumerate(generated):
        row = logits[:, i, :]
        for p in serial:
            row = p.process_last_token(g, row)
        want.append(row)
    got, _ = _windowed(make(), prompt, generated, logits, [(4, None)])
    assert all(mx.array_equal(g, w).item() for g, w in zip(got, want, strict=False))
