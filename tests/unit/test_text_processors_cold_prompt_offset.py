"""min_tokens and the frequency/presence penalties on a COLD prompt (no prefix-cache hit).

mlx-lm's ``generate_step`` hands its logits processors ``tokens`` = the WHOLE prompt that it
prefills itself, then the generated ids (older releases handed ``[last_prompt_token, gen1, ...]``).
The processors assumed the old shape (prompt offset 1), so on a cold request ``min_tokens`` saw
"enough tokens" at once (EOS was never masked) and the penalties counted the prompt. A prompt-cache
hit prefills only the last token, which is why the bug showed only on misses."""

import mlx.core as mx

from tests.unit.tiny_text_engine import make_engine

EOS = 1
PROMPT = "a" * 24  # FakeTokenizer: every 'a' is the same token (39)
A = 39


async def test_min_tokens_holds_on_a_cold_prompt():
    for floor in (1, 4):
        out = await make_engine()._generate_fast(
            prompt=PROMPT,
            max_tokens=10,
            temperature=0.0,
            logit_bias={EOS: 100.0},
            min_tokens=floor,
        )
        assert out.error is None
        assert out.completion_tokens == floor


def _greedy(model, ids, steps, freq, pres, bias, count_prompt):
    seq, counts, out = list(ids), {}, []
    if count_prompt:
        for t in ids:
            counts[t] = counts.get(t, 0) + 1
    for _ in range(steps):
        logits = model(mx.array([seq]))[0, -1].astype(mx.float32)
        for k, v in bias.items():
            logits[k] += v
        for t, c in counts.items():
            logits[t] -= freq * c + pres
        t = int(mx.argmax(logits).item())
        out.append(t)
        seq.append(t)
        counts[t] = counts.get(t, 0) + 1
    return out


async def test_penalties_count_only_generated_tokens_on_a_cold_prompt():
    steps, freq, pres, bias = 6, 0.9, 0.4, {A: 6.0}
    eng = make_engine()
    ids = eng._tokenizer.encode(PROMPT)
    mine = _greedy(eng._model, ids, steps, freq, pres, bias, count_prompt=False)
    wrong = _greedy(eng._model, ids, steps, freq, pres, bias, count_prompt=True)
    assert mine != wrong and mine[0] == A != wrong[0], "the reference must discriminate"

    out = await make_engine()._generate_fast(
        prompt=PROMPT,
        max_tokens=steps,
        temperature=0.0,
        frequency_penalty=freq,
        presence_penalty=pres,
        logit_bias=bias,
    )
    assert out.error is None
    assert out.completion_tokens == steps
    # the decoded text of the reference run (FakeTokenizer.decode) is the observable
    assert out.text == eng._tokenizer.decode(mine)


async def test_the_streaming_path_agrees():
    from tests.unit.tiny_text_engine import collect_stream

    kw = dict(prompt=PROMPT, max_tokens=6, temperature=0.0, logit_bias={EOS: 100.0})
    _, last = await collect_stream(make_engine(), min_tokens=3, **kw)
    assert last.completion_tokens == 3
    pen = dict(frequency_penalty=0.9, presence_penalty=0.4, logit_bias={A: 6.0})
    eng = make_engine()
    mine = _greedy(
        eng._model, eng._tokenizer.encode(PROMPT), 6, 0.9, 0.4, {A: 6.0}, False
    )
    text, _ = await collect_stream(
        make_engine(), prompt=PROMPT, max_tokens=6, temperature=0.0, **pen
    )
    assert text == eng._tokenizer.decode(mine)


def test_the_prompt_offset_is_read_off_the_first_call_in_both_mlx_lm_shapes():
    from yunshu_engine.text_logit_processors import (
        FrequencyPresencePenalty,
        TokenBan,
    )

    for prompt_len in (1, 7):  # after a cache hit / a cold prompt
        pen = FrequencyPresencePenalty(1.0, 2.0, n_prompt=None)
        ban = TokenBan([3], floor=2, n_prompt=None)
        toks = [5] * prompt_len  # prompt tokens: never counted
        for gen in range(4):
            logits = mx.zeros((1, 8))
            out = pen(list(toks), logits)
            assert out[0, 5].item() == 0.0, "prompt tokens are not penalised"
            if gen:
                assert out[0, 6].item() == -(1.0 * gen + 2.0)
            masked = ban(list(toks), logits)[0, 3].item() == float("-inf")
            assert masked == (gen < 2)
            toks.append(6)
