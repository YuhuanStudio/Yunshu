"""mlx-lm's generate_step hands its logits processors the WHOLE prefilled prompt suffix plus the
generated ids, so min_tokens and the frequency/presence penalties must count from that length
(not from 1). Real tiny engine, both text fast paths."""

from __future__ import annotations

import pytest

from yunshu_engine import text_logit_processors as tlp

from .tiny_text_engine import collect_stream, make_engine

PROMPT = "the quick brown fox jumps over the lazy dog!"  # one token per char


def _kw(**extra):
    # a huge bias on EOS (id 1) makes the model want to stop immediately
    return dict(prompt=PROMPT, temperature=0.0, logit_bias={1: 1000.0}, **extra)


async def test_min_tokens_holds_the_eos_ban_with_a_long_prompt_non_stream():
    out = await make_engine()._generate_fast(max_tokens=30, min_tokens=5, **_kw())
    assert out.error is None
    assert out.completion_tokens >= 5  # was 0: the ban never engaged


async def test_min_tokens_holds_the_eos_ban_with_a_long_prompt_stream():
    _, last = await collect_stream(make_engine(), max_tokens=30, min_tokens=5, **_kw())
    assert last.error is None
    assert last.completion_tokens >= 5


async def test_without_min_tokens_the_model_stops_at_once():
    out = await make_engine()._generate_fast(max_tokens=30, **_kw())
    assert out.completion_tokens == 0  # the bias test above is meaningful


@pytest.mark.parametrize("stream", [False, True])
async def test_penalties_count_generated_tokens_only(monkeypatch, stream):
    seen = []
    real = tlp.FrequencyPresencePenalty.__call__

    def spy(self, tokens, logits):
        seen.append((self.n_prompt, len(tokens)))
        return real(self, tokens, logits)

    monkeypatch.setattr(tlp.FrequencyPresencePenalty, "__call__", spy)
    kw = dict(
        prompt=PROMPT,
        max_tokens=4,
        temperature=0.0,
        frequency_penalty=0.5,
        presence_penalty=0.5,
    )
    if stream:
        await collect_stream(make_engine(), **kw)
    else:
        await make_engine()._generate_fast(**kw)
    assert seen
    n_prompt, first_len = seen[0]
    assert (
        first_len == n_prompt == len(PROMPT)
    )  # nothing generated yet: no prompt token is counted
    assert [ln - n for n, ln in seen] == list(range(len(seen)))
