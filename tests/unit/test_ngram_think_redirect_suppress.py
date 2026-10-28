"""The json_schema + open <think> redirect from the n-gram route to the plain fast path must carry
suppress_tokens (it forwarded every other field and dropped the ban)."""

from yunshu_engine.batched_engine import GenerationOutput

from .test_spec_routes_behaviour import CLOSE, OPEN, _spec_engine
from .tiny_text_engine import FakeTokenizer


class _ThinkTok(FakeTokenizer):
    think_start_id = OPEN

    def encode(self, text, add_special_tokens=True, **kw):
        if text == "<think>":
            return [OPEN]
        if text == "</think>":
            return [CLOSE]
        return super().encode(text, add_special_tokens=add_special_tokens)


async def test_redirect_forwards_suppress_tokens():
    eng = _spec_engine(0)
    eng._tokenizer = _ThinkTok()
    seen = {}

    async def fake_fast(**kw):
        seen.update(kw)
        return GenerationOutput(text="plain", finished=True)

    eng._generate_fast = fake_fast
    await eng._generate_ngram_spec(
        prompt="hello:",
        max_tokens=8,
        temperature=0.0,
        json_schema={"type": "choice", "choices": ["ab"]},
        suppress_tokens=[7],
    )
    assert seen.get("suppress_tokens") == [7]
