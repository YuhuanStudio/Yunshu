"""generation_config.json ``suppress_tokens`` is the text path's default mask (as on the VLM path);
a request's own list wins. Run end to end over the tiny Llama in both entry points."""

from __future__ import annotations

import json

from yunshu_engine.generation_suppress import default_suppress_ids

from .tiny_text_engine import FakeTokenizer, make_engine

PROMPT = "the quick brown fox jumps over the lazy dog"


def _ban_ids_for(letters):
    return [i for i in range(2, 64) if FakeTokenizer().decode([i]) in letters]


async def _gen(eng, **kw):
    out = await eng.generate(prompt=PROMPT, max_tokens=8, temperature=0.0, **kw)
    assert out.error is None
    return out.text


async def _stream(eng, **kw):
    text = ""
    async for o in eng.stream_generate(
        prompt=PROMPT, max_tokens=8, temperature=0.0, **kw
    ):
        assert o.error is None
        text += o.new_text or ""
    return text


async def test_model_default_suppress_applies_to_generate_and_stream(tmp_path):
    eng = make_engine()
    base = await _gen(eng)
    assert base
    banned = _ban_ids_for(set(base))
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"suppress_tokens": banned + [-1, 5000]})
    )
    eng.model_name = str(tmp_path)
    got = await _gen(eng)
    assert got and not (set(got) & set(base))
    streamed = await _stream(eng)
    assert streamed and not (set(streamed) & set(base))


async def test_request_suppress_tokens_win_over_the_model_default(tmp_path):
    eng = make_engine()
    base = await _gen(eng)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"suppress_tokens": _ban_ids_for(set(base))})
    )
    eng.model_name = str(tmp_path)
    # the request bans an id that no output uses: the model default must NOT also apply
    assert await _gen(eng, suppress_tokens=[63]) == base


def test_default_ids_drop_out_of_vocab_and_bad_entries(tmp_path):
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"suppress_tokens": [9, 5, True, "x", -3, 9, 100]})
    )
    assert default_suppress_ids(tmp_path, FakeTokenizer()) == [5, 9]
    assert default_suppress_ids(tmp_path / "missing", FakeTokenizer()) == []
