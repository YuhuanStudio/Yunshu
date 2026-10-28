"""suppress_tokens (the request's or the checkpoint's generation_config default) no longer takes a
request off the n-gram speculative route: the route bans the ids itself, and its output equals the
plain fast path's with the same ban. A request with min_tokens / ignore_eos still goes plain."""

import asyncio
import json

import mlx.core as mx
import pytest

pytest.importorskip("mlx_lm")

from mlx_lm.generate import generate_step  # noqa: E402

from yunshu_engine.text_logit_processors import TokenBan  # noqa: E402

from .test_ngram_default_routing import _engine as _routing_engine  # noqa: E402
from .test_ngram_default_routing import _route  # noqa: E402
from .test_ngram_spec_lossless import _engine, _tiny_model, _Tok  # noqa: E402

PROMPT = "abcabcabcabc"
N = 40


def _plain(model, ban=()):
    tok = _Tok()
    out = []
    kw = {"logits_processors": [TokenBan(ban)]} if ban else {}
    for token, _ in generate_step(
        mx.array(tok.encode(PROMPT)),
        model,
        max_tokens=N,
        sampler=lambda x: mx.argmax(x, axis=-1),
        **kw,
    ):
        if int(token) == tok.eos_token_id:
            break
        out.append(int(token))
    return out


def test_ngram_route_bans_suppressed_ids_and_matches_plain_greedy():
    model = _tiny_model(0)
    base = _plain(model)
    ban = sorted(set(base))[:2]  # ids the unconstrained continuation actually uses
    ref = _plain(model, ban)
    assert ref != base and not set(ref) & set(ban)
    eng = _engine(model)
    out = asyncio.run(
        eng._generate_ngram_spec(
            prompt=PROMPT, max_tokens=N, temperature=0.0, suppress_tokens=ban
        )
    )
    assert out.error is None
    assert out.text == _Tok().decode(ref)
    assert eng._ngram_stats["proposals"] > 0  # it really speculated


async def test_suppress_tokens_keep_the_ngram_route():
    eng = _routing_engine(greedy_default=True)
    assert await _route(eng, temperature=0.0, suppress_tokens=[5]) == "ngram"


async def test_checkpoint_default_suppress_keeps_the_ngram_route(tmp_path):
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"suppress_tokens": [5, 6]})
    )
    eng = _routing_engine(greedy_default=True)
    eng.model_name = str(tmp_path)
    assert await _route(eng, temperature=0.0) == "ngram"  # was "fast"


def test_hub_id_reads_generation_config_from_the_resolved_snapshot(
    tmp_path, monkeypatch
):
    import mlx_lm.utils

    from yunshu_engine import generation_suppress as gs

    (tmp_path / "generation_config.json").write_text(
        json.dumps({"suppress_tokens": [7, 3]})
    )
    seen = []
    monkeypatch.setattr(
        mlx_lm.utils, "hf_repo_to_path", lambda r: seen.append(r) or tmp_path
    )
    monkeypatch.setattr(gs, "_DIRS", {})
    assert gs.default_suppress_ids("org/some-model") == [
        3,
        7,
    ]  # was []: not a local dir
    assert seen == ["org/some-model"]
    # an unresolvable name still means "no default", not an error
    monkeypatch.setattr(
        mlx_lm.utils,
        "hf_repo_to_path",
        lambda r: (_ for _ in ()).throw(RuntimeError("not cached")),
    )
    assert gs.default_suppress_ids("org/other") == []


async def test_min_tokens_and_ignore_eos_still_take_the_plain_path():
    eng = _routing_engine(greedy_default=True)
    assert await _route(eng, temperature=0.0, min_tokens=3) == "fast"
    assert await _route(eng, temperature=0.0, ignore_eos=True) == "fast"
