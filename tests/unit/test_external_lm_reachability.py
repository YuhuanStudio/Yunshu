"""The legacy 'eagle' route accepts an ordinary LM, without EAGLE heads."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from yunshu_engine.batched_engine import BatchedEngine, GenerationOutput


def engine_with_plain_config(config):
    engine = BatchedEngine(model_name="plain-qwen2")
    engine._model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: config))
    engine._tokenizer = SimpleNamespace(
        eos_token_id=99, encode=lambda *_a, **_k: [1, 2, 3]
    )
    return engine


def route(engine, **kwargs):
    args = dict(
        spec_decode=True,
        stream=False,
        temperature=0.0,
        logprobs=False,
        use_engine_loop=False,
    )
    args.update(kwargs)
    return engine._spec_route(**args)


@pytest.mark.parametrize("path_source", ["setting", "config"])
@pytest.mark.parametrize("with_config", [False, True])
def test_plain_external_model_is_reachable(monkeypatch, path_source, with_config):
    from mlx_lm import utils

    path = "/ordinary/qwen2-draft"
    config = {"model_type": "qwen2"}
    monkeypatch.setenv("YUNSHU_SPEC_UNVERIFIED", "eagle")
    monkeypatch.setenv("YUNSHU_DRAFT_MODEL", path if path_source == "setting" else "")
    if path_source == "config":
        config["draft_model_path"] = path
    engine = engine_with_plain_config(config)
    draft = SimpleNamespace(config=SimpleNamespace(model_type="qwen2"))
    loaded = (
        (draft, engine._tokenizer, {}) if with_config else (draft, engine._tokenizer)
    )
    load = Mock(return_value=loaded)
    monkeypatch.setattr(utils, "load", load)
    engine._init_spec_decode()
    load.assert_called_once_with(path)
    assert engine._spec_decoder.draft is draft
    assert engine._spec_enabled is True
    assert route(engine) == "eagle"
    assert route(engine, stream=True) is None
    assert route(engine, temperature=0.7) is None
    assert route(engine, logprobs=True) is None
    assert route(engine, use_engine_loop=True) is None


def test_draft_path_alone_never_loads_a_second_model(monkeypatch):
    from mlx_lm import utils

    monkeypatch.setenv("YUNSHU_SPEC_UNVERIFIED", "")
    monkeypatch.setenv("YUNSHU_DRAFT_MODEL", "/ordinary/qwen2-draft")
    engine = engine_with_plain_config({"model_type": "qwen2"})
    load = Mock()
    monkeypatch.setattr(utils, "load", load)
    engine._init_spec_decode()
    load.assert_not_called()
    assert engine._spec_decoder is None
    assert route(engine) == "ngram"


@pytest.mark.asyncio
async def test_public_generate_selects_plain_external_draft(monkeypatch):
    from mlx_lm import utils

    monkeypatch.setenv("YUNSHU_SPEC_UNVERIFIED", "eagle")
    monkeypatch.setenv("YUNSHU_DRAFT_MODEL", "/ordinary/qwen2-draft")
    engine = engine_with_plain_config({"model_type": "qwen2"})
    monkeypatch.setattr(utils, "load", Mock(return_value=(object(), engine._tokenizer)))
    engine._init_spec_decode()
    engine._loaded = True
    expected = GenerationOutput(text="draft route", finished=True, finish_reason="stop")
    engine._generate_speculative = AsyncMock(return_value=expected)
    engine._generate_fast = AsyncMock()
    assert (
        await engine.generate(
            "hello", temperature=0.0, spec_decode=True, use_engine_loop=False
        )
        is expected
    )
    engine._generate_speculative.assert_awaited_once()
    engine._generate_fast.assert_not_called()


@pytest.mark.parametrize(
    "option",
    [
        {"json_schema": {"type": "object"}},
        {"logits_processors": [object()]},
        {"logit_bias": {2: -10}},
        {"stop": ["multi token stop"]},
        {"thinking_budget": 20},
        {"reasoning_effort": "high"},
        {"lora_adapter": "adapter"},
        {"repetition_penalty": 1.1},
        {"frequency_penalty": 0.1},
        {"presence_penalty": 0.1},
        {"top_p": 0.9},
        {"top_k": 10},
        {"min_p": 0.1},
        {"xtc_probability": 0.1},
        {"top_n_sigma": 2},
        {"min_tokens": 10},
        {"ignore_eos": True},
        {"suppress_tokens": [2]},
    ],
)
@pytest.mark.asyncio
async def test_external_draft_falls_back_for_unimplemented_parameters(
    monkeypatch, option
):
    monkeypatch.setenv("YUNSHU_SPEC_UNVERIFIED", "eagle")
    engine = engine_with_plain_config({"model_type": "qwen2"})
    engine._spec_decoder = object()
    engine._spec_enabled = True
    engine._loaded = True
    engine._generate_speculative = AsyncMock()
    expected = GenerationOutput(text="fast", finished=True, finish_reason="stop")
    engine._generate_fast = AsyncMock(return_value=expected)
    assert (
        await engine.generate(
            "hello", temperature=0.0, spec_decode=True, use_engine_loop=False, **option
        )
        is expected
    )
    engine._generate_speculative.assert_not_awaited()


@pytest.mark.asyncio
async def test_spec_prompt_array_and_seed_are_created_on_executor(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import mlx.core as mx

    from yunshu_engine import mlx_executor

    engine = engine_with_plain_config({"model_type": "qwen2"})
    thread_ids = []
    detok = SimpleNamespace(
        reset=lambda: None, add_token=lambda _t: None, finalize=lambda: None, text="ok"
    )
    engine._tokenizer.detokenizer = detok
    engine._spec_decoder = SimpleNamespace(
        constraint=None, generate=lambda **kw: [4, 5]
    )
    main_thread = threading.get_ident()

    def array(ids):
        thread_ids.append(threading.get_ident())
        assert threading.get_ident() != main_thread
        return SimpleNamespace(reshape=lambda *_: ids)

    monkeypatch.setattr(mx, "array", array)
    monkeypatch.setattr(
        mx.random, "seed", lambda _s: thread_ids.append(threading.get_ident())
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(mlx_executor, "get_mlx_executor", lambda: executor)
        result = await engine._generate_speculative("hello", max_tokens=2, seed=42)
    assert result.finish_reason == "length"
    assert len(thread_ids) == 2
    assert len(set(thread_ids)) == 1
