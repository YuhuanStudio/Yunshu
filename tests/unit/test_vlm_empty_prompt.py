"""A prompt that renders to zero tokens is refused before it can reach BatchGenerator.insert."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from tests.unit.test_vlm_runner_batching import FakeGen, runner  # noqa: F401,F811
from yunshu_engine.context_window import EmptyPromptError
from yunshu_engine.vlm_engine import VLMEngine


def test_runner_refuses_an_empty_prompt_before_queueing(runner):  # noqa: F811
    with pytest.raises(EmptyPromptError, match="empty prompt"):
        list(runner.iter_tokens([], max_tokens=3, prompt_kwargs={}))
    assert not runner.busy()
    assert all(g.next_uid == 0 for g in FakeGen.instances)


def test_engine_refuses_an_empty_tokenization():
    eng = SimpleNamespace(
        _tokenize_with_cache=lambda *a, **k: mx.array([], dtype=mx.int32),
        _non_empty_ids=VLMEngine._non_empty_ids,
    )
    with pytest.raises(EmptyPromptError, match="empty prompt"):
        VLMEngine._runner_input(
            eng, [{"role": "user", "content": ""}], [], [], None, None
        )
