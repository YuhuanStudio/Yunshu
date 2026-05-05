"""OpenAI Completions API endpoint tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from yunshu_gateway.routers.completions import CompletionRequest


class TestCompletionRequest:
    def test_defaults(self):
        req = CompletionRequest(model="test", prompt="hello")
        assert req.model == "test"
        assert req.prompt == "hello"
        assert req.max_tokens == 128
        assert req.temperature == 0.7
        assert req.top_p == 1.0
        assert req.stream is False
        assert req.stop is None
        assert req.echo is False
        assert req.logprobs == 0
        assert req.seed is None
        assert req.repetition_penalty == 1.0
        assert req.frequency_penalty == 0.0
        assert req.presence_penalty == 0.0
        assert req.logit_bias is None

    def test_with_all_params(self):
        req = CompletionRequest(
            model="qwen3",
            prompt="test prompt",
            max_tokens=256,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.2,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            logit_bias={100: -10, 200: 5.0},
            stream=True,
            stop=["###"],
            echo=True,
            logprobs=5,
            seed=42,
        )
        assert req.temperature == 0.5
        assert req.top_k == 50
        assert req.min_p == 0.1
        assert req.repetition_penalty == 1.2
        assert req.frequency_penalty == 0.5
        assert req.presence_penalty == 0.3
        assert req.logit_bias == {100: -10, 200: 5.0}
        assert req.stream is True
        assert req.stop == ["###"]
        assert req.echo is True
        assert req.logprobs == 5
        assert req.seed == 42

    def test_prompt_as_token_ids(self):
        req = CompletionRequest(model="test", prompt=[1, 2, 3, 4, 5])
        assert req.prompt == [1, 2, 3, 4, 5]

    def test_prompt_as_list_int(self):
        req = CompletionRequest(model="test", prompt=[100, 200])
        assert isinstance(req.prompt, list)
