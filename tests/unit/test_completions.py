"""OpenAI Completions API endpoint tests."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

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


class TestCompletionsEdgeCases:
    """Edge case tests for the Completions API."""

    def test_empty_prompt_string_rejected(self):
        """Empty string prompt should return 422 validation error."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            CompletionRequest(model="test", prompt="")

    def test_whitespace_only_prompt_rejected(self):
        """Whitespace-only prompt should return 422 validation error."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            CompletionRequest(model="test", prompt="   ")

    def test_empty_prompt_list_rejected(self):
        """Empty list prompt should return 422 validation error."""
        with pytest.raises(ValidationError, match="cannot be an empty list"):
            CompletionRequest(model="test", prompt=[])

    def test_max_tokens_zero_allowed(self):
        """max_tokens=0 should be accepted (returns prompt_tokens only)."""
        req = CompletionRequest(model="test", prompt="hello", max_tokens=0)
        assert req.max_tokens == 0
        assert req.effective_max_tokens() == 0

    def test_max_completion_tokens_zero_allowed(self):
        """max_completion_tokens=0 should be accepted."""
        req = CompletionRequest(model="test", prompt="hello", max_completion_tokens=0)
        assert req.effective_max_tokens() == 0

    def test_max_tokens_one_allowed(self):
        """max_tokens=1 should generate exactly 1 token."""
        req = CompletionRequest(model="test", prompt="hello", max_tokens=1)
        assert req.max_tokens == 1

    def test_n_gt_1_with_streaming_rejected(self):
        """n > 1 with streaming should return validation error."""
        with pytest.raises(
            ValidationError, match="n > 1 is not supported when stream is True"
        ):
            CompletionRequest(
                model="test",
                prompt="hello",
                n=3,
                stream=True,
            )

    def test_n_eq_1_with_streaming_allowed(self):
        """n=1 with streaming should be accepted."""
        req = CompletionRequest(
            model="test",
            prompt="hello",
            n=1,
            stream=True,
        )
        assert req.n == 1
        assert req.stream is True

    def test_n_gt_1_non_streaming_allowed(self):
        """n > 1 without streaming should be accepted."""
        req = CompletionRequest(
            model="test",
            prompt="hello",
            n=3,
            stream=False,
        )
        assert req.n == 3

    def test_max_tokens_zero_endpoint_returns_prompt_only(self):
        """Completions endpoint with max_tokens=0 returns empty completion."""

        from yunshu_gateway.routers.completions import create_completion

        req = CompletionRequest(model="test", prompt="hello world", max_tokens=0)

        # Mock the FastAPI request
        mock_request = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.state = MagicMock()
        mock_request.state.rbac_key = None
        mock_request.state.request_id = "test"

        # Mock engine with tokenizer
        mock_engine = MagicMock()
        mock_engine.is_loaded = False
        mock_engine.model_name = "other-model"
        mock_engine.resolve_model_id.return_value = False

        with (
            patch(
                "yunshu_gateway.routers.completions.get_engine",
                return_value=mock_engine,
            ),
            patch(
                "yunshu_gateway.routers.completions.get_engine_for_model",
                side_effect=Exception("no model"),
            ),
        ):
            # Even with no engine loaded, max_tokens=0 should return a response
            # (it hits the fast path before engine lookup)
            import asyncio

            result = asyncio.new_event_loop().run_until_complete(
                create_completion(req, mock_request)
            )
            # Should return JSONResponse with prompt_tokens and completion_tokens=0
            assert result.status_code == 200
            import json

            body = json.loads(result.body)
            assert body["usage"]["completion_tokens"] == 0
            assert body["choices"][0]["finish_reason"] == "length"


class TestPromptLogprobsSchemaW737:
    """CompletionRequest accepts prompt_logprobs (eval/perplexity)."""

    def test_accepts_prompt_logprobs(self):
        from yunshu_gateway.routers.completions import CompletionRequest

        r = CompletionRequest(model="m", prompt="hi", prompt_logprobs=5)
        assert r.prompt_logprobs == 5

    def test_default_none(self):
        from yunshu_gateway.routers.completions import CompletionRequest

        r = CompletionRequest(model="m", prompt="hi")
        assert r.prompt_logprobs is None


class TestGuidedAliasesW744:
    """guided_* aliases on /v1/completions (parity with chat)."""

    def _r(self, **kw):
        from yunshu_gateway.routers.completions import CompletionRequest

        return CompletionRequest(model="m", prompt="hi", **kw)

    def test_guided_regex(self):
        assert self._r(guided_regex="a+").grammar == {"type": "regex", "pattern": "a+"}

    def test_guided_choice(self):
        assert self._r(guided_choice=["x", "y"]).grammar == {
            "type": "choice",
            "choices": ["x", "y"],
        }

    def test_guided_grammar(self):
        assert self._r(guided_grammar='start: "a"').grammar == {
            "type": "cfg",
            "grammar": 'start: "a"',
        }

    def test_guided_json(self):
        s = {"type": "object"}
        assert self._r(guided_json=s).response_format == {
            "type": "json_schema",
            "json_schema": {"schema": s},
        }

    def test_native_grammar_wins(self):
        r = self._r(grammar={"type": "regex", "pattern": "z"}, guided_regex="a+")
        assert r.grammar == {"type": "regex", "pattern": "z"}
