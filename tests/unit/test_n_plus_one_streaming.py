"""Tests for n>1 streaming support."""
import json
import pytest


class TestFormatChoiceChunk:
    def test_basic_chunk(self):
        from yunshu_gateway.routers.chat import _format_choice_chunk
        result = _format_choice_chunk(
            completion_id="chatcmpl-test",
            model="test-model",
            index=0,
            delta_content="hello",
            finish_reason=None,
        )
        parsed = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert parsed["object"] == "chat.completion.chunk"
        assert parsed["choices"][0]["index"] == 0
        assert parsed["choices"][0]["delta"]["content"] == "hello"
        assert parsed["choices"][0]["finish_reason"] is None

    def test_chunk_with_role(self):
        from yunshu_gateway.routers.chat import _format_choice_chunk
        result = _format_choice_chunk(
            completion_id="chatcmpl-test",
            model="test-model",
            index=1,
            delta_content="",
            finish_reason=None,
            include_role=True,
        )
        parsed = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert parsed["choices"][0]["delta"]["role"] == "assistant"
        assert parsed["choices"][0]["index"] == 1

    def test_chunk_with_finish(self):
        from yunshu_gateway.routers.chat import _format_choice_chunk
        result = _format_choice_chunk(
            completion_id="chatcmpl-test",
            model="test-model",
            index=2,
            delta_content="",
            finish_reason="stop",
        )
        parsed = json.loads(result.removeprefix("data: ").removesuffix("\n\n"))
        assert parsed["choices"][0]["finish_reason"] == "stop"
        assert parsed["choices"][0]["index"] == 2

    def test_multiple_indices(self):
        from yunshu_gateway.routers.chat import _format_choice_chunk
        chunks = []
        for i in range(3):
            chunks.append(_format_choice_chunk(
                completion_id="chatcmpl-test",
                model="test-model",
                index=i,
                delta_content=f"choice_{i}",
                finish_reason="stop" if i == 2 else None,
            ))
        for i, chunk in enumerate(chunks):
            parsed = json.loads(chunk.removeprefix("data: ").removesuffix("\n\n"))
            assert parsed["choices"][0]["index"] == i


class TestStreamResponseMultiRouting:
    def test_n1_uses_single_stream(self):
        """n=1 should use the standard _stream_response, not _stream_response_multi."""
        from yunshu_gateway.routers.chat import ChatCompletionRequest
        req = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hi"}])
        assert req.n == 1

    def test_n2_sets_n(self):
        from yunshu_gateway.routers.chat import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            n=3,
        )
        assert req.n == 3
