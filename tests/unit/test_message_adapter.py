"""Tests for Message Adapters — model-specific message formatting."""
import pytest

from yunshu_engine.message_adapter import (
    DeepSeekMessageAdapter,
    Gemma4MessageAdapter,
    GenericMessageAdapter,
    HarmonyMessageAdapter,
    QwenMessageAdapter,
    adapt_messages,
    get_message_adapter,
    register_message_adapter,
)


class TestHarmonyMessageAdapter:
    def test_system_to_developer(self):
        adapter = HarmonyMessageAdapter()
        msgs = [{"role": "system", "content": "You are helpful."}]
        result = adapter.adapt(msgs)
        assert result[0]["role"] == "developer"
        assert result[0]["content"] == "You are helpful."

    def test_passthrough_user(self):
        adapter = HarmonyMessageAdapter()
        msgs = [{"role": "user", "content": "Hello"}]
        result = adapter.adapt(msgs)
        assert result[0]["role"] == "user"

    def test_tool_with_id(self):
        adapter = HarmonyMessageAdapter()
        msgs = [{"role": "tool", "content": "result", "tool_call_id": "tc-1"}]
        result = adapter.adapt(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tc-1"

    def test_assistant_with_tool_calls(self):
        adapter = HarmonyMessageAdapter()
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "tc-1"}]}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_calls"] == [{"id": "tc-1"}]

    def test_family_name(self):
        assert HarmonyMessageAdapter().family_name() == "harmony"


class TestGemma4MessageAdapter:
    def test_system_merged_into_first_user(self):
        adapter = Gemma4MessageAdapter()
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = adapter.adapt(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Be helpful." in result[0]["content"]
        assert "Hi" in result[0]["content"]

    def test_consecutive_same_role_merged(self):
        adapter = Gemma4MessageAdapter()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        result = adapter.adapt(msgs)
        assert len(result) == 1
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]

    def test_starts_with_user(self):
        adapter = Gemma4MessageAdapter()
        msgs = [{"role": "assistant", "content": "Hi"}]
        result = adapter.adapt(msgs)
        assert result[0]["role"] == "user"

    def test_family_name(self):
        assert Gemma4MessageAdapter().family_name() == "gemma4"


class TestDeepSeekMessageAdapter:
    def test_trailing_whitespace_trimmed(self):
        adapter = DeepSeekMessageAdapter()
        msgs = [{"role": "user", "content": "Hello   "}]
        result = adapter.adapt(msgs)
        assert result[0]["content"] == "Hello"

    def test_system_moved_first(self):
        adapter = DeepSeekMessageAdapter()
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Be helpful."},
        ]
        result = adapter.adapt(msgs)
        assert result[0]["role"] == "system"

    def test_tool_call_id_preserved(self):
        adapter = DeepSeekMessageAdapter()
        msgs = [{"role": "tool", "content": "data", "tool_call_id": "tc-1"}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_call_id"] == "tc-1"

    def test_family_name(self):
        assert DeepSeekMessageAdapter().family_name() == "deepseek"


class TestQwenMessageAdapter:
    def test_reasoning_content_preserved(self):
        adapter = QwenMessageAdapter()
        msgs = [
            {"role": "assistant", "content": "Answer", "reasoning_content": "I thought..."},
        ]
        result = adapter.adapt(msgs)
        assert result[0]["reasoning_content"] == "I thought..."

    def test_tool_call_id_preserved(self):
        adapter = QwenMessageAdapter()
        msgs = [{"role": "tool", "content": "data", "tool_call_id": "tc-1"}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_call_id"] == "tc-1"

    def test_family_name(self):
        assert QwenMessageAdapter().family_name() == "qwen"


class TestGenericMessageAdapter:
    def test_passthrough(self):
        adapter = GenericMessageAdapter()
        msgs = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]
        result = adapter.adapt(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"

    def test_family_name(self):
        assert GenericMessageAdapter().family_name() == "generic"


class TestGetMessageAdapter:
    def test_harmony_detection(self):
        adapter = get_message_adapter("harmony-v2")
        assert isinstance(adapter, HarmonyMessageAdapter)

    def test_gpt_oss_detection(self):
        adapter = get_message_adapter("gpt-oss-7b")
        assert isinstance(adapter, HarmonyMessageAdapter)

    def test_gemma4_detection(self):
        adapter = get_message_adapter("gemma-4-9b")
        assert isinstance(adapter, Gemma4MessageAdapter)

    def test_deepseek_detection(self):
        adapter = get_message_adapter("deepseek-v3")
        assert isinstance(adapter, DeepSeekMessageAdapter)

    def test_qwen_detection(self):
        adapter = get_message_adapter("qwen-2.5-7b")
        assert isinstance(adapter, QwenMessageAdapter)

    def test_unknown_falls_to_generic(self):
        adapter = get_message_adapter("llama-3-8b")
        assert isinstance(adapter, GenericMessageAdapter)

    def test_none_falls_to_generic(self):
        adapter = get_message_adapter(None)
        assert isinstance(adapter, GenericMessageAdapter)


class TestAdaptMessages:
    def test_convenience_function(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = adapt_messages(msgs, "unknown-model")
        assert len(result) == 1

    def test_harmony_through_convenience(self):
        msgs = [{"role": "system", "content": "Hi"}]
        result = adapt_messages(msgs, "harmony-v2")
        assert result[0]["role"] == "developer"


class TestRegisterMessageAdapter:
    def test_register_custom(self):
        class CustomAdapter(GenericMessageAdapter):
            def family_name(self):
                return "custom"

        register_message_adapter("custom", CustomAdapter)
        adapter = get_message_adapter("custom-model")
        # Shouldn't auto-detect (no hint), but registry has it
        assert "custom" in _REGISTRY

    def teardown_method(self):
        # Clean up registration
        from yunshu_engine.message_adapter import _REGISTRY
        _REGISTRY.pop("custom", None)


class TestGemma4MultiPartContent:
    """Bug fix: multi-part content (list format) was str()-repr'd."""

    def test_multipart_content_extracted(self):
        adapter = Gemma4MessageAdapter()
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]},
        ]
        result = adapter.adapt(msgs)
        assert len(result) == 1
        assert "Hello" in result[0]["content"]
        assert "World" in result[0]["content"]
        # Must NOT contain list repr garbage like "[{'type':"
        assert "[{'type':" not in result[0]["content"]

    def test_none_content_handled(self):
        adapter = Gemma4MessageAdapter()
        msgs = [
            {"role": "user", "content": None},
        ]
        result = adapter.adapt(msgs)
        assert result[0]["content"] == ""

    def test_string_content_unchanged(self):
        adapter = Gemma4MessageAdapter()
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = adapter.adapt(msgs)
        assert result[0]["content"] == "Be helpful.\n\nHello"


from yunshu_engine.message_adapter import _REGISTRY
