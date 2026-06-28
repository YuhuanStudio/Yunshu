"""Tests for Message Adapters — model-specific message formatting."""

from yunshu_engine.message_adapter import (
    CohereMessageAdapter,
    DeepSeekMessageAdapter,
    Gemma4MessageAdapter,
    GenericMessageAdapter,
    GLMMessageAdapter,
    HarmonyMessageAdapter,
    InternVLMessageAdapter,
    LLamaMessageAdapter,
    MistralMessageAdapter,
    PhiMessageAdapter,
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

    def test_system_only_request_keeps_system(self):
        """a system-only request (no user/assistant turn) must NOT drop the
        system prompt. The old `system_prefix and adapted` guard returned [] → Gemma template
        raised → plaintext fallback → system 100% lost. Now it injects a user turn."""
        result = Gemma4MessageAdapter().adapt([{"role": "system", "content": "You are a pirate."}])
        assert result == [{"role": "user", "content": "You are a pirate."}]

    def test_multi_system_only_accumulates(self):
        result = Gemma4MessageAdapter().adapt(
            [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}])
        assert len(result) == 1 and "A" in result[0]["content"] and "B" in result[0]["content"]

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


class TestMistralMessageAdapter:
    def test_system_merged_into_first_user(self):
        adapter = MistralMessageAdapter()
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = adapter.adapt(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Be helpful." in result[0]["content"]

    def test_strict_alternation(self):
        adapter = MistralMessageAdapter()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        result = adapter.adapt(msgs)
        # Should insert empty assistant between two user messages
        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "user"]

    def test_system_only_request_keeps_system(self):
        """the Gemma4 drop-on-empty sibling — a system-only request must surface
        the system prompt as an injected user turn, not return [] (→ template raise → lost)."""
        result = MistralMessageAdapter().adapt([{"role": "system", "content": "You are a pirate."}])
        assert result == [{"role": "user", "content": "You are a pirate."}]

    def test_tool_call_id_preserved(self):
        adapter = MistralMessageAdapter()
        msgs = [
            {"role": "user", "content": "Call f"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc-1"}]},
            {"role": "tool", "content": "data", "tool_call_id": "tc-1"},
        ]
        result = adapter.adapt(msgs)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "tc-1"

    def test_assistant_tool_calls_preserved(self):
        adapter = MistralMessageAdapter()
        msgs = [
            {"role": "user", "content": "Call f"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc-1"}]},
        ]
        result = adapter.adapt(msgs)
        asst_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(asst_msgs) >= 1
        assert asst_msgs[0]["tool_calls"] == [{"id": "tc-1"}]

    def test_reasoning_content_preserved(self):
        adapter = MistralMessageAdapter()
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Ans", "reasoning_content": "thinking..."},
        ]
        result = adapter.adapt(msgs)
        asst_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(asst_msgs) >= 1
        assert asst_msgs[0]["reasoning_content"] == "thinking..."

    def test_starts_with_user(self):
        adapter = MistralMessageAdapter()
        msgs = [{"role": "assistant", "content": "Hi"}]
        result = adapter.adapt(msgs)
        assert result[0]["role"] == "user"

    def test_family_name(self):
        assert MistralMessageAdapter().family_name() == "mistral"


class TestPhiMessageAdapter:
    def test_system_moved_first(self):
        adapter = PhiMessageAdapter()
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Be helpful."},
        ]
        result = adapter.adapt(msgs)
        assert result[0]["role"] == "system"

    def test_tool_call_id_preserved(self):
        adapter = PhiMessageAdapter()
        msgs = [{"role": "tool", "content": "data", "tool_call_id": "tc-1"}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_call_id"] == "tc-1"

    def test_assistant_tool_calls_preserved(self):
        adapter = PhiMessageAdapter()
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "tc-1"}]}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_calls"] == [{"id": "tc-1"}]

    def test_reasoning_content_preserved(self):
        adapter = PhiMessageAdapter()
        msgs = [{"role": "assistant", "content": "Ans", "reasoning_content": "thinking..."}]
        result = adapter.adapt(msgs)
        assert result[0]["reasoning_content"] == "thinking..."

    def test_family_name(self):
        assert PhiMessageAdapter().family_name() == "phi"


class TestCohereMessageAdapter:
    def test_tool_call_id_preserved(self):
        adapter = CohereMessageAdapter()
        msgs = [{"role": "tool", "content": "data", "tool_call_id": "tc-1"}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_call_id"] == "tc-1"

    def test_assistant_tool_calls_preserved(self):
        adapter = CohereMessageAdapter()
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "tc-1"}]}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_calls"] == [{"id": "tc-1"}]

    def test_reasoning_content_preserved(self):
        adapter = CohereMessageAdapter()
        msgs = [{"role": "assistant", "content": "Ans", "reasoning_content": "thinking..."}]
        result = adapter.adapt(msgs)
        assert result[0]["reasoning_content"] == "thinking..."

    def test_system_preserved(self):
        adapter = CohereMessageAdapter()
        msgs = [{"role": "system", "content": "You are helpful."}]
        result = adapter.adapt(msgs)
        assert result[0]["role"] == "system"

    def test_family_name(self):
        assert CohereMessageAdapter().family_name() == "cohere"


class TestLLamaMessageAdapter:
    def test_system_moved_first(self):
        adapter = LLamaMessageAdapter()
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Be helpful."},
        ]
        result = adapter.adapt(msgs)
        assert result[0]["role"] == "system"

    def test_tool_call_id_preserved(self):
        adapter = LLamaMessageAdapter()
        msgs = [{"role": "tool", "content": "data", "tool_call_id": "tc-1"}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_call_id"] == "tc-1"

    def test_assistant_tool_calls_preserved(self):
        adapter = LLamaMessageAdapter()
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "tc-1"}]}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_calls"] == [{"id": "tc-1"}]

    def test_reasoning_content_preserved(self):
        adapter = LLamaMessageAdapter()
        msgs = [{"role": "assistant", "content": "Ans", "reasoning_content": "thinking..."}]
        result = adapter.adapt(msgs)
        assert result[0]["reasoning_content"] == "thinking..."

    def test_family_name(self):
        assert LLamaMessageAdapter().family_name() == "llama"


class TestInternVLMessageAdapter:
    def test_passthrough(self):
        adapter = InternVLMessageAdapter()
        msgs = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]
        result = adapter.adapt(msgs)
        assert len(result) == 2

    def test_tool_call_id_preserved(self):
        adapter = InternVLMessageAdapter()
        msgs = [{"role": "tool", "content": "data", "tool_call_id": "tc-1"}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_call_id"] == "tc-1"

    def test_assistant_tool_calls_preserved(self):
        adapter = InternVLMessageAdapter()
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "tc-1"}]}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_calls"] == [{"id": "tc-1"}]

    def test_reasoning_content_preserved(self):
        adapter = InternVLMessageAdapter()
        msgs = [{"role": "assistant", "content": "Ans", "reasoning_content": "thinking..."}]
        result = adapter.adapt(msgs)
        assert result[0]["reasoning_content"] == "thinking..."

    def test_family_name(self):
        assert InternVLMessageAdapter().family_name() == "internvl"


class TestGLMMessageAdapter:
    def test_system_moved_first(self):
        adapter = GLMMessageAdapter()
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Be helpful."},
        ]
        result = adapter.adapt(msgs)
        assert result[0]["role"] == "system"

    def test_tool_call_id_preserved(self):
        adapter = GLMMessageAdapter()
        msgs = [{"role": "tool", "content": "data", "tool_call_id": "tc-1"}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_call_id"] == "tc-1"

    def test_assistant_tool_calls_preserved(self):
        adapter = GLMMessageAdapter()
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "tc-1"}]}]
        result = adapter.adapt(msgs)
        assert result[0]["tool_calls"] == [{"id": "tc-1"}]

    def test_reasoning_content_preserved(self):
        adapter = GLMMessageAdapter()
        msgs = [{"role": "assistant", "content": "Ans", "reasoning_content": "thinking..."}]
        result = adapter.adapt(msgs)
        assert result[0]["reasoning_content"] == "thinking..."

    def test_family_name(self):
        assert GLMMessageAdapter().family_name() == "glm"


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

    def test_mistral_detection(self):
        adapter = get_message_adapter("mistral-7b-instruct")
        assert isinstance(adapter, MistralMessageAdapter)

    def test_codestral_detection(self):
        adapter = get_message_adapter("codestral-22b")
        assert isinstance(adapter, MistralMessageAdapter)

    def test_mixtral_detection(self):
        adapter = get_message_adapter("mixtral-8x7b")
        assert isinstance(adapter, MistralMessageAdapter)

    def test_phi_detection(self):
        adapter = get_message_adapter("phi-3.5-mini-instruct")
        assert isinstance(adapter, PhiMessageAdapter)

    def test_phi4_detection(self):
        adapter = get_message_adapter("phi-4-mini")
        assert isinstance(adapter, PhiMessageAdapter)

    def test_cohere_detection(self):
        adapter = get_message_adapter("command-r-plus")
        assert isinstance(adapter, CohereMessageAdapter)

    def test_cohere_model_detection(self):
        adapter = get_message_adapter("cohere-for-ai")
        assert isinstance(adapter, CohereMessageAdapter)

    def test_llama_detection(self):
        adapter = get_message_adapter("llama-3.1-70b")
        assert isinstance(adapter, LLamaMessageAdapter)

    def test_internvl_detection(self):
        adapter = get_message_adapter("internvl-2.5-8b")
        assert isinstance(adapter, InternVLMessageAdapter)

    def test_glm_detection(self):
        adapter = get_message_adapter("glm-4-9b")
        assert isinstance(adapter, GLMMessageAdapter)

    def test_unknown_falls_to_generic(self):
        adapter = get_message_adapter("falcon-180b")
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
        get_message_adapter("custom-model")
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
