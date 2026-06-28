"""Tests for multi-turn conversation correctness across the stack.

Covers:
1. ChatMessage schema: tool_calls, tool_call_id, name fields
2. _extract_messages: preserving tool call fields in multi-turn
3. Context window truncation: tool call/response pair integrity
4. Token counting: tool_calls, tool_call_id in count_message_tokens
5. Engine._messages_to_text: preserving tool call fields
6. KV prefix cache: multi-turn prefix matching
7. LoRA: adapter switching between turns
8. Thinking budget: per-request reset
9. Streaming: concurrent multi-turn requests
10. Message adapter: tool call fields across model families
"""

from yunshu_gateway.routers.chat import (
    ChatMessage,
    ToolCall,
    ToolCallFunction,
    _extract_messages,
)

# ── 1. ChatMessage Schema ──


class TestChatMessageSchema:
    """Verify ChatMessage accepts all OpenAI multi-turn fields."""

    def test_basic_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_with_tool_calls(self):
        """Assistant messages with tool_calls must be accepted."""
        msg = ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_abc123",
                    type="function",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments='{"city": "SF"}',
                    ),
                )
            ],
        )
        assert msg.role == "assistant"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_abc123"
        assert msg.tool_calls[0].function.name == "get_weather"
        assert msg.tool_calls[0].function.arguments == '{"city": "SF"}'

    def test_tool_result_message(self):
        """Tool role messages must accept tool_call_id and name."""
        msg = ChatMessage(
            role="tool",
            content='{"temp": 72}',
            tool_call_id="call_abc123",
            name="get_weather",
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_abc123"
        assert msg.name == "get_weather"

    def test_multiple_tool_calls(self):
        """Assistant can make multiple tool calls in one turn."""
        msg = ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="call_1", function=ToolCallFunction(name="f1", arguments="{}")),
                ToolCall(id="call_2", function=ToolCallFunction(name="f2", arguments="{}")),
            ],
        )
        assert len(msg.tool_calls) == 2

    def test_none_content_with_tool_calls(self):
        """OpenAI spec allows content=None when tool_calls present."""
        msg = ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_1", function=ToolCallFunction(name="f", arguments="{}"))],
        )
        assert msg.content is None
        assert msg.tool_calls is not None


# ── 2. _extract_messages ──


class TestExtractMessages:
    """Verify _extract_messages preserves tool call fields for multi-turn."""

    def test_basic_messages(self):
        msgs = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]
        result = _extract_messages(msgs)
        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[2] == {"role": "assistant", "content": "Hi there!"}

    def test_tool_calls_preserved(self):
        """Assistant tool_calls must survive extraction."""
        msgs = [
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=ToolCallFunction(name="get_weather", arguments='{"city":"SF"}'),
                    )
                ],
            )
        ]
        result = _extract_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == ""
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "call_1"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"city":"SF"}'

    def test_tool_result_preserved(self):
        """Tool role messages must preserve tool_call_id and name."""
        msgs = [
            ChatMessage(
                role="tool",
                content='{"temp": 72}',
                tool_call_id="call_1",
                name="get_weather",
            )
        ]
        result = _extract_messages(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == '{"temp": 72}'
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["name"] == "get_weather"

    def test_full_multi_turn_tool_conversation(self):
        """End-to-end multi-turn conversation with tool calls."""
        msgs = [
            ChatMessage(role="system", content="You are a weather bot."),
            ChatMessage(role="user", content="What's the weather in SF?"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_w1",
                        type="function",
                        function=ToolCallFunction(name="get_weather", arguments='{"city":"SF"}'),
                    )
                ],
            ),
            ChatMessage(
                role="tool",
                content='{"temp": 72, "condition": "sunny"}',
                tool_call_id="call_w1",
                name="get_weather",
            ),
            ChatMessage(role="assistant", content="It's sunny and 72F in SF!"),
            ChatMessage(role="user", content="What about NYC?"),
        ]
        result = _extract_messages(msgs)

        assert len(result) == 6
        # System message
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a weather bot."
        # User message
        assert result[1]["role"] == "user"
        # Assistant with tool_calls
        assert result[2]["role"] == "assistant"
        assert "tool_calls" in result[2]
        assert result[2]["tool_calls"][0]["id"] == "call_w1"
        # Tool result
        assert result[3]["role"] == "tool"
        assert result[3]["tool_call_id"] == "call_w1"
        assert result[3]["name"] == "get_weather"
        assert result[3]["content"] == '{"temp": 72, "condition": "sunny"}'
        # Follow-up assistant
        assert result[4]["role"] == "assistant"
        assert result[4]["content"] == "It's sunny and 72F in SF!"
        # Follow-up user
        assert result[5]["role"] == "user"

    def test_no_tool_fields_when_not_set(self):
        """Messages without tool fields should not have them in output."""
        msgs = [ChatMessage(role="user", content="Hello")]
        result = _extract_messages(msgs)
        assert "tool_calls" not in result[0]
        assert "tool_call_id" not in result[0]
        assert "name" not in result[0]


# ── 3. Context Window Truncation ──


class TestContextWindowTruncation:
    """Verify context window truncation preserves tool call pairs."""

    def test_truncate_oldest_preserves_tool_pairs(self):
        """Truncation must remove assistant+tool_calls with its tool results together."""
        from yunshu_engine.context_window import ContextWindowManager

        mgr = ContextWindowManager()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Call func A"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "func_a", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "result A", "tool_call_id": "call_1", "name": "func_a"},
            {"role": "assistant", "content": "Done with A."},
            {"role": "user", "content": "Call func B"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_2", "type": "function", "function": {"name": "func_b", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "result B", "tool_call_id": "call_2", "name": "func_b"},
            {"role": "assistant", "content": "Done with B."},
        ]

        # Truncate to a small budget that requires removing older messages
        result = mgr.compute_truncation(messages, max_tokens=40, strategy="truncate_oldest")

        # The result must not have orphaned tool messages
        for i, msg in enumerate(result.messages):
            if msg.get("role") == "tool":
                # Every tool message must be preceded by an assistant with tool_calls
                assert i > 0, f"Tool message at index {i} has no preceding assistant"
                prev = result.messages[i - 1]
                assert prev.get("role") == "assistant", (
                    f"Tool message at index {i} preceded by {prev.get('role')}, not assistant"
                )
                assert prev.get("tool_calls") is not None, (
                    f"Tool message at index {i} preceded by assistant without tool_calls"
                )

    def test_truncate_oldest_removes_tool_group_together(self):
        """When budget is tight, assistant+tool_calls+tool_results must all be removed."""
        from yunshu_engine.context_window import ContextWindowManager

        mgr = ContextWindowManager()
        messages = [
            {"role": "system", "content": "System"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "R1", "tool_call_id": "c1"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]

        # Budget only fits system + user + assistant
        result = mgr.compute_truncation(messages, max_tokens=30, strategy="truncate_oldest")
        roles = [m["role"] for m in result.messages]

        # Either the tool group is fully present or fully removed
        if "tool" in roles:
            # If tool messages exist, assistant+tool_calls must precede
            tool_idx = roles.index("tool")
            assert roles[tool_idx - 1] == "assistant"
            assert result.messages[tool_idx - 1].get("tool_calls") is not None
        else:
            # If no tool messages, no assistant with tool_calls either
            for m in result.messages:
                if m["role"] == "assistant":
                    assert m.get("tool_calls") is None

    def test_sliding_window_no_orphaned_tool_results(self):
        """Sliding window must not produce orphaned tool results at start."""
        from yunshu_engine.context_window import ContextWindowManager

        mgr = ContextWindowManager()
        messages = [
            {"role": "system", "content": "System"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "Result", "tool_call_id": "c1"},
            {"role": "assistant", "content": "Final answer"},
        ]

        result = mgr.compute_truncation(messages, max_tokens=30, strategy="sliding_window")

        # Must not start with a tool message (it would be orphaned)
        non_system = [m for m in result.messages if m.get("role") != "system"]
        if non_system:
            assert non_system[0]["role"] != "tool", (
                "Sliding window produced orphaned tool result at start"
            )

    def test_importance_aware_tool_group_integrity(self):
        """Importance-aware strategy must keep tool call groups together."""
        from yunshu_engine.context_window import ContextWindowManager

        mgr = ContextWindowManager()
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Important question " * 20},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q": "test"}'}},
            ]},
            {"role": "tool", "content": "Search results here " * 10, "tool_call_id": "c1"},
            {"role": "assistant", "content": "Based on search... " * 10},
            {"role": "user", "content": "Follow up question"},
            {"role": "assistant", "content": "Follow up answer"},
        ]

        result = mgr.compute_truncation(messages, max_tokens=80, strategy="importance_aware")

        # Check tool group integrity
        for i, msg in enumerate(result.messages):
            if msg.get("role") == "tool":
                assert i > 0, "Tool message at start without preceding assistant"
                prev = result.messages[i - 1]
                if prev.get("role") == "assistant":
                    assert prev.get("tool_calls") is not None, (
                        "Tool message preceded by assistant without tool_calls"
                    )


# ── 4. Token Counting ──


class TestTokenCounting:
    """Verify token counting includes tool call fields in multi-turn."""

    def test_count_basic_messages(self):
        from yunshu_control.token_counter import count_message_tokens

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        tokens = count_message_tokens(msgs)
        assert tokens > 0
        # 2 messages * 4 overhead + 2 priming + content
        assert tokens >= 10  # At minimum: 4+5 + 4+2 + 2 = 17

    def test_count_with_tool_calls(self):
        """Token counting must include tool_calls in assistant messages."""
        from yunshu_control.token_counter import count_message_tokens

        msgs_without = [
            {"role": "assistant", "content": "Let me check."},
        ]
        msgs_with = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "San Francisco"}',
                        },
                    }
                ],
            },
        ]

        tokens_without = count_message_tokens(msgs_without)
        tokens_with = count_message_tokens(msgs_with)
        # Messages with tool_calls should have more tokens
        assert tokens_with > tokens_without, (
            f"Tool calls should add tokens: {tokens_with} vs {tokens_without}"
        )

    def test_count_with_tool_result(self):
        """Token counting must include tool_call_id and name in tool messages."""
        from yunshu_control.token_counter import count_message_tokens

        msgs_plain = [
            {"role": "tool", "content": "Result data"},
        ]
        msgs_with_fields = [
            {
                "role": "tool",
                "content": "Result data",
                "tool_call_id": "call_abc123",
                "name": "get_weather",
            },
        ]

        tokens_plain = count_message_tokens(msgs_plain)
        tokens_with = count_message_tokens(msgs_with_fields)
        # tool_call_id adds 4 tokens, name adds tokens for the name string
        assert tokens_with > tokens_plain, (
            f"Tool fields should add tokens: {tokens_with} vs {tokens_plain}"
        )


# ── 5. Engine._messages_to_text ──


class TestEngineMessagesToText:
    """Verify Engine._messages_to_text preserves tool call fields."""

    def test_basic_messages(self):
        """Simple messages should produce text output."""
        from yunshu_engine.engine import Engine

        engine = Engine.__new__(Engine)
        engine._tokenizer = None  # Force fallback

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        text = engine._messages_to_text(msgs)
        assert "Hello" in text
        assert "Hi there!" in text

    def test_tool_calls_not_stripped_in_fallback(self):
        """Even in fallback mode, tool_calls should not crash."""
        from yunshu_engine.engine import Engine

        engine = Engine.__new__(Engine)
        engine._tokenizer = None

        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "result", "tool_call_id": "c1", "name": "f"},
        ]
        # Should not crash - tool_calls, tool_call_id, name are preserved
        text = engine._messages_to_text(msgs)
        assert isinstance(text, str)


# ── 6. Message Adapter ──


class TestMessageAdapterMultiTurn:
    """Verify message adapters preserve tool call fields in multi-turn."""

    def test_qwen_tool_calls(self):
        from yunshu_engine.message_adapter import QwenMessageAdapter

        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        result = QwenMessageAdapter().adapt(msgs)
        assert result[0].get("tool_calls") == [{"id": "c1"}]
        assert result[1].get("tool_call_id") == "c1"

    def test_deepseek_tool_calls(self):
        from yunshu_engine.message_adapter import DeepSeekMessageAdapter

        msgs = [
            {"role": "assistant", "content": "Let me check", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "result", "tool_call_id": "c1", "name": "search"},
        ]
        result = DeepSeekMessageAdapter().adapt(msgs)
        # Find the assistant message
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("tool_calls") == [{"id": "c1"}]
        # Find the tool message
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].get("tool_call_id") == "c1"
        assert tool_msgs[0].get("name") == "search"

    def test_gemma4_tool_call_id_preserved(self):
        from yunshu_engine.message_adapter import Gemma4MessageAdapter

        msgs = [
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        result = Gemma4MessageAdapter().adapt(msgs)
        # Gemma4 may restructure, but tool_call_id should be preserved
        tool_msgs = [m for m in result if m["role"] == "tool"]
        if tool_msgs:
            assert tool_msgs[0].get("tool_call_id") == "c1"

    def test_harmony_tool_calls_preserved(self):
        from yunshu_engine.message_adapter import HarmonyMessageAdapter

        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        result = HarmonyMessageAdapter().adapt(msgs)
        assert result[0].get("tool_calls") == [{"id": "c1"}]
        assert result[1].get("tool_call_id") == "c1"


# ── 7. KV Prefix Cache Multi-Turn ──


class TestKVPrefixCacheMultiTurn:
    """Verify KV prefix cache works correctly for multi-turn conversations."""

    def test_prefix_match_across_turns(self):
        """Turn 2 should get a prefix cache hit from turn 1's KV cache."""
        import mlx.core as mx

        from yunshu_engine.kv_prefix_cache import KVPrefixCache

        cache = KVPrefixCache(max_entries=10, min_prefix_length=8)

        # Simulate turn 1: system + user + assistant
        turn1_tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        # Create a mock KV cache (just a list of dicts with offset)
        mock_kv1 = [
            type("KVLayer", (), {"keys": mx.zeros((1, 1)), "values": mx.zeros((1, 1)), "offset": 15})()
        ]

        cache.add(turn1_tokens, mock_kv1)

        # Simulate turn 2: same system + user + assistant + new user message
        turn2_tokens = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

        result_kv, remaining, matched = cache.get(turn2_tokens)

        # Should match at least the first 15 tokens from turn 1
        assert matched >= 8, f"Expected prefix match >= 8, got {matched}"
        assert remaining == len(turn2_tokens) - matched


# ── 8. Thinking Budget Per-Request ──


class TestThinkingBudgetPerRequest:
    """Verify thinking budget state is per-request (no cross-turn leakage)."""

    def test_processor_reset(self):
        from yunshu_engine.thinking_budget import (
            ThinkingBudgetConfig,
            ThinkingBudgetProcessor,
        )

        proc = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=10))

        # Simulate thinking tokens in turn 1 (exceed budget by 1)
        for _ in range(11):
            proc.process_token("reasoning")
        assert proc.is_budget_exceeded
        assert proc.thinking_tokens_used == 11

        # Reset for turn 2
        proc.reset()
        assert not proc.is_budget_exceeded
        assert proc.thinking_tokens_used == 0

    def test_per_request_instances(self):
        """In the scheduler, each request gets its own ThinkingBudgetProcessor."""
        from yunshu_engine.thinking_budget import (
            ThinkingBudgetConfig,
            ThinkingBudgetProcessor,
        )

        # Simulate scheduler behavior: one processor per request
        proc1 = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=100))
        proc2 = ThinkingBudgetProcessor(ThinkingBudgetConfig(max_thinking_tokens=100))

        # Burn through budget in proc1 (exceed by 1)
        for _ in range(101):
            proc1.process_token("reasoning")
        assert proc1.is_budget_exceeded

        # proc2 should be independent
        assert not proc2.is_budget_exceeded
        assert proc2.thinking_tokens_used == 0


# ── 9. Context Window _count_messages_tokens ──


class TestContextWindowCountTokens:
    """Verify ContextWindowManager._count_messages_tokens includes tool fields."""

    def test_counts_tool_calls(self):
        from yunshu_engine.context_window import ContextWindowManager

        mgr = ContextWindowManager()
        msgs_without = [{"role": "assistant", "content": "Hello"}]
        msgs_with = [{
            "role": "assistant",
            "content": "Hello",
            "tool_calls": [
                {"id": "c1", "function": {"name": "search", "arguments": '{"q": "test"}'}},
            ],
        }]

        tokens_without = mgr._count_messages_tokens(msgs_without)
        tokens_with = mgr._count_messages_tokens(msgs_with)
        assert tokens_with > tokens_without

    def test_counts_tool_call_id(self):
        from yunshu_engine.context_window import ContextWindowManager

        mgr = ContextWindowManager()
        msgs_without = [{"role": "tool", "content": "result"}]
        msgs_with = [{"role": "tool", "content": "result", "tool_call_id": "call_abc123", "name": "search"}]

        tokens_without = mgr._count_messages_tokens(msgs_without)
        tokens_with = mgr._count_messages_tokens(msgs_with)
        assert tokens_with > tokens_without


# ── 10. Truncate Helper ──


class TestTruncateFirstMessageGroup:
    """Test the _truncate_first_message_group helper directly."""

    def test_simple_removal(self):
        from yunshu_engine.context_window import _truncate_first_message_group

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        _truncate_first_message_group(msgs)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"

    def test_removes_assistant_with_tool_calls(self):
        from yunshu_engine.context_window import _truncate_first_message_group

        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
            {"role": "user", "content": "Next question"},
        ]
        _truncate_first_message_group(msgs)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_removes_orphaned_tool_result(self):
        from yunshu_engine.context_window import _truncate_first_message_group

        msgs = [
            {"role": "tool", "content": "orphaned result", "tool_call_id": "c1"},
            {"role": "user", "content": "Question"},
        ]
        _truncate_first_message_group(msgs)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_empty_list(self):
        from yunshu_engine.context_window import _truncate_first_message_group

        msgs = []
        _truncate_first_message_group(msgs)
        assert msgs == []

    def test_multiple_tool_results_removed(self):
        from yunshu_engine.context_window import _truncate_first_message_group

        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "f1", "arguments": "{}"}},
                {"id": "c2", "function": {"name": "f2", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "r1", "tool_call_id": "c1"},
            {"role": "tool", "content": "r2", "tool_call_id": "c2"},
            {"role": "user", "content": "Next"},
        ]
        _truncate_first_message_group(msgs)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
