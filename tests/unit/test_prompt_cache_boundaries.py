"""Renderer boundaries must use full tokenization, never isolated block counts."""

import pytest

from yunshu_engine.prompt_caching import rendered_boundaries, tool_boundaries


class SeamTokenizer:
    def encode(self, text, **kw):
        # 'ab' is one token: the block seam lies inside that token.
        out = []
        i = 0
        while i < len(text):
            if text[i : i + 2] == "ab":
                out.append(999)
                i += 2
            else:
                out.append(ord(text[i]))
                i += 1
        return out


def test_marker_maps_full_template_and_bpe_seam():
    tok = SeamTokenizer()
    prompt = "<system>ab尾巴</system><user>x</user>"
    marked = "<system>aMARKb尾巴</system><user>x</user>"
    points = rendered_boundaries(prompt, marked, {"MARK": 300}, tok, tok.encode(prompt))
    assert points == [(len("<system>"), 300)]  # do not split the merged token


def test_duplicate_content_maps_source_marker_not_first_text_match():
    tok = SeamTokenizer()
    prompt = "<s>same</s><u>same</u>"
    marked = "<s>same</s><u>sameMARK</u>"
    assert rendered_boundaries(
        prompt, marked, {"MARK": 3600}, tok, tok.encode(prompt)
    ) == [(18, 3600)]


def test_probe_must_reproduce_exact_original_prompt():
    tok = SeamTokenizer()
    with pytest.raises(ValueError, match="render"):
        rendered_boundaries(
            "original", "changedMARK", {"MARK": 300}, tok, tok.encode("original")
        )


def test_tools_boundary_includes_whole_serialized_definition():
    tools = [
        {
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object"}},
        }
    ]
    prompt = '<tools>\n{"name": "f", "parameters": {"type": "object"}}\n</tools>'
    assert tool_boundaries(prompt, tools, {0: 300}) == [
        (prompt.index("\n</tools>"), 300)
    ]


def test_prefill_resumes_at_absolute_chunk_end():
    from yunshu_engine.prompt_caching import canonical_step_end

    # A breakpoint split the first atom at 113. Warm and cold both finish it
    # at 2048, instead of warm forwarding through 2161.
    assert canonical_step_end(113, 8000, 2048, [113, 5001]) == 2048
    assert canonical_step_end(4096, 8000, 2048, [113, 5001]) == 5001
    assert canonical_step_end(5001, 8000, 2048, [113, 5001]) == 6144


def test_openai_block_marker_preserved_without_altering_input():
    from yunshu_engine.prompt_caching import openai_plan, strip_markers
    from yunshu_gateway.routers.responses import _extract_input_text

    source = [
        {
            "type": "input_text",
            "text": "stable",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
        {"type": "input_text", "text": "variable"},
    ]
    converted = _extract_input_text(source)
    assert isinstance(converted, list)
    messages = [{"role": "user", "content": converted}]
    plan = openai_plan(messages, {"mode": "explicit"})
    assert strip_markers(plan["messages"], plan["markers"]) == messages
    assert len(plan["markers"]) == 2
    assert len(plan["write_markers"]) == 1


def test_anthropic_automatic_uses_final_block_and_rejects_conflicting_ttl():
    from yunshu_engine.prompt_caching import mark_anthropic
    from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest

    req = AnthropicMessagesRequest(
        model="local",
        max_tokens=1,
        cache_control={"type": "ephemeral"},
        messages=[{"role": "user", "content": "hello"}],
    )
    plan = mark_anthropic(req)
    assert len(plan["markers"]) == 1
    req = AnthropicMessagesRequest(
        model="local",
        max_tokens=1,
        cache_control={"type": "ephemeral"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="TTL"):
        mark_anthropic(req)


def test_processor_media_expansion_maps_only_complete_media_boundary():
    from yunshu_engine.prompt_caching import expanded_boundaries

    plain = [1, 2, 900, 3, 4, 5]
    expanded = [1, 2, 900, 900, 900, 900, 3, 4, 5]
    assert expanded_boundaries(
        plain, expanded, [(2, 300), (3, 300), (5, 300)], {900}
    ) == [(2, 300), (6, 300), (8, 300)]
    with pytest.raises(ValueError, match="processor"):
        expanded_boundaries(plain, [1, 2, 901, 3, 4, 5], [(5, 300)], {900})


def test_diagnostic_marker_preserves_template_trimming():
    from yunshu_engine.prompt_caching import trim_safe_shadow

    value = "hello  MARK"
    shadow = trim_safe_shadow(value, {"MARK": 300})

    def render(text):
        return "<s>" + text.strip() + "</s>"

    assert render(shadow).replace("MARK", "") == render("hello  ")


def test_moving_anthropic_breakpoint_has_twenty_block_lookback():
    from yunshu_engine.prompt_caching import mark_anthropic, strip_markers
    from yunshu_gateway.routers.anthropic import (
        AnthropicMessagesRequest,
        _convert_anthropic_messages,
    )

    messages = [{"role": "user", "content": f"block {i}"} for i in range(27)]
    messages[-1]["content"] = [
        {"type": "text", "text": "block 26", "cache_control": {"type": "ephemeral"}}
    ]
    req = AnthropicMessagesRequest(model="local", max_tokens=1, messages=messages)
    plan = mark_anthropic(req)
    assert len(plan["markers"]) == 27
    assert len(plan["write_markers"]) == 1
    converted, _ = _convert_anthropic_messages(
        req.messages, has_images=False, temp_files=[]
    )
    shadow = "".join("<u>" + m["content"] + "</u>" for m in converted)
    prompt = strip_markers(shadow, plan["markers"])
    tok = SeamTokenizer()
    resolved = rendered_boundaries(
        prompt, shadow, plan["markers"], tok, tok.encode(prompt), selection=plan
    )
    assert len(resolved["points"]) == 27
    assert len(resolved["writes"]) == 1
    assert resolved["lookup_points"] == [n for n, _ in resolved["points"]][-21:]


def test_openai_implicit_keeps_a_reusable_developer_head():
    from yunshu_engine.prompt_caching import openai_plan

    plan = openai_plan(
        [
            {"role": "developer", "content": "stable instructions"},
            {"role": "user", "content": "changing question"},
        ],
        {"mode": "implicit", "ttl": "30m"},
    )
    assert len(plan["write_markers"]) == 2


def test_chat_conversion_preserves_explicit_content_breakpoint():
    from yunshu_gateway.routers.chat import ChatMessage, _extract_messages

    message = ChatMessage(
        role="user",
        content=[
            {
                "type": "text",
                "text": "stable",
                "prompt_cache_breakpoint": {"mode": "explicit"},
            },
            {"type": "text", "text": "tail"},
        ],
    )
    converted = _extract_messages([message])[0]["content"]
    assert isinstance(converted, list)
    assert converted[0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    from yunshu_engine.vlm_engine import VLMEngine

    assert VLMEngine._extract_text(converted) == "stable\ntail"
