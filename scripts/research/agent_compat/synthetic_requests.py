"""Spec-shaped requests the official SDKs and popular clients send, beyond what the three agent CLIs record.

Each case is replayed by replay.py as a session named `synthetic` and must answer 2xx with a valid body / stream.
They are small on purpose (thinking off or short answers): the point is the wire contract, not generation quality.
"""

from __future__ import annotations

WEATHER = {"name": "get_weather", "description": "Weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
NO_THINK = {"enable_thinking": False}


def cases() -> list[dict]:
    u = [{"role": "user", "content": "Reply with the single word: pong"}]
    t = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
    fn = {"type": "function", "function": WEATHER}
    rfn = {"type": "function", **WEATHER}
    return [
        dict(name="chat_nulls", path="/v1/chat/completions", body=dict(messages=u, temperature=None, top_p=None, max_tokens=None, n=None, stream=None, seed=None, stop=None, chat_template_kwargs=NO_THINK)),
        dict(name="chat_default_length_stream_usage", path="/v1/chat/completions", body=dict(messages=u, stream=True, stream_options={"include_usage": True, "include_obfuscation": False}, top_k=-1, max_completion_tokens=200000, chat_template_kwargs=NO_THINK)),
        dict(name="chat_tool_required_stream", path="/v1/chat/completions", body=dict(messages=t, tools=[fn], tool_choice="required", stream=True, chat_template_kwargs=NO_THINK)),
        dict(name="chat_allowed_tools", path="/v1/chat/completions", body=dict(messages=t, tools=[fn, {"type": "function", "function": {**WEATHER, "name": "other"}}], tool_choice={"type": "allowed_tools", "allowed_tools": {"mode": "auto", "tools": [fn]}}, chat_template_kwargs=NO_THINK)),
        dict(name="messages_nulls_adaptive_stream", path="/v1/messages", body=dict(max_tokens=200000, messages=u, stream=True, temperature=None, top_p=None, top_k=-1, thinking={"type": "adaptive", "display": "updates"}, metadata={"user_id": "x"})),
        dict(name="messages_tool_any_stream", path="/v1/messages", body=dict(max_tokens=512, messages=t, stream=True, tools=[{"name": "get_weather", "description": "w", "input_schema": WEATHER["parameters"]}], tool_choice={"type": "any", "disable_parallel_tool_use": True}, thinking={"type": "disabled"})),
        dict(name="responses_defaults_stream", path="/v1/responses", body=dict(input="Reply with the single word: pong", stream=True, temperature=None, max_output_tokens=None, reasoning={"effort": "low"}, include=["reasoning.encrypted_content"], store=False)),
        dict(name="responses_allowed_tools", path="/v1/responses", body=dict(input="What is the weather in Paris? Use the tool.", tools=[rfn, {**rfn, "name": "other"}], tool_choice={"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "get_weather"}]}, reasoning={"effort": "low"})),
        dict(name="responses_forced_function", path="/v1/responses", body=dict(input="What is the weather in Paris?", tools=[rfn], tool_choice={"type": "function", "name": "get_weather"}, stream=True, reasoning={"effort": "low"})),
    ]
