"""A chat template that drops a cache_control marker must not fail the request (Claude Code replay on Qwen3.5)."""

from yunshu_engine.vlm_engine import VLMEngine


def test_template_dropping_a_marker_serves_without_explicit_checkpoints():
    eng = object.__new__(VLMEngine)
    marker = "<<CACHE-MARK-1>>"
    messages = [{"role": "user", "content": "hi" + marker}]
    plan = {"markers": {marker: 300}, "tools": {}, "messages": messages}
    # the "template" renders the real prompt without the marker (as Qwen does for earlier-turn reasoning)
    eng._tokenizer = object()
    eng._format_prompt = lambda msgs, *_a, **_k: "<|im_start|>user\nhi<|im_end|>"
    out = eng._resolve_prompt_cache_plan(plan, messages, [1, 2, 3], None, False, None)
    assert out is None
