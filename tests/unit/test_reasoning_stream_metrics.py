"""two reasoning bugs.

HIGH (streaming CoT leak): the streaming fast path (_stream_generate_fast) only resolved
the <think>/</think> token ids when `thinking_budget is not None or enable_thinking`. But
enable_thinking defaults to None end-to-end while Qwen3/Qwen3.5/DeepSeek-R1 chat templates
are default-ON (they inject the opening <think> into the PROMPT). So a plain default-param
streaming request had think_end_token=None → the pre-seed was skipped → _in_thinking
never flipped → the ENTIRE chain-of-thought leaked into visible delta.content and
reasoning_tokens stayed 0 (non-stream was correct via the post-hoc parser). Now the think
tokens resolve unconditionally; the pre-seed remains gated on the prompt actually ending
with an open <think>.

MEDIUM (Anthropic metric double-count): /v1/messages recorded completion_tokens +
reasoning_tokens, but completion_tokens ALREADY includes reasoning (subset, not addend) —
double-counting the server metric for every thinking-model request. chat.py records it
correctly."""
from __future__ import annotations

import inspect

from yunshu_engine import batched_engine
from yunshu_gateway.routers import anthropic


def test_stream_fast_resolves_think_tokens_unconditionally():
    src = inspect.getsource(batched_engine.BatchedEngine._stream_generate_fast)
    # the pre-seed must be present and reachable without the enable_thinking gate
    assert "detect_needs_think_prefix" in src
    # think-token resolution now goes through _resolve_think_token_ids (bracketed
    # form) instead of the buggy bare encode("<think"). It must still be UNCONDITIONAL.
    i = src.index("_resolve_think_token_ids(tokenizer)")
    window = src[max(0, i - 400):i]
    assert "if thinking_budget is not None or enable_thinking:" not in window


def test_anthropic_does_not_double_count_reasoning():
    src = inspect.getsource(anthropic)
    # neither metric call re-adds reasoning tokens to completion
    assert "result.completion_tokens + (getattr(result, 'reasoning_tokens'" not in src
    assert "completion_toks + _reasoning_tok_legacy" not in src
    # the corrected calls are present
    assert "_record_metrics(result.prompt_tokens, result.completion_tokens)" in src
    assert "_record_metrics(prompt_toks, completion_toks)" in src
