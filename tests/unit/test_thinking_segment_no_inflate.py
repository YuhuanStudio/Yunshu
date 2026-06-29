"""the thinking-segment KV "reuse" is a non-functional stub (it finds a stored
segment but never injects its kv_data into the live cache). The old code still did
`cached_tokens += _best.num_tokens`, which over-reported cached_tokens AND, on the hybrid
path, made `_capture_hybrid_prefix(start_offset=cached_tokens)` skip that many real prompt
tokens → cache/RoPE misalignment → wrong output. Until the KV is genuinely injected, the
lookup must not touch cached_tokens.
"""

from __future__ import annotations

import inspect

from yunshu_engine.batched_engine import BatchedEngine


def test_thinking_segment_lookup_does_not_inflate_cached_tokens():
    src = inspect.getsource(BatchedEngine._generate_fast)
    # the stub no longer bumps cached_tokens by the (never-injected) segment length —
    # check ACTUAL code lines (a comment may still quote the old form for explanation)
    code_lines = [ln.split("#", 1)[0] for ln in src.splitlines()]
    assert not any("cached_tokens += _best.num_tokens" in ln for ln in code_lines), (
        "thinking-segment stub still inflates cached_tokens with un-injected KV"
    )
    # it still detects/logs the match for observability
    assert "_best.kv_data is not None" in src
    assert "not yet injected" in src
