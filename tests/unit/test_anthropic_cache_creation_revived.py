"""the cache_creation_input_tokens fix was DEAD on both Anthropic paths.
create_message collapses req.system to a joined STRING (so a canonical system message
reaches the engine), but _cacheable_prefix_token_count returns 0 unless system is a LIST —
so cache_creation_input_tokens was always 0 (non-stream) or capped at 1 (streaming passed
bool(breakpoints)). The cacheable prefix was billed entirely as input_tokens.

Fix: snapshot the original system list (req._anthropic_orig_system) before the overwrite,
use it at all three usage call sites, and compute the real prefix count on the streaming
path instead of bool().
"""

from __future__ import annotations

import inspect

from yunshu_gateway.routers import (
    anthropic as A,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers.anthropic import _cacheable_prefix_token_count


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return text.split()  # 1 token per whitespace-word, deterministic


def test_helper_counts_list_prefix_but_zero_for_string():
    system = [
        {
            "type": "text",
            "text": "alpha beta gamma",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "delta epsilon"},
    ]
    # breakpoint char offset = end of the first block's text ("alpha beta gamma" = 16 chars).
    # assembled = "alpha beta gamma\ndelta epsilon"; prefix[:16] = "alpha beta gamma" → 3 tokens.
    n = _cacheable_prefix_token_count(system, [16], _Tok())
    assert n == 3
    # the bug: a STRING system (what req.system becomes after the overwrite) → always 0
    assert (
        _cacheable_prefix_token_count("alpha beta gamma\ndelta epsilon", [16], _Tok())
        == 0
    )


def test_create_message_snapshots_original_system():
    src = inspect.getsource(A.create_message)
    # the original list is captured BEFORE req.system is collapsed to a string
    i_snap = src.index("req._anthropic_orig_system = req.system")
    i_join = src.index('req.system = "\\n\\n".join(_all_system)')
    assert i_snap < i_join


def test_all_usage_sites_use_original_and_streaming_not_bool():
    src = inspect.getsource(A)
    # all cacheable-count calls use the snapshot, never the collapsed req.system directly
    assert src.count('getattr(req, "_anthropic_orig_system", req.system)') >= 3
    # streaming no longer caps the cache-creation count at bool(breakpoints)
    assert "bool(kv_cache_breakpoints))" not in src
