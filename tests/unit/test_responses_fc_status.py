"""the Responses non-streaming function_call output item must include
`status` like the streaming path does (a client reconstructing output items shouldn't
see a shape that differs between stream and non-stream by a missing field). The
Responses _response_store surface was otherwise verified clean (locking, LRU eviction,
store=false gating, internal-key stripping, stored-vs-returned parity)."""

from __future__ import annotations

import pathlib


def test_both_function_call_items_include_status():
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "python/yunshu_gateway/routers/responses.py").read_text()
    # Each `"type": "function_call"` output item (non-stream + stream) must be followed,
    # within its dict literal, by a `"status": "completed"` field.
    idx = 0
    count = 0
    while True:
        i = src.find('"type": "function_call"', idx)
        if i == -1:
            break
        # the dict closes at the next standalone "})" or "}\n" — scan a generous window
        window = src[i : i + 800]
        assert '"status": "completed"' in window, (
            f"function_call item @ {i} missing status"
        )
        count += 1
        idx = i + 1
    assert count >= 2  # the non-stream builder + the streaming output_item.done
