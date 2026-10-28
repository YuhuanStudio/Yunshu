"""The MTP lane clears MLX's buffer cache every 256 emitted tokens.

A spec round emits several tokens, so the old ``emitted % 256 == 0`` test after the round
only fired when a round ended exactly on a multiple of 256 and the cache could grow for
thousands of tokens at long context (ollama hit the same pattern: >90 GB at 98k).
"""

from __future__ import annotations

import inspect

from yunshu_engine import mtp_lane


def test_clear_cache_fires_on_crossing_a_256_boundary():
    src = inspect.getsource(mtp_lane)
    assert "emitted % 256 == 0" not in src
    assert "if emitted // 256 > cleared_at:" in src
