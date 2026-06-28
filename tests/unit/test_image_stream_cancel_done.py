"""the streaming image endpoint's cancel path yielded a 'cancelled' event then
returned WITHOUT the SSE [DONE] terminator — every other terminal path (final-image,
error) emits [DONE], so a strict SSE client waiting for it could hang until socket close.
Now the cancel path emits [DONE] too."""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import (
    images as IMG,  # noqa: N812  # intentional short module alias
)


def test_cancel_path_emits_done():
    src = inspect.getsource(IMG.stream_image_generation)
    # locate the cancel branch and assert a [DONE] follows the cancelled event
    i = src.index("'type': 'cancelled'")
    after = src[i:i + 400]
    assert "[DONE]" in after, "cancel path must emit the SSE [DONE] terminator"
