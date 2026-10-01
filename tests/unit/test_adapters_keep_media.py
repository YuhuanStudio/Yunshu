"""Every family adapter keeps the media parts of a message (images, audio, video).

The vision path runs the family adapter; an adapter that flattens content to text drops
the image, and the model answers "I need an image" (Gemma 4 in the 2026-10-01 gate).
"""

import pytest

from yunshu_engine.message_adapter import _REGISTRY


def _media(messages):
    return [
        p
        for m in messages
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if isinstance(p, dict) and p.get("type") not in ("text", None)
    ]


IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


@pytest.mark.parametrize("family", sorted(_REGISTRY))
@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": [IMG, {"type": "text", "text": "what color?"}]}],
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "look"}, IMG]},
        ],
        [
            {"role": "user", "content": "first"},
            {"role": "user", "content": [IMG, {"type": "text", "text": "and this"}]},
        ],
    ],
    ids=["image-only-turn", "system-then-image", "merged-user-turns"],
)
def test_media_parts_survive(family, messages):
    out = _REGISTRY[family]().adapt([dict(m) for m in messages])
    assert len(_media(out)) == len(_media(messages)), out
