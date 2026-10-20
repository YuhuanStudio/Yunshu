"""Streaming image generation must always deliver its terminal event, even when
a slow consumer leaves the bounded queue full while the worker finishes."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from yunshu_engine import image_engine as ie


def _engine(monkeypatch):
    monkeypatch.setattr(ie, "mx", MagicMock())
    eng = ie.ImageGenEngine("/models/zimage")
    eng._transformer = MagicMock()
    eng._vae = MagicMock()
    eng._executor = ThreadPoolExecutor(max_workers=1)
    eng._encode_prompt = lambda p: MagicMock()
    eng._resolve_sigmas = lambda n, w, h: MagicMock()
    eng._to_png = lambda image: b"png"
    return eng


@pytest.mark.asyncio
async def test_final_event_survives_full_queue(monkeypatch):
    eng = _engine(monkeypatch)
    agen = eng.generate_image_stream("a cat", num_inference_steps=100)
    chunks = [await asyncio.wait_for(agen.__anext__(), 5)]
    await asyncio.sleep(0.5)  # slow client: worker finishes, queue stays full

    async def rest():
        async for c in agen:
            chunks.append(c)

    await asyncio.wait_for(rest(), 5)
    assert chunks[-1]["is_final"] is True
    assert chunks[-1]["image"] == b"png"
    eng._executor.shutdown()


@pytest.mark.asyncio
async def test_stream_uses_tiled_vae_for_large_images(monkeypatch):
    eng = _engine(monkeypatch)
    async for _ in eng.generate_image_stream(
        "a cat", width=2048, height=2048, num_inference_steps=2, preview_interval=1
    ):
        pass
    assert eng._vae.decode_tiled.called
    assert not eng._vae.decode.called
    eng._executor.shutdown()
