"""Serve a local checkpoint directory through the real gateway app (TestClient) on CPU.

``serve()`` runs the production lifespan: ``instantiate_engine`` picks BatchedEngine or VLMEngine
from the checkpoint's config, exactly as ``yunshu serve -m <dir>`` does."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


@contextlib.contextmanager
def serve(model_dir, **env) -> Iterator[tuple[TestClient, object]]:
    from yunshu_gateway import engine as engine_mod
    from yunshu_gateway import main as gm

    mp = pytest.MonkeyPatch()
    mp.setenv("YUNSHU_AUTH_DISABLED", "true")
    mp.setenv("YUNSHU_DRAIN_TIMEOUT", "0")
    for key, value in env.items():
        mp.setenv(key, str(value))
    mp.setattr(gm, "DEFAULT_MODEL", str(model_dir))
    engine_mod._engine = None
    try:
        with TestClient(gm.create_app()) as client:
            yield client, engine_mod.get_engine()
    finally:
        engine_mod._engine = None
        mp.undo()


def chat(client, messages, headers=None, **kw):
    body = {"model": "x", "messages": messages, "max_tokens": 8, "temperature": 0}
    body.update(kw)
    return client.post("/v1/chat/completions", json=body, headers=headers or {})


def user(content):
    return [{"role": "user", "content": content}]


def sse_events(resp) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def cached_tokens(resp) -> int:
    return resp.json()["usage"]["prompt_tokens_details"]["cached_tokens"]


def logprob_rows(resp) -> list[dict]:
    """Per generated token: bytes, logprob and the top_logprobs list (full precision)."""
    return resp.json()["choices"][0]["logprobs"]["content"]


def install_cpu_bitmask(monkeypatch) -> None:
    """llguidance's token-mask kernel is Metal-only (``mx.fast.metal_kernel``). This is the same
    operation on any device: tokens whose bit is 0 go to -inf."""
    import llguidance.mlx as llm
    import mlx.core as mx
    import numpy as np

    def apply_token_bitmask(logits, mask_np):
        mask = np.asarray(mask_np, dtype=np.int32).view(np.uint32)
        if logits.ndim == 1:
            logits = mx.expand_dims(logits, axis=0)
        batch, vocab = logits.shape
        bits = (mask[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1
        bits = bits.reshape(mask.shape[0], -1)[:, :vocab]
        if bits.shape[1] < vocab:  # the kernel reads a missing word as 0
            bits = np.pad(bits, ((0, 0), (0, vocab - bits.shape[1])))
        assert bits.shape[0] == batch
        return mx.where(mx.array(bits.astype(bool)), logits, -float("inf"))

    monkeypatch.setattr(llm, "apply_token_bitmask", apply_token_bitmask)
