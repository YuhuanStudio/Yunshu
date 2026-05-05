"""Yunshu SDK — Embeddings namespace."""

from __future__ import annotations

from typing import Optional

import httpx


class Embedding:
    def __init__(self, data: dict):
        self.object = data.get("object", "embedding")
        self.embedding = data.get("embedding", [])
        self.index = data.get("index", 0)


class EmbeddingList:
    def __init__(self, data: dict):
        self.object = data.get("object", "list")
        self.data = [Embedding(e) for e in data.get("data", [])]
        self.model = data.get("model", "")
        self.usage = data.get("usage", {})

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class _EmbeddingsCreate:
    def __init__(self, http: httpx.Client):
        self._http = http

    def create(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str = "float",
        **kwargs,
    ) -> EmbeddingList:
        payload: dict = {
            "model": model,
            "input": input,
            "encoding_format": encoding_format,
        }
        payload.update(kwargs)
        resp = self._http.post("/v1/embeddings", json=payload)
        resp.raise_for_status()
        return EmbeddingList(resp.json())


class EmbeddingsNamespace:
    """Embeddings namespace (client.embeddings)."""

    def __init__(self, http: httpx.Client):
        self._create = _EmbeddingsCreate(http)

    def create(self, **kwargs) -> EmbeddingList:
        return self._create.create(**kwargs)
