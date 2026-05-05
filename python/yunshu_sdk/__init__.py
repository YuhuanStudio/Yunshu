"""Yunshu Python SDK — OpenAI-compatible client library.

Provides a drop-in replacement for the OpenAI Python SDK when using
Yunshu as the backend. Supports all inference endpoints plus
Yunshu-specific admin operations.

Usage:
    from yunshu_sdk import YunshuClient

    client = YunshuClient(base_url="http://localhost:8000")

    # Chat completion (OpenAI-compatible)
    response = client.chat.completions.create(
        model="Qwen3.5-9B-MLX-4bit",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=256,
        stream=True,
    )

    # Text completion
    resp = client.completions.create(
        model="Qwen3.5-9B-MLX-4bit",
        prompt="Once upon a time",
        max_tokens=128,
    )

    # Embeddings
    emb = client.embeddings.create(
        model="text-embedding-model",
        input="Hello world",
    )

    # Images
    img = client.images.generations.create(
        model="Z-Image-Turbo-MLX-4bit",
        prompt="A cat in space",
    )

    # Audio
    audio = client.audio.speech.create(
        model="Qwen3-TTS-12Hz-1.7B",
        input="Hello world",
        voice="Chelsie",
    )

    # Model management
    models = client.models.list()
"""

from __future__ import annotations

from typing import Optional

import httpx

from .chat import Chat
from .completions import CompletionsNamespace
from .models import Models
from .audio import Audio
from .images import Images
from .embeddings import EmbeddingsNamespace
from .realtime import RealtimeNamespace
from .admin import Admin
from .monitoring import Monitoring


class YunshuClient:
    """Yunshu inference platform client.

    OpenAI-compatible interface with Yunshu-specific extensions.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        self._async_http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

        # Namespaces (OpenAI SDK pattern)
        self.chat = Chat(self._http, self._async_http)
        self.completions = CompletionsNamespace(self._http, self._async_http)
        self.models = Models(self._http)
        self.audio = Audio(self._http)
        self.images = Images(self._http)
        self.embeddings = EmbeddingsNamespace(self._http)
        self.realtime = RealtimeNamespace(self._base_url)
        self.admin = Admin(self._http)
        self.monitoring = Monitoring(self._http)

    def health(self) -> dict:
        """Check server health."""
        resp = self._http.get("/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the client."""
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncYunshuClient:
    """Async version of the Yunshu client."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

        self.chat = Chat(None, self._http)
        self.completions = CompletionsNamespace(None, self._http)
        self.models = Models(self._http)
        self.audio = Audio(self._http)
        self.images = Images(self._http)
        self.embeddings = EmbeddingsNamespace(self._http)
        self.realtime = RealtimeNamespace(self._base_url)
        self.admin = Admin(self._http)
        self.monitoring = Monitoring(self._http)

    async def health(self) -> dict:
        resp = await self._http.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
