"""Yunshu SDK — Chat completions namespace.

OpenAI-compatible chat.completions.create() with streaming support.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterator, Optional

import httpx


class ChatCompletion:
    """A single chat completion result."""

    def __init__(self, data: dict):
        self.id = data.get("id", "")
        self.object = data.get("object", "")
        self.created = data.get("created", 0)
        self.model = data.get("model", "")
        self.choices = data.get("choices", [])
        self.usage = data.get("usage", {})

    @property
    def content(self) -> str:
        if self.choices:
            msg = self.choices[0].get("message", {})
            return msg.get("content", "")
        return ""

    @property
    def finish_reason(self) -> str:
        if self.choices:
            return self.choices[0].get("finish_reason", "")
        return ""


class ChatCompletionChunk:
    """A streaming chat completion chunk."""

    def __init__(self, data: dict):
        self.id = data.get("id", "")
        self.model = data.get("model", "")
        self.choices = data.get("choices", [])

    @property
    def delta_content(self) -> str:
        if self.choices:
            delta = self.choices[0].get("delta", {})
            return delta.get("content", "")
        return ""

    @property
    def delta_reasoning(self) -> str:
        if self.choices:
            delta = self.choices[0].get("delta", {})
            return delta.get("reasoning_content", "")
        return ""

    @property
    def finish_reason(self) -> Optional[str]:
        if self.choices:
            return self.choices[0].get("finish_reason")
        return None


class _Completions:
    def __init__(self, http: httpx.Client, async_http: httpx.AsyncClient):
        self._http = http
        self._async_http = async_http

    def create(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
        stream: bool = False,
        stop: Optional[list[str]] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        payload.update(kwargs)

        if stream:
            return self._stream(payload)

        resp = self._http.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return ChatCompletion(resp.json())

    def _stream(self, payload: dict) -> Iterator[ChatCompletionChunk]:
        with self._http.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        yield ChatCompletionChunk(json.loads(data))
                    except json.JSONDecodeError:
                        continue

    async def acreate(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
        stream: bool = False,
        stop: Optional[list[str]] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        payload.update(kwargs)

        if stream:
            return self._astream(payload)

        resp = await self._async_http.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return ChatCompletion(resp.json())

    async def _astream(self, payload: dict) -> AsyncIterator[ChatCompletionChunk]:
        async with self._async_http.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        yield ChatCompletionChunk(json.loads(data))
                    except json.JSONDecodeError:
                        continue


class Chat:
    """Chat namespace (client.chat)."""

    def __init__(self, http: httpx.Client, async_http: httpx.AsyncClient):
        self.completions = _Completions(http, async_http)
