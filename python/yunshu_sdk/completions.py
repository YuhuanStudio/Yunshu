"""Yunshu SDK — Completions namespace (text completions)."""

from __future__ import annotations

import json
from typing import AsyncIterator, Iterator, Optional

import httpx


class TextCompletion:
    def __init__(self, data: dict):
        self.id = data.get("id", "")
        self.object = data.get("object", "")
        self.created = data.get("created", 0)
        self.model = data.get("model", "")
        self.choices = data.get("choices", [])
        self.usage = data.get("usage", {})

    @property
    def text(self) -> str:
        if self.choices:
            return self.choices[0].get("text", "")
        return ""

    @property
    def finish_reason(self) -> str:
        if self.choices:
            return self.choices[0].get("finish_reason", "")
        return ""


class _Completions:
    def __init__(self, http: httpx.Client, async_http: httpx.AsyncClient):
        self._http = http
        self._async_http = async_http

    def create(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> TextCompletion | Iterator[TextCompletion]:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        if stream:
            return self._stream(payload)

        resp = self._http.post("/v1/completions", json=payload)
        resp.raise_for_status()
        return TextCompletion(resp.json())

    def _stream(self, payload: dict) -> Iterator[TextCompletion]:
        with self._http.stream("POST", "/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        yield TextCompletion(json.loads(data))
                    except json.JSONDecodeError:
                        continue


class CompletionsNamespace:
    """Completions namespace (client.completions)."""

    def __init__(self, http: httpx.Client, async_http: httpx.AsyncClient):
        self._completions = _Completions(http, async_http)

    def create(self, **kwargs) -> TextCompletion | Iterator[TextCompletion]:
        return self._completions.create(**kwargs)
