"""Yunshu SDK — Images namespace (image generation)."""

from __future__ import annotations

import base64
from typing import Optional

import httpx


class GeneratedImage:
    def __init__(self, data: dict):
        raw = data.get("data", [{}])[0] if data.get("data") else data
        self.b64_json = raw.get("b64_json")
        self.url = raw.get("url", "")
        self.revised_prompt = raw.get("revised_prompt", "")

    def save(self, path: str):
        if self.b64_json:
            with open(path, "wb") as f:
                f.write(base64.b64decode(self.b64_json))
        elif self.url:
            raise ValueError("Cannot save URL-based images directly. Use download().")


class Images:
    """Images namespace (client.images)."""

    def __init__(self, http: httpx.Client):
        self._http = http
        self.generations = _Generations(http)


class _Generations:
    def __init__(self, http: httpx.Client):
        self._http = http

    def create(
        self,
        model: str,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "b64_json",
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> GeneratedImage:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
        }
        if num_inference_steps is not None:
            payload["num_inference_steps"] = num_inference_steps
        if seed is not None:
            payload["seed"] = seed
        payload.update(kwargs)

        resp = self._http.post("/v1/images/generations", json=payload)
        resp.raise_for_status()
        return GeneratedImage(resp.json())
