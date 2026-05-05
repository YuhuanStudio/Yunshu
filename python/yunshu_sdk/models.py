"""Yunshu SDK — Models namespace."""

from __future__ import annotations

from typing import Optional

import httpx


class Model:
    """A model object."""

    def __init__(self, data: dict):
        self.id = data.get("id", "")
        self.object = data.get("object", "model")
        self.created = data.get("created", 0)
        self.owned_by = data.get("owned_by", "yunshu")

    def __repr__(self):
        return f"Model(id={self.id!r})"


class ModelList:
    """A list of models."""

    def __init__(self, data: dict):
        self.object = data.get("object", "list")
        self.data = [Model(m) for m in data.get("data", [])]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class Models:
    """Models namespace (client.models)."""

    def __init__(self, http: httpx.Client):
        self._http = http

    def list(self) -> ModelList:
        resp = self._http.get("/v1/models")
        resp.raise_for_status()
        return ModelList(resp.json())

    def retrieve(self, model_id: str) -> Model:
        resp = self._http.get(f"/v1/models/{model_id}")
        resp.raise_for_status()
        return Model(resp.json())

    def load(self, model_id: str) -> dict:
        resp = self._http.post("/v1/models/load", json={"model_id": model_id})
        resp.raise_for_status()
        return resp.json()
