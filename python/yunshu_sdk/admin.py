"""Yunshu SDK — Admin namespace (model management, key management, config).

Provides administrative operations for managing models, API keys,
and server configuration via Yunshu's management API.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Default retry settings
_MAX_RETRIES = 3
_RETRY_DELAY_S = 0.5


class Admin:
    """Admin namespace (client.admin) — Yunshu-specific management API.

    Provides model lifecycle management (register, load, unload, delete),
    API key management, and server configuration endpoints.
    """

    def __init__(self, http: httpx.Client):
        self._http = http
        self.models = _AdminModels(http)
        self.keys = _AdminKeys(http)
        self.config = _AdminConfig(http)

    def health(self) -> dict:
        """Check server health status.

        Returns:
            Dict with 'status' key and optional engine stats.
        """
        return _request_with_retry(self._http.get, "/health")

    def info(self) -> dict:
        """Get server information (version, uptime, config summary).

        Returns:
            Dict with server metadata.
        """
        return _request_with_retry(self._http.get, "/api/v1/admin/info")


class _AdminModels:
    """Model lifecycle management."""

    def __init__(self, http: httpx.Client):
        self._http = http

    def list(self) -> list[dict]:
        """List all registered models.

        Returns:
            List of model dicts with id, path, status, and metadata.
        """
        data = _request_with_retry(self._http.get, "/api/v1/admin/models")
        return data.get("models", [])

    def register(self, model_id: str, model_path: str, **kwargs) -> dict:
        """Register a new model for serving.

        Args:
            model_id: Unique identifier for the model.
            model_path: Local path or HuggingFace repo ID.
            **kwargs: Additional options (quantization, max_context, etc.)

        Returns:
            Registration confirmation dict.
        """
        payload = {"model_id": model_id, "model_path": model_path}
        payload.update(kwargs)
        return _request_with_retry(
            self._http.post, "/api/v1/admin/models/register", json=payload,
        )

    def load(self, model_id: str) -> dict:
        """Load a registered model into GPU memory.

        Args:
            model_id: Model to load.

        Returns:
            Load confirmation with memory stats.
        """
        return _request_with_retry(
            self._http.post, "/api/v1/admin/models/load",
            json={"model_id": model_id},
        )

    def unload(self, model_id: str, force: bool = False) -> dict:
        """Unload a model from GPU memory.

        Args:
            model_id: Model to unload.
            force: Force unload even if requests are in progress.

        Returns:
            Unload confirmation with freed memory stats.
        """
        return _request_with_retry(
            self._http.post, "/api/v1/admin/models/unload",
            json={"model_id": model_id, "force": force},
        )

    def delete(self, model_id: str) -> dict:
        """Delete a model registration (must be unloaded first).

        Args:
            model_id: Model to delete.

        Returns:
            Deletion confirmation.
        """
        return _request_with_retry(
            self._http.delete, f"/api/v1/admin/models/{model_id}",
        )

    def get(self, model_id: str) -> dict:
        """Get details for a specific model.

        Args:
            model_id: Model to query.

        Returns:
            Model details dict.
        """
        return _request_with_retry(
            self._http.get, f"/api/v1/admin/models/{model_id}",
        )


class _AdminKeys:
    """API key management."""

    def __init__(self, http: httpx.Client):
        self._http = http

    def list(self) -> list[dict]:
        """List all API keys (masked).

        Returns:
            List of key dicts with id, prefix, created_at, permissions.
        """
        data = _request_with_retry(self._http.get, "/api/v1/admin/keys")
        return data.get("keys", [])

    def create(
        self,
        name: str = "",
        permissions: list[str] | None = None,
        expires_in_days: int | None = None,
    ) -> dict:
        """Create a new API key.

        Args:
            name: Human-readable name for the key.
            permissions: List of permission strings (e.g. ['inference', 'admin']).
            expires_in_days: Key expiration in days (None = never expires).

        Returns:
            Dict with 'key' (full key, shown only once) and 'key_id'.
        """
        payload: dict = {"name": name}
        if permissions is not None:
            payload["permissions"] = permissions
        if expires_in_days is not None:
            payload["expires_in_days"] = expires_in_days
        return _request_with_retry(
            self._http.post, "/api/v1/admin/keys", json=payload,
        )

    def revoke(self, key_id: str) -> dict:
        """Revoke an API key.

        Args:
            key_id: The key ID to revoke.

        Returns:
            Revocation confirmation.
        """
        return _request_with_retry(
            self._http.delete, f"/api/v1/admin/keys/{key_id}",
        )


class _AdminConfig:
    """Server configuration management."""

    def __init__(self, http: httpx.Client):
        self._http = http

    def get_engine(self) -> dict:
        """Get current engine configuration.

        Returns:
            Engine config dict with scheduler, memory, and model settings.
        """
        return _request_with_retry(self._http.get, "/api/v1/admin/config/engine")

    def update_engine(self, **kwargs) -> dict:
        """Update engine configuration (hot-reload where supported).

        Args:
            **kwargs: Config keys to update (e.g. completion_batch_size=64).

        Returns:
            Updated config dict.
        """
        return _request_with_retry(
            self._http.patch, "/api/v1/admin/config/engine", json=kwargs,
        )

    def get_scheduler(self) -> dict:
        """Get current scheduler configuration.

        Returns:
            Scheduler config dict.
        """
        return _request_with_retry(self._http.get, "/api/v1/admin/config/scheduler")


def _request_with_retry(
    method,
    url: str,
    *,
    max_retries: int = _MAX_RETRIES,
    **kwargs,
):
    """Execute an HTTP request with retries on transient errors.

    Args:
        method: httpx method (self._http.get, .post, etc.)
        url: Request URL path.
        max_retries: Maximum number of retry attempts.
        **kwargs: Additional arguments passed to the method.

    Returns:
        Parsed JSON response.

    Raises:
        httpx.HTTPStatusError: On non-retryable HTTP errors.
        httpx.ConnectError: If server is unreachable after retries.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = method(url, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = _RETRY_DELAY_S * (2 ** attempt)
                logger.debug("Connection error, retrying in %.1fs: %s", delay, url)
                time.sleep(delay)
        except httpx.HTTPStatusError as e:
            # Retry on 502, 503, 504 (gateway/overloaded)
            if e.response.status_code in (502, 503, 504) and attempt < max_retries - 1:
                delay = _RETRY_DELAY_S * (2 ** attempt)
                logger.debug(
                    "Server error %d, retrying in %.1fs: %s",
                    e.response.status_code, delay, url,
                )
                time.sleep(delay)
                last_exc = e
            else:
                raise
    raise last_exc
