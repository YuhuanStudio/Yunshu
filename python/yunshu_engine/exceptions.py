"""Yunshu exception hierarchy (oMLX pattern).

Structured exceptions for better error handling, debugging, and recovery.
Cache corruption patterns enable automatic scheduler recovery.
"""
from __future__ import annotations

from typing import Any, Optional


class YunshuError(Exception):
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


# ── Cache ──

class CacheError(YunshuError):
    pass

class CacheCorruptionError(CacheError):
    def __init__(self, message: str, request_id: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, details)
        self.request_id = request_id

class CacheMissError(CacheError):
    pass

class CacheStorageError(CacheError):
    def __init__(self, message: str, path: Optional[str] = None, operation: Optional[str] = None):
        super().__init__(message)
        self.path = path
        self.operation = operation


# ── Scheduler ──

class SchedulerError(YunshuError):
    pass

class RequestError(SchedulerError):
    def __init__(self, message: str, request_id: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, details)
        self.request_id = request_id

class RequestNotFoundError(RequestError):
    pass

class RequestAbortedError(RequestError):
    pass

class BatchingError(SchedulerError):
    pass


# ── Model ──

class ModelError(YunshuError):
    pass

class ModelLoadError(ModelError):
    def __init__(self, message: str, model_name: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, details)
        self.model_name = model_name

class ModelInferenceError(ModelError):
    pass

class TokenizerError(ModelError):
    pass


# ── Memory ──

class MemoryError(YunshuError):
    pass

class OutOfMemoryError(MemoryError):
    def __init__(self, message: str, requested_bytes: Optional[int] = None, available_bytes: Optional[int] = None):
        super().__init__(message)
        self.requested_bytes = requested_bytes
        self.available_bytes = available_bytes

class PrefillMemoryExceededError(MemoryError):
    def __init__(self, message: str, request_id: Optional[str] = None, estimated_bytes: Optional[int] = None, limit_bytes: Optional[int] = None):
        super().__init__(message)
        self.request_id = request_id
        self.estimated_bytes = estimated_bytes
        self.limit_bytes = limit_bytes


# ── Engine Pool ──

class EnginePoolError(YunshuError):
    pass

class ModelNotFoundError(EnginePoolError):
    def __init__(self, model_id: str, available: list[str]):
        self.model_id = model_id
        msg = f"Model '{model_id}' not found. Available: {', '.join(available) or '(none)'}"
        super().__init__(msg)

class ModelTooLargeError(EnginePoolError):
    def __init__(self, model_id: str, size: int, max_memory: int):
        from .utils.hardware import format_bytes
        super().__init__(
            f"Model '{model_id}' ({format_bytes(size)}) exceeds limit ({format_bytes(max_memory)})"
        )
        self.model_id = model_id


# ── API ──

class APIError(YunshuError):
    pass

class InvalidRequestError(APIError):
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.field = field

class RateLimitError(APIError):
    pass

class AuthenticationError(APIError):
    pass


# ── Cache corruption detection ──

CACHE_CORRUPTION_PATTERNS = [
    "'NoneType' object is not subscriptable",
    "'NoneType' object is not iterable",
    "BatchKVCache",
    "KVCache",
    "cache.keys",
    "cache.values",
    "'NoneType' object has no attribute",
    "not broadcastable",
    "cannot be broadcast",
    "shape mismatch",
]


def is_cache_corruption_error(error: Exception) -> bool:
    return any(p in str(error) for p in CACHE_CORRUPTION_PATTERNS)
