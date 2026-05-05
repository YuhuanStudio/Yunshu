"""Tests for Yunshu exception hierarchy."""
import pytest

from yunshu_engine.exceptions import (
    APIError,
    AuthenticationError,
    BatchingError,
    CacheCorruptionError,
    CacheError,
    CacheMissError,
    CacheStorageError,
    EnginePoolError,
    InvalidRequestError,
    MemoryError,
    ModelError,
    ModelInferenceError,
    ModelLoadError,
    ModelNotFoundError,
    ModelTooLargeError,
    OutOfMemoryError,
    PrefillMemoryExceededError,
    RateLimitError,
    RequestAbortedError,
    RequestError,
    RequestNotFoundError,
    SchedulerError,
    TokenizerError,
    YunshuError,
    is_cache_corruption_error,
)


class TestExceptionHierarchy:
    def test_base_error(self):
        e = YunshuError("test error")
        assert str(e) == "test error"

    def test_base_error_with_details(self):
        e = YunshuError("test", details={"key": "val"})
        assert "key" in str(e)

    def test_cache_error_hierarchy(self):
        assert issubclass(CacheCorruptionError, CacheError)
        assert issubclass(CacheMissError, CacheError)
        assert issubclass(CacheStorageError, CacheError)
        assert issubclass(CacheError, YunshuError)

    def test_scheduler_error_hierarchy(self):
        assert issubclass(RequestError, SchedulerError)
        assert issubclass(RequestNotFoundError, RequestError)
        assert issubclass(RequestAbortedError, RequestError)
        assert issubclass(BatchingError, SchedulerError)

    def test_model_error_hierarchy(self):
        assert issubclass(ModelLoadError, ModelError)
        assert issubclass(ModelInferenceError, ModelError)
        assert issubclass(TokenizerError, ModelError)

    def test_memory_error_hierarchy(self):
        assert issubclass(OutOfMemoryError, MemoryError)
        assert issubclass(PrefillMemoryExceededError, MemoryError)

    def test_engine_pool_hierarchy(self):
        assert issubclass(ModelNotFoundError, EnginePoolError)
        assert issubclass(ModelTooLargeError, EnginePoolError)

    def test_api_hierarchy(self):
        assert issubclass(InvalidRequestError, APIError)
        assert issubclass(RateLimitError, APIError)
        assert issubclass(AuthenticationError, APIError)


class TestCacheCorruption:
    def test_detect_none_subscriptable(self):
        e = RuntimeError("'NoneType' object is not subscriptable")
        assert is_cache_corruption_error(e)

    def test_detect_none_iterable(self):
        e = TypeError("'NoneType' object is not iterable")
        assert is_cache_corruption_error(e)

    def test_detect_shape_mismatch(self):
        e = ValueError("shape mismatch in matmul")
        assert is_cache_corruption_error(e)

    def test_no_false_positive(self):
        e = ValueError("invalid input")
        assert not is_cache_corruption_error(e)

    def test_detect_batch_kv_cache(self):
        e = RuntimeError("BatchKVCache index out of range")
        assert is_cache_corruption_error(e)


class TestExceptionAttributes:
    def test_cache_corruption_with_request_id(self):
        e = CacheCorruptionError("corrupted", request_id="req-123")
        assert e.request_id == "req-123"

    def test_cache_storage_with_path(self):
        e = CacheStorageError("io error", path="/tmp/cache", operation="write")
        assert e.path == "/tmp/cache"
        assert e.operation == "write"

    def test_model_load_error(self):
        e = ModelLoadError("not found", model_name="test-model")
        assert e.model_name == "test-model"

    def test_model_not_found(self):
        e = ModelNotFoundError("test-model", ["model-a", "model-b"])
        assert "test-model" in str(e)
        assert "model-a" in str(e)

    def test_model_too_large(self):
        e = ModelTooLargeError("big-model", 10 * 1024 ** 3, 8 * 1024 ** 3)
        assert "big-model" in str(e)

    def test_oom_error(self):
        e = OutOfMemoryError("oom", requested_bytes=100, available_bytes=50)
        assert e.requested_bytes == 100
        assert e.available_bytes == 50

    def test_prefill_memory_exceeded(self):
        e = PrefillMemoryExceededError(
            "too large", request_id="req-1", estimated_bytes=100, limit_bytes=80,
        )
        assert e.request_id == "req-1"

    def test_invalid_request_with_field(self):
        e = InvalidRequestError("bad input", field="temperature")
        assert e.field == "temperature"
