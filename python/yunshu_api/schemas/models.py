"""Yunshu Control Plane — Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Model Management ──


class ModelRegisterRequest(BaseModel):
    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Local path or HuggingFace repo ID")
    model_type: Optional[str] = Field(None, description="Force model type (LLM/VLM/TTS/ASR/IMAGE_GEN)")
    priority: int = Field(0, description="Loading priority (higher = more important)")
    pinned: bool = Field(False, description="Pin in memory (prevent eviction)")
    auto_load: bool = Field(False, description="Load immediately after registration")


class ModelLoadRequest(BaseModel):
    model_id: str
    gpu_memory_limit_gb: Optional[float] = None


class ModelUnloadRequest(BaseModel):
    model_id: str
    force: bool = Field(False, description="Force unload even if active requests exist")


class ModelResponse(BaseModel):
    model_id: str
    model_type: str
    status: str  # "registered", "loading", "loaded", "error"
    size_bytes: int = 0
    pinned: bool = False
    loaded_at: Optional[datetime] = None
    error: Optional[str] = None


class ModelListResponse(BaseModel):
    models: list[ModelResponse]
    total: int


# ── Monitoring ──


class GPUMemoryStats(BaseModel):
    total_bytes: int
    active_bytes: int
    peak_bytes: int
    cache_bytes: int
    available_bytes: int
    utilization_pct: float


class EngineStatsResponse(BaseModel):
    model: Optional[str] = None
    loaded: bool
    running: bool
    active_requests: int
    waiting_requests: int
    step_counter: int
    requests_processed: int
    total_prompt_tokens: int
    total_completion_tokens: int
    uptime_seconds: float
    gpu_memory: Optional[GPUMemoryStats] = None


class SystemStatsResponse(BaseModel):
    cpu_percent: float
    memory_total_bytes: int
    memory_used_bytes: int
    memory_available_bytes: int
    gpu: GPUMemoryStats
    uptime_seconds: float
    python_version: str
    mlx_version: str


class RequestStatsResponse(BaseModel):
    total_requests: int
    active_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    tokens_per_second: float
    requests_per_second: float


# ── Config ──


class EngineConfigUpdate(BaseModel):
    completion_batch_size: Optional[int] = None
    prefill_batch_size: Optional[int] = None
    max_kv_size: Optional[int] = None
    deferred_clear_delay: Optional[int] = None
    cache_cleanup_interval: Optional[int] = None


class AuthTokenCreate(BaseModel):
    name: str = Field(..., description="Token name/description")
    expires_days: Optional[int] = Field(None, description="Days until expiration (None = never)")


class AuthTokenResponse(BaseModel):
    token: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None


class RBACTokenCreate(BaseModel):
    name: str = Field(..., description="Key name/description")
    role: str = Field("user", description="Role: admin, developer, user")
    slo_class: str = Field("standard", description="SLO class: best_effort, standard, premium")
    expires_days: Optional[int] = Field(None, description="Days until expiration (None = never)")
    requests_per_minute: Optional[int] = Field(None, description="Override rate limit (requests/min)")
    tokens_per_minute: Optional[int] = Field(None, description="Override rate limit (tokens/min)")


class RBACTokenResponse(BaseModel):
    key: str = Field(..., description="Raw API key (shown only once)")
    name: str
    role: str
    slo_class: str
    created_at: float
    expires_at: Optional[float] = None
