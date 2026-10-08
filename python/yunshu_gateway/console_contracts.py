"""OpenAPI contracts for single-node console metadata."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DraftDepth(BaseModel):
    position: int
    drafted: int
    accepted: int
    acceptance_rate: float | None


class SpecMode(BaseModel):
    mode: str
    num_drafts: int
    num_draft_tokens: int
    num_accepted_tokens: int
    per_depth: list[DraftDepth]


class SpecAggregate(BaseModel):
    object: str
    scope: str
    position_basis: str
    data: list[SpecMode]


class CacheEntry(BaseModel):
    model: str
    id: str
    namespace: str
    tokens: int
    bytes_logical: int | None
    bytes_physical: int | None = Field(
        description="Attributed storage bytes, not allocator footprint"
    )
    tier: str
    last_hit: float | None
    hits: int


class CacheEvent(BaseModel):
    model: str
    id: int
    t: float
    action: str
    entry_id: str
    tokens: int
    tier: str
    reason: str
    request_id: str | None


class CacheView(BaseModel):
    object: str
    data: list[CacheEntry]
    count: int
    events: list[CacheEvent]
    event_capacity: int
    models: list[str]
    scope: str


class CacheClear(BaseModel):
    object: str
    models: list[str]
    scope: str
    persistent_cleared: bool


class RequestMetadata(BaseModel):
    t: float | None
    request_id: str
    model: str | None
    path: str | None
    stream: bool
    status: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    prefill_tps: float | None
    decode_tps: float | None
    ttft_ms: float | None
    latency: dict | None
    energy: dict | None
    speculative: dict | None
    cache: dict | None
    structured_output: dict | None = None
    reasons: dict | None = None


class HistoryPage(BaseModel):
    object: str
    enabled: bool
    data: list[RequestMetadata]
    count: int
    next_cursor: str | None


class BundleManifest(BaseModel):
    object: str
    format: str
    included: list[str]
    redacted: list[str]
    excluded: list[str]
    error_line_limit: int
    error_line_max_chars: int
    generator: str


class DiagnosticsBundle(BaseModel):
    created: str
    version: str
    platform: dict
    packages: dict
    settings_changed: list[dict]
    paths: dict
    doctor: list[dict]
    caches: list[dict]
    errors: dict
    excluded: str


class UnloadImpact(BaseModel):
    in_flight_policy: str
    waits: bool
    interrupts: bool


class LoadImpact(BaseModel):
    would_evict: list[str]
    blocked: bool
    advisory: bool
    post_load_pressure: str


class ModelImpact(BaseModel):
    object: str
    model: str
    unload: UnloadImpact
    load: LoadImpact
