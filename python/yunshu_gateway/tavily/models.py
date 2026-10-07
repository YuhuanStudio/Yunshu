"""Tavily 2026-10-07 requests, including legacy SDK fields."""

from datetime import date
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CompatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    api_key: str | None = None
    include_usage: bool = False

    @model_validator(mode="before")
    @classmethod
    def empty_optional_values(cls, value):
        if isinstance(value, dict):
            # Older MCP versions send empty strings/arrays for optional scalar fields.
            value = dict(value)
            for key, field in cls.model_fields.items():
                if (
                    key in value
                    and not field.is_required()
                    and value[key] in (None, "", [])
                ):
                    value.pop(key)
        return value


class SearchRequest(CompatRequest):
    query: str = Field(min_length=1, max_length=1500)
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic"
    chunks_per_source: int = Field(3, ge=1, le=3)
    max_results: int = Field(10, ge=0, le=20)
    topic: Literal["general", "news", "finance"] = "general"
    time_range: Literal["day", "week", "month", "year", "d", "w", "m", "y"] | None = (
        None
    )
    start_date: date | None = None
    end_date: date | None = None
    days: int | None = Field(None, ge=0)
    max_age_hours: float | None = Field(None, ge=0)
    fetch_timeout: float | None = Field(None, gt=0, le=60)
    cache_fallback: bool = False
    include_published_date: bool = False
    filter_by_published_date: bool = False
    include_answer: bool | Literal["basic", "advanced"] = False
    include_raw_content: bool | Literal["markdown", "text"] = False
    include_images: bool = False
    include_image_descriptions: bool = False
    include_favicon: bool = False
    include_domains: list[str] = Field(default_factory=list, max_length=300)
    exclude_domains: list[str] = Field(default_factory=list, max_length=150)
    include_domains_mode: Literal["restrict", "prefer"] = "restrict"
    country: str | None = None
    language: str | None = None
    filter_by_language: bool = False
    auto_parameters: bool = False
    exact_match: bool = False
    safe_search: bool = False

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("query is required")
        return value.strip()

    @field_validator("include_domains", "exclude_domains")
    @classmethod
    def domains(cls, values):
        out = []
        for value in values:
            parsed = urlsplit(value if "://" in value else "https://" + value)
            if (
                not parsed.hostname
                or parsed.path not in ("", "/")
                or parsed.query
                or parsed.fragment
                or "*" in value
                or parsed.username
            ):
                raise ValueError(
                    "Domains must be plain hosts, without paths or wildcards"
                )
            out.append(parsed.hostname.lower().removeprefix("www."))
        return out

    @field_validator("country")
    @classmethod
    def full_country_name(cls, value):
        if value is not None and len(value.strip()) < 4:
            raise ValueError("country must be a full country name")
        return value.lower().strip() if value else value

    @model_validator(mode="after")
    def relationships(self):
        if "include_domains_mode" in self.model_fields_set and not self.include_domains:
            raise ValueError("include_domains_mode requires include_domains")
        if self.filter_by_language and not self.language:
            raise ValueError("filter_by_language requires language")
        if self.country and self.topic != "general":
            raise ValueError("country is only available with topic=general")
        if self.time_range and (self.start_date or self.end_date):
            raise ValueError("time_range cannot be combined with dates")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must precede end_date")
        return self


class ExtractRequest(CompatRequest):
    urls: list[str] = Field(min_length=1, max_length=20)
    query: str | None = Field(None, max_length=1500)
    chunks_per_source: int = Field(3, ge=1, le=5)
    extract_depth: Literal["basic", "advanced"] = "basic"
    include_images: bool = False
    include_favicon: bool = False
    format: Literal["markdown", "text"] = "markdown"
    timeout: float | None = Field(None, ge=1, le=60)

    @field_validator("urls", mode="before")
    @classmethod
    def single_url(cls, value):
        return [value] if isinstance(value, str) else value


class CrawlRequest(CompatRequest):
    url: str = Field(min_length=1, max_length=2048)
    max_depth: int = Field(1, ge=1, le=5)
    max_breadth: int = Field(20, ge=1, le=500)
    limit: int = Field(50, ge=1, le=10000)
    instructions: str | None = Field(None, max_length=1500)
    select_paths: list[str] = Field(default_factory=list, max_length=50)
    select_domains: list[str] = Field(default_factory=list, max_length=50)
    exclude_paths: list[str] = Field(default_factory=list, max_length=50)
    exclude_domains: list[str] = Field(default_factory=list, max_length=50)
    allow_external: bool = True
    timeout: float = Field(150, ge=10, le=150)
    chunks_per_source: int = Field(3, ge=1, le=5)
    include_images: bool = False
    include_favicon: bool = False
    extract_depth: Literal["basic", "advanced"] = "basic"
    format: Literal["markdown", "text"] = "markdown"
    categories: list[str] | None = None

    @field_validator(
        "select_paths", "select_domains", "exclude_paths", "exclude_domains"
    )
    @classmethod
    def valid_regex(cls, values):
        import regex

        for value in values:
            if len(value) > 1024:
                raise ValueError("Selector exceeds 1024 characters")
            try:
                regex.compile(value)
            except regex.error as exc:
                raise ValueError("Invalid regex selector") from exc
        return values


class ResearchFile(BaseModel):
    name: str = Field(max_length=200)
    data: str = Field(max_length=4_000_000)
    type: Literal["base64"] = "base64"


class ResearchRequest(CompatRequest):
    input: str = Field(min_length=1, max_length=20000)
    model: Literal["mini", "pro", "auto"] = "auto"
    stream: bool = False
    output_schema: dict | None = None
    citation_format: Literal["numbered", "mla", "apa", "chicago"] = "numbered"
    include_domains: list[str] = Field(default_factory=list, max_length=20)
    exclude_domains: list[str] = Field(default_factory=list, max_length=20)
    output_length: Literal["short", "standard", "long"] = "standard"
    files: list[ResearchFile] = Field(default_factory=list, max_length=5)

    @field_validator("include_domains", "exclude_domains")
    @classmethod
    def domains(cls, values):
        return SearchRequest.domains(values)

    @field_validator("output_schema")
    @classmethod
    def schema(cls, value):
        if value is not None:
            from jsonschema import Draft202012Validator, SchemaError

            if not isinstance(value.get("properties"), dict):
                raise ValueError("output_schema requires properties")
            value = {"type": "object", **value}
            try:
                Draft202012Validator.check_schema(value)
            except SchemaError as exc:
                raise ValueError("Invalid output_schema") from exc
        return value


class LabeledScore(BaseModel):
    label: str = Field(max_length=64)
    value: float | str

    @field_validator("value", mode="before")
    @classmethod
    def score_length(cls, value):
        if isinstance(value, str) and len(value) > 64:
            raise ValueError("Score exceeds 64 characters")
        return value


class URLScore(BaseModel):
    id: str | None = Field(None, max_length=200)
    url: str | None = Field(None, max_length=2000)
    agent_score: float | str | None = None
    scores: list[LabeledScore] = Field(default_factory=list, max_length=20)
    comment: str | None = Field(None, max_length=2000)

    @model_validator(mode="after")
    def identity(self):
        if not self.id and not self.url:
            raise ValueError("urls_scores item requires id or url")
        return self


class FeedbackRequest(CompatRequest):
    request_id: str | None = Field(None, max_length=200)
    session_id: str | None = Field(None, max_length=200)
    agent_score: float | str | None = None
    human_score: float | str | None = None
    extra_scores: list[LabeledScore] = Field(default_factory=list, max_length=50)
    comment: str | None = Field(None, max_length=10000)
    response_delivered: str | None = Field(None, max_length=50000)
    used_ids: list[str] = Field(default_factory=list, max_length=100)
    used_urls: list[str] = Field(default_factory=list, max_length=100)
    used_citations: list[str] = Field(default_factory=list, max_length=100)
    urls_scores: list[URLScore] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def limits(self):
        if not self.request_id and not self.session_id:
            raise ValueError("request_id or session_id is required")
        for values, cap in (
            (self.used_ids, 200),
            (self.used_urls, 2000),
            (self.used_citations, 2000),
        ):
            if any(len(v) > cap for v in values):
                raise ValueError("Feedback item exceeds length limit")
        for score in (self.agent_score, self.human_score):
            if isinstance(score, str) and len(score) > 64:
                raise ValueError("Score exceeds 64 characters")
        return self


class LogsRequest(CompatRequest):
    limit: int = Field(10, ge=1, le=10000)
    start_date: date | None = None
    end_date: date | None = None
    endpoints: list[Literal["search", "extract", "map", "crawl", "research"]] = Field(
        default_factory=list
    )
    project_id: str | None = None
    filter_by_api_key: bool = False
