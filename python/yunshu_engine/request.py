from __future__ import annotations

"""Yunshu request management.

Deep MLX integration:
- RequestStatus enum for lifecycle tracking
- SamplingParams with all mlx-lm sampler options
- Request with per-request state, token tracking, detokenizer
- RequestOutput with incremental + cumulative output
"""

import contextlib
import enum
import time
from dataclasses import dataclass, field
from typing import Any


class RequestStatus(enum.IntEnum):
    """Request lifecycle states.

    WAITING -> PREFILLING -> RUNNING -> FINISHED_*
    PREFILLING is used during external prefill (chunked progress tracking).
    """

    WAITING = enum.auto()
    PREFILLING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_TIMEOUT = enum.auto()


# Attach helper methods after class definition to avoid NameError in Python 3.13
# (IntEnum class body cannot reference itself in method bodies during definition)
def _rs_is_finished(status: int) -> bool:
    """Check if a request status represents a finished state."""
    return status >= RequestStatus.FINISHED_STOPPED


def _rs_finish_reason(status: int) -> str | None:
    """Map a request status to its finish reason string."""
    mapping = {
        RequestStatus.FINISHED_STOPPED: "stop",
        RequestStatus.FINISHED_LENGTH: "length",
        RequestStatus.FINISHED_ABORTED: "abort",
        RequestStatus.FINISHED_ERROR: "error",
        RequestStatus.FINISHED_TIMEOUT: "timeout",
    }
    return mapping.get(status)


RequestStatus.is_finished = staticmethod(_rs_is_finished)
RequestStatus.finish_reason = staticmethod(_rs_finish_reason)


@dataclass
class SamplingParams:
    """Generation parameters — maps to mlx-lm's make_sampler."""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[int, float] | None = None
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    logprobs: bool = False
    top_logprobs: int | None = None
    seed: int | None = None
    priority: int = 0
    thinking_budget: int | None = None
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.0
    # Structured output (JSON schema constrained generation)
    json_schema: dict | str | None = None
    # Grammar constraint (regex, choice, CFG — processed by ConstraintFactory)
    grammar: dict | str | None = None
    # Custom user-provided logits processors
    # Each processor is a callable(tokens: list[int], logits: mx.array) -> mx.array
    logits_processors: list | None = None
    # min_tokens — floor on generated tokens; EOS/stop ids are masked to -inf
    # until this many tokens have been generated (prevents empty/too-short output).
    # ignore_eos — keep generating past EOS up to max_tokens (throughput benchmarking,
    # forced-length generation).
    # suppress_tokens — hard-ban these token ids from the output (logits → -inf).
    min_tokens: int = 0
    ignore_eos: bool = False
    suppress_tokens: list[int] | None = None


@dataclass
class RequestOutput:
    """Per-step output from the engine.

    Supports both incremental (new_text) and cumulative (output_text) output.
    """

    request_id: str
    new_token_ids: list[int] = field(default_factory=list)
    new_text: str = ""
    output_token_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    finished: bool = False
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprobs: Any = None
    reasoning_tokens: int = 0
    current_state: str = "normal"
    error: str | None = None
    cached_tokens: int = 0
    ttft_ms: float = 0.0

    # Chunked prefill progress
    # Tuple of (processed_tokens, total_tokens) for chunked prefill progress reporting.
    # Non-None during prefill phase, None once generation starts.
    prefill_progress: tuple[int, int] | None = None

    # Backward compat aliases for legacy Engine path
    @property
    def token_text(self) -> str:
        return self.new_text

    @property
    def token_id(self) -> int:
        return self.new_token_ids[-1] if self.new_token_ids else -1

    @property
    def logprob(self) -> Any:
        return self.logprobs

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


@dataclass
class Request:
    """Per-request state.

    Tracks the full lifecycle: prompt tokenization → generation → output.
    Integrates with mlx-lm's BatchGenerator via batch_uid.
    """

    request_id: str
    prompt: str | list[int] | list[dict]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    arrival_time: float = field(default_factory=time.monotonic)
    priority: int = 0

    # Tokenization state
    prompt_token_ids: list[int] = field(default_factory=list)
    num_prompt_tokens: int = 0
    num_computed_tokens: int = 0

    # Generation state
    status: RequestStatus = RequestStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    finish_reason: str | None = None

    # BatchGenerator integration
    batch_uid: int | None = None
    num_preemptions: int = 0

    # Per-request detokenizer (never pool — reset() leaks byte buffers)
    detokenizer: Any = None

    # Streaming output queue (asyncio.Queue[RequestOutput | None])
    output_queue: Any = None
    done_event: Any = None

    # Chat context
    enable_thinking: bool | None = None

    # VLM fields
    rope_deltas: float = (
        0.0  # mRoPE position delta for multi-modal models (Qwen3 Omni, etc.)
    )
    vlm_inputs_embeds: Any | None = (
        None  # Precomputed vision embeddings from VLM encoder
    )
    vlm_extra_kwargs: dict | None = None  # Extra kwargs for VLM-specific generation
    vlm_image_hash: str | None = (
        None  # Content hash of input images for feature cache lookup
    )

    # Prefix cache fields
    prompt_cache: Any = None
    cached_tokens: int = 0
    remaining_tokens: list[int] | None = None

    # Multimodal content
    images: list[Any] | None = None
    videos: list[Any] | None = None  # Video input paths/frames for VLM understanding

    # Timing (for ServerMetrics integration)
    prefill_start: float = 0.0
    prefill_end: float = 0.0
    generation_start: float = 0.0
    generation_end: float = 0.0

    # Priority aging — time.monotonic() when request enters the waiting queue.
    # Used by scheduler._schedule_waiting() to compute effective_priority =
    # original_priority + (now - _submit_time) * aging_weight.
    # Set by scheduler._add_request() and reset on preemption.
    _submit_time: float = 0.0

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def max_tokens(self) -> int:
        return self.sampling_params.max_tokens

    @property
    def prefill_duration(self) -> float:
        return max(0.0, self.prefill_end - self.prefill_start)

    @property
    def generation_duration(self) -> float:
        return max(0.0, self.generation_end - self.generation_start)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    # Valid predecessor states for each finished state.
    # PREEMPTED/WAITING can also go to FINISHED_* (queue_full, timeout, abort).
    _FINISH_VALID_PREDECESSORS = frozenset(
        {
            RequestStatus.WAITING,
            RequestStatus.PREFILLING,
            RequestStatus.RUNNING,
            RequestStatus.PREEMPTED,
        }
    )

    def set_finished(self, status: RequestStatus, reason: str | None = None) -> None:
        """Transition to a finished state with optional reason.

        Validates the transition: already-finished requests cannot transition
        to a different finished state (prevents masking bugs).  Non-finished
        states that are not valid predecessors (e.g. another FINISHED_* state)
        are silently rejected.
        """
        if RequestStatus.is_finished(self.status):
            # Already finished — first reason is authoritative; don't overwrite.
            import logging as _logging

            _logging.getLogger(__name__).debug(
                f"Request {self.request_id}: ignoring re-finish "
                f"({self.status.name} -> {status.name}), "
                f"original reason={self.finish_reason}"
            )
            return
        if not RequestStatus.is_finished(self.status):
            # Validate the current state is a legal predecessor for a finish.
            if self.status not in self._FINISH_VALID_PREDECESSORS:
                return
        self.status = status
        self.finish_reason = reason or RequestStatus.finish_reason(status)
        if not self.generation_end:
            self.generation_end = time.monotonic()
        # Ensure done_event is set so no coroutine hangs waiting on it
        if self.done_event is not None:
            with contextlib.suppress(Exception):
                self.done_event.set()

    def release_resources(self) -> None:
        """Release heavy references (MLX arrays, embeddings, media, caches).

        Call this when the request is finalized to avoid holding GPU memory
        via stale references in long-lived data structures. Safe to call
        multiple times — sets fields to None after releasing.
        """
        self.prompt_cache = None
        self.vlm_inputs_embeds = None
        self.vlm_extra_kwargs = None
        self.images = None
        self.videos = None
        self.detokenizer = None
        if hasattr(self, "_spec_draft_cache"):
            self._spec_draft_cache = None
        # Clear large lists that may hold many tokens
        self.prompt_token_ids = []
        self.output_token_ids = []
        self.remaining_tokens = None

    def __lt__(self, other: Request) -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.arrival_time < other.arrival_time

    def __hash__(self) -> int:
        return hash(self.request_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Request):
            return False
        return self.request_id == other.request_id
