from __future__ import annotations

"""OpenAI Completions API compatible router (text completions, not chat).

Supports:
- Text completions (non-chat)
- Streaming and non-streaming
- Logprobs
- Echo mode
"""
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from yunshu_engine.tracing import get_inference_tracer, get_structured_logger

from ..engine import get_engine, get_engine_for_model
from .chat import (
    _apply_lora_adapter,
    _normalize_finish_reason,
    _parse_response_format,
    _release_lora_adapter,
)

logger = logging.getLogger(__name__)

_MAX_STREAMING_TEXT_BUFFER = 1 * 1024 * 1024
_TRUNCATE_KEEP = 512 * 1024
import contextlib

from ..streaming import (
    format_openai_completion_chunk,
    format_openai_completion_usage_chunk,
    format_openai_done,
    run_with_disconnect_guard,
    validate_context_window,
    validate_prefill_memory,
)
from .chat import _validate_sampling_params
from .models import _check_permission

router = APIRouter(tags=["completions"])


def _classify_prompts_lite(prompt) -> list[str]:
    """Lightweight prompt classifier used when no engine/tokenizer is available.

    Returns a list of string prompts. For token-id forms (list[int],
    list[list[int]]) the caller must decode via a tokenizer; this helper
    falls back to a textual representation only as a last resort.
    """
    if isinstance(prompt, str):
        return [prompt]
    if isinstance(prompt, list):
        if not prompt:
            return [""]
        # list[str]
        if all(isinstance(p, str) for p in prompt):
            return list(prompt)
        # list[int] — single token-id prompt
        if all(isinstance(p, int) for p in prompt):
            return [" ".join(str(t) for t in prompt)]
        # list[list[int]] — batched token-id prompts
        if all(isinstance(p, list) for p in prompt):
            return [" ".join(str(t) for t in inner) for inner in prompt]
    return [str(prompt)]


def _normalize_prompts(prompt, tokenizer) -> list[str]:
    """Normalize an OpenAI-style prompt field into a list of string prompts.

    OpenAI accepts: str | list[str] | list[int] | list[list[int]].
    list[int] is a single token-id prompt; list[list[int]] is N batched
    token-id prompts. Each element becomes its own choice in the response.
    """
    if isinstance(prompt, str):
        return [prompt]
    if not isinstance(prompt, list) or not prompt:
        return [""]
    # list[str]
    if all(isinstance(p, str) for p in prompt):
        return list(prompt)
    # list[int]
    if all(isinstance(p, int) for p in prompt):
        if tokenizer is not None:
            try:
                return [tokenizer.decode(prompt)]
            except Exception:
                pass
        return [" ".join(str(t) for t in prompt)]
    # list[list[int]]
    if all(isinstance(p, list) for p in prompt):
        out: list[str] = []
        for inner in prompt:
            if tokenizer is not None:
                try:
                    out.append(tokenizer.decode(inner))
                    continue
                except Exception:
                    pass
            out.append(" ".join(str(t) for t in inner))
        return out
    return [str(prompt)]


def _record_metrics(prompt_tokens: int, completion_tokens: int) -> None:
    """Record token counts to metrics middleware + tracing counters for completions.

    Does NOT call ServerMetrics.record_request_complete — the engine
    already records that with full detail; the old duplicate here doubled the
    billing-feeding totals (see chat.py _record_metrics).
    """
    try:
        from ..middleware.metrics import get_metrics

        get_metrics().record_tokens(prompt_tokens, completion_tokens)
        get_metrics().record_inference()
    except Exception:
        logger.debug("metrics recording failed", exc_info=True)
    try:
        from yunshu_engine.tracing import get_metrics_v2

        get_metrics_v2().counter(
            "yunshu_tokens_total", {"type": "prompt"}, prompt_tokens
        )
        get_metrics_v2().counter(
            "yunshu_tokens_total", {"type": "completion"}, completion_tokens
        )
    except Exception:
        logger.debug("metrics recording failed", exc_info=True)
    # feed the per-request TPM box (see usage_context).
    try:
        from ..usage_context import record_billed_tokens

        record_billed_tokens((prompt_tokens or 0) + (completion_tokens or 0))
    except Exception:
        logger.debug("billed-token accounting failed", exc_info=True)


class StreamOptions(BaseModel):
    """OpenAI stream_options parameter."""

    include_usage: bool = False


class CompletionRequest(BaseModel):
    model: str
    # OpenAI Completions API permits prompt as: string, list[string], list[int],
    # or list[list[int]] (batched token-id prompts). Each list element becomes
    # its own choice in the response.
    prompt: str | list[str] | list[int] | list[list[int]]
    max_tokens: int = Field(default=128, ge=0, le=131072)
    max_completion_tokens: int | None = Field(default=None, ge=0, le=131072)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: dict[int, float] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    # OpenAI permits stop as either a single string or a list[str] (up to 4).
    # Normalize to list[str] internally in @field_validator below.
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    echo: bool = False
    logprobs: int = Field(default=0, ge=0, le=5)
    top_logprobs: int | None = Field(default=None, ge=0, le=5)
    seed: int | None = None
    spec_decode: bool = False
    enable_thinking: bool | None = None
    thinking_budget: int | None = Field(default=None, ge=1, le=32768)
    response_format: dict | None = None
    reasoning_effort: str | None = None
    xtc_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    xtc_threshold: float = Field(
        default=0.0, ge=0.0, le=0.5
    )  # engine requires [0,0.5]; le=1.0 made out-of-range 500 not 422
    # serving parity:
    min_tokens: int = Field(default=0, ge=0)
    ignore_eos: bool = False
    suppress_tokens: list[int] | None = None
    # per-prompt-token logprobs (eval/perplexity). int = number of top alternatives.
    prompt_logprobs: int | None = None
    lora_adapter: str | None = None
    grammar: dict | None = None  # {"type": "regex", "pattern": "..."} etc.
    # guided-decoding aliases (mirror chat completions).
    # SECURITY: length-capped to bound regex/grammar construction (DoS).
    guided_regex: str | None = Field(default=None, max_length=2048)
    guided_choice: list[str] | None = None
    guided_grammar: str | None = Field(default=None, max_length=32768)
    guided_json: dict | None = None
    user: str | None = None
    suffix: str | None = None  # OpenAI: suffix after inserted text completion
    best_of: int | None = Field(
        default=None, ge=1, le=128
    )  # OpenAI: server-side best-of selection
    priority: int = Field(default=0, ge=0, le=100)
    n: int = Field(default=1, ge=1, le=128)
    logits_processors: list | None = (
        None  # SAMP-2: User-provided custom logits processors
    )
    timeout: float | None = Field(
        default=None, ge=1.0, le=600.0
    )  # Request timeout in seconds

    def effective_max_tokens(self) -> int:
        """Return max_completion_tokens if set, else max_tokens (OpenAI SDK compat)."""
        return (
            self.max_completion_tokens
            if self.max_completion_tokens is not None
            else self.max_tokens
        )

    @field_validator("stop", mode="before")
    @classmethod
    def _normalize_stop(cls, v):
        """Normalize OpenAI's str|list[str] `stop` to internal list[str]."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        # fold guided_* aliases into grammar/response_format (native wins).
        if self.grammar is None:
            if self.guided_regex is not None:
                self.grammar = {"type": "regex", "pattern": self.guided_regex}
            elif self.guided_choice:
                self.grammar = {"type": "choice", "choices": self.guided_choice}
            elif self.guided_grammar is not None:
                self.grammar = {"type": "cfg", "grammar": self.guided_grammar}
        if self.guided_json is not None and self.response_format is None:
            self.response_format = {
                "type": "json_schema",
                "json_schema": {"schema": self.guided_json},
            }
        # Validate prompt: string must be non-empty, list must have elements
        if isinstance(self.prompt, str) and not self.prompt.strip():
            raise ValueError("prompt: cannot be empty or whitespace-only")
        if isinstance(self.prompt, list) and not self.prompt:
            raise ValueError("prompt: cannot be an empty list")
        if self.top_logprobs is not None and self.logprobs <= 0:
            raise ValueError("top_logprobs: can only be set when logprobs > 0")
        if self.stop and len(self.stop) > 16:
            raise ValueError("stop: maximum 16 stop sequences")
        if self.stop_token_ids and len(self.stop_token_ids) > 16:
            raise ValueError("stop_token_ids: maximum 16 stop token IDs")
        # Validate response_format type if provided
        if self.response_format is not None:
            rf_type = (
                self.response_format.get("type")
                if isinstance(self.response_format, dict)
                else None
            )
            if rf_type not in ("json_object", "json_schema", "text", None):
                raise ValueError(
                    f"response_format.type: must be 'json_object', 'json_schema', or 'text', got '{rf_type}'"
                )
        # Validate grammar type if provided
        if self.grammar is not None:
            gtype = self.grammar.get("type") if isinstance(self.grammar, dict) else None
            if gtype not in ("json", "regex", "choice", "cfg", None):
                raise ValueError(
                    f"grammar.type: must be one of 'json', 'regex', 'choice', 'cfg', got '{gtype}'"
                )
        # Validate best_of: must be >= n, and not used with streaming
        if self.best_of is not None:
            if self.best_of < self.n:
                raise ValueError(f"best_of ({self.best_of}) must be >= n ({self.n})")
            if self.stream:
                raise ValueError("best_of is not supported when stream is True")
        # n > 1 with streaming is not supported
        if self.stream and self.n > 1:
            raise ValueError(
                "n > 1 is not supported when stream is True. "
                "Use non-streaming mode for multiple choices."
            )
        # validate logit_bias values (chat.py + responses.py do this;
        # completions was missing it). A NaN/Inf bias makes softmax all-NaN → garbage output
        # instead of a clean 422. Excludes bool (subclass of int).
        if self.logit_bias:
            import math

            for k, v in self.logit_bias.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(f"logit_bias[{k}]: must be a finite number")
                if math.isnan(v) or math.isinf(v):
                    raise ValueError(f"logit_bias[{k}]: must be a finite number")
                if v < -100.0 or v > 100.0:
                    raise ValueError(
                        f"logit_bias[{k}]={v}: must be between -100 and 100"
                    )
        # bound seed to 64-bit signed range (parity with chat.py:449 — downstream
        # samplers seed numpy/Gumbel PRNGs that overflow on an out-of-range int).
        if self.seed is not None and (self.seed < -(2**63) or self.seed >= 2**63):
            raise ValueError("seed: must be within the 64-bit signed integer range")
        return self


@router.post("/completions", response_model=None)
async def create_completion(req: CompletionRequest, request: Request):
    """OpenAI-compatible text completion endpoint."""
    _check_permission(request, "can_infer")
    _validate_sampling_params(req.temperature, req.effective_max_tokens(), req.top_p)
    _rbac_key = getattr(request.state, "rbac_key", None)
    if _rbac_key is not None and not _rbac_key.can_access_model(req.model):
        raise HTTPException(
            status_code=403,
            detail=f"Model '{req.model}' not accessible with this API key",
        )

    # Fast path: max_tokens=0 returns prompt_tokens only (OpenAI API behavior).
    _effective_mt = req.effective_max_tokens()
    if _effective_mt == 0:
        # Classify prompt into list-of-string-prompts for token counting.
        _fp_prompts = _classify_prompts_lite(req.prompt)
        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        prompt_tok = 0
        # Resolve tokenizer: single-engine first, then multi-model manager.
        # Multi-model mode returns None from get_engine() — must look up via
        # the model_manager so prompt_tok isn't silently 0.
        tokenizer = None
        engine = get_engine()
        if engine and engine.is_loaded:
            tokenizer = getattr(engine, "_tokenizer", None)
        if tokenizer is None:
            try:
                from ..engine import get_model_manager

                _mm = get_model_manager()
                if _mm is not None:
                    _entry = _mm.get_entry(req.model)
                    if (
                        _entry is not None
                        and _entry.is_loaded
                        and _entry.engine is not None
                    ):
                        tokenizer = getattr(_entry.engine, "_tokenizer", None)
            except Exception:
                logger.debug(
                    "max_tokens=0 multi-model tokenizer lookup failed", exc_info=True
                )
        if tokenizer is not None:
            try:
                if (
                    isinstance(req.prompt, list)
                    and req.prompt
                    and isinstance(req.prompt[0], int)
                ):
                    prompt_tok = len(req.prompt)
                elif (
                    isinstance(req.prompt, list)
                    and req.prompt
                    and isinstance(req.prompt[0], list)
                ):
                    # list[list[int]] — sum per-prompt token count
                    prompt_tok = sum(len(p) for p in req.prompt)
                else:
                    # list[str] or str — sum across all string prompts
                    prompt_tok = sum(len(tokenizer.encode(p)) for p in _fp_prompts)
            except Exception:
                logger.debug("max_tokens=0 token encode failed", exc_info=True)
        # echo text must be the DECODED prompt. For a token-id
        # array prompt, _classify_prompts_lite returns the raw numeric string ("785 3489
        # …"), so echo previously returned numeric IDs instead of the prompt text. Decode
        # token-array prompts here (the tokenizer is already resolved above for counting).
        _echo_texts = _fp_prompts
        if (
            req.echo
            and tokenizer is not None
            and isinstance(req.prompt, list)
            and req.prompt
        ):
            try:
                if isinstance(req.prompt[0], int):
                    _echo_texts = [tokenizer.decode(req.prompt)]
                elif isinstance(req.prompt[0], list):
                    _echo_texts = [tokenizer.decode(p) for p in req.prompt]
            except Exception:
                logger.debug("max_tokens=0 echo decode failed", exc_info=True)
                _echo_texts = _fp_prompts
        return JSONResponse(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                # n choices PER prompt (OpenAI returns n completions even at max_tokens=0;
                # the old loop ignored req.n and returned only one per prompt).
                "choices": [
                    {
                        "index": idx,
                        "text": (
                            _echo_texts[idx // max(req.n, 1)]
                            if req.echo and (idx // max(req.n, 1)) < len(_echo_texts)
                            else ""
                        ),
                        "finish_reason": "length",
                    }
                    for idx in range(len(_fp_prompts) * max(req.n, 1))
                ],
                "usage": {
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": 0,
                    "total_tokens": prompt_tok,
                },
            }
        )

    # Validate stop strings: reject empty strings (would match immediately)
    if req.stop:
        req.stop = [s for s in req.stop if s]
        if not req.stop:
            req.stop = None
    engine = get_engine()

    if engine is None or not engine.is_loaded or not engine.resolve_model_id(req.model):
        try:
            engine = await get_engine_for_model(req.model)
        except (KeyError, Exception):
            raise HTTPException(
                status_code=404, detail=f"Model '{req.model}' not found"
            ) from None

    # OpenAI accepts str | list[str] | list[int] | list[list[int]] for prompt.
    # Normalize into a list of string prompts; each becomes its own choice.
    _prompts = _normalize_prompts(req.prompt, getattr(engine, "_tokenizer", None))
    # Streaming with multi-prompt produces interleaved choice_index output; keep
    # the streaming path single-prompt-only for now and reject the combo.
    if req.stream and len(_prompts) > 1:
        raise HTTPException(
            status_code=400,
            detail="Multi-prompt (list[str] / list[list[int]]) is not supported when stream is True. "
            "Use non-streaming mode for multiple prompts.",
        )
    # Single-prompt fast path (streaming + most non-streaming clients).
    prompt = _prompts[0]

    # context-window + prefill-memory validation (chat/responses/
    # anthropic all do this; completions was missing it — an over-window prompt produced
    # garbage/degraded output and monopolized the executor with no clean 400).
    try:
        _ctx_tok = getattr(engine, "_tokenizer", None)
        if _ctx_tok is not None:
            _est_tokens = max((len(_ctx_tok.encode(p)) for p in _prompts), default=0)
            validate_context_window(_est_tokens, req.model, engine)
            validate_prefill_memory(_est_tokens)
    except HTTPException:
        raise
    except Exception:
        logger.debug("completions context window validation failed", exc_info=True)

    # Extract JSON schema from response_format or grammar (shared with chat router)
    json_schema = _parse_response_format(req.response_format, req.grammar)

    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    # Structured tracing + logging
    tracer = get_inference_tracer()
    slog = get_structured_logger()
    trace_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    tracer.start_trace(
        trace_id,
        metadata={
            "model": req.model,
            "max_tokens": req.effective_max_tokens(),
            "temperature": req.temperature,
            "stream": req.stream,
            "endpoint": "/completions",
        },
    )
    tracer.span(trace_id, "prefill", {"model": req.model})
    slog.info(
        "inference_request",
        model=req.model,
        trace_id=trace_id,
        max_tokens=req.effective_max_tokens(),
        stream=req.stream,
    )

    if req.stream:
        return StreamingResponse(
            _stream_completion(
                engine,
                prompt,
                req,
                completion_id,
                request,
                json_schema=json_schema,
                trace_id=trace_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming
    from yunshu_engine.batched_engine import BatchedEngine

    is_batched = isinstance(engine, BatchedEngine)

    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)

    # Register with request tracker for cancellation support in non-streaming path
    _ns_tracker = None
    _ns_gen = None
    _ns_cancel_event = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker

        _ns_tracker = get_request_tracker()
        _ns_gen = _ns_tracker.register(completion_id, req.model)
        _ns_cancel_event = _ns_gen.cancel_event
    except Exception:
        _ns_tracker = None

    try:
        # prompt_logprobs is a per-request property (same prompt → same
        # for all n choices), so capture it in an outer holder instead of
        # threading it through the best_of/multi-prompt result tuple machinery.
        _prompt_lp_holder: dict = {}

        async def _gen_one(idx: int, _prompt: str):
            # when best_of > n, the server must score candidates to
            # return the best — so force the engine to compute logprobs internally even if
            # the client didn't request them in the response (else _avg_logprob got -inf
            # for every candidate → best_of degenerated to "keep the first", wasting the
            # extra generations). The forced lp is stripped from the response below.
            _need_sel = req.best_of is not None and req.best_of > max(req.n, 1)
            _want_lp = (req.logprobs > 0) or _need_sel
            if is_batched:
                result = await engine.generate(
                    prompt=_prompt,
                    max_tokens=req.effective_max_tokens(),
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    min_p=req.min_p,
                    repetition_penalty=req.repetition_penalty,
                    frequency_penalty=req.frequency_penalty,
                    presence_penalty=req.presence_penalty,
                    logit_bias=req.logit_bias,
                    stop=req.stop,
                    stop_token_ids=req.stop_token_ids,
                    seed=(req.seed + idx) if req.seed is not None else None,
                    spec_decode=req.spec_decode,
                    enable_thinking=req.enable_thinking,
                    thinking_budget=req.thinking_budget,
                    json_schema=json_schema,
                    reasoning_effort=req.reasoning_effort,
                    xtc_probability=req.xtc_probability,
                    xtc_threshold=req.xtc_threshold,
                    logprobs=_want_lp,
                    top_logprobs=(
                        req.top_logprobs
                        if req.top_logprobs is not None
                        else req.logprobs
                    ),
                    priority=req.priority,
                    logits_processors=req.logits_processors,
                    cancel_event=_ns_cancel_event,
                    timeout_seconds=req.timeout,
                    lora_adapter=loaded_adapter,
                    min_tokens=req.min_tokens,
                    ignore_eos=req.ignore_eos,
                    suppress_tokens=req.suppress_tokens,
                    # echo=true + logprobs=N must return logprobs for the PROMPT
                    # tokens too (OpenAI contract, used by perplexity/eval clients). That
                    # needs a prompt-token forward, which the prompt_logprobs machinery
                    # already does — so force it on (top_k = the legacy logprobs count) for
                    # the echo case even when the client didn't set the separate
                    # prompt_logprobs field. _format_logprobs then prepends the prompt
                    # tokens. (Single-prompt only; batched-prompt echo+logprobs stays a
                    # documented gap, like the existing per-prompt prompt_logprobs limit.)
                    prompt_logprobs=(
                        req.prompt_logprobs
                        if req.prompt_logprobs is not None
                        else (
                            req.logprobs
                            if (req.echo and req.logprobs and len(_prompts) == 1)
                            else None
                        )
                    ),
                )
                if (
                    req.prompt_logprobs is not None
                    and getattr(result, "prompt_logprobs", None) is not None
                ):
                    _prompt_lp_holder["v"] = result.prompt_logprobs
                text = result.text
                pt = result.prompt_tokens
                ct = result.completion_tokens
                fr = _normalize_finish_reason(result.finish_reason)
                rt = getattr(result, "reasoning_tokens", 0)
                lp = None
                if _want_lp:
                    lp = _format_logprobs(
                        result,
                        getattr(engine, "_tokenizer", None),
                        req.top_logprobs
                        if req.top_logprobs is not None
                        else req.logprobs,
                        echo=req.echo,
                        prompt=_prompt,
                    )
            else:
                state = await engine.generate(
                    prompt=_prompt,
                    max_tokens=req.effective_max_tokens(),
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    min_p=req.min_p,
                    repetition_penalty=req.repetition_penalty,
                    frequency_penalty=req.frequency_penalty,
                    presence_penalty=req.presence_penalty,
                    logit_bias=req.logit_bias,
                    stop=req.stop,
                    stop_token_ids=req.stop_token_ids,
                    seed=(req.seed + idx) if req.seed is not None else None,
                    enable_thinking=req.enable_thinking,
                    thinking_budget=req.thinking_budget,
                    reasoning_effort=req.reasoning_effort,
                    xtc_probability=req.xtc_probability,
                    xtc_threshold=req.xtc_threshold,
                    spec_decode=req.spec_decode,
                    json_schema=json_schema,
                    logprobs=_want_lp,
                    top_logprobs=(
                        req.top_logprobs
                        if req.top_logprobs is not None
                        else req.logprobs
                    ),
                    priority=req.priority,
                    logits_processors=req.logits_processors,
                    cancel_event=_ns_cancel_event,
                    timeout_seconds=req.timeout,
                    lora_adapter=loaded_adapter,
                    # parity with chat.py + the batched branch — these were dropped
                    # on the non-batched/streaming completions paths, so ignore_eos/min_tokens
                    # were no-ops for the default (fast-path) case (generation still stopped at
                    # EOS) and suppress_tokens was ignored.
                    min_tokens=req.min_tokens,
                    ignore_eos=req.ignore_eos,
                    suppress_tokens=req.suppress_tokens,
                )
                # VLMEngine.generate() returns dict, not object with attributes
                if isinstance(state, dict):
                    text = state.get("text", "")
                    pt = state.get("prompt_tokens", 0)
                    ct = state.get("completion_tokens", 0)
                    fr = _normalize_finish_reason(state.get("finish_reason", "stop"))
                    rt = state.get("reasoning_tokens", 0)
                else:
                    # CRITICAL: GenerationOutput uses text/
                    # prompt_tokens/completion_tokens (batched_engine.py:84-87).
                    # Prior code used non-existent generated_text/*_token_count
                    # → AttributeError on every BatchedEngine completion.
                    text = getattr(state, "text", None) or getattr(
                        state, "generated_text", ""
                    )
                    pt = getattr(state, "prompt_tokens", None) or getattr(
                        state, "prompt_token_count", 0
                    )
                    ct = getattr(state, "completion_tokens", None) or getattr(
                        state, "completion_token_count", 0
                    )
                    fr = _normalize_finish_reason(
                        getattr(state, "finish_reason", None) or "stop"
                    )
                    rt = getattr(state, "reasoning_tokens", 0)
                lp = None
                # gate on _want_lp, not req.logprobs>0, so the
                # legacy (non-batched) branch ALSO populates logprobs when best_of needs
                # them to score candidates (matches the batched branch at line ~456).
                # Without this, best_of on a legacy engine left every candidate's lp=None
                # → _avg_logprob returned -inf for all → best_of degenerated to "keep the
                # first n". The strip-back at line ~610 removes them when the client didn't ask.
                if _want_lp:
                    lp = _format_logprobs(
                        state,
                        getattr(engine, "_tokenizer", None),
                        req.top_logprobs
                        if req.top_logprobs is not None
                        else req.logprobs,
                        echo=req.echo,
                        prompt=_prompt,
                    )

            if req.echo:
                text = _prompt + text
            # OpenAI's `suffix` is fill-in-the-middle CONTEXT (the text AFTER the
            # insertion point) used only to condition generation — it is NEVER returned in
            # choices[].text. The old code appended it to the output, corrupting the
            # completion (glued the user's suffix onto the end) AND the model never saw it as
            # context. This engine has no FIM template, so drop it rather than corrupt the
            # output. (Implement real FIM here if/when a FIM-capable model path is added.)
            # Determine the generation result object for extracting cached_tokens.
            # Both branches assign to either `result` (batched) or `state` (legacy),
            # but the variable names are different — use a unified accessor.
            _gen_result = result if is_batched else state
            # VLM legacy path returns a dict (no attributes) — getattr would always
            # yield 0, dropping cached_tokens for VLM-via-/completions cache hits.
            if isinstance(_gen_result, dict):
                _cached = _gen_result.get("cached_tokens", 0)
            else:
                _cached = (
                    getattr(_gen_result, "cached_tokens", 0)
                    if _gen_result is not None
                    else 0
                )
            # Stop-sequence overcount correction
            if req.stop and fr == "stop":
                _raw = text
                if req.echo and _prompt:
                    _raw = text[len(_prompt) :]
                # do NOT strip req.suffix from _raw. The
                # suffix is never appended to `text` (this engine has no FIM
                # template), so chopping len(req.suffix) chars here truncated the
                # ACTUAL completion before the stop search → recomputed
                # completion_tokens (ct) undercounted (a billing bug; output text
                # at the return is `text` and was unaffected).
                for _seq in req.stop:
                    if _seq and _seq in _raw:
                        _corrected = _raw[: _raw.find(_seq)]
                        _tok = getattr(engine, "_tokenizer", None)
                        if _tok:
                            try:
                                _cc = len(_tok.encode(_corrected))
                                if _cc < ct:
                                    ct = _cc
                            except Exception:
                                pass
                        break
            return idx, pt, ct, fr, rt, lp, text, _cached

        n = max(req.n, 1)
        # best_of: generate more completions than returned, keep best by logprob
        # OpenAI semantics: total choices = len(prompts) * n, with best_of
        # operating per-prompt. For multi-prompt requests we disallow best_of
        # to avoid ambiguity (matches OpenAI server-side behavior).
        if req.best_of is not None and len(_prompts) > 1:
            raise HTTPException(
                status_code=400,
                detail="best_of is not supported with multi-prompt requests",
            )
        _per_prompt = max(req.best_of, n) if req.best_of is not None else n
        # Flat list of (prompt_idx, prompt_str, choice_local_idx) tuples.
        _tasks: list[tuple[int, str, int]] = []
        for _pi, _p in enumerate(_prompts):
            for _ci in range(_per_prompt):
                _tasks.append((_pi, _p, _ci))
        # Sequential execution — single-threaded MLX executor.
        results = []
        for _gi, (_pi, _p, _ci) in enumerate(_tasks):
            try:
                # wrap each generation in the disconnect guard so a client that
                # drops mid-request actually SETS _ns_cancel_event and the engine decode loop
                # stops — chat.py had this (run_with_disconnect_guard) but completions only
                # registered the cancel_event and never polled is_disconnected(), so a
                # disconnect ran to max_tokens / the 300s timeout, head-of-line-blocking the
                # serial executor. None return == disconnected → stop.
                _r = await run_with_disconnect_guard(
                    request, _gen_one(_gi, _p), cancel_event=_ns_cancel_event
                )
                if _r is None:
                    logger.info(
                        "Completions: client disconnected mid-generation; aborting"
                    )
                    break
                # Tag result with prompt_idx for grouping post-hoc.
                results.append((_pi, _ci, _r))
            except Exception as exc:
                # was `except BaseException` which swallowed
                # CancelledError/KeyboardInterrupt/SystemExit. Client
                # disconnect → CancelledError → silently logged as "choice
                # failed" → server kept generating. Use Exception so
                # cancellation propagates.
                logger.error(f"Choice generation failed: {exc}", exc_info=exc)
        # Sort by (prompt_idx, choice_local_idx) to match OpenAI choice ordering.
        results.sort(key=lambda x: (x[0], x[1]))
        # capture one prompt_tokens per DISTINCT prompt index BEFORE collapsing away
        # the (_pi, _ci) tag. The old prompt_tokens sum sampled results[i*_per_prompt] by
        # arithmetic stride, which breaks when a partial generation failure removes a result
        # (the stride no longer lands on each prompt's first choice → the wrong prompts are
        # summed → wrong billed prompt_tokens). _r[1] is pt in the inner tuple.
        _prompt_tokens_by_pi: dict[int, int] = {}
        for _pi_r, _ci_r, _r_r in results:
            if _pi_r not in _prompt_tokens_by_pi:
                _prompt_tokens_by_pi[_pi_r] = _r_r[1]
        # Reduce to (idx, pt, ct, fr, rt, lp, text, _cached) with global idx.
        results = [(gi, *r[2][1:]) for gi, r in enumerate(results)]

        if not results:
            raise HTTPException(
                status_code=500, detail="All choices failed to generate"
            )

        # OpenAI bills ALL best_of generations, not just the
        # returned n. Capture the full completion/reasoning totals BEFORE best_of
        # truncation (the old code summed AFTER truncating → under-billed the discarded
        # candidates' tokens, which were really generated).
        total_completion_tokens = sum(r[2] for r in results)
        total_reasoning_tokens = sum(r[4] for r in results)

        # best_of: select top n results by average log probability per token
        if req.best_of is not None and len(results) > n:

            def _avg_logprob(r):
                """Compute average log probability for a result tuple."""
                _, pt, ct, fr, rt, lp, text, _cached = r
                if lp and "token_logprobs" in lp:
                    probs = lp["token_logprobs"]
                    # with echo, token_logprobs has the PROMPT tokens'
                    # logprobs PREPENDED. best_of must rank on the COMPLETION
                    # ONLY — the prompt forward is identical across all candidates, so
                    # averaging the (constant) prompt logprobs in divides them by each
                    # candidate's (prompt+completion) length, diluting more for longer
                    # completions and FLIPPING the ranking (a worse, longer completion
                    # can win). With echo the completion entries are exactly those whose
                    # text_offset >= the prompt's char length (the boundary set in
                    # _format_logprobs); slice them off before averaging. Guard on a
                    # str prompt (echo logprobs require string prompts) + aligned lists.
                    if (
                        req.echo
                        and isinstance(lp.get("text_offset"), list)
                        and _prompts
                        and isinstance(_prompts[0], str)
                    ):
                        _offs = lp["text_offset"]
                        _plen = len(_prompts[0])
                        if len(_offs) == len(probs):
                            probs = [
                                p
                                for p, o in zip(probs, _offs, strict=True)
                                if o >= _plen
                            ]
                    # Filter out None values (some tokens may have null logprobs)
                    valid_probs = [p for p in probs if p is not None]
                    if valid_probs:
                        return sum(valid_probs) / len(valid_probs)
                return float("-inf")  # no logprobs -> lowest priority

            results.sort(key=_avg_logprob, reverse=True)
            results = results[:n]
            # Re-index choices after best_of selection
            results = [(i, *r[1:]) for i, r in enumerate(results)]
            # Strip the internally-forced selection logprobs when the client didn't ask
            # for them (we forced logprobs on only to score best_of candidates).
            if req.logprobs <= 0:
                results = [
                    (r[0], r[1], r[2], r[3], r[4], None, r[6], r[7]) for r in results
                ]

        # For multi-prompt requests, each unique prompt's tokens are counted
        # once (n choices share the same prompt). For single-prompt, this is
        # equivalent to results[0][1]. Use _per_prompt to coalesce duplicates.
        if len(_prompts) > 1 and _per_prompt > 0:
            # sum one prompt_tokens per distinct prompt that actually produced a
            # result (failure-robust), not arithmetic stride. best_of is forbidden with
            # multi-prompt (rejected above), so _prompt_tokens_by_pi is complete here.
            prompt_tokens = sum(_prompt_tokens_by_pi.values())
        else:
            prompt_tokens = results[0][1]
        max_cached_tokens = max(r[7] for r in results) if results else 0
        max_finish_reason = results[0][3]

        choices = []
        # only emit prompt_logprobs for a SINGLE-prompt request. The
        # holder is overwritten per generation, so for a multi-prompt request it
        # holds only the LAST prompt's logprobs — attaching that to every choice
        # (including other prompts' choices) is WRONG. Omit rather than mislead;
        # per-prompt prompt_logprobs for batched prompts is a future enhancement.
        _plp = _prompt_lp_holder.get("v") if len(_prompts) == 1 else None
        for idx, _pt, _ct, fr, _rt, lp, text, _cached in results:
            choices.append(
                {
                    "index": idx,
                    "text": text,
                    "finish_reason": fr,
                    **({"logprobs": lp} if lp else {}),
                    **({"prompt_logprobs": _plp} if _plp is not None else {}),
                }
            )

        # End tracing
        tracer.end_span(trace_id, "prefill")
        tracer.end_trace(
            trace_id,
            result={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "finish_reason": max_finish_reason,
            },
        )
        slog.info(
            "inference_complete",
            model=req.model,
            trace_id=trace_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=total_completion_tokens,
        )

        # Record metrics for completions endpoint
        # total_completion_tokens (engine count) already includes reasoning.
        _record_metrics(prompt_tokens, total_completion_tokens)

        usage = {
            "prompt_tokens": prompt_tokens,
            # total_completion_tokens (engine count) already includes reasoning;
            # reasoning is the detail subset below (was double-added — the
            # tracing/slog above correctly report it WITHOUT the add).
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_tokens + total_completion_tokens,
        }
        if total_reasoning_tokens:
            usage["completion_tokens_details"] = {
                "reasoning_tokens": total_reasoning_tokens
            }
        if max_cached_tokens > 0:
            usage["prompt_tokens_details"] = {"cached_tokens": max_cached_tokens}

        return JSONResponse(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": choices,
                "usage": usage,
            }
        )
    except MemoryError:
        return JSONResponse(
            status_code=507,
            content={"error": {"message": "Out of GPU memory", "type": "memory_error"}},
        )
    except Exception as e:
        logger.error(f"Completions generation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {"message": "Internal server error", "type": "internal_error"}
            },
        )
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        if _ns_tracker is not None:
            with contextlib.suppress(Exception):
                _ns_tracker.unregister(completion_id)


async def _stream_completion(
    engine, prompt, req, completion_id, request, json_schema=None, trace_id=None
) -> AsyncIterator[bytes]:
    """SSE streaming for text completions with keepalive and disconnect detection."""
    from yunshu_engine.batched_engine import BatchedEngine

    from ..streaming import with_sse_keepalive

    is_batched = isinstance(engine, BatchedEngine)
    include_usage = req.stream_options is not None and req.stream_options.include_usage
    prompt_tok = 0
    completion_tok = 0
    # Track reasoning tokens per-choice to avoid overwrite across n>1 choices
    reasoning_tok_per_choice: dict[int, int] = {}
    completion_tok_per_choice: dict[int, int] = {}
    cached_tok = 0
    n = max(req.n, 1)
    # echo+logprobs prompt-logprob block, computed once (same prompt across n>1
    # choices) on the first echoed choice. _ECHO_LP_UNSET distinguishes "not yet computed"
    # from a computed None (compute failed / no logprobs).
    _echo_logprobs_cache = _ECHO_LP_UNSET

    async def _stream_choice(choice_idx: int):
        nonlocal prompt_tok, cached_tok
        choice_finish_reason = None
        _choice_streamed_text = ""  # track emitted text for stop-sequence correction
        # Per-choice text offset tracker for logprobs text_offset field.
        # When echo=True, the prompt text is emitted first, so the completion
        # text offsets must account for the prompt length.
        _choice_text_offset = len(prompt) if req.echo else 0
        if req.echo:
            # echo+logprobs streaming previously emitted the prompt with NO logprobs
            # (asymmetric with non-streaming, which prepends them). Compute prompt_logprobs up
            # front and emit them in the echo chunk via the SAME shared helper the non-stream
            # path uses (so BOS-collapse/null-first can't drift). Cached across n>1 choices
            # (same prompt) — choices stream sequentially. Batched engine only (the prompt-
            # logprob forward lives on BatchedEngine); legacy path keeps the bare echo chunk.
            nonlocal _echo_logprobs_cache
            if (
                req.logprobs
                and req.logprobs > 0
                and is_batched
                and _echo_logprobs_cache is _ECHO_LP_UNSET
            ):
                _echo_logprobs_cache = None
                with contextlib.suppress(Exception):
                    _ptok = getattr(engine, "_tokenizer", None)
                    _plp = await engine._compute_prompt_logprobs_for(
                        prompt,
                        req.enable_thinking,
                        req.logprobs,
                    )
                    _resolved_top = (
                        req.top_logprobs
                        if req.top_logprobs is not None
                        else req.logprobs
                    )
                    _pe = _format_prompt_logprob_entries(
                        _plp, _ptok, _resolved_top, prompt
                    )
                    if _pe:
                        _po = 0
                        _toks, _lps, _tops, _offs = [], [], [], []
                        for _ent in _pe:
                            _toks.append(_ent["token"])
                            _lps.append(_ent["logprob"])
                            _tops.append(_ent["top_logprobs"])
                            _offs.append(_po)
                            _po += len(_ent["token"])
                        _echo_logprobs_cache = {
                            "tokens": _toks,
                            "token_logprobs": _lps,
                            "top_logprobs": _tops,
                            "text_offset": _offs,
                        }
            _echo_lp = (
                _echo_logprobs_cache
                if _echo_logprobs_cache is not _ECHO_LP_UNSET
                else None
            )
            yield format_openai_completion_chunk(
                completion_id=completion_id,
                model=req.model,
                text=prompt,
                logprobs=_echo_lp,
                choice_index=choice_idx,
            )

        if is_batched:
            async for output in engine.stream_generate(
                prompt=prompt,
                max_tokens=req.effective_max_tokens(),
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                min_p=req.min_p,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                logit_bias=req.logit_bias,
                stop=req.stop,
                stop_token_ids=req.stop_token_ids,
                seed=(req.seed + choice_idx) if req.seed is not None else None,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                json_schema=json_schema,
                reasoning_effort=req.reasoning_effort,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                spec_decode=req.spec_decode,
                logprobs=req.logprobs > 0,
                # Completions API: `logprobs` is the alt-count; fall back to it when
                # top_logprobs is unset (matches the non-streaming path at line 397 —
                # streaming previously passed None → engine emitted no alternatives).
                top_logprobs=(
                    req.top_logprobs if req.top_logprobs is not None else req.logprobs
                ),
                priority=req.priority,
                logits_processors=req.logits_processors,
                cancel_event=_comp_cancel_evt,
                timeout_seconds=req.timeout,
                lora_adapter=loaded_adapter,
                # parity — see the non-streaming branch.
                min_tokens=req.min_tokens,
                ignore_eos=req.ignore_eos,
                suppress_tokens=req.suppress_tokens,
            ):
                if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if (
                    hasattr(output, "completion_tokens")
                    and output.completion_tokens is not None
                    and output.completion_tokens > 0
                ):
                    completion_tok_per_choice[choice_idx] = output.completion_tokens
                elif (
                    output.new_text
                    and getattr(output, "current_state", None) != "reasoning"
                ):
                    # Only count non-reasoning tokens toward completion_tok
                    completion_tok_per_choice[choice_idx] = (
                        completion_tok_per_choice.get(choice_idx, 0) + 1
                    )
                _choice_reasoning = getattr(output, "reasoning_tokens", 0)
                reasoning_tok_per_choice[choice_idx] = max(
                    reasoning_tok_per_choice.get(choice_idx, 0), _choice_reasoning
                )
                if hasattr(output, "cached_tokens") and output.cached_tokens:
                    cached_tok = max(cached_tok, output.cached_tokens)
                # Track finish_reason from engine; only emit on final chunk
                if output.finish_reason is not None:
                    choice_finish_reason = output.finish_reason
                # emit prefill progress as SSE comment for
                # client-side progress bars during long chunked prefills.
                _pf_prog = getattr(output, "prefill_progress", None)
                if _pf_prog is not None:
                    yield f": prefill-progress {_pf_prog[0]}/{_pf_prog[1]}\n\n"
                    continue  # progress outputs carry no text
                # Track emitted text for stop-sequence overcount correction
                if output.new_text:
                    _choice_streamed_text += output.new_text
                if len(_choice_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
                    logger.error(
                        "Choice streaming text buffer exceeded 1MB — truncating"
                    )
                    _choice_streamed_text = _choice_streamed_text[-_TRUNCATE_KEEP:]
                # Detect stop-sequence overcount on final output
                if (
                    req.stop
                    and choice_finish_reason == "stop"
                    and getattr(output, "finished", False)
                ):
                    for _seq in req.stop:
                        if _seq and _seq in _choice_streamed_text:
                            _idx = _choice_streamed_text.find(_seq)
                            _choice_streamed_text = _choice_streamed_text[:_idx]
                            _tok = getattr(engine, "_tokenizer", None)
                            if _tok:
                                try:
                                    _correct_count = len(
                                        _tok.encode(_choice_streamed_text)
                                    )
                                    if _correct_count < completion_tok_per_choice.get(
                                        choice_idx, 0
                                    ):
                                        completion_tok_per_choice[choice_idx] = (
                                            _correct_count
                                        )
                                except Exception:
                                    pass
                            break
                # Format logprobs for this token if present
                _chunk_logprobs = None
                if output.logprobs:
                    _chunk_logprobs, _choice_text_offset = _format_streaming_logprobs(
                        output.logprobs,
                        text_offset_start=_choice_text_offset,
                        top_logprobs=req.top_logprobs
                        if req.top_logprobs is not None
                        else req.logprobs,
                        tokenizer=getattr(engine, "_tokenizer", None),
                    )
                yield format_openai_completion_chunk(
                    completion_id=completion_id,
                    model=req.model,
                    text=output.new_text,
                    finish_reason=None,  # intermediate: always None
                    choice_index=choice_idx,
                    logprobs=_chunk_logprobs,
                )
        else:
            async for output in engine.generate_stream(
                prompt=prompt,
                max_tokens=req.effective_max_tokens(),
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                min_p=req.min_p,
                repetition_penalty=req.repetition_penalty,
                frequency_penalty=req.frequency_penalty,
                presence_penalty=req.presence_penalty,
                logit_bias=req.logit_bias,
                stop=req.stop,
                seed=(req.seed + choice_idx) if req.seed is not None else None,
                enable_thinking=req.enable_thinking,
                thinking_budget=req.thinking_budget,
                reasoning_effort=req.reasoning_effort,
                stop_token_ids=req.stop_token_ids,
                xtc_probability=req.xtc_probability,
                xtc_threshold=req.xtc_threshold,
                spec_decode=req.spec_decode,
                json_schema=json_schema,
                priority=req.priority,
                logprobs=req.logprobs > 0,
                # Fall back to `logprobs` (the alt-count) when top_logprobs is unset
                # — matches non-streaming (line 397). See the batched-path note above.
                top_logprobs=(
                    req.top_logprobs if req.top_logprobs is not None else req.logprobs
                ),
                logits_processors=req.logits_processors,
                cancel_event=_comp_cancel_evt,
                timeout_seconds=req.timeout,
                lora_adapter=loaded_adapter,
                # parity — see the non-streaming branch.
                min_tokens=req.min_tokens,
                ignore_eos=req.ignore_eos,
                suppress_tokens=req.suppress_tokens,
            ):
                if hasattr(output, "prompt_tokens") and output.prompt_tokens:
                    prompt_tok = output.prompt_tokens
                if (
                    hasattr(output, "completion_tokens")
                    and output.completion_tokens is not None
                    and output.completion_tokens > 0
                ):
                    completion_tok_per_choice[choice_idx] = output.completion_tokens
                elif (
                    output.token_text
                    and getattr(output, "current_state", None) != "reasoning"
                ):
                    # Only count non-reasoning tokens toward completion_tok
                    completion_tok_per_choice[choice_idx] = (
                        completion_tok_per_choice.get(choice_idx, 0) + 1
                    )
                # Track reasoning + cached tokens (the batched branch does this; the
                # legacy branch omitted both → reasoning_tokens detail always 0 and
                # cached_tokens never reported for engine-loop streaming completions).
                _choice_reasoning = getattr(output, "reasoning_tokens", 0)
                reasoning_tok_per_choice[choice_idx] = max(
                    reasoning_tok_per_choice.get(choice_idx, 0), _choice_reasoning
                )
                if getattr(output, "cached_tokens", 0):
                    cached_tok = max(cached_tok, output.cached_tokens)
                if output.finish_reason is not None:
                    choice_finish_reason = output.finish_reason
                _chunk_lp = None
                if req.logprobs and hasattr(output, "logprobs"):
                    _chunk_lp, _choice_text_offset = _format_streaming_logprobs(
                        output.logprobs,
                        text_offset_start=_choice_text_offset,
                        top_logprobs=req.top_logprobs
                        if req.top_logprobs is not None
                        else req.logprobs,
                        tokenizer=getattr(engine, "_tokenizer", None),
                    )
                # Track emitted text for stop-sequence overcount correction
                if output.token_text:
                    _choice_streamed_text += output.token_text
                if len(_choice_streamed_text) > _MAX_STREAMING_TEXT_BUFFER:
                    logger.error(
                        "Choice streaming text buffer exceeded 1MB — truncating"
                    )
                    _choice_streamed_text = _choice_streamed_text[-_TRUNCATE_KEEP:]
                # Detect stop-sequence overcount on final output
                if (
                    req.stop
                    and choice_finish_reason == "stop"
                    and getattr(output, "finished", False)
                ):
                    for _seq in req.stop:
                        if _seq and _seq in _choice_streamed_text:
                            _idx = _choice_streamed_text.find(_seq)
                            _choice_streamed_text = _choice_streamed_text[:_idx]
                            _tok = getattr(engine, "_tokenizer", None)
                            if _tok:
                                try:
                                    _correct_count = len(
                                        _tok.encode(_choice_streamed_text)
                                    )
                                    if _correct_count < completion_tok_per_choice.get(
                                        choice_idx, 0
                                    ):
                                        completion_tok_per_choice[choice_idx] = (
                                            _correct_count
                                        )
                                except Exception:
                                    pass
                            break
                yield format_openai_completion_chunk(
                    completion_id=completion_id,
                    model=req.model,
                    text=output.token_text,
                    finish_reason=None,  # intermediate: always None
                    choice_index=choice_idx,
                    logprobs=_chunk_lp,
                )

        # do NOT emit req.suffix as a completion chunk — `suffix` is FIM context,
        # never part of the returned text (see the non-streaming path).

        # Emit final chunk with finish_reason for this choice (even if zero tokens)
        yield format_openai_completion_chunk(
            completion_id=completion_id,
            model=req.model,
            text="",
            finish_reason=_normalize_finish_reason(choice_finish_reason),
            choice_index=choice_idx,
        )

    async def _token_source():
        nonlocal _done_emitted, metrics_recorded
        # Stream each choice sequentially (matches OpenAI spec behavior)
        for choice_idx in range(n):
            async for chunk in _stream_choice(choice_idx):
                yield chunk

        if include_usage:
            # Sum reasoning tokens across all choices for total usage
            _total_reasoning = sum(reasoning_tok_per_choice.values())
            _total_completion = (
                sum(completion_tok_per_choice.values())
                if completion_tok_per_choice
                else completion_tok
            )
            yield format_openai_completion_usage_chunk(
                completion_id=completion_id,
                model=req.model,
                prompt_tokens=prompt_tok,
                completion_tokens=_total_completion,
                reasoning_tokens=_total_reasoning,
                cached_tokens=cached_tok,
            )

        # Record metrics for completions streaming path
        _total_completion = (
            sum(completion_tok_per_choice.values())
            if completion_tok_per_choice
            else completion_tok
        )
        _total_reasoning = (
            sum(reasoning_tok_per_choice.values()) if reasoning_tok_per_choice else 0
        )
        if prompt_tok > 0 or _total_completion > 0:
            _record_metrics(prompt_tok, _total_completion)  # already incl. reasoning

        metrics_recorded = True
        _done_emitted = True
        yield format_openai_done()

    metrics_recorded = False
    _done_emitted = False
    loaded_adapter = _apply_lora_adapter(engine, req.lora_adapter)
    # Register with request tracker for cancellation support
    _tracker = None
    _tracker_gen = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker

        _tracker = get_request_tracker()
        _tracker_gen = _tracker.register(completion_id, req.model)
    except Exception:
        _tracker = None
    _comp_cancel_evt = _tracker_gen.cancel_event if _tracker_gen is not None else None

    # SSE line-buffer: accumulate partial data and only yield complete
    # \n\n-terminated SSE events.  Prevents clients from receiving partial
    # SSE lines when TCP chunk boundaries split an event mid-way.
    _sse_buffer = ""

    def _drain_sse_buffer():
        """Return list of complete SSE events from the buffer, keeping any trailing partial line."""
        nonlocal _sse_buffer
        chunks = []
        while "\n\n" in _sse_buffer:
            event, _sse_buffer = _sse_buffer.split("\n\n", 1)
            chunks.append((event + "\n\n").encode("utf-8"))
        return chunks

    try:
        async for event in with_sse_keepalive(
            _token_source(),
            http_request=request,
            cancel_event=_comp_cancel_evt,
        ):
            _sse_buffer += event
            for chunk in _drain_sse_buffer():
                yield chunk
        # Flush any remaining complete event in buffer
        for chunk in _drain_sse_buffer():
            yield chunk
        # If buffer still has residual content without \n\n terminator,
        # append terminator and flush
        if _sse_buffer.strip():
            _sse_buffer += "\n\n"
            for chunk in _drain_sse_buffer():
                yield chunk
    except MemoryError:
        if _comp_cancel_evt is not None:
            _comp_cancel_evt.set()
        # Flush any buffered partial data before error
        if _sse_buffer.strip():
            _sse_buffer += "\n\n"
            for chunk in _drain_sse_buffer():
                yield chunk
        yield b'data: {"error": {"message": "Insufficient GPU memory", "type": "memory_error", "code": "oom"}}\n\n'
        if not _done_emitted:
            yield b"data: [DONE]\n\n"
    except Exception as e:
        if _comp_cancel_evt is not None:
            _comp_cancel_evt.set()
        # Flush any buffered partial data before error
        if _sse_buffer.strip():
            _sse_buffer += "\n\n"
            for chunk in _drain_sse_buffer():
                yield chunk
        logger.error(f"Completions streaming error: {e}", exc_info=True)
        err_payload = {
            "error": {"message": "Internal server error", "type": "internal_error"}
        }
        yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n".encode()
        if not _done_emitted:
            yield b"data: [DONE]\n\n"
    finally:
        _release_lora_adapter(engine, loaded_adapter)
        if _tracker is not None:
            with contextlib.suppress(Exception):
                _tracker.unregister(completion_id)
        # Fallback metrics recording if generator raised before completing
        if not metrics_recorded:
            _total = (
                sum(completion_tok_per_choice.values())
                if completion_tok_per_choice
                else completion_tok
            )
            _total_reasoning = (
                sum(reasoning_tok_per_choice.values())
                if reasoning_tok_per_choice
                else 0
            )
            if prompt_tok > 0 or _total > 0:
                with contextlib.suppress(Exception):
                    _record_metrics(prompt_tok, _total)  # already incl. reasoning


_ECHO_LP_UNSET = object()  # sentinel: echo prompt-logprobs not yet computed


def _format_prompt_logprob_entries(
    prompt_lp, tokenizer, top_logprobs: int, prompt: str
) -> list[dict]:
    """Shared prompt-token logprob formatter for echo=true. Used by BOTH the
    non-streaming _format_logprobs prepend AND the streaming echo chunk, so the prepend
    + BOS-collapse logic lives in ONE place (no stream↔non-stream drift — the streaming
    path previously emitted the echoed prompt with NO logprobs at all, an asymmetry with the
    non-streaming path that does return them).

    `prompt_lp` is the engine's list aligned to input_ids:
    [None, {token_id, logprob, top_logprobs?}, ...] (see _compute_prompt_logprobs_sync).
    Returns a list of {token, logprob, top_logprobs} entries aligned to the USER prompt tokens
    (the leading BOS slot collapsed for Llama/Gemma/Mistral; the first prompt token's logprob
    is null — no preceding context). Empty list when prompt_lp is not a usable list. The caller
    computes text_offset by walking token lengths.
    """
    if not (
        isinstance(prompt_lp, (list, tuple))
        and len(prompt_lp) >= 1
        and tokenizer is not None
    ):
        return []
    _tail = []
    for _e in prompt_lp[1:]:
        _tid = _e.get("token_id") if isinstance(_e, dict) else None
        try:
            _tail.append(tokenizer.decode([_tid]) if _tid is not None else "")
        except Exception:
            _tail.append("")
    _head = prompt[: max(0, len(prompt) - sum(len(s) for s in _tail))]
    # When _head=="" the tail tokens already cover the whole prompt → position 0 contributed
    # no visible text → it is a prepended BOS (Llama/Gemma/Mistral). Collapse it: list only the
    # real tokens and null the first real token's logprob. No-BOS models (Qwen) keep
    # _head=first token, so this branch is skipped and behaviour is unchanged.
    _bos_collapse = (
        _head == "" and len(prompt_lp) >= 2 and not isinstance(prompt_lp[0], dict)
    )
    if _bos_collapse:
        _src = list(prompt_lp[1:])
        _ptoks = list(_tail)
    else:
        _src = list(prompt_lp)
        _ptoks = [_head] + _tail
    _out: list[dict] = []
    for _i, _e in enumerate(_src):
        _tok = _ptoks[_i] if _i < len(_ptoks) else ""
        if isinstance(_e, dict) and not (_bos_collapse and _i == 0):
            _lpv = _e.get("logprob", 0.0)
            _top = {}
            for _t in (
                (_e.get("top_logprobs") or [])[:top_logprobs] if top_logprobs else []
            ):
                _ttid = _t.get("token_id")
                if _ttid is not None:
                    with contextlib.suppress(Exception):
                        _top[tokenizer.decode([_ttid])] = _t.get("logprob", 0.0)
        else:
            _lpv = None  # first prompt token (or BOS slot): no preceding context → null
            _top = {}
        _out.append({"token": _tok, "logprob": _lpv, "top_logprobs": _top})
    return _out


def _format_logprobs(
    state, tokenizer, top_logprobs: int, echo: bool = False, prompt: str = ""
) -> dict | None:
    """Format logprobs from request state into OpenAI Completions format.

    OpenAI Completions API returns logprobs as a flat structure:
      {
        "tokens": ["tok1", "tok2", ...],
        "token_logprobs": [-0.5, -1.2, ...],
        "top_logprobs": [{"tok_a": -0.5, "tok_b": -1.0}, ...],  # Dict[str, float], NOT chat-style
        "text_offset": [0, 3, ...]
      }

    Note: top_logprobs entries are Dict[str, float] (token -> logprob),
    NOT the Chat Completions format with token/logprob/bytes keys.

    Args:
        state: Generation result with logprobs attribute.
        tokenizer: Tokenizer for decoding token IDs.
        top_logprobs: Maximum number of top logprobs to return per token.
        echo: Whether echo mode is enabled (shifts text_offset by prompt length).
        prompt: The prompt text, used for text_offset shift when echo=True.
    """
    raw_logprobs = getattr(state, "logprobs", None)
    # Guard on type/length, not truthiness: the engine-loop/legacy paths can hand
    # back a raw mx.array, and `not <multi-element array>` raises ValueError.
    # Non-list formats → no logprobs rather than a 500.
    if not isinstance(raw_logprobs, (list, tuple)) or len(raw_logprobs) == 0:
        return None

    token_logprobs = []
    text_offsets = []
    _offset = len(prompt) if echo else 0
    # when echo=true, PREPEND the prompt tokens' logprobs (one entry per
    # prompt token, first=null; BOS slot collapsed for Llama/Gemma/Mistral) via the shared
    # helper below — the same helper feeds the streaming echo chunk so the two paths can't drift.
    _prompt_lp = getattr(state, "prompt_logprobs", None) if echo else None
    # prompt-token logprob formatting (prepend + BOS-collapse) lives in
    # the shared _format_prompt_logprob_entries so the streaming echo path reuses it (no drift).
    _pentries = _format_prompt_logprob_entries(
        _prompt_lp, tokenizer, top_logprobs, prompt
    )
    if _pentries:
        _poffset = 0
        for _ent in _pentries:
            token_logprobs.append(_ent)
            text_offsets.append(_poffset)
            _poffset += len(_ent["token"])
        # Completion tokens continue from the END of the echoed prompt so they stay aligned
        # with `prompt + completion`, regardless of any tiny token-length rounding above.
        _offset = len(prompt)
    if isinstance(raw_logprobs, (list, tuple)):
        for lp_entry in raw_logprobs:
            if isinstance(lp_entry, dict):
                token_str = lp_entry.get("token", "")
                if not token_str and tokenizer and "token_id" in lp_entry:
                    try:
                        token_str = tokenizer.decode([lp_entry["token_id"]])
                    except Exception:
                        logger.debug("tokenizer decode failed", exc_info=True)
                top_lps = lp_entry.get("top_logprobs", [])
                # OpenAI Completions API: top_logprobs is List[Dict[str, float]]
                # Each dict maps token string -> logprob float value.
                decoded_top = {}
                for tlp in top_lps[:top_logprobs] if top_logprobs else []:
                    tlp_token = tlp.get("token", "")
                    if not tlp_token and tokenizer and "token_id" in tlp:
                        with contextlib.suppress(Exception):
                            tlp_token = tokenizer.decode([tlp["token_id"]])
                    if tlp_token:
                        decoded_top[tlp_token] = tlp.get("logprob", 0.0)
                token_logprobs.append(
                    {
                        "token": token_str,
                        "logprob": lp_entry.get("logprob", 0.0),
                        "top_logprobs": decoded_top,
                    }
                )
                text_offsets.append(_offset)
                _offset += len(token_str)
            elif isinstance(lp_entry, (int, float)) and not isinstance(lp_entry, bool):
                # bool ⊂ int → True/False would become 1.0/0.0.
                token_logprobs.append(
                    {
                        "token": "",
                        "logprob": float(lp_entry),
                        "top_logprobs": {},
                    }
                )
                text_offsets.append(_offset)

    if not token_logprobs:
        return None

    return {
        "tokens": [e["token"] for e in token_logprobs],
        "token_logprobs": [e["logprob"] for e in token_logprobs],
        "top_logprobs": [e["top_logprobs"] for e in token_logprobs],
        "text_offset": text_offsets,
    }


def _format_streaming_logprobs(
    logprobs_list: list[dict],
    *,
    text_offset_start: int = 0,
    top_logprobs: int | None = None,
    tokenizer: object | None = None,
) -> tuple[dict | None, int]:
    """Format per-token logprobs from streaming GenerationOutput into OpenAI Completions format.

    In streaming mode, each GenerationOutput has at most 1 logprob entry.
    Returns (logprobs_dict, new_text_offset) where new_text_offset is the
    running offset to pass into the next call.

    Per the OpenAI Completions API, logprobs must include ``text_offset``
    (character offset of each token in the output text).

    OpenAI Completions top_logprobs format: Dict[str, float] per token,
    NOT the Chat Completions style with token/logprob/bytes keys.
    """
    if not isinstance(logprobs_list, (list, tuple)) or len(logprobs_list) == 0:
        return None, text_offset_start
    entries = []
    offsets = []
    _offset = text_offset_start
    # When top_logprobs is not specified, include all available top logprobs.
    # Use a large default to avoid truncation (engine already limits this).
    _max_top = top_logprobs if top_logprobs is not None else 999
    for lp_entry in logprobs_list:
        if not isinstance(lp_entry, dict):
            continue
        token_str = lp_entry.get("token", "")
        if not token_str and "token_id" in lp_entry:
            if tokenizer is not None:
                try:
                    token_str = tokenizer.decode([lp_entry["token_id"]])
                except Exception:
                    token_str = str(lp_entry["token_id"])
            else:
                token_str = str(lp_entry["token_id"])
        top_lps = lp_entry.get("top_logprobs", [])
        # OpenAI Completions API: top_logprobs is Dict[str, float]
        decoded_top = {}
        for tlp in top_lps[:_max_top]:
            if not isinstance(tlp, dict):
                continue
            tlp_token = tlp.get("token", "")
            if not tlp_token and "token_id" in tlp:
                if tokenizer is not None:
                    try:
                        tlp_token = tokenizer.decode([tlp["token_id"]])
                    except Exception:
                        tlp_token = str(tlp["token_id"])
                else:
                    tlp_token = str(tlp["token_id"])
            if tlp_token:
                decoded_top[tlp_token] = tlp.get("logprob", 0.0)
        entries.append(
            {
                "token": token_str,
                "logprob": lp_entry.get("logprob", 0.0),
                "top_logprobs": decoded_top,
            }
        )
        offsets.append(_offset)
        _offset += len(token_str)
    if not entries:
        return None, text_offset_start
    return {
        "tokens": [e["token"] for e in entries],
        "token_logprobs": [e["logprob"] for e in entries],
        "top_logprobs": [e["top_logprobs"] for e in entries],
        "text_offset": offsets,
    }, _offset
