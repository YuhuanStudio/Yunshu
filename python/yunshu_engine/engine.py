"""Yunshu L4 Engine — MLX-native continuous batching.

Deeply adapted to MLX's actual execution model by studying:
- mlx-lm's BatchGenerator (generate.py): continuous batching with PromptProcessingBatch + GenerationBatch
- oMLX's Scheduler + EngineCore: request lifecycle, deferred cache clearing, output parsing
- vLLM's request management: waiting/running queues, per-request state

Architecture:
  Gateway → add_request() → Engine._step_loop()
    → BatchGenerator.insert(samplers=per_request, state_machines=per_request) / next()
    → GenerationBatch.Response → _distribute_responses()
    → Per-request detokenizer: add_token(r.token) → last_segment
  Gateway streams from per-request queue as SSE

Key oMLX lessons incorporated:
- Deferred cache clearing: wait 8 steps after completion before mx.clear_cache()
- EOS token handling: skip add_token() for stop tokens
- Per-request detokenizer: never pool (reset() leaks byte buffers)
- Output token ID tracking for cumulative decode
- Step counter and stats for monitoring
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncIterator, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


# ── Enums and Data Classes ──


class RequestPhase(Enum):
    WAITING = auto()
    PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()


@dataclass
class RequestOutput:
    """One token of output from the engine.

    Follows oMLX's RequestOutput pattern: incremental text + cumulative counts.
    token_text comes from the per-request detokenizer (add_token → last_segment).
    """

    request_id: str
    token_text: str
    token_id: int
    finish_reason: Optional[str] = None  # "stop" | "length" | "abort" | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprob: float = 0.0
    current_state: str = "normal"  # "normal" | "reasoning" | "tool"

    @property
    def usage(self) -> dict[str, int]:
        """OpenAI-compatible usage stats."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


@dataclass
class RequestState:
    """Per-request state managed by the engine.

    Follows oMLX's Request + vLLM's request tracking pattern.
    """

    request_id: str
    prompt_tokens: list[int]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)

    # Runtime state
    phase: RequestPhase = RequestPhase.WAITING
    uid: Optional[int] = None  # BatchGenerator-assigned UID
    prompt_token_count: int = 0
    completion_token_count: int = 0
    generated_text: str = ""
    finish_reason: Optional[str] = None

    # Output token IDs for cumulative decode (oMLX pattern)
    output_token_ids: list[int] = field(default_factory=list)

    # Per-request detokenizer (from tokenizer.detokenizer)
    detokenizer: Any = None

    # Streaming output queue (gateway reads from this)
    output_queue: asyncio.Queue[RequestOutput | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=1024)
    )
    # Completion event (for non-streaming generate)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Timing
    arrival_time: float = field(default_factory=time.monotonic)


@dataclass
class EngineConfig:
    """Engine tuning parameters.

    Maps to BatchGenerator constructor + oMLX's SchedulerConfig.
    """

    completion_batch_size: int = 32
    prefill_batch_size: int = 8
    prefill_step_size: int = 2048
    max_kv_size: Optional[int] = None
    step_interval_ms: float = 1.0
    # oMLX deferred cache clearing: wait N steps after completion
    deferred_clear_delay: int = 8
    # Periodic cache cleanup interval (steps between mx.clear_cache())
    cache_cleanup_interval: int = 512


# ── Engine Core ──

# Module-level deferred clear state (oMLX pattern: #435, #557)
# After a request finishes, we wait a few steps before clearing the Metal
# buffer cache to prevent IOKit kernel panics from async completeMemory().


class Engine:
    """MLX-native continuous batching engine.

    Threading model (from mlx-lm's server.py and oMLX):
    All MLX GPU work goes through the global MLX executor (single thread).
    This is REQUIRED because mlx-lm uses a module-level Metal stream
    (`generation_stream`). The executor thread gets its own stream via
    mx.new_thread_local_stream() in mlx_executor._init_mlx_thread().

    Two backends (selectable via use_engine_core flag):
    - Legacy: inline step loop with per-request asyncio.Queue
    - EngineCore: oMLX-style Scheduler + RequestOutputCollector orchestration
    """

    def __init__(self, config: EngineConfig | None = None, *, use_engine_core: bool = True) -> None:
        self.config = config or EngineConfig()

        # Model state
        self._model = None
        self._tokenizer = None
        self._batch_gen = None
        self._model_name: Optional[str] = None

        # Global MLX executor (single-thread, shared across all engines)
        from .mlx_executor import get_mlx_executor
        self._executor = get_mlx_executor()

        # Lifecycle
        self._running = False
        self._step_task: Optional[asyncio.Task] = None

        # Request tracking (legacy backend)
        self._waiting: deque[RequestState] = deque()
        self._active: dict[str, RequestState] = {}  # request_id → state
        self._uid_to_req: dict[int, str] = {}  # BatchGenerator UID → request_id
        self._abort_set: set[str] = set()

        # oMLX-style deferred cache clearing (#435)
        self._step_counter: int = 0
        self._deferred_clear_at: Optional[int] = None

        # Stats (oMLX pattern)
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._num_requests_processed: int = 0
        self._start_time: Optional[float] = None

        # Memory monitor (oMLX pattern)
        from .memory_monitor import MemoryMonitor
        self._memory_monitor = MemoryMonitor()

        # EngineCore backend (oMLX EngineCore pattern)
        self._engine_core: Any = None
        self._use_engine_core = use_engine_core

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_running(self) -> bool:
        return self._running

    def has_active_requests(self) -> bool:
        """Check if any requests are currently being processed.

        Used by ModelManager's LRU eviction to avoid interrupting in-flight work.
        oMLX pattern: EnginePool._find_lru_victim skips engines with active requests.
        """
        if self._engine_core is not None:
            return self._engine_core.has_active_requests
        return any(
            s.phase in (RequestPhase.PREFILLING, RequestPhase.DECODING)
            for s in self._active.values()
        )

    @property
    def model_name(self) -> str | None:
        return getattr(self, '_model_display', self._model_name)

    def resolve_model_id(self, model_id: str) -> bool:
        """Check if a requested model ID matches the loaded model.

        Supports multiple match modes (oMLX pattern):
        - Exact match on display name
        - Exact match on full path
        - Case-insensitive match
        - Strip provider prefix (e.g. "yunshu/Qwen3.5" → "Qwen3.5")
        """
        if not self.is_loaded or not self._model_name:
            return False

        display = getattr(self, '_model_display', '') or ''
        full = self._model_name

        known = {display, full}
        known_lower = {k.lower() for k in known if k}

        if model_id in known or model_id.lower() in known_lower:
            return True

        if '/' in model_id:
            stripped = model_id.rsplit('/', 1)[-1]
            if stripped in known or stripped.lower() in known_lower:
                return True

        return False

    def load(self, model_name: str) -> None:
        """Load model and create BatchGenerator or EngineCore."""
        from mlx_lm.utils import load as load_model
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.generate import BatchGenerator, generation_stream

        self._model_name = model_name
        self._model_display = model_name.rsplit("/", 1)[-1] if "/" in model_name else model_name
        self._model, self._tokenizer = load_model(model_name)

        if self._use_engine_core:
            from .engine_core import EngineCore, EngineCoreConfig
            self._engine_core = EngineCore(
                model=self._model,
                tokenizer=self._tokenizer,
                config=EngineCoreConfig(
                    completion_batch_size=self.config.completion_batch_size,
                    prefill_batch_size=self.config.prefill_batch_size,
                    prefill_step_size=self.config.prefill_step_size,
                    max_kv_size=self.config.max_kv_size,
                    deferred_clear_delay=self.config.deferred_clear_delay,
                    cache_cleanup_interval=self.config.cache_cleanup_interval,
                ),
                executor=self._executor,
            )
            self._engine_core.scheduler.config.model_name = model_name
        else:
            sampler = make_sampler(temp=1.0)
            self._batch_gen = BatchGenerator(
                self._model,
                max_tokens=self.config.completion_batch_size,
                sampler=sampler,
                prefill_batch_size=self.config.prefill_batch_size,
                completion_batch_size=self.config.completion_batch_size,
                prefill_step_size=self.config.prefill_step_size,
                max_kv_size=self.config.max_kv_size,
                stream=generation_stream,
            )

        logger.info(f"Engine loaded model: {model_name} (engine_core={self._use_engine_core})")

        # Initialize memory monitor with model architecture info (oMLX pattern)
        self._init_memory_monitor()

    async def start(self) -> None:
        """Start the async step loop."""
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()

        if self._engine_core is not None:
            await self._engine_core.start()
        else:
            self._step_task = asyncio.get_running_loop().create_task(self._step_loop())

        logger.info("Engine step loop started")

    async def stop(self) -> None:
        """Stop the engine gracefully and release all GPU resources.

        oMLX EngineCore.close() pattern:
        - Release model/tokenizer references for GC
        - Clear scheduler/batch_gen state
        - gc.collect() + mx.synchronize() + mx.clear_cache() on MLX executor
        Without this, model weights stay in Metal memory after "unloading",
        causing OOM crashes when loading subsequent models.
        """
        self._running = False

        if self._engine_core is not None:
            await self._engine_core.stop()
            self._engine_core = None
            self._model = None
            self._tokenizer = None
            return

        if self._step_task:
            self._step_task.cancel()
            try:
                await self._step_task
            except asyncio.CancelledError:
                pass

        for req in list(self._active.values()):
            req.finish_reason = "abort"
            req.output_queue.put_nowait(None)
            req.done_event.set()
        self._active.clear()
        self._waiting.clear()
        self._uid_to_req.clear()
        self._abort_set.clear()

        if self._batch_gen is not None:
            try:
                self._batch_gen.close()
            except Exception:
                pass
            self._batch_gen = None

        # Release model and tokenizer references (oMLX EngineCore.close pattern)
        self._model = None
        self._tokenizer = None

        # Force GC + synchronize + clear cache on MLX executor thread
        gc.collect()
        loop = asyncio.get_running_loop()
        from .mlx_executor import sync_and_clear_cache
        try:
            await loop.run_in_executor(self._executor, sync_and_clear_cache)
        except Exception:
            pass

        logger.info("Engine stopped and GPU memory released")

    def _create_detokenizer(self):
        """Create a fresh streaming detokenizer for one request.

        Key lesson from oMLX: do NOT pool detokenizer instances.
        reset()/finalize() does NOT fully clear internal byte buffers,
        causing text corruption. Always create fresh per-request.
        """
        if self._tokenizer is None:
            return None
        detok = self._tokenizer.detokenizer
        detok.reset()
        return detok

    def _make_sampler(self, temperature: float = 0.7, top_p: float = 1.0,
                      top_k: int = 0, min_p: float = 0.0,
                      repetition_penalty: float = 1.0):
        """Create a per-request sampler with the given parameters.

        Supports all mlx-lm sampler params (oMLX SamplingParams pattern).
        """
        from mlx_lm.sample_utils import make_sampler
        return make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

    def _make_state_machine(self, stop: list[str] | None = None):
        """Create a SequenceStateMachine for stop sequence detection.

        Follows mlx-lm server.py's _make_state_machine pattern:
        - EOS tokens always stop generation
        - User stop words encoded to token sequences
        - Reasoning transitions (think_start/think_end) if tokenizer supports it
        - All managed via Aho-Corasick trie for O(k) matching
        """
        from mlx_lm.generate import SequenceStateMachine

        tokenizer = self._tokenizer

        eos_ids = list(tokenizer.eos_token_ids) if hasattr(tokenizer, 'eos_token_ids') else []

        common_stops = [((t,), None) for t in eos_ids]
        for w in (stop or []):
            t = tuple(tokenizer.encode(w, add_special_tokens=False))
            common_stops.append((t, None))

        transitions = {}
        transitions["normal"] = list(common_stops)

        if getattr(tokenizer, 'has_thinking', False):
            try:
                ts = tokenizer.think_start_tokens
                te = tokenizer.think_end_tokens
                transitions["normal"].append((ts, "reasoning"))
                transitions["reasoning"] = [(te, "normal")]
                transitions["reasoning"].extend(common_stops)
            except (AttributeError, TypeError):
                pass

        return SequenceStateMachine(transitions, initial="normal")

    async def add_request(
        self,
        prompt: str | list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        request_id: str | None = None,
        enable_thinking: bool | None = None,
        **kwargs,
    ) -> RequestState:
        """Add a new generation request to the engine."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded")

        # EngineCore path: returns request_id, wraps as RequestState for compatibility
        if self._engine_core is not None:
            req_id = await self._engine_core.add_request(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logit_bias=logit_bias,
                stop=stop,
                request_id=request_id,
                enable_thinking=enable_thinking,
            )
            # Wrap as RequestState for backward compatibility with gateway
            if isinstance(prompt, str):
                token_ids = self._tokenizer.encode(prompt)
            else:
                text = self._messages_to_text(prompt, enable_thinking=enable_thinking)
                token_ids = self._tokenizer.encode(text)

            state = RequestState(
                request_id=req_id,
                prompt_tokens=token_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                stop=stop or [],
                prompt_token_count=len(token_ids),
                phase=RequestPhase.WAITING,
            )
            self._active[req_id] = state
            return state

        # Legacy path
        req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

        if isinstance(prompt, str):
            token_ids = self._tokenizer.encode(prompt)
        else:
            text = self._messages_to_text(prompt, enable_thinking=enable_thinking)
            token_ids = self._tokenizer.encode(text)

        detokenizer = self._create_detokenizer()

        state = RequestState(
            request_id=req_id,
            prompt_tokens=token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            stop=stop or [],
            prompt_token_count=len(token_ids),
            detokenizer=detokenizer,
        )

        self._waiting.append(state)
        self._active[req_id] = state
        return state

    async def abort_request(self, request_id: str) -> None:
        """Request abortion (processed at next step boundary)."""
        if self._engine_core is not None:
            await self._engine_core.abort_request(request_id)
            return

        self._abort_set.add(request_id)
        state = self._active.get(request_id)
        if state and state.phase == RequestPhase.WAITING:
            state.finish_reason = "abort"
            state.output_queue.put_nowait(None)
            state.done_event.set()
            self._waiting = deque(
                r for r in self._waiting if r.request_id != request_id
            )

    async def generate(self, **kwargs) -> RequestState:
        """Non-streaming generate: add request and wait for completion."""
        if self._engine_core is not None:
            result = await self._engine_core.generate(**kwargs)
            if result is None:
                raise RuntimeError("Generation failed")
            # Convert to RequestState for backward compatibility
            state = RequestState(
                request_id=result.request_id,
                prompt_tokens=[],
                max_tokens=kwargs.get("max_tokens", 512),
                phase=RequestPhase.FINISHED,
                prompt_token_count=result.prompt_tokens,
                completion_token_count=result.completion_tokens,
                generated_text=result.output_text,
                finish_reason=result.finish_reason,
            )
            self._active.pop(result.request_id, None)
            return state

        state = await self.add_request(**kwargs)
        await state.done_event.wait()
        return state

    async def generate_stream(self, **kwargs) -> AsyncIterator[RequestOutput]:
        """Streaming generate: yields RequestOutput for each token."""
        if self._engine_core is not None:
            req_id = await self._engine_core.add_request(**kwargs)
            async for output in self._engine_core.stream_outputs(req_id):
                yield RequestOutput(
                    request_id=output.request_id,
                    token_text=output.new_text,
                    token_id=output.new_token_ids[-1] if output.new_token_ids else 0,
                    finish_reason=output.finish_reason,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    current_state=output.current_state,
                )
            self._active.pop(req_id, None)
            return

        state = await self.add_request(**kwargs)
        while True:
            output = await state.output_queue.get()
            if output is None:
                break
            yield output

    # ── Step Loop ──

    async def _step_loop(self) -> None:
        """Main async step loop — drives continuous batching.

        Incorporates oMLX's deferred cache clearing pattern:
        After requests finish, schedule a cache clear 8 steps later
        to prevent IOKit kernel panics from async completeMemory() callbacks.
        """
        loop = asyncio.get_running_loop()

        while self._running:
            # 1. Process deferred aborts
            self._process_aborts()

            # 2. Insert waiting requests into BatchGenerator
            self._schedule_waiting()

            if self._batch_gen is None:
                await asyncio.sleep(self.config.step_interval_ms / 1000)
                continue

            # 3. Run one BatchGenerator step on the Metal thread
            try:
                prompt_responses, gen_responses = await loop.run_in_executor(
                    self._executor,
                    self._batch_gen.next,
                )
            except Exception as e:
                logger.error(f"Step error: {e}", exc_info=True)
                for req in list(self._active.values()):
                    if req.phase in (RequestPhase.PREFILLING, RequestPhase.DECODING):
                        req.finish_reason = "error"
                        req.output_queue.put_nowait(None)
                        req.done_event.set()
                self._active.clear()
                self._uid_to_req.clear()
                await asyncio.sleep(0.1)
                continue

            # 4. Distribute generation responses
            if gen_responses:
                self._distribute_responses(gen_responses)

            # 5. Deferred Metal cache cleanup (oMLX pattern #435)
            self._step_counter += 1
            should_clear = False
            if (
                self.config.cache_cleanup_interval > 0
                and self._step_counter % self.config.cache_cleanup_interval == 0
            ):
                should_clear = True
            if (
                self._deferred_clear_at is not None
                and self._step_counter >= self._deferred_clear_at
            ):
                should_clear = True
                self._deferred_clear_at = None

            if should_clear:
                try:
                    from .mlx_executor import sync_and_clear_cache
                    await loop.run_in_executor(self._executor, sync_and_clear_cache)
                except Exception:
                    pass

            # 6. Yield to event loop
            await asyncio.sleep(0)

    def _schedule_waiting(self) -> None:
        """Move waiting requests into BatchGenerator."""
        if self._batch_gen is None:
            return

        to_insert = []
        while self._waiting:
            state = self._waiting.popleft()
            if state.request_id in self._abort_set:
                self._abort_set.discard(state.request_id)
                continue
            to_insert.append(state)

        for state in to_insert:
            tokens = state.prompt_tokens
            try:
                sampler = self._make_sampler(
                    state.temperature, state.top_p,
                    state.top_k, state.min_p,
                    state.repetition_penalty,
                )
                sm = self._make_state_machine(state.stop)

                uids = self._batch_gen.insert(
                    prompts=[tokens],
                    max_tokens=[state.max_tokens],
                    samplers=[sampler],
                    state_machines=[sm],
                )
                state.uid = uids[0]
                state.phase = RequestPhase.PREFILLING
                self._uid_to_req[state.uid] = state.request_id

                # Track prompt tokens
                self._total_prompt_tokens += state.prompt_token_count

            except Exception as e:
                logger.error(f"Failed to insert request {state.request_id}: {e}")
                state.finish_reason = "error"
                state.output_queue.put_nowait(None)
                state.done_event.set()
                self._active.pop(state.request_id, None)

    def _process_aborts(self) -> None:
        """Process pending abort requests."""
        if not self._abort_set:
            return

        abort_uids = []
        for req_id in list(self._abort_set):
            state = self._active.get(req_id)
            if state and state.uid is not None:
                abort_uids.append(state.uid)

        if abort_uids and self._batch_gen:
            mx.synchronize()
            self._batch_gen.remove(abort_uids)
            for uid in abort_uids:
                req_id = self._uid_to_req.pop(uid, None)
                if req_id:
                    state = self._active.pop(req_id, None)
                    if state:
                        state.finish_reason = "abort"
                        state.output_queue.put_nowait(None)
                        state.done_event.set()

        self._abort_set.clear()

    def _distribute_responses(self, responses: list) -> None:
        """Distribute GenerationBatch.Response objects to per-request queues.

        Incorporates oMLX's output processing pattern:
        - Skip add_token() for stop tokens (EOS)
        - Track output_token_ids for cumulative decode
        - Deferred cache clearing on completion
        - Proper finalize() for detokenizer
        """
        for resp in responses:
            uid = resp.uid
            req_id = self._uid_to_req.get(uid)
            if req_id is None:
                continue

            state = self._active.get(req_id)
            if state is None:
                continue

            # Check finish reason first (oMLX pattern)
            is_stop = resp.finish_reason == "stop"
            is_finished = resp.finish_reason is not None

            # Decode token via per-request detokenizer (mlx-lm pattern)
            token_text = ""

            if not is_stop:
                # Track output token ID (oMLX: append_output_token)
                state.output_token_ids.append(resp.token)
                state.completion_token_count += 1

                if state.detokenizer is not None:
                    state.detokenizer.add_token(resp.token)
                    token_text = state.detokenizer.last_segment
                else:
                    token_text = self._tokenizer.decode([resp.token]) if self._tokenizer else ""

                state.generated_text += token_text
            elif is_finished:
                # Stop token — don't decode, just count
                state.completion_token_count += 1

            # Extract logprob if available
            logprob = 0.0
            if hasattr(resp, 'logprobs') and resp.logprobs is not None:
                try:
                    logprob = float(resp.logprobs[resp.token].item())
                except Exception:
                    pass

            # Get state machine state
            current_state = getattr(resp, 'current_state', 'normal') or 'normal'

            # finish_reason from BatchGenerator
            finish_reason = resp.finish_reason

            output = RequestOutput(
                request_id=req_id,
                token_text=token_text,
                token_id=resp.token,
                finish_reason=finish_reason,
                prompt_tokens=state.prompt_token_count,
                completion_tokens=state.completion_token_count,
                logprob=logprob,
                current_state=current_state,
            )

            try:
                state.output_queue.put_nowait(output)
            except asyncio.QueueFull:
                pass

            if finish_reason:
                # Finalize detokenizer to flush remaining bytes (oMLX pattern)
                if state.detokenizer is not None:
                    try:
                        state.detokenizer.finalize()
                        final_text = state.detokenizer.last_segment
                        if final_text:
                            state.generated_text += final_text
                            final_output = RequestOutput(
                                request_id=req_id,
                                token_text=final_text,
                                token_id=resp.token,
                                finish_reason=None,
                                prompt_tokens=state.prompt_token_count,
                                completion_tokens=state.completion_token_count,
                                current_state=current_state,
                            )
                            try:
                                state.output_queue.put_nowait(final_output)
                            except asyncio.QueueFull:
                                pass
                    except Exception:
                        pass

                state.finish_reason = finish_reason
                state.phase = RequestPhase.FINISHED
                state.output_queue.put_nowait(None)  # sentinel
                state.done_event.set()
                self._uid_to_req.pop(uid, None)

                # oMLX deferred cache clearing (#435)
                self._deferred_clear_at = (
                    self._step_counter + self.config.deferred_clear_delay
                )

                # Update stats
                self._total_completion_tokens += state.completion_token_count
                self._num_requests_processed += 1

                # Server metrics integration (oMLX pattern)
                try:
                    from .server_metrics import get_server_metrics
                    get_server_metrics().record_request_complete(
                        prompt_tokens=state.prompt_token_count,
                        completion_tokens=state.completion_token_count,
                        model_id=self._model_display or self._model_name or "",
                    )
                except Exception:
                    pass

    def _messages_to_text(
        self,
        messages: list[dict],
        enable_thinking: bool | None = None,
    ) -> str:
        """Convert chat messages to text using the model's chat template.

        Follows mlx-lm's TokenizerWrapper.apply_chat_template pattern:
        - If enable_thinking is None, uses tokenizer's default (has_thinking)
        - If enable_thinking is explicitly set, passes it to the chat template
        - This controls whether Qwen3/DeepSeek-R1 emit <think/> tags
        """
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                clean_messages = []
                for msg in messages:
                    clean_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    })
                kwargs: dict[str, Any] = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if enable_thinking is not None:
                    kwargs["enable_thinking"] = enable_thinking

                text = self._tokenizer.apply_chat_template(
                    clean_messages,
                    **kwargs,
                )
                if text:
                    return text
            except Exception:
                pass

        # Generic fallback
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _init_memory_monitor(self) -> None:
        """Initialize memory monitor with model architecture info (oMLX pattern).

        Extracts model dimensions from config for accurate memory estimation.
        """
        if self._model is None:
            return

        config = getattr(self._model, 'config', None)
        if config is None:
            return

        # Extract architecture params
        num_layers = (
            getattr(config, 'num_hidden_layers', None)
            or (config.get('num_hidden_layers') if isinstance(config, dict) else None)
        )
        num_kv_heads = (
            getattr(config, 'num_key_value_heads', None)
            or (config.get('num_key_value_heads') if isinstance(config, dict) else None)
        )
        num_attn_heads = (
            getattr(config, 'num_attention_heads', None)
            or (config.get('num_attention_heads') if isinstance(config, dict) else None)
        )
        head_dim = (
            getattr(config, 'head_dim', None)
            or (config.get('head_dim') if isinstance(config, dict) else None)
        )

        # Infer head_dim from hidden_size / num_heads if not explicit
        if head_dim is None and num_attn_heads:
            hidden = (
                getattr(config, 'hidden_size', None)
                or (config.get('hidden_size') if isinstance(config, dict) else None)
            )
            if hidden:
                head_dim = hidden // num_attn_heads

        if num_layers and num_kv_heads and head_dim:
            self._memory_monitor.set_model_info(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_attention_heads=num_attn_heads,
            )

        # Set baseline memory after model weights are loaded
        self._memory_monitor.set_baseline_memory()

    def get_stats(self) -> dict:
        """Return engine stats (oMLX Scheduler.get_stats pattern)."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0
        base = {
            "model": self._model_name,
            "loaded": self.is_loaded,
            "running": self._running,
            "engine_core": self._engine_core is not None,
            "uptime_seconds": round(uptime, 1),
            "gpu_memory": self._memory_monitor.get_stats() if self.is_loaded else None,
        }
        if self._engine_core is not None:
            base.update(self._engine_core.get_stats())
        else:
            base.update({
                "waiting": len(self._waiting),
                "active": len(self._active),
                "active_uids": len(self._uid_to_req),
                "pending_aborts": len(self._abort_set),
                "step_counter": self._step_counter,
                "num_requests_processed": self._num_requests_processed,
                "total_prompt_tokens": self._total_prompt_tokens,
                "total_completion_tokens": self._total_completion_tokens,
            })
        return base
