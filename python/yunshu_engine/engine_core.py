from __future__ import annotations
"""Yunshu EngineCore — continuous batching orchestrator (oMLX pattern).

Studied from oMLX's engine_core.py, written from scratch:
- EngineCore orchestrates Scheduler + RequestOutputCollector + RequestStreamState
- All scheduler.step() calls run on the MLX executor thread (serialized GPU work)
- Output distribution happens on the event loop (low latency)
- stream_outputs() uses get_nowait() or await get() pattern from vLLM
- generate() waits on asyncio.Event then drains collector
- Request lifecycle: add → schedule → step → distribute → collect

Architecture:
  Gateway → EngineCore.add_request()
    → Scheduler.add_request() on MLX executor
  EngineCore._engine_loop()
    → Scheduler.step() on MLX executor
    → distribute outputs to per-request RequestOutputCollector
  Gateway → EngineCore.stream_outputs(request_id)
    → collector.get_nowait() or await collector.get()
"""

import asyncio
import gc
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class EngineCoreConfig:
    """EngineCore tuning parameters (maps to oMLX's EngineConfig)."""
    step_interval: float = 0.001
    stream_interval: int = 1
    completion_batch_size: int = 32
    prefill_batch_size: int = 8
    prefill_step_size: int = 2048
    max_kv_size: int | None = None
    deferred_clear_delay: int = 8
    cache_cleanup_interval: int = 512
    # Paged KV cache (oMLX PagedAttention pattern)
    enable_paged_kv: bool = True  # C11: enabled by default for radix tree + memory efficiency
    kv_block_size: int = 64
    kv_cache_ratio: float = 0.25  # fraction of UMA for KV cache
    # Model architecture (for KV cache memory budget)
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    kv_num_blocks: int = 0  # pre-computed block count (0 = auto-compute)
    # C18: CPU/GPU overlap scheduling (SGLang pattern)
    enable_cpu_gpu_overlap: bool = False  # Disabled by default; enable via YUNSHU_CPU_GPU_OVERLAP=1
    # N-gram speculative decoding in batch path (model-free, zero GPU overhead)
    ngram_spec_enabled: bool = False
    ngram_spec_min_n: int = 1
    ngram_spec_max_n: int = 5
    ngram_spec_k: int = 5
    ngram_spec_mode: str = "lps"
    # External prefill (memory preflight, chunked progress, mid-prefill abort)
    use_external_prefill: bool = False
    prefill_chunk_size: int = 2048
    # Sarathi-style hybrid chunked prefill (interleave prefill chunks with decode)
    enable_hybrid_prefill: bool = False
    hybrid_chunk_size: int = 512
    # Chunked prefill production hardening
    chunked_prefill_budget: int = 4     # Max chunks per scheduling round (fairness)
    chunked_prefill_timeout_seconds: float = 30.0  # Per-request prefill timeout (0 = no timeout)
    chunked_prefill_abort_on_timeout: bool = True  # Abort request on timeout (vs force-feed)
    # Per-request generation timeout (seconds, 0 = no timeout)
    request_timeout_seconds: float = 300.0


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute from a config object or dict."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


class EngineCore:
    """MLX-native continuous batching engine core (oMLX EngineCore pattern).

    Orchestrates:
    - Scheduler: manages BatchGenerator + request lifecycle
    - RequestOutputCollector: per-request output buffer with smart aggregation
    - RequestStreamState: stream_interval batching
    - asyncio.Event: per-request completion signaling

    Threading: scheduler.step() runs on MLX executor (single GPU thread).
    Everything else runs on the asyncio event loop.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: EngineCoreConfig | None = None,
        executor: Any = None,
    ) -> None:
        from .scheduler import Scheduler, SchedulerConfig
        from .mlx_executor import get_mlx_executor

        self.config = config or EngineCoreConfig()
        self._executor = executor or get_mlx_executor()

        scheduler_config = SchedulerConfig(
            completion_batch_size=self.config.completion_batch_size,
            prefill_batch_size=self.config.prefill_batch_size,
            prefill_step_size=self.config.prefill_step_size,
            max_kv_size=self.config.max_kv_size,
            deferred_clear_delay=self.config.deferred_clear_delay,
            cache_cleanup_interval=self.config.cache_cleanup_interval,
            stream_interval=self.config.stream_interval,
            step_interval=self.config.step_interval,
            ngram_spec_enabled=self.config.ngram_spec_enabled,
            ngram_spec_min_n=self.config.ngram_spec_min_n,
            ngram_spec_max_n=self.config.ngram_spec_max_n,
            ngram_spec_k=self.config.ngram_spec_k,
            ngram_spec_mode=self.config.ngram_spec_mode,
            enable_hybrid_prefill=self.config.enable_hybrid_prefill,
            hybrid_chunk_size=self.config.hybrid_chunk_size,
            use_external_prefill=self.config.use_external_prefill,
            prefill_chunk_size=self.config.prefill_chunk_size,
            request_timeout_seconds=self.config.request_timeout_seconds,
            chunked_prefill_budget=self.config.chunked_prefill_budget,
            chunked_prefill_timeout_seconds=self.config.chunked_prefill_timeout_seconds,
            chunked_prefill_abort_on_timeout=self.config.chunked_prefill_abort_on_timeout,
        )

        if self.config.enable_paged_kv:
            # Paged KV requires model architecture info
            has_arch = (
                self.config.num_layers > 0
                and self.config.num_kv_heads > 0
                and self.config.head_dim > 0
            )
            if not has_arch:
                logger.info(
                    "Paged KV requested but model arch info missing "
                    f"(layers={self.config.num_layers}, kv_heads={self.config.num_kv_heads}, "
                    f"head_dim={self.config.head_dim}). Falling back to non-paged scheduler."
                )
                self.config.enable_paged_kv = False
            else:
                try:
                    from .paged_scheduler import PagedScheduler
                    from yunshu_kv.manager import KVCacheManager, KVCacheConfig, compute_num_blocks

                    kv_config = KVCacheConfig(
                        block_size=self.config.kv_block_size,
                        num_layers=self.config.num_layers,
                        num_kv_heads=self.config.num_kv_heads,
                        head_dim=self.config.head_dim,
                    )

                    # Determine block count
                    num_blocks = self.config.kv_num_blocks
                    if num_blocks <= 0:
                        # Auto-compute from UMA budget minus model weights
                        try:
                            from .utils.hardware import get_hardware_info
                            _hw = get_hardware_info()
                            uma_bytes = _hw.total_memory_bytes
                            # Compute actual model weight bytes so we don't
                            # over-allocate KV cache.  On 8GB M1 with a 4GB
                            # model the old code (model_weight_bytes=0) assumed
                            # 6.8 GB available when only ~2.8 GB remained.
                            _model_bytes = 0
                            try:
                                if model is not None:
                                    _model_bytes = sum(
                                        p.nbytes
                                        for p in model.parameters()
                                    )
                            except Exception:
                                logger.debug("model parameter scan failed", exc_info=True)
                            if uma_bytes > 0:
                                num_blocks = compute_num_blocks(
                                    kv_config, uma_bytes, _model_bytes,
                                )
                            else:
                                num_blocks = 1024  # safe default
                        except Exception:
                            logger.debug("memory monitor unavailable, using default block count", exc_info=True)
                            num_blocks = 1024  # safe default

                    kv_manager = KVCacheManager(kv_config, num_blocks=num_blocks)

                    # Wrap with TieredKVCacheManager if SSD cache is configured
                    ssd_dir = os.environ.get("YUNSHU_SSD_CACHE_DIR")
                    if ssd_dir:
                        from yunshu_kv.tiered import TieredKVCacheManager, SSDCacheStore
                        ssd_store = SSDCacheStore(
                            cache_dir=ssd_dir,
                            block_size=self.config.kv_block_size,
                        )
                        kv_manager = TieredKVCacheManager(
                            hot_manager=kv_manager,
                            ssd_store=ssd_store,
                            warm_tier=kv_manager._warm_tier,
                        )
                        logger.info(f"TieredKVCacheManager enabled: SSD dir={ssd_dir}")

                    self.scheduler = PagedScheduler(model, tokenizer, scheduler_config, kv_manager)
                    self._kv_manager = kv_manager
                    logger.info(
                        f"PagedScheduler enabled: block_size={self.config.kv_block_size}, "
                        f"num_blocks={num_blocks}"
                    )
                except Exception as e:
                    logger.warning(f"Paged KV init failed ({e}), falling back to non-paged")
                    self.config.enable_paged_kv = False

        if not self.config.enable_paged_kv:
            self.scheduler = Scheduler(model, tokenizer, scheduler_config)
            self._kv_manager = None
        else:
            # Wire KV optimization modules into PagedScheduler
            try:
                from .kv_optimizations import KVBlockCompactor, KVEvictionPredictor
                compactor = KVBlockCompactor()
                predictor = KVEvictionPredictor()
                if hasattr(self.scheduler, 'set_compactor'):
                    self.scheduler.set_compactor(compactor)
                if hasattr(self.scheduler, 'set_eviction_predictor'):
                    self.scheduler.set_eviction_predictor(predictor)
                logger.info("KVBlockCompactor + KVEvictionPredictor wired into PagedScheduler")
            except Exception:
                logger.debug("KV optimization wiring skipped", exc_info=True)

        # Wire ServerMetrics + PrefillProgressTracker into scheduler
        try:
            from .server_metrics import get_server_metrics
            self.scheduler.set_server_metrics(get_server_metrics())
        except Exception:
            logger.debug("server_metrics unavailable", exc_info=True)
        try:
            from .prefill_progress import get_prefill_tracker
            self.scheduler.set_prefill_tracker(get_prefill_tracker())
        except Exception:
            logger.debug("prefill_progress tracker unavailable", exc_info=True)

        # Per-request output management
        self._output_collectors: dict[str, Any] = {}
        self._stream_states: dict[str, Any] = {}
        self._finished_events: dict[str, asyncio.Event] = {}

        # Tokenizer reference (for chat template, encoding)
        self._tokenizer = tokenizer
        self._model = model

        # Memory guard (created after model info is available)
        self._memory_guard: Any = None

        # C18: CPU/GPU overlap scheduler
        from .cpu_gpu_overlap import OverlapConfig, OverlapScheduler
        overlap_cfg = OverlapConfig.from_env()
        if self.config.enable_cpu_gpu_overlap:
            overlap_cfg.enabled = True
        self._overlap_scheduler = OverlapScheduler(overlap_cfg)

        # §14.1: Two-Batch Overlap scheduler (TBO — SGLang pattern)
        from .two_batch_overlap import TwoBatchOverlapScheduler, TBOConfig
        tbo_cfg = TBOConfig.from_env()
        self._tbo_scheduler = TwoBatchOverlapScheduler(tbo_cfg)

        # Adaptive batch scheduler (load-aware batch sizing)
        from .adaptive_batch import AdaptiveBatchScheduler, AdaptiveBatchConfig
        self._adaptive_batch = AdaptiveBatchScheduler(AdaptiveBatchConfig())

        # Telemetry (sampled metric collection)
        from .telemetry import TelemetryCollector, TelemetryConfig
        telemetry_enabled = os.environ.get("YUNSHU_TELEMETRY", "0") == "1"
        self._telemetry = TelemetryCollector(TelemetryConfig(enabled=telemetry_enabled))

        # KV offload manager (async tier-to-tier block migration, §12.3)
        from .kv_offload import KVOffloadConfig, KVOffloadManager
        kv_offload_cfg = KVOffloadConfig.from_env()
        self._kv_offload_manager = KVOffloadManager(kv_offload_cfg, kv_manager=self._kv_manager)

        # Pass offload manager to scheduler for periodic sync offload checks
        self.scheduler.set_kv_offload_manager(self._kv_offload_manager)

        # External prefill server/client (disaggregated prefill, §16.2)
        from .external_prefill import (
            ExternalPrefillConfig,
            ExternalPrefillServer,
            ExternalPrefillClient,
            get_prefill_role,
        )
        self._prefill_server: ExternalPrefillServer | None = None
        self._prefill_client: ExternalPrefillClient | None = None
        self._prefill_task: asyncio.Task | None = None
        self._prefill_role: str | None = None
        prefill_role = get_prefill_role()
        self._prefill_role = prefill_role
        if prefill_role == "server":
            prefill_config = ExternalPrefillConfig.from_env()
            self._prefill_server = ExternalPrefillServer(
                model, tokenizer, prefill_config,
            )
            logger.info("ExternalPrefillServer configured (disaggregated prefill)")
        elif prefill_role == "client":
            prefill_config = ExternalPrefillConfig.from_env()
            self._prefill_client = ExternalPrefillClient(prefill_config)
            logger.info("ExternalPrefillClient configured (remote prefill)")

        # KV transfer server (receives KV blocks on decode nodes, §12.2)
        self._kv_transfer_server: Any | None = None
        try:
            from .kv_transfer import KVTransferServer, KVTransferConfig, is_kv_transfer_enabled
            if is_kv_transfer_enabled():
                kv_xfer_config = KVTransferConfig.from_env()
                self._kv_transfer_server = KVTransferServer(
                    kv_xfer_config,
                    kv_cache_manager=self._kv_manager,
                )
                logger.info(
                    "KVTransferServer configured on port %d",
                    kv_xfer_config.listen_port,
                )
        except Exception:
            logger.debug("KV transfer server setup skipped", exc_info=True)

        # ── Wave 42: 實現-整合 wiring ──

        # Request lifecycle orchestrator (QUEUED→PREFILLING→DECODING→FINISHED)
        from .request_lifecycle import RequestLifecycleOrchestrator, AdaptiveConcurrencyController
        concurrency_ctrl = AdaptiveConcurrencyController.from_env()
        self._lifecycle_orchestrator = RequestLifecycleOrchestrator(
            concurrency_controller=concurrency_ctrl,
        )
        logger.info(f"RequestLifecycleOrchestrator wired (max_concurrent={concurrency_ctrl._maximum})")

        # Inference budget manager (token/time/cost/thinking 4-dimension budgets)
        from .inference_budget import InferenceBudgetManager
        self._budget_manager = InferenceBudgetManager.from_env()
        logger.info("InferenceBudgetManager wired")

        # Request deduplication (SHA-256 content-hash dedup with fan-out)
        from .request_dedup import RequestDeduplicator
        self._request_dedup: RequestDeduplicator | None = None
        self._dedup_hashes: dict[str, str] = {}  # req_id → content_hash
        self._dedup_shadows: dict[str, str] = {}  # shadow_req_id → primary_req_id
        self._finalized_ids: set[str] = set()  # idempotency guard for _finalize_request
        self._ttft_done: set[str] = set()  # TTFT deduplication guard
        if os.environ.get("YUNSHU_REQUEST_DEDUP", "").strip() in ("1", "true", "yes"):
            self._request_dedup = RequestDeduplicator.from_env()
            logger.info("RequestDeduplicator wired (SHA-256 content-hash dedup)")

        # KV lifecycle manager (4-tier hot/warm/cool/cold admission/migration/eviction)
        from .kv_lifecycle import KVLifecycleManager
        self._kv_lifecycle = KVLifecycleManager()
        logger.info("KVLifecycleManager wired (4-tier KV lifecycle)")

        # Token-level scheduler (WFQ token budget + priority inversion guard)
        from .token_scheduler import TokenLevelScheduler, PriorityInversionGuard, FairnessTracker
        self._token_scheduler = TokenLevelScheduler()
        self._priority_guard = PriorityInversionGuard()
        self._fairness_tracker = FairnessTracker()

        # Auto-tuner (adaptive config: performance profiler + SLO monitor + hill-climbing)
        from .auto_tuner import AutoTuner, PerformanceProfiler, SLOMonitor, AdaptiveBatchSizer
        self._profiler = PerformanceProfiler()
        self._profiler_started = False  # started in start(), stopped in stop()
        self._slo_monitor = SLOMonitor()
        self._auto_tuner = AutoTuner(profiler=self._profiler, slo_monitor=self._slo_monitor)
        self._adaptive_batch_sizer = AdaptiveBatchSizer()

        # Composition scheduler mixins (SGLang §14.1 pattern)
        from .scheduler_mixins import (
            CompositionScheduler, MetricsMixin, MemoryPressureMixin,
            ProfilingMixin, DisaggregationMixin, DataParallelMixin,
            PipelineParallelMixin, SpecDecodeMixin,
        )
        try:
            self._composition_scheduler = CompositionScheduler(self.scheduler)
            self._composition_scheduler.add_mixin(MetricsMixin())
            self._composition_scheduler.add_mixin(MemoryPressureMixin.from_env())

            # ProfilingMixin — YUNSHU_SCHEDULER_PROFILING=1
            if os.environ.get("YUNSHU_SCHEDULER_PROFILING", "").strip() == "1":
                self._composition_scheduler.add_mixin(ProfilingMixin())

            # DisaggregationMixin — YUNSHU_DISAGGREGATED=1 with node lists
            if os.environ.get("YUNSHU_DISAGGREGATED", "").strip() == "1":
                prefill_nodes = os.environ.get("YUNSHU_PREFILL_NODES", "").split(",") if os.environ.get("YUNSHU_PREFILL_NODES") else []
                decode_nodes = os.environ.get("YUNSHU_DECODE_NODES", "").split(",") if os.environ.get("YUNSHU_DECODE_NODES") else []
                self._composition_scheduler.add_mixin(DisaggregationMixin(
                    prefill_nodes=[n.strip() for n in prefill_nodes if n.strip()],
                    decode_nodes=[n.strip() for n in decode_nodes if n.strip()],
                ))

            # DataParallelMixin — YUNSHU_DATA_PARALLEL=1 with replica count
            dp_replicas = int(os.environ.get("YUNSHU_DP_REPLICAS", "1"))
            if dp_replicas > 1:
                self._composition_scheduler.add_mixin(DataParallelMixin(
                    num_replicas=dp_replicas,
                    strategy=os.environ.get("YUNSHU_DP_STRATEGY", "least_loaded"),
                ))

            # PipelineParallelMixin — YUNSHU_PIPELINE_PARALLEL=1
            if os.environ.get("YUNSHU_PIPELINE_PARALLEL", "").strip() == "1":
                self._composition_scheduler.add_mixin(PipelineParallelMixin(
                    num_stages=int(os.environ.get("YUNSHU_PP_STAGES", "1")),
                    stage_id=int(os.environ.get("YUNSHU_PP_STAGE_ID", "0")),
                    micro_batch_size=int(os.environ.get("YUNSHU_PP_MICRO_BATCH", "1")),
                ))

            # SpecDecodeMixin — YUNSHU_SPEC_DECODE_TRACKING=1
            if os.environ.get("YUNSHU_SPEC_DECODE_TRACKING", "").strip() == "1":
                self._composition_scheduler.add_mixin(SpecDecodeMixin())
        except Exception:
            logger.debug("CompositionScheduler setup skipped", exc_info=True)
            self._composition_scheduler = None

        # Lifecycle — vLLM 3-state shutdown pattern:
        # RUNNING → REQUESTED (drain in-flight) → SHUTTING_DOWN (force stop)
        self._running = False
        self._shutdown_requested = False
        self._was_started = False  # True after first start(), even if later stopped
        self._loop_task: asyncio.Task | None = None
        self._start_time: float | None = None
        self._wake_event: asyncio.Event | None = None  # Event-driven wake-up for idle loop
        self._stopped = False  # True after stop() completes; reset by start()

        # ── Wave 43: Additional production wiring ──

        # Forward batch hierarchy (ScheduleBatch → ForwardBatch → BatchResult)
        from .forward_batch import BatchComposer
        self._batch_composer = BatchComposer()

        # Memory-aware scheduler (admission control with memory budget)
        from .memory_aware_scheduler import MemoryAwareScheduler
        self._memory_aware_scheduler = MemoryAwareScheduler()

        # Configure memory-aware scheduler with model parameters for accurate estimation
        try:
            model_config = getattr(model, 'config', model) if model else None
            if model_config is not None:
                layers = _safe_get(model_config, 'num_hidden_layers', 0)
                heads = _safe_get(model_config, 'num_key_value_heads',
                                  _safe_get(model_config, 'num_attention_heads', 0))
                h_dim = _safe_get(model_config, 'head_dim', 0)
                self._memory_aware_scheduler.set_model_config(
                    num_layers=layers, num_kv_heads=heads, head_dim=h_dim,
                )
        except Exception:
            logger.debug("memory_aware_scheduler model config skipped", exc_info=True)

        # Context window manager (4 truncation strategies for long prompts)
        from .context_window import ContextWindowManager
        self._context_window_mgr = ContextWindowManager()

        # KV prefix compression (mean_pool/top_k/frequency_aware strategies)
        from .kv_prefix_compression import KVPrefixCompressor, SlidingWindowKVManager
        self._kv_compressor = KVPrefixCompressor()
        self._sliding_window_mgr: SlidingWindowKVManager | None = None
        # Auto-detect sliding window from model config
        try:
            model_cfg = getattr(model, 'config', model) if model else None
            if model_cfg is not None:
                sw = getattr(model_cfg, 'sliding_window', None)
                if sw is not None and sw > 0:
                    self._sliding_window_mgr = SlidingWindowKVManager(window_size=sw)
                    logger.info(f"SlidingWindowKVManager enabled: window={sw}")
        except Exception:
            logger.debug("sliding window detection skipped", exc_info=True)

        # KV migration manager (multi-tier migration with temperature tracking)
        from .kv_migration import KVMigrationManager
        self._kv_migration = KVMigrationManager()

        # Mamba/Hybrid KV cache (SSM state management)
        from .mamba_cache import HybridKVCache
        self._hybrid_kv = HybridKVCache()

        # Batch sampler (vectorized batch sampling + logits processing)
        from .batch_sampler import BatchSampler
        self._batch_sampler = BatchSampler()

        # Model optimizations (RoPE scaling, attention type detection, MoE efficiency)
        from .model_optimizations import (
            RoPEScalingOptimizer, AttentionOptimizer, MoEEfficiencyOptimizer, ModelWarmupManager,
        )
        self._rope_optimizer = RoPEScalingOptimizer()
        self._attention_optimizer = AttentionOptimizer()
        self._moe_optimizer = MoEEfficiencyOptimizer()
        self._warmup_manager = ModelWarmupManager()

        # Process isolation (opt-in via YUNSHU_PROCESS_ISOLATION=1)
        self._isolation_enabled = False
        if os.environ.get("YUNSHU_PROCESS_ISOLATION", "").lower() in ("1", "true", "yes"):
            from .process_isolation import is_isolation_enabled
            self._isolation_enabled = is_isolation_enabled()
            if self._isolation_enabled:
                logger.info("Process isolation enabled (YUNSHU_PROCESS_ISOLATION=1)")

        # Inference checkpoint/restore (opt-in via YUNSHU_AUTO_CHECKPOINT)
        self._checkpoint_mgr = None
        _cp_interval = int(os.environ.get("YUNSHU_CHECKPOINT_INTERVAL", "0"))
        if _cp_interval > 0:
            from .checkpoint import InferenceCheckpoint, AutoCheckpointPolicy
            self._checkpoint_mgr = InferenceCheckpoint(
                auto_checkpoint_interval=_cp_interval,
                auto_checkpoint_policy=AutoCheckpointPolicy.EVERY_N_TOKENS,
            )
            logger.info(f"Auto-checkpoint enabled: every {_cp_interval} tokens")

        # Output parser (model-specific output extraction)
        from .output_parser import parse_output
        self._parse_output = parse_output

        # Model preprocessor registry (auto-detects model family for multimodal input)
        from .model_preprocessor import PreprocessorRegistry
        self._preprocessor_registry = PreprocessorRegistry()

        # SpecPrefill engine (priority prefill queue for GPU idle time)
        from .spec_prefill_engine import SpecPrefillEngine
        self._spec_prefill_engine = SpecPrefillEngine()

        # TurboQuant (fast quantization utilities)
        from .turbo_quant import TurboQuantManager, TurboQuantConfig
        self._turbo_quant = TurboQuantManager(TurboQuantConfig())

        # Staged multimodal pipeline coordinator (7-stage processing)
        from .staged_pipeline import MultimodalPipelineCoordinator
        self._multimodal_pipeline = MultimodalPipelineCoordinator()

        # Stats
        self._num_requests_processed: int = 0
        self._request_timestamps: dict[str, float] = {}  # req_id → monotonic start time
        self._ttft_timestamps: dict[str, float] = {}  # req_id → monotonic first-token time
        self._request_lora_adapters: dict[str, str] = {}  # req_id → lora_adapter_id

        # Cache-locality request reordering (SGLang/vLLM pattern)
        # Maps request_id → KV prefix hash for grouping requests with shared
        # prefixes. Requests sharing the same prefix hash are scheduled
        # consecutively to maximize KV cache block locality and reduce thrashing.
        self._kv_prefix_hashes: dict[str, int] = {}

        # MON-2/4/5: Scheduler monitoring gauges (read by Prometheus exporter)
        self._last_batch_size: int = 0
        self._last_queue_depth: int = 0
        self._last_step_wall_ms: float = 0.0
        self._total_step_time_ms: float = 0.0
        self._total_idle_time_ms: float = 0.0
        self._monitoring_start_time: float = time.monotonic()

    def set_prefix_cache(self, cache: Any) -> None:
        """Set KV prefix cache for batch-path insert_segments (C16)."""
        self.scheduler.set_prefix_cache(cache)
        # Wire MemoryGuard.should_evict_block into the prefix cache's eviction
        # path so prediction-based eviction decisions are respected.
        if self._memory_guard is not None and hasattr(cache, '_block_evict_checker'):
            cache._block_evict_checker = self._memory_guard.should_evict_block

    def set_metal_kernel_manager(self, manager: Any) -> None:
        """Set Metal kernel manager for custom GPU kernel operations.

        Wires the MetalKernelManager into the scheduler so batch-path
        operations can use Metal-accelerated paged attention, GEMV,
        and KIVI 2-bit KV compression.
        """
        self.scheduler.set_metal_kernel_manager(manager)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def has_active_requests(self) -> bool:
        return self.scheduler.has_requests()

    def get_compute_utilization(self) -> float:
        """Return percentage of time spent in active scheduler steps vs total time.

        Returns 0.0 when no steps have been taken yet. Value is 0–100.
        """
        total_ms = self._total_step_time_ms + self._total_idle_time_ms
        if total_ms <= 0:
            return 0.0
        return min(self._total_step_time_ms / total_ms * 100.0, 100.0)

    def _apply_tuning_to_config(self, decisions: list) -> None:
        """Apply auto-tuner decisions to the live engine configuration.

        Propagates parameter changes from AutoTuner's internal TunableParams
        back into EngineCoreConfig so they take effect on subsequent steps.
        """
        try:
            params = self._auto_tuner.params
            # Apply batch size tuning (clamped by AdaptiveBatchSizer bounds)
            if params.batch_size != self.config.completion_batch_size:
                old = self.config.completion_batch_size
                self.config.completion_batch_size = max(1, min(params.batch_size, 64))
                if self.config.completion_batch_size != old:
                    logger.info(
                        "AutoTuner: batch_size %d -> %d",
                        old, self.config.completion_batch_size,
                    )
            # Apply prefill chunk size tuning
            if params.prefill_chunk_size != self.config.prefill_chunk_size:
                old = self.config.prefill_chunk_size
                self.config.prefill_chunk_size = params.prefill_chunk_size
                if self.config.prefill_chunk_size != old:
                    logger.info(
                        "AutoTuner: prefill_chunk_size %d -> %d",
                        old, self.config.prefill_chunk_size,
                    )
        except Exception:
            logger.debug("auto-tuner config application failed", exc_info=True)

    def setup_memory_guard(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_attention_heads: int | None = None,
        max_concurrent_requests: int = 64,
    ) -> None:
        """Create and configure the MemoryGuard after model info is available."""
        from .memory_monitor import MemoryMonitor
        from .memory_guard import MemoryGuard

        # Compute KV cache budget so MemoryMonitor.get_memory_info()
        # reports available memory relative to the actual KV budget rather
        # than the full working set — prevents over-admitting requests.
        kv_budget = 0
        try:
            from .utils.hardware import get_hardware_info
            _hw = get_hardware_info()
            kv_budget = int(_hw.max_working_set_bytes * self.config.kv_cache_ratio)
        except Exception:
            logger.debug("KV budget computation failed, using default", exc_info=True)

        monitor = MemoryMonitor(max_kv_cache_memory=kv_budget)
        monitor.set_model_info(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
        )
        monitor.set_baseline_memory()

        self._memory_guard = MemoryGuard(
            memory_monitor=monitor,
            max_concurrent_requests=max_concurrent_requests,
        )
        logger.info(
            f"MemoryGuard configured: {num_layers}L, {num_kv_heads} KV heads, "
            f"{head_dim}d, max_concurrent={max_concurrent_requests}"
        )

    def setup_turbo_quant(
        self,
        total_layers: int,
        kv_quant_bits: int | None = None,
        kv_quant_start_layer: int = 0,
        kv_quant_group_size: int = 64,
    ) -> None:
        """Configure TurboQuant per-layer mixed-precision KV quantization."""
        if total_layers <= 0:
            return
        from .turbo_quant import TurboQuantConfig
        if kv_quant_bits is not None:
            config = TurboQuantConfig(
                enabled=True,
                total_layers=total_layers,
                fp16_end_layer=max(kv_quant_start_layer - 1, 0),
                int8_end_layer=min(kv_quant_start_layer + total_layers // 3, total_layers - 1),
                int4_group_size=kv_quant_group_size,
            )
        else:
            # Default: lightweight mixed-precision profile
            config = TurboQuantConfig(
                enabled=True,
                total_layers=total_layers,
                fp16_end_layer=min(3, total_layers - 1),
                int8_end_layer=min(total_layers // 2, total_layers - 1),
                int4_group_size=64,
            )
        self._turbo_quant = TurboQuantManager(config)
        logger.info(
            f"TurboQuant configured: {total_layers} layers, "
            f"compression={config.expected_compression_ratio:.1f}x"
        )

    def setup_hybrid_kv(
        self,
        model: Any,
        total_layers: int,
    ) -> None:
        """Register model layer types in HybridKVCache for Mamba/hybrid models.

        Scans model layers for SSM (state-space model) vs attention types and
        registers them so the scheduler can route allocate/free correctly.
        """
        from .mamba_cache import CacheBlockType
        registered = 0
        for idx in range(total_layers):
            try:
                layer = model.layers[idx] if hasattr(model, 'layers') else None
                if layer is None:
                    continue
                # Detect SSM layers by presence of state attribute or class name
                layer_cls = type(layer).__name__.lower()
                has_ssm = (
                    hasattr(layer, 'state')
                    or 'mamba' in layer_cls
                    or 'ssm' in layer_cls
                    or 'deltanet' in layer_cls
                )
                if has_ssm:
                    self._hybrid_kv.register_layer(
                        idx, CacheBlockType.MAMBA_SSM,
                        cache_shape=(48, 16),
                    )
                else:
                    model_cfg = getattr(model, 'config', model)
                    num_heads = getattr(model_cfg, 'num_key_value_heads', 1)
                    head_dim = getattr(model_cfg, 'hidden_size', 1) // max(
                        getattr(model_cfg, 'num_attention_heads', 1), 1
                    )
                    # Detect sliding window attention
                    sw = getattr(model_cfg, 'sliding_window', None)
                    # Detect MLA (DeepSeek) by kv_lora_rank or q_lora_rank
                    has_mla = (
                        hasattr(layer, 'kv_lora_rank')
                        or 'mla' in layer_cls
                        or getattr(model_cfg, 'kv_lora_rank', None) is not None
                    )
                    if has_mla:
                        latent_dim = getattr(model_cfg, 'kv_lora_rank', head_dim)
                        self._hybrid_kv.register_layer(
                            idx, CacheBlockType.MLA,
                            cache_shape=(latent_dim, self.config.kv_block_size),
                        )
                    elif sw is not None and sw > 0:
                        self._hybrid_kv.register_layer(
                            idx, CacheBlockType.SLIDING_WINDOW,
                            cache_shape=(num_heads, head_dim, self.config.kv_block_size),
                        )
                    else:
                        self._hybrid_kv.register_layer(
                            idx, CacheBlockType.ATTENTION,
                            cache_shape=(num_heads, head_dim, self.config.kv_block_size),
                        )
                registered += 1
            except Exception:
                logger.debug("hybrid KV layer registration failed", exc_info=True)
                break
        if registered > 0:
            logger.info(
                f"HybridKVCache registered {registered}/{total_layers} layers "
                f"({len(self._hybrid_kv._pools)} pools)"
            )

    async def start(self) -> None:
        """Start the engine loop."""
        if self._running:
            return
        # Guard against stale _loop_task from a failed/incomplete stop()
        if self._loop_task is not None and not self._loop_task.done():
            logger.warning("start() called with stale _loop_task, cancelling")
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        self._running = True
        self._was_started = True
        self._stopped = False
        self._shutdown_requested = False
        self._start_time = time.monotonic()
        self._wake_event = asyncio.Event()
        # Start profiler (deferred from __init__ to avoid resource leak if init fails)
        if not self._profiler_started:
            try:
                self._profiler.start_profiling()
                self._profiler_started = True
            except Exception:
                logger.debug("profiler start failed", exc_info=True)
        # Start KV offload manager (async tier migration, §12.3)
        if self._kv_offload_manager is not None:
            try:
                await self._kv_offload_manager.start()
            except Exception:
                logger.debug("KV offload manager start failed", exc_info=True)
        self._loop_task = asyncio.get_running_loop().create_task(self._engine_loop())

        # Start KV migration background thread
        try:
            self._kv_migration.start()
        except Exception:
            logger.debug("KV migration start failed", exc_info=True)

        # Start external prefill server if configured
        # NOTE: task is fire-and-forget but will be stopped cleanly via
        # _prefill_server.stop() in our stop() method.
        if self._prefill_server is not None:
            self._prefill_task = asyncio.get_running_loop().create_task(
                self._prefill_server.serve()
            )
            logger.info("ExternalPrefillServer started")

        # Start KV transfer server if configured (decode node receives KV blocks)
        if self._kv_transfer_server is not None:
            try:
                await self._kv_transfer_server.start()
                logger.info("KVTransferServer started")
            except Exception:
                logger.debug("KV transfer server start failed", exc_info=True)

        logger.info("EngineCore started")

        # Checkpoint recovery: restore in-flight requests from previous crash
        if self._checkpoint_mgr is not None:
            try:
                saved_ids = self._checkpoint_mgr.list_checkpoints()
                if saved_ids:
                    logger.info(f"Checkpoint recovery: {len(saved_ids)} saved states found")
                    for ckpt_id in saved_ids:
                        state = self._checkpoint_mgr.load(ckpt_id)
                        if state is not None:
                            logger.debug(f"Restored checkpoint for {ckpt_id}")
            except Exception:
                logger.debug("checkpoint recovery failed", exc_info=True)

    async def stop(self) -> None:
        """Stop the engine with graceful drain (vLLM 3-state shutdown pattern).

        RUNNING → REQUESTED: reject new requests, let in-flight finish.
        REQUESTED → SHUTTING_DOWN: after drain timeout, force-stop remaining.

        Idempotent: safe to call multiple times or before start().
        """
        # Guard: already stopped or never started
        if not self._running and self._loop_task is None:
            return

        # Phase 1: REQUESTED — signal graceful shutdown, reject new requests
        self._shutdown_requested = True

        # Wake the engine loop so it notices shutdown_requested immediately
        if self._wake_event is not None:
            self._wake_event.set()

        # Phase 2: wait for in-flight requests to drain (up to 30s)
        if self._loop_task is not None:
            drain_timeout = float(os.environ.get("YUNSHU_SHUTDOWN_DRAIN_TIMEOUT", "30"))
            if self.scheduler.has_requests():
                logger.info(
                    f"Graceful shutdown requested, waiting up to {drain_timeout}s "
                    f"for in-flight requests to complete"
                )
            try:
                await asyncio.wait_for(self._loop_task, timeout=drain_timeout)
            except asyncio.TimeoutError:
                active = len(self.scheduler.running) + len(self.scheduler.waiting)
                logger.warning(
                    f"Graceful shutdown timeout ({drain_timeout}s), "
                    f"force-stopping with {active} active requests"
                )
            except asyncio.CancelledError:
                pass

            # If loop task is still running, force-cancel
            if not self._loop_task.done():
                self._loop_task.cancel()
                try:
                    await self._loop_task
                except asyncio.CancelledError:
                    pass
            self._loop_task = None

        # Phase 3: SHUTTING_DOWN — full cleanup
        self._running = False
        self._start_time = None  # Reset so get_stats() uptime is 0 after stop
        self._wake_event = None  # Clear stale event; recreated by start()

        # Stop KV migration background thread
        try:
            self._kv_migration.stop()
        except Exception:
            logger.debug("KV migration stop failed", exc_info=True)

        # Stop performance profiler
        if self._profiler_started:
            try:
                self._profiler.stop_profiling()
                self._profiler_started = False
            except Exception:
                logger.debug("profiler stop failed", exc_info=True)

        # Stop KV offload manager
        if self._kv_offload_manager is not None:
            try:
                await self._kv_offload_manager.stop()
            except Exception:
                logger.debug("KV offload manager stop failed", exc_info=True)

        # Flush KV prefix cache to SSD for persistence across restarts
        _prefix_cache = getattr(self.scheduler, '_prefix_cache', None)
        if _prefix_cache is not None and hasattr(_prefix_cache, 'flush_to_ssd'):
            try:
                _loop = asyncio.get_running_loop()
                blocks_flushed = await _loop.run_in_executor(
                    self._executor, _prefix_cache.flush_to_ssd,
                )
                if blocks_flushed > 0:
                    logger.info(f"KV prefix cache flushed {blocks_flushed} blocks to SSD")
            except Exception:
                logger.debug("KV prefix cache SSD flush failed", exc_info=True)

        # ── Wave 42: Shutdown wired modules ──
        if self._composition_scheduler is not None:
            try:
                self._composition_scheduler.shutdown()
            except Exception:
                logger.debug("composition scheduler shutdown failed", exc_info=True)

        # Stop external prefill server and its background task
        if self._prefill_server is not None:
            try:
                await self._prefill_server.stop()
            except Exception:
                logger.debug("prefill server stop failed", exc_info=True)
        if self._prefill_task is not None and not self._prefill_task.done():
            self._prefill_task.cancel()
            try:
                await self._prefill_task
            except asyncio.CancelledError:
                pass
            self._prefill_task = None

        # Stop KV transfer server
        if self._kv_transfer_server is not None:
            try:
                await self._kv_transfer_server.stop()
            except Exception:
                logger.debug("kv transfer server stop failed", exc_info=True)

        # Signal all active collectors with sentinel
        for collector in list(self._output_collectors.values()):
            try:
                collector.put(None)
            except Exception:
                logger.debug("collector sentinel put failed", exc_info=True)
        for event in list(self._finished_events.values()):
            event.set()

        # Clean up all running requests (inflight prefix sharing, etc.)
        for req_id in list(self._output_collectors.keys()):
            self._cleanup_request(req_id)
        self._output_collectors.clear()
        self._stream_states.clear()
        self._finished_events.clear()
        self._request_timestamps.clear()
        if hasattr(self, '_ttft_timestamps'):
            self._ttft_timestamps.clear()
        self._request_lora_adapters.clear()
        self._kv_prefix_hashes.clear()
        if self._request_dedup is not None:
            self._dedup_hashes.clear()
            self._dedup_shadows.clear()
        self._finalized_ids.clear()
        self._ttft_done.clear()

        # Release model/tokenizer refs + GC + cache clear on executor
        self._model = None
        self._tokenizer = None
        # Also clear scheduler's model refs to prevent GC leak on hot-reload
        if hasattr(self.scheduler, 'model'):
            self.scheduler.model = None
        if hasattr(self.scheduler, 'tokenizer'):
            self.scheduler.tokenizer = None
        gc.collect()
        loop = asyncio.get_running_loop()
        from .mlx_executor import sync_and_clear_cache
        try:
            await loop.run_in_executor(self._executor, sync_and_clear_cache)
        except Exception:
            logger.debug("cache clear on executor failed", exc_info=True)
        self._executor = None

        # Mark fully stopped AFTER all cleanup is done.
        # This must come last so concurrent add_request() calls during the
        # await yields above don't create collectors that we then clear.
        # add_request() checks _stopped to decide whether to reject.
        self._stopped = True

        logger.info("EngineCore stopped")

    async def add_request(
        self,
        prompt: str | list[int] | list[dict],
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
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        request_id: str | None = None,
        enable_thinking: bool | None = None,
        json_schema: dict | str | None = None,
        thinking_budget: int | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        lora_adapter: str | None = None,
        priority: int = 0,
        reasoning_effort: str | None = None,
        logits_processors: list | None = None,
        grammar: dict | str | None = None,
        **kwargs,
    ) -> str:
        """Add a generation request. Returns request_id for streaming/abort.

        If MemoryGuard preflight check fails, creates an error output
        immediately instead of adding to the scheduler.
        """
        from .request import Request, SamplingParams

        req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

        # Reject new requests when engine was started but has since stopped.
        # Note: we do NOT reject when engine was never started -- the batched
        # engine test suite calls add_request on an EngineCore without start(),
        # driving the scheduler loop manually. Only reject when the engine was
        # explicitly stopped (i.e., stop() was called after a successful start).
        # Use _stopped flag (set atomically at the END of stop()) instead of
        # _running to avoid a race: stop() sets _running=False early but clears
        # collectors later (with await yields in between). A concurrent
        # add_request during that window would create collectors that stop()
        # then clears, orphaning the consumer. _stopped is only set after all
        # cleanup is complete.
        if self._stopped:
            from .output_collector import RequestOutputCollector, RequestStreamState
            from .request import RequestOutput
            self._output_collectors[req_id] = RequestOutputCollector(aggregate=True)
            self._stream_states[req_id] = RequestStreamState(
                stream_interval=self.config.stream_interval
            )
            self._finished_events[req_id] = asyncio.Event()
            error_output = RequestOutput(
                request_id=req_id,
                finished=True,
                finish_reason="error",
                error="Engine is not running",
                prompt_tokens=0,
                completion_tokens=0,
            )
            self._output_collectors[req_id].put(error_output)
            self._output_collectors[req_id].put(None)
            self._finished_events[req_id].set()
            return req_id

        # Reject new requests during graceful shutdown (vLLM REQUESTED state)
        if self._shutdown_requested:
            from .output_collector import RequestOutputCollector, RequestStreamState
            from .request import RequestOutput
            self._output_collectors[req_id] = RequestOutputCollector(aggregate=True)
            self._stream_states[req_id] = RequestStreamState(
                stream_interval=self.config.stream_interval
            )
            self._finished_events[req_id] = asyncio.Event()
            error_output = RequestOutput(
                request_id=req_id,
                finished=True,
                finish_reason="error",
                error="Engine is shutting down, new requests rejected",
                prompt_tokens=0,
                completion_tokens=0,
            )
            self._output_collectors[req_id].put(error_output)
            self._output_collectors[req_id].put(None)
            self._finished_events[req_id].set()
            return req_id

        # Save original prompt before any transformation for message-level truncation
        _original_prompt = prompt

        # Apply per-request LoRA adapter (acquire ref before generation, release after)
        loaded_lora = None
        if lora_adapter:
            try:
                from .lora_manager import get_lora_manager
                lora_mgr = get_lora_manager(engine_id=getattr(self.scheduler, 'model_id', 'default') or 'default')
                if lora_mgr is not None:
                    loaded_lora = lora_mgr.acquire_adapter(lora_adapter)
                    if loaded_lora:
                        logger.debug(f"LoRA adapter '{lora_adapter}' acquired for request {req_id}")
            except Exception:
                logger.debug(f"LoRA adapter acquire failed: {lora_adapter}", exc_info=True)

        # Model preprocessor: detect and preprocess for model-specific input
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            has_multimodal = any(
                isinstance(c.get("content"), list)
                for c in prompt
                if isinstance(c.get("content"), list)
            )
            if has_multimodal and self._preprocessor_registry is not None:
                try:
                    model_config = {"model_type": getattr(self.scheduler, 'model_id', '') or ""}
                    if self._model is not None:
                        cfg = getattr(self._model, 'config', self._model)
                        if hasattr(cfg, 'model_type'):
                            model_config["model_type"] = cfg.model_type
                    preprocessor = self._preprocessor_registry.detect(model_config)
                    if preprocessor is not None:
                        processed = preprocessor.preprocess(prompt, self._tokenizer)
                        if processed.token_ids:
                            prompt = processed.token_ids
                except Exception:
                    logger.debug("model preprocessor failed, using raw prompt", exc_info=True)

        # Encode prompt
        if isinstance(prompt, str):
            token_ids = self._tokenizer.encode(prompt)
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._messages_to_text(prompt, enable_thinking)
            token_ids = self._tokenizer.encode(text)
        else:
            token_ids = list(prompt)

        # Guard against empty prompt: inject BOS/EOS so the scheduler
        # doesn't crash with a zero-length token sequence.
        if not token_ids:
            bos_id = getattr(self._tokenizer, 'bos_token_id', None)
            if bos_id is not None:
                token_ids = [bos_id]
            else:
                eos_id = getattr(self._tokenizer, 'eos_token_id', 1)
                token_ids = [eos_id]

        num_prompt_tokens = len(token_ids)

        # Inflight prefix sharing: register for concurrent KV prefix reuse (SGLang pattern).
        # Token IDs are registered here so concurrent requests can match the prefix.
        # KV cache ref starts as None — updated by scheduler after the first forward
        # pass produces a prompt_cache (see scheduler response processing loop).
        try:
            from .inflight_prefix_sharing import get_inflight_tracker
            _inflight_tracker = get_inflight_tracker()
            _inflight_tracker.register(
                req_id,
                token_ids,
                None,  # KV cache ref updated by scheduler after first response
                getattr(self.scheduler, 'model_id', '') or '',
            )
        except Exception:
            logger.debug("inflight prefix register failed", exc_info=True)

        # Multimodal pipeline: process images/audio/video via staged pipeline
        # (staged_pipeline.py MultimodalPipelineCoordinator)
        _mm_images = kwargs.get('images') or kwargs.get('image')
        _mm_audio = kwargs.get('audio')
        _mm_video = kwargs.get('video')
        if _mm_images or _mm_audio or _mm_video:
            try:
                from .staged_pipeline import PipelineRequest
                _mm_req = PipelineRequest(
                    request_id=req_id,
                    model_id=getattr(self.scheduler, 'model_id', ''),
                    text=prompt if isinstance(prompt, str) else None,
                    images=_mm_images if isinstance(_mm_images, list) else [_mm_images] if _mm_images else None,
                    audio=_mm_audio if isinstance(_mm_audio, list) else [_mm_audio] if _mm_audio else None,
                    video=_mm_video if isinstance(_mm_video, list) else [_mm_video] if _mm_video else None,
                )
                _mm_results = self._multimodal_pipeline.process(_mm_req)
                _mm_errors = [r for r in _mm_results if r.error is not None]
                if _mm_errors:
                    logger.warning(
                        f"Multimodal pipeline errors for {req_id}: "
                        f"{[r.error for r in _mm_errors]}"
                    )
            except Exception:
                logger.debug("multimodal pipeline processing failed", exc_info=True)

        # ── Context window truncation (prevent garbage output from overlength prompts) ──
        max_seq_len = self._get_max_seq_len()
        # Thinking tokens also consume context window positions.  When
        # enable_thinking is True the model will produce thinking tokens
        # *in addition to* the max_tokens completion tokens, so they must
        # be subtracted from the available prompt budget to avoid generating
        # garbage when prompt + thinking + completion > max_seq_len.
        _thinking_overhead = thinking_budget if (thinking_budget and enable_thinking) else 0
        # Total tokens that the generation will consume beyond the prompt.
        _generation_budget = max_tokens + _thinking_overhead
        if max_seq_len > 0 and num_prompt_tokens + _generation_budget > max_seq_len:
            excess = num_prompt_tokens + _generation_budget - max_seq_len
            if excess > 0 and num_prompt_tokens > excess:
                # Prefer message-level truncation when we still have the
                # original messages — it preserves system/developer messages
                # and keeps tool-call/response pairs intact.
                truncated_messages = None
                if isinstance(_original_prompt, list) and _original_prompt and isinstance(_original_prompt[0], dict):
                    try:
                        trunc_result = self._context_window_mgr.compute_truncation(
                            messages=_original_prompt,
                            max_tokens=max_seq_len - _generation_budget,
                            strategy="importance_aware",
                        )
                        truncated_messages = trunc_result.messages
                    except Exception:
                        logger.debug("message-level truncation failed, falling back to token-level", exc_info=True)

                if truncated_messages is not None:
                    # Re-encode the truncated messages
                    try:
                        text = self._messages_to_text(truncated_messages, enable_thinking)
                        token_ids = self._tokenizer.encode(text)
                        num_prompt_tokens = len(token_ids)
                        prompt = truncated_messages
                        logger.info(
                            f"Context window truncation (message-level): "
                            f"{num_prompt_tokens} tokens remaining "
                            f"(max_seq_len={max_seq_len})"
                        )
                    except Exception:
                        # Fallback to token-level truncation
                        logger.debug("re-encoding truncated messages failed, falling back to token-level", exc_info=True)
                        token_ids = token_ids[excess:]
                        num_prompt_tokens = len(token_ids)
                        try:
                            prompt = self._tokenizer.decode(token_ids)
                        except Exception:
                            prompt = token_ids
                        logger.info(
                            f"Context window truncation (token-level): {excess} tokens removed "
                            f"(max_seq_len={max_seq_len})"
                        )
                else:
                    # Token-level fallback for string prompts or when message-level fails
                    token_ids = token_ids[excess:]
                    num_prompt_tokens = len(token_ids)
                    try:
                        prompt = self._tokenizer.decode(token_ids)
                    except Exception:
                        prompt = token_ids
                    # Record truncation via ContextWindowManager
                    if self._context_window_mgr is not None:
                        self._context_window_mgr._stats.truncations_applied += 1
                        self._context_window_mgr._stats.total_tokens_saved += excess
                    logger.info(
                        f"Context window truncation (token-level): {excess} tokens removed "
                        f"(max_seq_len={max_seq_len})"
                    )
            elif excess > 0 and num_prompt_tokens <= excess:
                # Prompt alone exceeds the context window — truncation would
                # consume the entire prompt.  Return an error immediately so
                # the caller gets a clear message instead of silently running
                # with an overlength prompt that produces garbage output.
                from .output_collector import RequestOutputCollector, RequestStreamState
                from .request import RequestOutput
                self._output_collectors[req_id] = RequestOutputCollector(aggregate=True)
                self._stream_states[req_id] = RequestStreamState(
                    stream_interval=self.config.stream_interval
                )
                self._finished_events[req_id] = asyncio.Event()
                error_output = RequestOutput(
                    request_id=req_id,
                    finished=True,
                    finish_reason="error",
                    error=(
                        f"Prompt exceeds context window "
                        f"({num_prompt_tokens} prompt tokens + "
                        f"{_generation_budget} generation budget > {max_seq_len} max_seq_len)"
                    ),
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=0,
                )
                self._output_collectors[req_id].put(error_output)
                self._output_collectors[req_id].put(None)  # sentinel
                self._finished_events[req_id].set()
                # Clean up resources acquired before this point
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker
                    get_inflight_tracker().unregister(req_id)
                except Exception:
                    logger.debug(f"inflight unregister failed in context window rejection for {req_id}", exc_info=True)
                self._budget_manager.remove(req_id)
                if loaded_lora and lora_adapter:
                    try:
                        from .lora_manager import get_lora_manager
                        lora_mgr = get_lora_manager(engine_id=getattr(self.scheduler, 'model_id', 'default') or 'default')
                        if lora_mgr is not None:
                            lora_mgr.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(f"LoRA release failed in context window rejection for {req_id}", exc_info=True)
                self._fail_dedup_shadows(req_id, "Prompt exceeds context window", "error")
                return req_id

        # ── Wave 42: Budget check (token/time/cost/thinking) ──
        budget = self._budget_manager.register(
            request_id=req_id,
            max_tokens=max_tokens,
            prompt_tokens=num_prompt_tokens,
            thinking_budget=thinking_budget,
        )
        if budget.is_exhausted:
            budget_reason = budget.exhaustion_reason or "budget_exceeded"
            self._budget_manager.remove(req_id)
            # Bug 5 fix: release LoRA adapter on early return
            if loaded_lora and lora_adapter:
                try:
                    from .lora_manager import get_lora_manager
                    lora_mgr = get_lora_manager(engine_id=getattr(self.scheduler, 'model_id', 'default') or 'default')
                    if lora_mgr is not None:
                        lora_mgr.release_adapter(lora_adapter)
                except Exception:
                    logger.debug(f"LoRA release failed in budget rejection for {req_id}", exc_info=True)
            try:
                from .inflight_prefix_sharing import get_inflight_tracker
                get_inflight_tracker().unregister(req_id)
            except Exception:
                logger.debug(f"inflight unregister failed in budget rejection for {req_id}", exc_info=True)
            from .output_collector import RequestOutputCollector, RequestStreamState
            from .request import RequestOutput
            self._output_collectors[req_id] = RequestOutputCollector(aggregate=True)
            self._stream_states[req_id] = RequestStreamState(
                stream_interval=self.config.stream_interval
            )
            self._finished_events[req_id] = asyncio.Event()
            error_output = RequestOutput(
                request_id=req_id,
                finished=True,
                finish_reason=budget_reason,
                error=f"Budget exceeded: {budget_reason}",
                prompt_tokens=num_prompt_tokens,
                completion_tokens=0,
            )
            self._output_collectors[req_id].put(error_output)
            self._output_collectors[req_id].put(None)
            self._finished_events[req_id].set()
            self._fail_dedup_shadows(req_id, f"Budget exceeded: {budget_reason}", budget_reason)
            return req_id

        # ── Wave 42: Request dedup ──
        if self._request_dedup is not None:
            from .request_dedup import RequestDeduplicator
            content_hash = RequestDeduplicator.compute_hash(
                model=str(getattr(self.scheduler, 'model_id', '') or ''),
                prompt=prompt, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p,
                top_k=top_k, min_p=min_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop,
                stop_token_ids=stop_token_ids,
                json_schema=str(json_schema) if json_schema else None,
                thinking_budget=thinking_budget,
                reasoning_effort=reasoning_effort,
            )
            dedup_result = self._request_dedup.check(req_id, content_hash)
            if dedup_result is not None:
                # Shadow request: don't add to scheduler, wait for primary's output
                primary_id = dedup_result.primary_request_id
                logger.debug(
                    f"Request {req_id} dedup hit, shadowing {primary_id}"
                )
                self._dedup_shadows[req_id] = primary_id
                self._dedup_hashes[req_id] = content_hash
                # Clean up registrations that won't be used by shadow path
                self._budget_manager.remove(req_id)
                # Bug 5 fix: release LoRA adapter on early return
                if loaded_lora and lora_adapter:
                    try:
                        from .lora_manager import get_lora_manager
                        lora_mgr = get_lora_manager(engine_id=getattr(self.scheduler, 'model_id', 'default') or 'default')
                        if lora_mgr is not None:
                            lora_mgr.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(f"LoRA release failed in dedup shadow for {req_id}", exc_info=True)
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker
                    get_inflight_tracker().unregister(req_id)
                except Exception:
                    logger.debug(f"inflight unregister failed in dedup shadow for {req_id}", exc_info=True)
                # Create output collector + finished event so the caller can await
                from .output_collector import RequestOutputCollector, RequestStreamState
                shadow_collector = RequestOutputCollector(aggregate=True)
                self._output_collectors[req_id] = shadow_collector
                self._stream_states[req_id] = RequestStreamState(
                    stream_interval=self.config.stream_interval
                )
                self._finished_events[req_id] = asyncio.Event()
                self._request_timestamps[req_id] = time.monotonic()

                # Forward primary's accumulated output so the shadow doesn't
                # miss tokens produced before the shadow registered.  Peek at
                # the primary's collector buffered output (not consumed yet by
                # its own consumer).  This may be partial — the primary could
                # have already had tokens consumed by its stream — but it's the
                # best we can do without maintaining a separate replay buffer.
                primary_collector = self._output_collectors.get(primary_id)
                if primary_collector is not None and primary_collector.output is not None:
                    from .request import RequestOutput as _RO
                    primary_out = primary_collector.output
                    shadow_collector.put(_RO(
                        request_id=req_id,
                        new_token_ids=list(primary_out.new_token_ids),
                        new_text=primary_out.new_text,
                        output_text=primary_out.output_text,
                        output_token_ids=list(primary_out.output_token_ids) if primary_out.output_token_ids else [],
                        completion_tokens=primary_out.completion_tokens,
                        finished=False,
                        prompt_tokens=primary_out.prompt_tokens,
                        logprobs=primary_out.logprobs,
                        current_state=primary_out.current_state,
                        reasoning_tokens=primary_out.reasoning_tokens,
                        cached_tokens=primary_out.cached_tokens,
                        error=None,
                        ttft_ms=0.0,  # TTFT is for primary, not shadow
                    ))

                return req_id
            else:
                dedup_entry = self._request_dedup.register(req_id, content_hash)
            self._dedup_hashes[req_id] = dedup_entry.content_hash

        # ── Wave 42: Lifecycle tracking ──
        self._lifecycle_orchestrator.on_request_added(req_id)

        # ── Wave 46: KV lifecycle admission ──
        _block_id = hash(req_id) % (10**9)
        try:
            # Use model-aware estimate: per-token KV = num_layers * num_kv_heads * head_dim * dtype_size * 2 (K+V).
            # Fall back to the memory_aware_scheduler's formula when available, which
            # accounts for the actual model architecture.  The old hardcoded 2048
            # bytes/token was ~64x too small for typical 32-layer models, causing
            # kv_lifecycle to wildly under-count and migration tier decisions to be
            # based on incorrect size information.
            estimated_kv_bytes = self._memory_aware_scheduler.estimate_kv_memory(num_prompt_tokens)
            if estimated_kv_bytes <= 0:
                # Fallback: rough estimate if memory-aware scheduler has no model info
                estimated_kv_bytes = num_prompt_tokens * 2048
            self._kv_lifecycle.admit(
                block_id=_block_id,
                size_bytes=estimated_kv_bytes,
                prefix_hash="",
            )
            # Register block in migration manager for temperature-based tier management
            from .kv_migration import KVTier
            self._kv_migration.register_block(_block_id, tier=KVTier.HOT, byte_size=estimated_kv_bytes)
        except Exception:
            logger.debug("kv_lifecycle admit failed", exc_info=True)

        # Sliding window KV registration for windowed attention models
        if self._sliding_window_mgr is not None:
            try:
                self._sliding_window_mgr.register_request(req_id)
            except Exception:
                logger.debug("sliding window registration failed", exc_info=True)

        # ── Wave 43: Memory-aware admission control ──
        try:
            estimated_bytes = self._memory_aware_scheduler.estimate_kv_memory(
                num_prompt_tokens, max_tokens=max_tokens,
            )
            if isinstance(estimated_bytes, int) and estimated_bytes > 0:
                admitted = self._memory_aware_scheduler.reserve_memory(
                    request_id=req_id,
                    num_bytes=estimated_bytes,
                    num_tokens=num_prompt_tokens,
                )
                if not admitted:
                    logger.warning(f"Memory-aware scheduler rejected request {req_id}: estimated {estimated_bytes} bytes")
                    # Release all resources acquired before this point
                    try:
                        from .inflight_prefix_sharing import get_inflight_tracker
                        get_inflight_tracker().unregister(req_id)
                    except Exception:
                        logger.debug(f"inflight unregister failed in mem-aware rejection for {req_id}", exc_info=True)
                    self._budget_manager.remove(req_id)
                    if loaded_lora and lora_adapter:
                        try:
                            from .lora_manager import get_lora_manager
                            lora_mgr = get_lora_manager(engine_id=getattr(self.scheduler, 'model_id', 'default') or 'default')
                            if lora_mgr is not None:
                                lora_mgr.release_adapter(lora_adapter)
                        except Exception:
                            logger.debug(f"LoRA release failed in mem-aware rejection for {req_id}", exc_info=True)
                    try:
                        self._lifecycle_orchestrator.on_request_failed(
                            req_id, error="Memory-aware scheduler: insufficient memory", retryable=True,
                        )
                    except Exception:
                        logger.debug(f"lifecycle cleanup failed in mem-aware rejection for {req_id}", exc_info=True)
                    try:
                        self._kv_lifecycle.release(_block_id)
                    except Exception:
                        logger.debug(f"kv_lifecycle release failed in mem-aware rejection for {req_id}", exc_info=True)
                    try:
                        self._kv_migration.unregister_block(_block_id)
                    except Exception:
                        logger.debug(f"kv_migration unregister failed in mem-aware rejection for {req_id}", exc_info=True)
                    if self._sliding_window_mgr is not None:
                        try:
                            self._sliding_window_mgr.remove_request(req_id)
                        except Exception:
                            logger.debug(f"sliding window cleanup failed in mem-aware rejection for {req_id}", exc_info=True)
                    # Set up output collector with error response
                    from .output_collector import RequestOutputCollector, RequestStreamState
                    from .request import RequestOutput
                    self._output_collectors[req_id] = RequestOutputCollector(aggregate=True)
                    self._stream_states[req_id] = RequestStreamState(
                        stream_interval=self.config.stream_interval
                    )
                    self._finished_events[req_id] = asyncio.Event()

                    error_output = RequestOutput(
                        request_id=req_id,
                        finished=True,
                        finish_reason="memory_exceeded",
                        error=f"Memory-aware scheduler rejected: insufficient memory (estimated {estimated_bytes} bytes)",
                        prompt_tokens=num_prompt_tokens,
                        completion_tokens=0,
                    )
                    self._output_collectors[req_id].put(error_output)
                    self._output_collectors[req_id].put(None)  # sentinel
                    self._finished_events[req_id].set()
                    # Fail any dedup shadows waiting for this primary
                    self._fail_dedup_shadows(req_id, "Memory-aware scheduler: insufficient memory", "memory_exceeded")
                    return req_id
        except Exception:
            logger.debug("memory-aware admission check skipped", exc_info=True)

        # Memory guard preflight check — reject before adding to scheduler
        if self._memory_guard is not None:
            ok, reason = self._memory_guard.preflight_check(
                num_prompt_tokens=num_prompt_tokens,
                max_tokens=max_tokens,
            )
            if not ok:
                # Clean up all registrations before early return
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker
                    get_inflight_tracker().unregister(req_id)
                except Exception:
                    logger.debug(f"inflight unregister failed in memguard rejection for {req_id}", exc_info=True)
                self._memory_aware_scheduler.release_memory(req_id)
                self._budget_manager.remove(req_id)
                # Bug 5 fix: release LoRA adapter on early return
                if loaded_lora and lora_adapter:
                    try:
                        from .lora_manager import get_lora_manager
                        lora_mgr = get_lora_manager(engine_id=getattr(self.scheduler, 'model_id', 'default') or 'default')
                        if lora_mgr is not None:
                            lora_mgr.release_adapter(lora_adapter)
                    except Exception:
                        logger.debug(f"LoRA release failed in memguard rejection for {req_id}", exc_info=True)
                try:
                    self._lifecycle_orchestrator.on_request_failed(
                        req_id, error=f"Memory guard rejected: {reason}", retryable=False,
                    )
                except Exception:
                    logger.debug(f"lifecycle cleanup failed in memguard rejection for {req_id}", exc_info=True)
                try:
                    self._kv_lifecycle.release(_block_id)
                except Exception:
                    logger.debug(f"kv_lifecycle release failed in memguard rejection for {req_id}", exc_info=True)
                try:
                    self._kv_migration.unregister_block(_block_id)
                except Exception:
                    logger.debug(f"kv_migration unregister failed in memguard rejection for {req_id}", exc_info=True)
                if self._sliding_window_mgr is not None:
                    try:
                        self._sliding_window_mgr.remove_request(req_id)
                    except Exception:
                        logger.debug(f"sliding window cleanup failed in memguard rejection for {req_id}", exc_info=True)
                # Set up output collector with error response
                from .output_collector import RequestOutputCollector, RequestStreamState
                from .request import RequestOutput
                self._output_collectors[req_id] = RequestOutputCollector(aggregate=True)
                self._stream_states[req_id] = RequestStreamState(
                    stream_interval=self.config.stream_interval
                )
                self._finished_events[req_id] = asyncio.Event()

                error_output = RequestOutput(
                    request_id=req_id,
                    finished=True,
                    finish_reason="memory_exceeded",
                    error=f"Memory guard rejected: {reason}",
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=0,
                )
                self._output_collectors[req_id].put(error_output)
                self._output_collectors[req_id].put(None)  # sentinel
                self._finished_events[req_id].set()
                # Fail any dedup shadows waiting for this primary
                self._fail_dedup_shadows(req_id, f"Memory guard rejected: {reason}", "memory_exceeded")
                return req_id

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            stop=stop or [],
            stop_token_ids=stop_token_ids or [],
            seed=seed,
            json_schema=json_schema,
            grammar=grammar,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            priority=priority,
            xtc_probability=kwargs.get('xtc_probability', 0.0),
            xtc_threshold=kwargs.get('xtc_threshold', 0.0),
            reasoning_effort=reasoning_effort,
            logits_processors=logits_processors,
        )

        request = Request(
            request_id=req_id,
            prompt=prompt if isinstance(prompt, str) else token_ids,
            sampling_params=sampling_params,
            prompt_token_ids=token_ids,
            num_prompt_tokens=num_prompt_tokens,
            enable_thinking=enable_thinking,
            images=_mm_images if _mm_images else None,
        )

        # Inflight prefix sharing (SGLang pattern): check for in-flight
        # prefills with matching prefix to share partial KV blocks.
        # This complements the scheduler's own inflight lookup (which runs
        # inside _schedule_waiting) by catching matches at enqueue time.
        # Both checks are needed: this one catches matches between add_request
        # calls, while the scheduler's catches matches at scheduling time.
        if token_ids:
            try:
                from .inflight_prefix_sharing import get_inflight_tracker
                _tracker = get_inflight_tracker()
                _inflight_match = _tracker.find_prefix(
                    token_ids,
                    getattr(self.scheduler, 'model_id', '') or '',
                )
                if _inflight_match is not None and _inflight_match.kv_cache_ref is not None:
                    request.prompt_cache = _inflight_match.kv_cache_ref
                    shared_len = min(len(_inflight_match.token_ids), len(token_ids))
                    request.cached_tokens = shared_len
                    request.remaining_tokens = token_ids[shared_len:]
                    logger.debug(
                        "inflight prefix reuse at enqueue: %d tokens from req=%s for %s",
                        shared_len,
                        _inflight_match.request_id[:12],
                        req_id[:12],
                    )
            except Exception:
                logger.debug("inflight prefix lookup at enqueue failed", exc_info=True)

        # Set up per-request output management (oMLX EngineCore pattern)
        from .output_collector import RequestOutputCollector, RequestStreamState
        self._output_collectors[req_id] = RequestOutputCollector(aggregate=True)
        self._stream_states[req_id] = RequestStreamState(
            stream_interval=self.config.stream_interval
        )
        self._finished_events[req_id] = asyncio.Event()
        self._request_timestamps[req_id] = time.monotonic()

        # Compute and store KV prefix hash for cache-locality reordering
        prefix_hash = self._compute_prefix_hash_for_request(req_id, token_ids)
        if prefix_hash is not None:
            self._kv_prefix_hashes[req_id] = prefix_hash
            # Also set on scheduler so _schedule_waiting can reorder by prefix
            self.scheduler.set_kv_prefix_hash(req_id, prefix_hash)

        if loaded_lora and lora_adapter:
            self._request_lora_adapters[req_id] = lora_adapter

        # Add to scheduler on MLX executor (thread-safe)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor, self.scheduler.add_request, request
        )

        # Check if the scheduler rejected the request (queue full, etc.)
        # If rejected, generate error output immediately so resources are
        # cleaned up via the normal finalize path.
        # NOTE: do NOT call _cleanup_request() here — the consumer (generate()
        # or stream_outputs()) must drain the collector before cleanup.
        # _finalize_request() is called to release scheduler-side resources,
        # but consumer-side state (collector, event) must remain until the
        # consumer reads them.
        from .request import RequestStatus
        if RequestStatus.is_finished(request.status):
            error_reason = request.finish_reason or "rejected"
            from .request import RequestOutput
            collector = self._output_collectors.get(req_id)
            if collector is not None:
                collector.put(RequestOutput(
                    request_id=req_id,
                    finished=True,
                    finish_reason=error_reason,
                    error=f"Request rejected by scheduler: {error_reason}",
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=0,
                ))
                collector.put(None)  # sentinel
            self._signal_finished(req_id)
            self._finalize_request(req_id)
            self._fail_dedup_shadows(req_id, f"Scheduler rejected: {error_reason}", "error")
            return req_id

        # Wake engine loop from idle sleep (event-driven scheduling)
        if self._wake_event is not None:
            self._wake_event.set()

        return req_id

    async def abort_request(self, request_id: str) -> None:
        """Deferred abort (oMLX pattern: enqueued, processed at next step).

        Also fails any dedup shadow requests so their consumers don't hang
        waiting for output from a primary that will never produce more tokens.
        """
        from .request import RequestOutput
        # Look up request to preserve token counts in abort output
        req = self.scheduler.requests.get(request_id)
        prompt_tok = req.num_prompt_tokens if req else 0
        completion_tok = req.num_output_tokens if req else 0
        self.scheduler.abort_request(request_id)
        # Put error output to wake up any waiting consumer
        collector = self._output_collectors.get(request_id)
        if collector is not None:
            collector.put(RequestOutput(
                request_id=request_id,
                finished=True,
                finish_reason="abort",
                error="Request aborted",
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
            ))
            collector.put(None)  # sentinel

        # Bug fix: fail dedup shadows so their consumers don't hang forever.
        # Without this, shadow requests whose primary is aborted would never
        # receive a sentinel or finished event, causing generate()/stream_outputs()
        # to hang until the engine-wide timeout enforcement rescues them.
        if self._request_dedup is not None:
            shadow_ids = [
                sid for sid, pid in list(self._dedup_shadows.items())
                if pid == request_id
            ]
            for sid in shadow_ids:
                s_collector = self._output_collectors.get(sid)
                if s_collector is not None:
                    s_collector.put(RequestOutput(
                        request_id=sid,
                        finished=True,
                        finish_reason="abort",
                        error=f"Primary request {request_id} was aborted",
                    ))
                    s_collector.put(None)
                self._signal_finished(sid)
                self._cleanup_request(sid)

        self._signal_finished(request_id)
        self._cleanup_request(request_id)

    async def abort_all_requests(self) -> None:
        """Abort all active requests (error recovery).

        Also fails dedup shadow requests that are not tracked by the scheduler
        but have collectors waiting for output from a primary.
        """
        from .request import RequestOutput
        failed_ids = self.scheduler.fail_all_requests()
        for req_id in failed_ids:
            collector = self._output_collectors.get(req_id)
            if collector is not None:
                collector.put(RequestOutput(
                    request_id=req_id,
                    finished=True,
                    finish_reason="abort",
                    error="All requests aborted",
                ))
                collector.put(None)  # sentinel
            self._signal_finished(req_id)
            self._cleanup_request(req_id)

        # Bug fix: also fail dedup shadow requests not tracked by the scheduler.
        # fail_all_requests() only returns scheduler-tracked IDs, so shadow
        # requests (short-circuited in add_request, never added to scheduler)
        # are missed. Without this, shadow consumers hang forever.
        if self._request_dedup is not None:
            for sid in list(self._dedup_shadows.keys()):
                if sid not in failed_ids:
                    s_collector = self._output_collectors.get(sid)
                    if s_collector is not None:
                        s_collector.put(RequestOutput(
                            request_id=sid,
                            finished=True,
                            finish_reason="abort",
                            error="All requests aborted (dedup shadow)",
                        ))
                        s_collector.put(None)
                    self._signal_finished(sid)
                    self._cleanup_request(sid)

    async def stream_outputs(
        self,
        request_id: str,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[Any]:
        """Stream outputs for a request (oMLX/vLLM pattern).

        Fast path: collector.get_nowait() avoids task switch under load.
        Slow path: await collector.get() for efficient waiting when idle.
        Sentinel (None) signals stream end.

        When cancel_event is provided (from gateway disconnect detection),
        the stream monitors it while waiting for new output. If cancelled,
        abort_request() is called to propagate cancellation down to the
        scheduler (removing from running/waiting queues, freeing KV blocks).
        This prevents GPU slots and KV blocks from being consumed by requests
        whose clients have already disconnected (SGLang pattern).
        """
        collector = self._output_collectors.get(request_id)
        if collector is None:
            return

        _cancelled = False
        _cleaned_up = False
        try:
            while True:
                # Check cancel_event at the top of each iteration.
                # This catches cancellation during chunked prefill when
                # no outputs have been produced yet.
                if cancel_event is not None:
                    cev = cancel_event
                    if isinstance(cev, asyncio.Event):
                        _cancelled = cev.is_set()
                    else:
                        _cancelled = cev.is_set()
                    if _cancelled:
                        break

                output = collector.get_nowait()
                if output is not None:
                    yield output
                    if output.finished:
                        break
                    continue

                # Wait for new output from engine loop, but also watch
                # cancel_event so we don't block indefinitely when the
                # client disconnects during chunked prefill.
                # Do NOT check collector._sentinel here — there is a race
                # between get_nowait() returning None and the sentinel check
                # where the engine could have put new output.  The await
                # collector.get() handles both sentinel and new output correctly.
                if cancel_event is not None:
                    # Race collector.get() against cancel_event
                    _get_task = asyncio.ensure_future(collector.get())
                    _cancel_waiter = asyncio.ensure_future(
                        cancel_event.wait() if isinstance(cancel_event, asyncio.Event)
                        else asyncio.sleep(0.05)  # polling for non-Event types
                    )
                    try:
                        done, pending = await asyncio.wait(
                            {_get_task, _cancel_waiter},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for p in pending:
                            p.cancel()
                            try:
                                await p
                            except (asyncio.CancelledError, Exception):
                                pass
                        if _cancel_waiter in done:
                            if isinstance(cancel_event, asyncio.Event):
                                _cancelled = cancel_event._value
                            else:
                                _cancelled = cancel_event.is_set()
                            if _cancelled:
                                break
                            # Poll interval expired but not cancelled — retry
                            continue
                        if _get_task in done:
                            output = _get_task.result()
                    except asyncio.CancelledError:
                        for t in (_get_task, _cancel_waiter):
                            if not t.done():
                                t.cancel()
                        raise
                else:
                    output = await collector.get()

                if output is None:
                    break
                yield output
                if output.finished:
                    break
        except asyncio.CancelledError:
            raise
        finally:
            if _cancelled:
                # Propagate cancellation to scheduler (SGLang pattern).
                # This removes the request from running/waiting queues and
                # frees KV blocks. Safe to call even if already finalized.
                # abort_request calls _cleanup_request internally, so mark
                # _cleaned_up to avoid double cleanup below.
                _cleaned_up = True
                try:
                    await self.abort_request(request_id)
                except Exception:
                    logger.debug("cancel-driven abort failed", exc_info=True)
            if not _cleaned_up:
                self._cleanup_request(request_id)

    async def generate(
        self,
        **kwargs,
    ) -> Any:
        """Non-streaming generate: add request, wait for completion, return result."""
        from .request import RequestOutput

        # Extract cancel_event before passing kwargs to add_request — it is
        # not a scheduler parameter but we need it to detect early cancellation
        # during event.wait().
        _cancel_event = kwargs.pop('cancel_event', None)

        req_id = await self.add_request(**kwargs)
        _cleaned_up = False

        try:
            # Store local reference to collector BEFORE event.wait() to avoid
            # TOCTOU race with abort_request()/_cleanup_request() which remove
            # the collector from _output_collectors.
            collector = self._output_collectors.get(req_id)

            # Wait for completion with timeout protection
            event = self._finished_events.get(req_id)
            if event:
                timeout_s = kwargs.get('timeout_seconds')
                if timeout_s is None:
                    timeout_s = self.config.request_timeout_seconds
                try:
                    # Race completion event against cancel_event so that
                    # gateway disconnects don't block until timeout.
                    if _cancel_event is not None:
                        _wait_task = asyncio.ensure_future(event.wait())
                        if isinstance(_cancel_event, asyncio.Event):
                            _cancel_waiter = asyncio.ensure_future(_cancel_event.wait())
                        else:
                            # Use short polling interval (matching stream_outputs)
                            # instead of full timeout — otherwise cancel is never
                            # detected and the request always runs to timeout.
                            _cancel_waiter = asyncio.ensure_future(asyncio.sleep(0.05))
                        try:
                            done, pending = await asyncio.wait(
                                {_wait_task, _cancel_waiter},
                                timeout=timeout_s,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            for p in pending:
                                p.cancel()
                                try:
                                    await p
                                except (asyncio.CancelledError, Exception):
                                    pass
                            if _cancel_waiter in done:
                                # Polling sleep completed — re-check if cancel
                                # is actually set before aborting (same pattern as
                                # stream_outputs).
                                if isinstance(_cancel_event, asyncio.Event):
                                    _cancelled = _cancel_event.is_set()
                                else:
                                    _cancelled = _cancel_event.is_set()
                                if _cancelled:
                                    await self.abort_request(req_id)
                                    _cleaned_up = True
                                    return RequestOutput(
                                        request_id=req_id,
                                        finished=True,
                                        finish_reason="stop",
                                        error="Request cancelled",
                                    )
                        except asyncio.CancelledError:
                            for t in (_wait_task, _cancel_waiter):
                                if not t.done():
                                    t.cancel()
                            raise
                    else:
                        await asyncio.wait_for(event.wait(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    logger.warning(f"generate() timeout ({timeout_s}s) for {req_id}")
                    # Abort the timed-out request so scheduler releases its slot.
                    # abort_request() calls _cleanup_request() internally, so
                    # mark _cleaned_up to prevent double cleanup in finally.
                    _cleaned_up = True
                    await self.abort_request(req_id)
                    return RequestOutput(
                        request_id=req_id,
                        finished=True,
                        finish_reason="timeout",
                        error=f"Generation timed out after {timeout_s}s",
                    )

            # Compute TTFT from tracked first-token timestamp (set by engine loop)
            # not total wall time (which would include full generation).
            _start_ts = self._request_timestamps.get(req_id)
            _ttft_ts = self._ttft_timestamps.get(req_id) if hasattr(self, '_ttft_timestamps') else None
            _ttft_ms = 0.0
            if _start_ts is not None and _ttft_ts is not None:
                _ttft_ms = round((_ttft_ts - _start_ts) * 1000, 1)

            # Drain collector (use local reference captured before event.wait())
            result = None
            if collector:
                while True:
                    output = collector.get_nowait()
                    if output is None:
                        break
                    if output.finished:
                        result = output
                    elif result is None:
                        result = output
                    else:
                        result = collector._merge(result, output)

            # Attach TTFT to result so downstream consumers can use it
            if result is not None:
                result.ttft_ms = _ttft_ms

            return result
        finally:
            if not _cleaned_up:
                self._cleanup_request(req_id)

    # ── Engine Loop ──

    async def _engine_loop(self) -> None:
        """Main engine loop — drives continuous batching (oMLX _engine_loop pattern).

        Uses event-driven wake-up: when idle (no requests), waits on _wake_event
        instead of polling at step_interval. This eliminates CPU waste during idle
        periods. New requests signal _wake_event from add_request().
        """
        loop = asyncio.get_running_loop()
        use_simple_streaming = self.config.stream_interval == 1

        while self._running:
            # Graceful shutdown: if requested, exit loop once all requests finish
            if self._shutdown_requested and not self.scheduler.has_requests():
                logger.info("Graceful shutdown: all in-flight requests completed")
                break

            if not self.scheduler.has_requests():
                # Event-driven idle: wait for wake signal instead of polling
                _idle_start = time.monotonic()
                if self._wake_event is not None:
                    try:
                        await asyncio.wait_for(
                            self._wake_event.wait(),
                            timeout=self.config.step_interval * 100,  # 100ms fallback
                        )
                        self._wake_event.clear()
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(self.config.step_interval)
                self._total_idle_time_ms += (time.monotonic() - _idle_start) * 1000
                continue

            _step_start = time.monotonic()
            scheduler_output = None  # Bug 1 fix: initialize before try block
            # Capture queue depth BEFORE scheduler step — the step may promote
            # waiting requests to running, so reading after underestimates
            # queue pressure reported to Prometheus.
            try:
                _pre_step_queue_depth = len(self.scheduler.waiting)
            except Exception:
                _pre_step_queue_depth = 0

            try:
                # Cache hardware info once per step (avoid 3+ repeated syscalls per step)
                _hw_info = None
                _total_mem_bytes = 0

                # Wave 43: CompositionScheduler pre_step hooks (metrics, memory pressure)
                if self._composition_scheduler is not None:
                    try:
                        self._composition_scheduler.pre_step(self.scheduler)
                        # Apply memory pressure recommendations
                        from .scheduler_mixins import MemoryPressureMixin
                        for m in self._composition_scheduler._mixins:
                            if isinstance(m, MemoryPressureMixin):
                                if m.is_admission_paused:
                                    logger.debug("MemoryPressureMixin: admission paused")
                                rec_batch = m.recommended_batch_size
                                if 0 < rec_batch < self.config.completion_batch_size:
                                    self.config.completion_batch_size = max(4, rec_batch)
                                break
                    except Exception:
                        logger.debug("composition pre_step failed", exc_info=True)

                # Priority inversion guard: detect and resolve priority inversion
                # before scheduling (token_scheduler.py PriorityInversionGuard)
                try:
                    from .token_scheduler import SchedulableRequest as _SReq
                    _running = [
                        _SReq(request_id=rid, priority=r.sampling_params.priority if r.sampling_params else 0,
                              wait_time=0.0, context_length=r.num_prompt_tokens,
                              output_length=len(r.output_token_ids) if r.output_token_ids else 0)
                        for rid, r in self.scheduler.running.items()
                    ]
                    _waiting = [
                        _SReq(request_id=r.request_id, priority=r.sampling_params.priority if r.sampling_params else 0,
                              wait_time=time.monotonic() - r._submit_time if r._submit_time > 0 else 0.0,
                              context_length=r.num_prompt_tokens)
                        for r in self.scheduler.waiting
                    ] if hasattr(self.scheduler.waiting, '__iter__') else []
                    inversions = self._priority_guard.check_inversion(_running, _waiting)
                    for inv in inversions:
                        low_req = next((r for r in _running if r.request_id == inv.low_request_id), None)
                        high_req = next((r for r in _waiting if r.request_id == inv.high_request_id), None)
                        if low_req and high_req:
                            self._priority_guard.apply_inheritance(low_req, high_req)
                            # Boost effective priority on the actual running request.
                            # Save the original priority so it can be restored when
                            # the boost expires (PriorityInversionGuard._expire_boosts).
                            actual = self.scheduler.running.get(inv.low_request_id)
                            if actual and actual.sampling_params:
                                if not hasattr(actual.sampling_params, '_original_priority'):
                                    actual.sampling_params._original_priority = actual.sampling_params.priority
                                actual.sampling_params.priority = low_req.effective_priority
                            logger.debug(f"Priority inheritance: {inv.low_request_id} boosted to {low_req.effective_priority}")
                except Exception:
                    logger.debug("priority inversion guard failed", exc_info=True)

                # Restore expired priority boosts on actual running requests.
                # PriorityInversionGuard._expire_boosts removes the boost record
                # but doesn't restore the original priority on sampling_params.
                try:
                    for rid in list(self.scheduler.running.keys()):
                        req = self.scheduler.running[rid]
                        if req and hasattr(req, 'sampling_params') and hasattr(req.sampling_params, '_original_priority'):
                            boost = self._priority_guard.get_boost(rid)
                            if boost is None:
                                # Boost expired or was cleared — restore original priority
                                req.sampling_params.priority = req.sampling_params._original_priority
                                del req.sampling_params._original_priority
                                logger.debug(f"Priority boost restored for {rid}")
                except Exception:
                    logger.debug("priority boost restoration failed", exc_info=True)

                # Token-level scheduling: convert running requests to schedulable form
                try:
                    from .token_scheduler import SchedulableRequest as _SReq
                    _sched_requests = [
                        _SReq(
                            request_id=rid,
                            priority=r.sampling_params.priority if r.sampling_params else 0,
                            wait_time=0.0,
                            context_length=r.num_prompt_tokens,
                            output_length=len(r.output_token_ids) if r.output_token_ids else 0,
                            is_prefilling=r.status.name == 'PREFILLING',
                        )
                        for rid, r in self.scheduler.running.items()
                    ]
                    _budget = self.config.completion_batch_size * 64
                    allocations = self._token_scheduler.compute_token_budget(
                        _sched_requests, _budget,
                    )
                    self._token_scheduler._stats["steps_with_allocations"] += 1
                    # Note: fairness tracker records are based on actual output tokens
                    # (recorded below in profiler section), not budget allocations.
                except Exception:
                    logger.debug("token-level scheduling failed", exc_info=True)

                # Run scheduler step on MLX executor thread
                # §14.1: TBO takes priority when enabled; else C18 overlap; else plain
                # _gpu_step_start isolates GPU kernel time from pre-step hooks
                # (composition scheduler, priority inversion guard, token scheduling)
                # so _last_step_wall_ms and ITL/TTFT estimates are accurate.
                _gpu_step_start = time.monotonic()
                if self._tbo_scheduler.config.enabled:
                    scheduler_output = await loop.run_in_executor(
                        self._executor, self._tbo_step,
                    )
                elif self._overlap_scheduler.config.enabled:
                    scheduler_output = await loop.run_in_executor(
                        self._executor, self._overlap_step,
                    )
                else:
                    scheduler_output = await loop.run_in_executor(
                        self._executor, self.scheduler.step
                    )
                _gpu_step_ms = (time.monotonic() - _gpu_step_start) * 1000

                # Wave 43: CompositionScheduler post_step hooks
                if self._composition_scheduler is not None and hasattr(scheduler_output, 'outputs') and scheduler_output.outputs:
                    try:
                        self._composition_scheduler.post_step(self.scheduler, scheduler_output)
                    except Exception:
                        logger.debug("composition post_step failed", exc_info=True)

                # AdaptiveBatchSizer: adjust batch size using ACTUAL GPU step wall time
                # (not total step time including pre-step hooks)
                _step_wall_ms = _gpu_step_ms
                try:
                    # Use pre-step queue depth (captured before scheduler.step()
                    # promoted waiting requests to running). Post-step queue depth
                    # underestimates pressure and causes unnecessary batch size
                    # reductions.
                    queue_depth = _pre_step_queue_depth
                    if _hw_info is None:
                        from .utils.hardware import get_hardware_info as _ghw
                        _hw_info = _ghw()
                        _total_mem_bytes = _hw_info.total_memory_bytes
                    import mlx.core as _mx
                    _mem_avail = 1.0 - (_mx.get_active_memory() / max(_total_mem_bytes, 1))
                    suggested = self._adaptive_batch_sizer.compute_optimal_batch(
                        queue_depth=queue_depth,
                        memory_available=_mem_avail,
                        slo_latency_ms=200.0,
                        current_latency_ms=_step_wall_ms,
                    )
                    if suggested < self.config.completion_batch_size:
                        self.config.completion_batch_size = suggested
                except Exception:
                    logger.debug("adaptive batch sizing failed", exc_info=True)

                # MON-2/4/5: Track monitoring gauges for Prometheus export
                try:
                    # _step_wall_ms measures GPU-bound scheduler.step() time only
                    # (set from _gpu_step_ms above). Total active time including
                    # output distribution + profiler overhead is accumulated
                    # separately from _step_start at the end of the loop.
                    self._last_step_wall_ms = _step_wall_ms
                    self._last_batch_size = len(scheduler_output.outputs) if hasattr(scheduler_output, 'outputs') else 0
                    self._last_queue_depth = _pre_step_queue_depth
                    # Record batch size in ServerMetrics for histogram distribution
                    try:
                        from .server_metrics import get_server_metrics
                        get_server_metrics().record_batch_size(self._last_batch_size)
                    except Exception:
                        pass
                except Exception:
                    pass
            except asyncio.CancelledError:
                self._total_step_time_ms += (time.monotonic() - _step_start) * 1000
                logger.info("Engine loop cancelled, failing all in-flight requests")
                self._fail_active_requests("Engine loop cancelled")
                raise
            except Exception as e:
                logger.error(f"Scheduler step error: {e}", exc_info=True)
                self._fail_active_requests(f"Scheduler step error: {e}")
                # Back-off to avoid tight loop if scheduler is persistently broken.
                # If no requests remain after failing, the loop will idle-wait
                # on _wake_event instead of spinning.
                self._total_step_time_ms += (time.monotonic() - _step_start) * 1000
                await asyncio.sleep(0.1)
                continue

            try:
                # Bug 1 fix: guard against stale/uninitialized scheduler_output
                if scheduler_output is None:
                    self._total_step_time_ms += (time.monotonic() - _step_start) * 1000
                    continue

                # Guard: scheduler_output must have 'outputs' attribute (defensive
                # against TBO/overlap step returning non-standard types)
                if not hasattr(scheduler_output, 'outputs'):
                    logger.warning("scheduler_output missing 'outputs' attribute (type=%s), skipping post-step",
                                   type(scheduler_output).__name__)
                    self._total_step_time_ms += (time.monotonic() - _step_start) * 1000
                    continue

                # Distribute outputs to per-request collectors
                for req_output in scheduler_output.outputs:
                    try:
                        rid = req_output.request_id
                        collector = self._output_collectors.get(rid)
                        if collector is None:
                            continue

                        # Lifecycle: transition QUEUED → PREFILLING on first output
                        if rid not in self._finalized_ids:
                            _lc_state = self._lifecycle_orchestrator.get_state(rid)
                            if _lc_state is not None and _lc_state.phase.name == "QUEUED":
                                self._lifecycle_orchestrator.on_prefill_start(rid)

                        # Output parser: extract reasoning/tool_calls from raw text
                        # (output_parser.py parse_output — model-specific extraction)
                        if req_output.finished and req_output.output_text:
                            try:
                                model_name = getattr(self.scheduler, 'model_id', None)
                                parsed = self._parse_output(req_output.output_text, model_name)
                                if parsed.reasoning and parsed.content != req_output.output_text:
                                    req_output.output_text = parsed.content
                                if parsed.finish_reason:
                                    req_output.finish_reason = parsed.finish_reason
                            except Exception:
                                logger.debug("output parser failed", exc_info=True)

                        if use_simple_streaming:
                            collector.put(req_output)
                        else:
                            # Always put output to collector to prevent token loss
                            # when stream_interval > 1.  The SSE emission layer
                            # handles throttling — skipping here would discard
                            # intermediate tokens permanently.
                            collector.put(req_output)
                            stream_state = self._stream_states.get(rid)
                            if stream_state is not None:
                                stream_state.mark_sent(req_output.completion_tokens)

                        # Forward intermediate output to dedup shadow requests
                        if not req_output.finished and self._request_dedup is not None:
                            shadow_ids = [
                                sid for sid, pid in self._dedup_shadows.items()
                                if pid == rid
                            ]
                            for sid in shadow_ids:
                                s_collector = self._output_collectors.get(sid)
                                if s_collector is not None:
                                    from .request import RequestOutput as _RO
                                    s_collector.put(_RO(
                                        request_id=sid,
                                        # Bug fix: deep-copy mutable list fields so shadow
                                        # collector doesn't share state with primary. Without
                                        # this, _merge or downstream mutation would corrupt
                                        # the primary's accumulated output.
                                        new_token_ids=list(req_output.new_token_ids) if req_output.new_token_ids else req_output.new_token_ids,
                                        new_text=req_output.new_text,
                                        output_token_ids=list(req_output.output_token_ids) if req_output.output_token_ids else req_output.output_token_ids,
                                        output_text=req_output.output_text,
                                        completion_tokens=req_output.completion_tokens,
                                        finished=False,
                                        prompt_tokens=req_output.prompt_tokens,
                                        logprobs=list(req_output.logprobs) if isinstance(req_output.logprobs, list) else req_output.logprobs,
                                        current_state=req_output.current_state,
                                        reasoning_tokens=req_output.reasoning_tokens,
                                        cached_tokens=req_output.cached_tokens,
                                        prefill_progress=req_output.prefill_progress,
                                    ))

                        if req_output.finished:
                            self._num_requests_processed += 1
                            # FairnessTracker: record completion before finalize pops timestamp
                            _start_ts = self._request_timestamps.get(rid)
                            if _start_ts is not None:
                                try:
                                    self._fairness_tracker.record_completion(
                                        rid, 0.0, time.monotonic() - _start_ts,
                                    )
                                except Exception:
                                    logger.debug("fairness record_completion failed", exc_info=True)
                            # Checkpoint: save final state for crash recovery
                            if self._checkpoint_mgr is not None:
                                try:
                                    from .checkpoint import InferenceState
                                    self._checkpoint_mgr.save(rid, InferenceState(
                                        request_id=rid,
                                        output_text=req_output.output_text or "",
                                        position=req_output.prompt_tokens + req_output.completion_tokens,
                                    ))
                                except Exception:
                                    logger.debug("checkpoint save failed", exc_info=True)
                            # Dedup fan-out: deliver output to shadow requests before finalize
                            if self._request_dedup is not None:
                                content_hash = self._dedup_hashes.get(rid)
                                if content_hash:
                                    all_ids = self._request_dedup.complete(content_hash)
                                    from .request import RequestOutput as _RO
                                    for shadow_id in all_ids:
                                        if shadow_id == rid:
                                            continue
                                        shadow_collector = self._output_collectors.get(shadow_id)
                                        if shadow_collector is not None:
                                            shadow_output = _RO(
                                                request_id=shadow_id,
                                                # Bug fix: deep-copy mutable list fields to
                                                # prevent shared-state corruption between
                                                # primary and shadow collectors.
                                                new_token_ids=list(req_output.new_token_ids) if req_output.new_token_ids else req_output.new_token_ids,
                                                new_text=req_output.new_text,
                                                output_token_ids=list(req_output.output_token_ids) if req_output.output_token_ids else req_output.output_token_ids,
                                                output_text=req_output.output_text,
                                                finished=True,
                                                finish_reason=req_output.finish_reason,
                                                prompt_tokens=req_output.prompt_tokens,
                                                completion_tokens=req_output.completion_tokens,
                                                logprobs=list(req_output.logprobs) if isinstance(req_output.logprobs, list) else req_output.logprobs,
                                                current_state=req_output.current_state,
                                                reasoning_tokens=req_output.reasoning_tokens,
                                                cached_tokens=req_output.cached_tokens,
                                                prefill_progress=req_output.prefill_progress,
                                            )
                                            shadow_collector.put(shadow_output)
                                            shadow_collector.put(None)  # sentinel
                                            # Signal shadow finished — don't cleanup here,
                                            # let the consumer's finally block handle it to
                                            # avoid racing with the consumer reading the collector.
                                            self._signal_finished(shadow_id)
                            # Signal request completion before finalize so generate()
                            # consumers waiting on the event can wake up.
                            self._signal_finished(rid)
                            # Finalize: release scheduler-side resources for this request
                            self._finalize_request(
                                rid,
                                completion_tokens=req_output.completion_tokens,
                                finish_reason=req_output.finish_reason or "stop",
                            )
                    except Exception as _output_err:
                        logger.error(
                            "Output distribution error for %s: %s",
                            getattr(req_output, 'request_id', '?'), _output_err,
                            exc_info=True,
                        )
                        # Ensure the request is finalized even if distribution failed
                        _rid = getattr(req_output, 'request_id', None)
                        if _rid:
                            # Put error output + sentinel into collector so
                            # generate()/stream_outputs() consumers don't hang
                            # forever waiting for output that will never arrive.
                            _err_collector = self._output_collectors.get(_rid)
                            if _err_collector is not None:
                                try:
                                    from .request import RequestOutput
                                    _err_collector.put(RequestOutput(
                                        request_id=_rid,
                                        finished=True,
                                        finish_reason="error",
                                        error=f"Output distribution failed: {_output_err}",
                                    ))
                                    _err_collector.put(None)  # sentinel
                                except Exception:
                                    logger.debug(
                                        "error collector put failed for %s",
                                        _rid, exc_info=True,
                                    )
                            self._signal_finished(_rid)
                            self._finalize_request(_rid)

                # Update adaptive batch scheduler metrics
                if scheduler_output.outputs:
                    # ── Wave 42: Lifecycle decode tracking for active requests ──
                    for req_output in scheduler_output.outputs:
                        rid = req_output.request_id
                        if req_output.completion_tokens > 0:
                            # Lifecycle: transition PREFILLING → DECODING on first
                            # output token.  Must also fire for finished requests
                            # (e.g. max_tokens=1 where the first token is the last)
                            # so lifecycle state is accurate for metrics and TTFT.
                            if rid not in self._finalized_ids:
                                state = self._lifecycle_orchestrator.get_state(rid)
                                if state is not None and state.phase.name in ("PREFILLING",):
                                    self._lifecycle_orchestrator.on_decode_start(rid)
                            # Skip budget/sliding-window work for finished requests —
                            # their lifecycle ends at FINISHED, not DECODING.
                            if req_output.finished:
                                continue
                            # Guard: skip if this request was already finalized earlier
                            # in this step (e.g., budget exhaustion on a previous output
                            # for the same request when stream_interval > 1 produces
                            # multiple outputs per step).  Without this, consume() returns
                            # "budget_not_found" and triggers duplicate error handling.
                            if rid in self._finalized_ids:
                                continue
                            # Use incremental token count (new_token_ids length), not
                            # cumulative completion_tokens.  completion_tokens is the
                            # total generated so far; feeding it to consume() on every
                            # step would over-count by 1+2+3+...+N instead of N.
                            _incr_tokens = len(req_output.new_token_ids) if req_output.new_token_ids else 1
                            budget_result = self._budget_manager.consume(rid, tokens=_incr_tokens)
                            if budget_result is not None:
                                # Budget exhausted — abort request so scheduler stops generating
                                logger.info(f"Budget exhausted for {rid}: {budget_result}")
                                self.scheduler.abort_request(rid)
                                from .request import RequestOutput as _RO
                                _bc = self._output_collectors.get(rid)
                                if _bc is not None:
                                    _bc.put(_RO(
                                        request_id=rid,
                                        finished=True,
                                        finish_reason=budget_result,
                                        error=f"Budget exhausted: {budget_result}",
                                        prompt_tokens=req_output.prompt_tokens,
                                        completion_tokens=req_output.completion_tokens,
                                    ))
                                    _bc.put(None)
                                # Fail dedup shadows so their consumers don't hang
                                if self._request_dedup is not None:
                                    _shadow_ids = [
                                        sid for sid, pid in self._dedup_shadows.items()
                                        if pid == rid
                                    ]
                                    for _sid in _shadow_ids:
                                        _sc = self._output_collectors.get(_sid)
                                        if _sc is not None:
                                            _sc.put(_RO(
                                                request_id=_sid,
                                                finished=True,
                                                finish_reason=budget_result,
                                                error=f"Primary request {rid} budget exhausted",
                                                prompt_tokens=req_output.prompt_tokens,
                                                completion_tokens=req_output.completion_tokens,
                                            ))
                                            _sc.put(None)
                                        self._signal_finished(_sid)
                                        self._finalize_request(_sid)
                                self._signal_finished(rid)
                                self._finalize_request(rid, completion_tokens=req_output.completion_tokens, finish_reason=budget_result)
                                # Request is fully finalized — skip remaining per-output
                                # processing (sliding window, lifecycle) to avoid operating
                                # on a request whose scheduler state has already been
                                # removed by abort_request + _finalize_request.
                                continue
                            # Sliding window tracking
                            if self._sliding_window_mgr is not None:
                                try:
                                    total_pos = req_output.prompt_tokens + req_output.completion_tokens
                                    evicted = self._sliding_window_mgr.on_new_token(
                                        token_position=total_pos,
                                        request_id=rid,
                                    )
                                    # Trim real KV cache for sliding window models.
                                    # When blocks slide out of the window, the KV cache
                                    # arrays must be trimmed to free memory and keep
                                    # attention computation correct.
                                    if evicted:
                                        req = self.scheduler.running.get(rid)
                                        if req is not None:
                                            # Trim the per-request prompt cache (MLX KV arrays)
                                            if req.prompt_cache is not None:
                                                self._sliding_window_mgr.trim_kv_cache(
                                                    req.prompt_cache, request_id=rid,
                                                )
                                            # Invalidate prefix cache entries whose blocks
                                            # have slid out of the window. Without this,
                                            # new requests may get prefix cache hits with
                                            # stale KV blocks that the model will never
                                            # attend to.
                                            prefix_cache = getattr(
                                                self.scheduler, '_prefix_cache', None,
                                            )
                                            if prefix_cache is not None:
                                                self._sliding_window_mgr.invalidate_prefix_cache(
                                                    prefix_cache, request_id=rid,
                                                )
                                except Exception:
                                    logger.debug("sliding window tracking failed", exc_info=True)

                    # Auto-checkpoint: save inference state periodically for crash recovery
                    if self._checkpoint_mgr is not None:
                        try:
                            for req_output in scheduler_output.outputs:
                                if not req_output.finished and req_output.completion_tokens > 0:
                                    if self._checkpoint_mgr.should_auto_checkpoint(
                                        req_output.request_id,
                                        req_output.completion_tokens,
                                    ):
                                        from .checkpoint import InferenceState
                                        self._checkpoint_mgr.save(
                                            req_output.request_id,
                                            InferenceState(
                                                request_id=req_output.request_id,
                                                position=req_output.prompt_tokens + req_output.completion_tokens,
                                                generated_tokens=[],
                                                output_text=getattr(req_output, 'output_text', ''),
                                                model_name=getattr(self.scheduler, 'model_id', ''),
                                            ),
                                        )
                        except Exception:
                            logger.debug("auto-checkpoint failed", exc_info=True)

                    # ── Wave 42: Profiler + auto-tuner + fairness ──
                    try:
                        batch_size = len(scheduler_output.outputs)
                        _step_wall_ms = self._last_step_wall_ms
                        # Per-step token count: each scheduler step produces 1 new token
                        # per decode request (and possibly multiple for prefill).  Use
                        # new_token_ids length (incremental) rather than cumulative
                        # completion_tokens to avoid over-counting.
                        _tokens_gen = sum(
                            len(o.new_token_ids) for o in scheduler_output.outputs if o.new_token_ids
                        )
                        # Fallback: if new_token_ids is empty (some scheduler paths
                        # don't populate it), use batch_size as a reasonable estimate.
                        if _tokens_gen == 0 and batch_size > 0:
                            _tokens_gen = batch_size
                        _throughput = _tokens_gen / (_step_wall_ms / 1000) if _step_wall_ms > 0 else 0.0

                        # Estimate per-step ITL from step wall time and tokens generated
                        _est_itl_ms = 0.0
                        if _tokens_gen > 0 and batch_size > 0:
                            _est_itl_ms = _step_wall_ms / _tokens_gen

                        # Estimate TTFT from requests that just transitioned from
                        # PREFILLING to DECODING (completion_tokens > 0 and not yet
                        # recorded in _ttft_done). This measures actual first-token
                        # latency, not end-to-end request latency.
                        _est_ttft_ms = 0.0
                        _ttft_count = 0
                        for o in scheduler_output.outputs:
                            rid = getattr(o, 'request_id', None)
                            if not rid or rid in self._ttft_done:
                                continue
                            # Do NOT skip finished requests — a request that
                            # finishes on its first decode step (e.g. max_tokens=1)
                            # still has a valid TTFT that should be recorded.
                            # Skipping it biases TTFT estimates upward.
                            if getattr(o, 'completion_tokens', 0) > 0:
                                _start_ts = self._request_timestamps.get(rid)
                                if _start_ts is not None:
                                    _est_ttft_ms += (time.monotonic() - _start_ts) * 1000
                                    _ttft_count += 1
                                    self._ttft_done.add(rid)
                                    self._ttft_timestamps[rid] = time.monotonic()
                        if _ttft_count > 0:
                            _est_ttft_ms /= _ttft_count

                        # GPU memory utilisation for bottleneck classification
                        _gpu_mem_util = 0.0
                        try:
                            import mlx.core as mx
                            active_mem = mx.get_active_memory()
                            if _hw_info is None:
                                from .utils.hardware import get_hardware_info as _ghw
                                _hw_info = _ghw()
                                _total_mem_bytes = _hw_info.total_memory_bytes
                            _gpu_mem_util = active_mem / max(_total_mem_bytes, 1)
                        except Exception:
                            pass

                        from .auto_tuner import StepMetrics
                        step_metrics = StepMetrics(
                            batch_size=batch_size,
                            tokens_generated=_tokens_gen,
                            wall_time_ms=_step_wall_ms,
                            throughput_tok_s=_throughput,
                            ttft_ms=_est_ttft_ms,
                            itl_ms=_est_itl_ms,
                            gpu_memory_util=_gpu_mem_util,
                        )
                        self._profiler.record_step(step_metrics)
                        # Auto-tune every 100 steps
                        if self._profiler._total_steps % 100 == 0:
                            # Evaluate previous tuning decisions for regression
                            for prev in self._auto_tuner._history[-1:]:
                                if getattr(prev, 'after_metrics', None) is None and prev.before_metrics is not None:
                                    self._auto_tuner.evaluate_tuning(prev, step_metrics)
                                    break
                            tuning_decisions = self._auto_tuner.auto_tune()
                            if tuning_decisions:
                                self._apply_tuning_to_config(tuning_decisions)
                                logger.debug(f"AutoTuner applied: {[d.param_name for d in tuning_decisions]}")
                        # SLO checks
                        if _est_ttft_ms > 0:
                            self._slo_monitor.check_slo("ttft", _est_ttft_ms)
                        if _est_itl_ms > 0:
                            self._slo_monitor.check_slo("itl", _est_itl_ms)
                        # Only check throughput SLO when the system is active.
                        # When idle (throughput=0 and no running requests), recording
                        # a violation inflates the violation rate and triggers
                        # spurious auto-tuning decisions.
                        if step_metrics.throughput_tok_s > 0 or len(self.scheduler.running) > 0:
                            self._slo_monitor.check_slo("throughput", step_metrics.throughput_tok_s)
                        # Fairness tracker: use incremental tokens (new_token_ids length)
                        # not cumulative completion_tokens, which grows every step.
                        for req_output in scheduler_output.outputs:
                            _alloc = len(req_output.new_token_ids) if req_output.new_token_ids else 1
                            self._fairness_tracker.record_allocation(
                                req_output.request_id,
                                tokens_allocated=_alloc,
                            )
                    except Exception:
                        logger.debug("profiler/auto-tuner failed", exc_info=True)

                    # ── Per-request generation timeout enforcement ──
                    try:
                        timeout_s = self.config.request_timeout_seconds
                        if timeout_s > 0:
                            now = time.monotonic()
                            for rid in list(self._request_timestamps.keys()):
                                # Skip requests already finalized in this step
                                # (normal completion or earlier timeout processing)
                                if rid not in self.scheduler.running:
                                    continue
                                start = self._request_timestamps[rid]
                                if (now - start) > timeout_s:
                                    logger.warning(f"Request {rid} timed out ({now - start:.0f}s > {timeout_s}s)")
                                    self.scheduler.abort_request(rid)
                                    collector = self._output_collectors.get(rid)
                                    if collector is not None:
                                        from .request import RequestOutput
                                        timeout_output = RequestOutput(
                                            request_id=rid,
                                            finished=True,
                                            finish_reason="timeout",
                                            error=f"Request exceeded timeout ({timeout_s}s)",
                                        )
                                        collector.put(timeout_output)
                                        collector.put(None)
                                    # Dedup fan-out: deliver timeout to shadow requests
                                    if self._request_dedup is not None:
                                        shadow_ids = [
                                            sid for sid, pid in self._dedup_shadows.items()
                                            if pid == rid
                                        ]
                                        for sid in shadow_ids:
                                            s_collector = self._output_collectors.get(sid)
                                            if s_collector is not None:
                                                from .request import RequestOutput
                                                s_collector.put(RequestOutput(
                                                    request_id=sid,
                                                    finished=True,
                                                    finish_reason="timeout",
                                                    error=f"Primary request {rid} timed out",
                                                ))
                                                s_collector.put(None)
                                            self._signal_finished(sid)
                                            self._cleanup_request(sid)
                                    self._signal_finished(rid)
                                    self._finalize_request(rid, finish_reason="timeout")

                            # ── Shadow request timeout enforcement ──
                            # Shadow requests are never in scheduler.running, so the loop
                            # above skips them.  If the primary request's output distribution
                            # never fires (e.g. engine loop crashed between scheduler step
                            # and output distribution, or primary was aborted externally
                            # without fan-out), the shadow's event is never set and its
                            # generate()/stream_outputs() hangs forever.  Check shadows
                            # directly against their timestamps.
                            if self._request_dedup is not None:
                                for sid in list(self._dedup_shadows.keys()):
                                    start = self._request_timestamps.get(sid)
                                    if start is None:
                                        continue
                                    if (now - start) > timeout_s:
                                        logger.warning(
                                            f"Dedup shadow {sid} timed out "
                                            f"({now - start:.0f}s > {timeout_s}s)"
                                        )
                                        s_collector = self._output_collectors.get(sid)
                                        if s_collector is not None:
                                            from .request import RequestOutput
                                            s_collector.put(RequestOutput(
                                                request_id=sid,
                                                finished=True,
                                                finish_reason="timeout",
                                                error=f"Dedup shadow timed out after {timeout_s}s (primary never completed)",
                                            ))
                                            s_collector.put(None)
                                        self._signal_finished(sid)
                                        self._cleanup_request(sid)
                    except Exception:
                        logger.debug("timeout enforcement failed", exc_info=True)

                    try:
                        import mlx.core as mx
                        active_mem = mx.get_active_memory()
                        if _hw_info is None:
                            from .utils.hardware import get_hardware_info
                            _hw_info = get_hardware_info()
                            _total_mem_bytes = _hw_info.total_memory_bytes
                        mem_usage = active_mem / max(_total_mem_bytes, 1)
                        # Use total step wall time (including output distribution
                        # and profiler overhead) for adaptive batch scheduling,
                        # NOT GPU-only time. GPU-only time understates latency
                        # and causes the batch sizer to over-recommend batch sizes.
                        _total_step_wall_ms = (time.monotonic() - _step_start) * 1000
                        self._adaptive_batch.update_metrics(
                            latency_ms=_total_step_wall_ms,
                            memory_usage=mem_usage,
                            batch_size=len(scheduler_output.outputs),
                        )
                        # KV compression under memory pressure: compress old blocks
                        # instead of outright eviction when usage > 85%
                        if mem_usage > 0.85:
                            # Reduce prefill batch size to prevent new prefill
                            # requests from consuming freed blocks faster than
                            # eviction can release them.  Without this, the
                            # scheduler admits large prefills that immediately
                            # re-fill the KV pool and perpetuate OOM.
                            if self.config.prefill_batch_size > 1:
                                old_pbs = self.config.prefill_batch_size
                                self.config.prefill_batch_size = max(1, old_pbs // 2)
                                logger.info(
                                    "Memory pressure: reducing prefill_batch_size %d -> %d (mem_usage=%.1f%%)",
                                    old_pbs, self.config.prefill_batch_size, mem_usage * 100,
                                )
                            if self._kv_compressor is not None:
                                try:
                                    kv_mgr = getattr(self.scheduler, '_kv_manager', None)
                                    if kv_mgr is not None:
                                        evicted = kv_mgr.memory_pressure_evict(0.90)
                                        if evicted > 0:
                                            logger.debug(
                                                f"KV memory pressure eviction: {evicted} blocks"
                                            )
                                except Exception:
                                    logger.debug("KV pressure eviction failed", exc_info=True)
                        # Recovery: when memory usage drops below 70%, the auto-tuner
                        # and AdaptiveBatchSizer will naturally restore batch sizes.
                        # No explicit recovery needed here — prefill_batch_size stays
                        # reduced until the auto-tuner evaluates the next tuning cycle.
                        # Telemetry: record step-level metrics
                        self._telemetry.collect(
                            "engine_step_batch_size",
                            float(len(scheduler_output.outputs)),
                            tags={"model": getattr(self.scheduler, 'model_id', '')},
                        )
                        self._telemetry.collect(
                            "engine_step_memory_usage",
                            mem_usage,
                        )
                        # SpecPrefill: use GPU idle time for priority prefill queue
                        if mem_usage < 0.7 and self._spec_prefill_engine is not None:
                            try:
                                batch_size = len(scheduler_output.outputs)
                                budget = max(0, self.config.completion_batch_size - batch_size) * 512
                                if budget > 0:
                                    entry = self._spec_prefill_engine.try_prefill(budget)
                                    if entry is not None and entry.status == "completed":
                                        self._spec_prefill_engine.remove_entry(entry.request_id)
                            except Exception:
                                logger.debug("spec prefill attempt failed", exc_info=True)
                    except Exception:
                        logger.debug("memory telemetry collection failed", exc_info=True)

                # Accumulate total active time (GPU step + output distribution
                # + profiler overhead).  This must happen after all post-step
                # processing so compute_utilization accurately reflects the
                # fraction of wall time spent doing useful work (not just the
                # GPU kernel time).
                self._total_step_time_ms += (time.monotonic() - _step_start) * 1000

                await asyncio.sleep(0)
            except Exception as _post_step_err:
                # Catch-all for post-step processing errors.  Individual output
                # distribution errors are already handled per-request, but
                # structural errors (e.g., scheduler_output.outputs not iterable,
                # attribute errors on scheduler state after shutdown race) would
                # otherwise crash the engine loop, leaving all active requests
                # hanging forever.  Log and continue to the next iteration.
                logger.error(
                    "Post-step processing error: %s", _post_step_err,
                    exc_info=True,
                )
                # Best-effort cleanup: finalize any unfinalized requests
                try:
                    if hasattr(scheduler_output, 'outputs'):
                        for _ro in scheduler_output.outputs:
                            _rid = getattr(_ro, 'request_id', None)
                            if _rid and _rid not in self._finalized_ids:
                                _fc = self._output_collectors.get(_rid)
                                if _fc is not None:
                                    try:
                                        from .request import RequestOutput
                                        _fc.put(RequestOutput(
                                            request_id=_rid,
                                            finished=True,
                                            finish_reason="error",
                                            error=f"Post-step processing error: {_post_step_err}",
                                        ))
                                        _fc.put(None)
                                    except Exception:
                                        pass
                                self._signal_finished(_rid)
                                self._finalize_request(_rid)
                except Exception:
                    logger.debug("post-step error cleanup failed", exc_info=True)

    def _compute_prefix_hash_for_request(self, req_id: str, prompt_token_ids: list[int]) -> int | None:
        """Compute and store a KV prefix hash for a request.

        Uses the first complete KV block's worth of tokens as the prefix hash.
        This approximates the system prompt / conversation prefix that determines
        KV block sharing. Returns the hash, or None if the prompt is too short.
        """
        block_size = self.config.kv_block_size
        if len(prompt_token_ids) < block_size:
            return None
        try:
            from yunshu_kv.hash import compute_block_hash
            first_block_tokens = prompt_token_ids[:block_size]
            return compute_block_hash(None, first_block_tokens)
        except Exception:
            logger.debug("block hash computation failed", exc_info=True)
            return None

    def _signal_finished(self, request_id: str) -> None:
        """Signal request completion."""
        event = self._finished_events.get(request_id)
        if event:
            event.set()

    def _finalize_request(self, request_id: str, completion_tokens: int = 0, finish_reason: str = "stop") -> None:
        """Release scheduler-side per-request resources (NOT consumer-side state).

        Called from:
        - Engine loop finish path (normal completion)
        - abort_request() / abort_all_requests()
        - Engine loop exception handlers
        - Consumer-side _cleanup_request() (which also pops consumer state)

        This is idempotent: safe to call multiple times.
        Consumer-side state (collectors, events, stream states, timestamps)
        is only removed by _cleanup_request() to ensure generate() and
        stream_outputs() can drain the collector after this call.
        """
        if not hasattr(self, '_finalized_ids'):
            self._finalized_ids = set()
        if request_id in self._finalized_ids:
            return
        self._finalized_ids.add(request_id)
        _block_id = hash(request_id) % (10**9)
        # Inflight prefix sharing
        try:
            from .inflight_prefix_sharing import get_inflight_tracker
            get_inflight_tracker().unregister(request_id)
        except Exception:
            logger.debug("inflight prefix unregister failed", exc_info=True)
        # Release LoRA adapter (decrement ref count, doesn't unload unless LRU evicts)
        lora_id = self._request_lora_adapters.pop(request_id, None)
        if lora_id:
            try:
                from .lora_manager import get_lora_manager
                lora_mgr = get_lora_manager(engine_id=getattr(self.scheduler, 'model_id', 'default') or 'default')
                if lora_mgr is not None:
                    lora_mgr.release_adapter(lora_id)
            except Exception:
                logger.debug("LoRA cleanup failed", exc_info=True)
        # Lifecycle + budget + memory + KV lifecycle
        try:
            self._lifecycle_orchestrator.on_request_finished(request_id, completion_tokens=completion_tokens, finish_reason=finish_reason)
        except Exception:
            logger.debug("lifecycle orchestrator finish failed", exc_info=True)
        try:
            self._budget_manager.remove(request_id)
        except Exception:
            logger.debug("budget manager remove failed", exc_info=True)
        try:
            self._memory_aware_scheduler.release_memory(request_id)
        except Exception:
            logger.debug("memory-aware scheduler release failed", exc_info=True)
        try:
            self._kv_lifecycle.release(_block_id)
        except Exception:
            logger.debug("kv_lifecycle release failed", exc_info=True)
        # KV migration
        try:
            self._kv_migration.unregister_block(_block_id)
        except Exception:
            logger.debug("kv_migration unregister failed", exc_info=True)
        # Dedup: only the primary request calls complete(). Shadows
        # are fanned-out by the engine loop before reaching here, so
        # calling complete() again would be redundant (and the entry
        # may already be removed by TTL pruning).
        is_shadow = request_id in self._dedup_shadows
        if self._request_dedup is not None:
            content_hash = self._dedup_hashes.pop(request_id, None)
            self._dedup_shadows.pop(request_id, None)
            if content_hash and not is_shadow:
                self._request_dedup.complete(content_hash)
        else:
            self._dedup_shadows.pop(request_id, None)
        # Checkpoint
        if self._checkpoint_mgr is not None:
            self._checkpoint_mgr.delete(request_id)
        # Sliding window
        if self._sliding_window_mgr is not None:
            try:
                self._sliding_window_mgr.remove_request(request_id)
            except Exception:
                logger.debug("sliding window cleanup failed", exc_info=True)
        # Fairness tracker: remove per-request allocation accumulators
        try:
            self._fairness_tracker.remove_request(request_id)
        except Exception:
            logger.debug("fairness tracker cleanup failed", exc_info=True)
        # Remove from scheduler (only if actually added — dedup shadows
        # are short-circuited before reaching scheduler.add_request)
        if not is_shadow:
            self.scheduler.remove_finished_request(request_id)

    def _fail_active_requests(self, error_msg: str) -> None:
        """Fail all active requests (scheduler + dedup shadows) with an error.

        Used by engine loop exception handlers to ensure ALL requests —
        including dedup shadow requests that are not in the scheduler —
        receive error outputs and sentinel values so consumers don't hang.
        """
        from .request import RequestOutput

        # Fail scheduler-tracked requests first
        failed = self.scheduler.fail_all_requests()

        # Also collect dedup shadow request IDs that have collectors but
        # are NOT in the scheduler (short-circuited in add_request).
        shadow_ids = []
        if self._request_dedup is not None:
            for sid, primary_id in list(self._dedup_shadows.items()):
                if sid not in failed:
                    shadow_ids.append(sid)

        all_ids = failed + shadow_ids

        for req_id in all_ids:
            collector = self._output_collectors.get(req_id)
            if collector is not None:
                collector.put(RequestOutput(
                    request_id=req_id,
                    finished=True,
                    finish_reason="error",
                    error=error_msg,
                ))
                collector.put(None)  # sentinel
            self._signal_finished(req_id)
            self._finalize_request(req_id)

        # Count failed requests so get_stats() and Prometheus report
        # accurate totals.  Without this, engine-loop crash recovery
        # under-counts because the normal finish path (line 2195) is
        # never reached for these requests.
        if all_ids:
            self._num_requests_processed += len(all_ids)

    def _fail_dedup_shadows(self, primary_id: str, error_msg: str, finish_reason: str = "error") -> None:
        """Deliver error output to all dedup shadows of a failed primary request.

        When a primary request is rejected before entering the scheduler (memory
        guard, budget, etc.), its shadow requests have collectors but no output.
        This method delivers error outputs to all of them so consumers don't hang.
        """
        from .request import RequestOutput
        shadow_ids = [
            sid for sid, pid in list(self._dedup_shadows.items())
            if pid == primary_id
        ]
        for sid in shadow_ids:
            collector = self._output_collectors.get(sid)
            if collector is not None:
                collector.put(RequestOutput(
                    request_id=sid,
                    finished=True,
                    finish_reason=finish_reason,
                    error=error_msg,
                    prompt_tokens=0,
                    completion_tokens=0,
                ))
                collector.put(None)
            self._signal_finished(sid)
            # Finalize scheduler-side resources (lifecycle, budget, KV) but
            # do NOT call _cleanup_request here — that removes consumer-side
            # state (collector, event) which the shadow's consumer (generate/
            # stream_outputs) still needs to read the error output we just put.
            # The consumer's finally block will call _cleanup_request.
            self._finalize_request(sid)

    def _cleanup_request(self, request_id: str) -> None:
        """Remove per-request state (consumer-side entry point).

        Called from stream_outputs() finally block and generate() finally block.
        Also called for dedup shadow requests that have no consumer.
        Releases scheduler-side resources via _finalize_request, then removes
        consumer-side state (collector, event, stream state, timestamps).
        """
        self._finalize_request(request_id)
        # Consumer-side state: only removed here so that generate() and
        # stream_outputs() can drain the collector before cleanup.
        self._output_collectors.pop(request_id, None)
        self._stream_states.pop(request_id, None)
        self._finished_events.pop(request_id, None)
        self._request_timestamps.pop(request_id, None)
        if hasattr(self, '_ttft_timestamps'):
            self._ttft_timestamps.pop(request_id, None)
        self._kv_prefix_hashes.pop(request_id, None)
        # Bug fix: do NOT discard from _finalized_ids here.  The idempotency
        # guard in _finalize_request relies on _finalized_ids persisting
        # across calls.  Discarding it here means a second _cleanup_request
        # call (e.g., from abort_request + stream_outputs finally) would
        # bypass the guard and double-release LoRA adapters, memory, KV
        # blocks, budget, etc.  Bounded growth is acceptable — cleared in
        # stop() along with all other per-request state.
        self._ttft_done.discard(request_id)

    def _get_max_seq_len(self) -> int:
        """Get the model's maximum sequence length from config."""
        try:
            model = self._model
            config = getattr(model, 'config', model)
            for attr in ('max_seq_len', 'max_position_embeddings', 'n_positions',
                         'max_sequence_length', 'seq_length'):
                val = _safe_get(config, attr, 0)
                if val and isinstance(val, int) and val > 0:
                    return val
        except Exception:
            logger.debug("max_seq_len detection failed", exc_info=True)
        return 0

    def _messages_to_text(
        self,
        messages: list[dict],
        enable_thinking: bool | None = None,
    ) -> str:
        """Convert chat messages to text using the model's chat template."""
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                clean = []
                for m in messages:
                    content = m.get("content", "")
                    # When content is a list (multimodal: text + image_url),
                    # extract only text parts so apply_chat_template gets a string.
                    if isinstance(content, list):
                        text_parts = [
                            p.get("text", "") for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        content = "\n".join(t for t in text_parts if t)
                    clean.append({"role": m.get("role", "user"), "content": content})
                kwargs: dict[str, Any] = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if enable_thinking is not None:
                    kwargs["enable_thinking"] = enable_thinking
                try:
                    text = self._tokenizer.apply_chat_template(clean, **kwargs)
                except TypeError as e:
                    if 'enable_thinking' in str(e):
                        logger.warning("Model doesn't support enable_thinking, retrying without")
                        kwargs.pop('enable_thinking', None)
                        text = self._tokenizer.apply_chat_template(clean, **kwargs)
                    else:
                        raise
                if text:
                    return text
            except Exception:
                logger.debug("chat template failed, using fallback", exc_info=True)

        # Generic fallback
        parts = []
        for m in messages:
            parts.append(f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Return engine core stats (oMLX pattern)."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0
        scheduler_stats = self.scheduler.get_stats()
        stats = {
            "running": self._running,
            "num_requests_processed": self._num_requests_processed,
            "active_collectors": len(self._output_collectors),
            "uptime_seconds": round(uptime, 1),
            "compute_utilization_pct": round(self.get_compute_utilization(), 2),
            "last_step_duration_ms": round(self._last_step_wall_ms, 2),
            "cpu_gpu_overlap": self._overlap_scheduler.get_stats(),
            "tbo": self._tbo_scheduler.get_stats(),
            "adaptive_batch": self._adaptive_batch.get_stats(),
            **{f"scheduler_{k}": v for k, v in scheduler_stats.items()},
        }
        # KV offload stats (§12.3)
        if self._kv_offload_manager is not None:
            stats["kv_offload"] = self._kv_offload_manager.get_stats()
        # §12.2: encoder-decoder cache stats
        if hasattr(self.scheduler, '_encoder_cache'):
            stats["encoder_cache"] = self.scheduler._encoder_cache.get_stats()
        # §16.2: external prefill server/client stats
        if self._prefill_server is not None:
            stats["external_prefill_server"] = self._prefill_server.get_stats()
        if self._prefill_client is not None:
            stats["external_prefill_client"] = self._prefill_client.get_stats()
        # §12.2: KV transfer server stats (decode node)
        if self._kv_transfer_server is not None:
            stats["kv_transfer_server"] = self._kv_transfer_server.get_stats()
        # ── Wave 42: Wired module stats ──
        stats["lifecycle"] = self._lifecycle_orchestrator.get_stats()
        stats["budget"] = self._budget_manager.get_stats()
        stats["kv_lifecycle"] = self._kv_lifecycle.get_stats()
        stats["token_scheduler"] = self._token_scheduler.get_stats()
        stats["auto_tuner"] = self._auto_tuner.get_stats()
        stats["fairness"] = self._fairness_tracker.get_stats()
        stats["profiler"] = self._profiler.get_stats()
        stats["slo"] = self._slo_monitor.get_stats()
        if self._request_dedup is not None:
            stats["request_dedup"] = self._request_dedup.get_stats()
        if self._composition_scheduler is not None:
            stats["composition_scheduler"] = self._composition_scheduler.get_stats()
        # Wave 43: Additional module stats
        stats["forward_batch"] = self._batch_composer.get_stats()
        stats["memory_aware_scheduler"] = self._memory_aware_scheduler.get_stats().__dict__
        stats["context_window"] = self._context_window_mgr.get_stats()
        stats["kv_prefix_compression"] = self._kv_compressor.get_stats()
        if self._checkpoint_mgr is not None:
            stats["checkpoint"] = self._checkpoint_mgr.get_stats()
        if self._sliding_window_mgr is not None:
            stats["sliding_window"] = self._sliding_window_mgr.get_stats()
        stats["kv_migration"] = self._kv_migration.get_stats()
        stats["hybrid_kv"] = self._hybrid_kv.get_stats()
        stats["batch_sampler"] = self._batch_sampler.get_stats()
        stats["model_optimizations"] = {
            "rope": self._rope_optimizer.get_stats(),
            "attention": self._attention_optimizer.get_stats(),
            "moe": self._moe_optimizer.get_stats(),
            "warmup": self._warmup_manager.get_stats(),
        }
        stats["process_isolation"] = {"enabled": self._isolation_enabled}
        stats["spec_prefill_engine"] = self._spec_prefill_engine.get_stats()
        stats["turbo_quant"] = self._turbo_quant.get_stats()
        # Model preprocessor stats
        if self._preprocessor_registry is not None:
            stats["model_preprocessor"] = self._preprocessor_registry.get_stats()
        # Inflight prefix sharing stats
        try:
            from .inflight_prefix_sharing import get_inflight_tracker
            stats["inflight_prefix_sharing"] = get_inflight_tracker().get_stats()
        except Exception:
            logger.debug("inflight prefix stats unavailable", exc_info=True)
            stats["inflight_prefix_sharing"] = {"enabled": False}
        return stats

    def _overlap_step(self) -> Any:
        """Run one scheduler step with CPU/GPU overlap (C18).

        Called on the MLX executor thread. Overlaps CPU post-processing
        from the previous step with GPU forward of the current step.
        """
        self._overlap_scheduler.step_async(self.scheduler)
        # CPU can do other work here while GPU is computing
        return self._overlap_scheduler.step_sync()

    def _tbo_step(self) -> Any:
        """Run one scheduler step with Two-Batch Overlap (§14.1).

        Called on the MLX executor thread. Uses double-buffering: while GPU
        processes the active batch, CPU prepares the pending batch.
        """
        return self._tbo_scheduler.step(self.scheduler)

    def get_kv_cache_stats(self) -> dict:
        """Return KV prefix cache statistics (for admin endpoint)."""
        if self._kv_manager is None:
            return {"enabled": False}
        mgr = self._kv_manager
        pool = mgr.block_pool
        result = {
            "enabled": True,
            "block_size": mgr.block_size,
            "total_blocks": pool.num_blocks,
            "free_blocks": mgr.num_free_blocks,
            "used_blocks": pool.num_blocks - 1 - mgr.num_free_blocks,
            "usage": round(mgr.usage, 3),
            "cached_hashes": len(pool._hash_to_block),
            "hit_rate": round(mgr.hit_rate, 4),
            "total_lookups": mgr._total_lookups,
            "total_hits": mgr._total_hits,
            "active_block_tables": len(
                getattr(self.scheduler, '_block_tables', {})
            ),
        }
        # Append KV offload stats (§12.3)
        if self._kv_offload_manager is not None:
            result["offload"] = self._kv_offload_manager.get_stats()
        return result


class AsyncEngineCore:
    """Async context manager wrapper around EngineCore (oMLX AsyncEngineCore pattern).

    Usage:
        async with AsyncEngineCore(model, tokenizer) as engine:
            req_id = await engine.add_request("Hello")
            async for output in engine.stream_outputs_corrected(req_id):
                print(output.new_text)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: EngineCoreConfig | None = None,
    ) -> None:
        self._core = EngineCore(model, tokenizer, config)

    async def __aenter__(self) -> AsyncEngineCore:
        await self._core.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self._core.stop()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._core, name)
