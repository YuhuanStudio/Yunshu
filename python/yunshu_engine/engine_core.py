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
from __future__ import annotations

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
                        # Auto-compute from UMA budget
                        try:
                            from .utils.hardware import get_hardware_info
                            _hw = get_hardware_info()
                            uma_bytes = _hw.total_memory_bytes
                            if uma_bytes > 0:
                                num_blocks = compute_num_blocks(kv_config, uma_bytes, 0)
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
        self._profiler.start_profiling()
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
        self._loop_task: asyncio.Task | None = None
        self._start_time: float | None = None
        self._wake_event: asyncio.Event | None = None  # Event-driven wake-up for idle loop

        # ── Wave 43: Additional production wiring ──

        # Forward batch hierarchy (ScheduleBatch → ForwardBatch → BatchResult)
        from .forward_batch import ScheduleBatch, ForwardBatch, BatchResult, BatchComposer
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
        self._request_lora_adapters: dict[str, str] = {}  # req_id → lora_adapter_id

        # Cache-locality request reordering (SGLang/vLLM pattern)
        # Maps request_id → KV prefix hash for grouping requests with shared
        # prefixes. Requests sharing the same prefix hash are scheduled
        # consecutively to maximize KV cache block locality and reduce thrashing.
        self._kv_prefix_hashes: dict[str, int] = {}

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

        monitor = MemoryMonitor()
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
        self._running = True
        self._start_time = time.monotonic()
        self._wake_event = asyncio.Event()
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
        if self._prefill_server is not None:
            asyncio.get_running_loop().create_task(
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
        """
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
        self._shutdown_requested = False

        # Stop KV migration background thread
        try:
            self._kv_migration.stop()
        except Exception:
            logger.debug("KV migration stop failed", exc_info=True)

        # Stop performance profiler
        try:
            self._profiler.stop_profiling()
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

        # Stop external prefill server
        if self._prefill_server is not None:
            try:
                await self._prefill_server.stop()
            except Exception:
                logger.debug("prefill server stop failed", exc_info=True)

        # Stop KV transfer server
        if self._kv_transfer_server is not None:
            try:
                await self._kv_transfer_server.stop()
            except Exception:
                logger.debug("kv transfer server stop failed", exc_info=True)

        # Signal all active collectors with sentinel
        for collector in self._output_collectors.values():
            try:
                collector.put(None)
            except Exception:
                logger.debug("collector sentinel put failed", exc_info=True)
        for event in self._finished_events.values():
            event.set()

        # Clean up all running requests (inflight prefix sharing, etc.)
        for req_id in list(self._output_collectors.keys()):
            self._cleanup_request(req_id)
        self._output_collectors.clear()
        self._stream_states.clear()
        self._finished_events.clear()

        self.scheduler.shutdown()

        # Release model/tokenizer refs + GC + cache clear on executor
        self._model = None
        self._tokenizer = None
        gc.collect()
        loop = asyncio.get_running_loop()
        from .mlx_executor import sync_and_clear_cache
        try:
            await loop.run_in_executor(self._executor, sync_and_clear_cache)
        except Exception:
            logger.debug("cache clear on executor failed", exc_info=True)

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
        **kwargs,
    ) -> str:
        """Add a generation request. Returns request_id for streaming/abort.

        If MemoryGuard preflight check fails, creates an error output
        immediately instead of adding to the scheduler.
        """
        from .request import Request, SamplingParams

        req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

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

        # Apply per-request LoRA adapter (load before generation, unload after)
        loaded_lora = None
        if lora_adapter:
            try:
                from .lora_manager import get_lora_manager
                lora_mgr = get_lora_manager()
                if lora_mgr is not None:
                    loaded_lora = lora_mgr.load_adapter(lora_adapter)
                    logger.debug(f"LoRA adapter '{lora_adapter}' loaded for request {req_id}")
            except Exception:
                logger.debug(f"LoRA adapter load failed: {lora_adapter}", exc_info=True)

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

        num_prompt_tokens = len(token_ids)

        # Inflight prefix sharing: register for concurrent KV prefix reuse (SGLang pattern)
        try:
            from .inflight_prefix_sharing import get_inflight_tracker
            _inflight_tracker = get_inflight_tracker()
            _inflight_tracker.register(
                req_id,
                token_ids,
                None,  # KV cache ref not yet available
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
        if max_seq_len > 0 and num_prompt_tokens + max_tokens > max_seq_len:
            excess = num_prompt_tokens + max_tokens - max_seq_len
            if excess > 0 and num_prompt_tokens > excess:
                token_ids = token_ids[excess:]
                num_prompt_tokens = len(token_ids)
                # Record truncation via ContextWindowManager
                if self._context_window_mgr is not None:
                    self._context_window_mgr._stats.truncations_applied += 1
                    self._context_window_mgr._stats.total_tokens_saved += excess
                logger.info(
                    f"Context window truncation: {excess} tokens removed from prompt "
                    f"(max_seq_len={max_seq_len})"
                )

        # ── Wave 42: Budget check (token/time/cost/thinking) ──
        budget = self._budget_manager.register(
            request_id=req_id,
            max_tokens=max_tokens,
            prompt_tokens=num_prompt_tokens,
            thinking_budget=thinking_budget or 0,
        )
        if budget.is_exhausted:
            budget_reason = budget.exhaustion_reason or "budget_exceeded"
            self._budget_manager.remove(req_id)
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
            return req_id

        # ── Wave 42: Request dedup ──
        if self._request_dedup is not None:
            from .request_dedup import RequestDeduplicator
            content_hash = RequestDeduplicator.compute_hash(
                model="", prompt=str(prompt), max_tokens=max_tokens,
                temperature=temperature, top_p=top_p,
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
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker
                    get_inflight_tracker().unregister(req_id)
                except Exception:
                    logger.debug(f"inflight unregister failed in dedup shadow for {req_id}", exc_info=True)
                # Create output collector + finished event so the caller can await
                from .output_collector import RequestOutputCollector, RequestStreamState
                self._output_collectors[req_id] = RequestOutputCollector(aggregate=True)
                self._stream_states[req_id] = RequestStreamState(
                    stream_interval=self.config.stream_interval
                )
                self._finished_events[req_id] = asyncio.Event()
                self._request_timestamps[req_id] = time.monotonic()
                return req_id
            else:
                self._request_dedup.register(req_id, content_hash)
            self._dedup_hashes[req_id] = content_hash

        # ── Wave 42: Lifecycle tracking ──
        self._lifecycle_orchestrator.on_request_added(req_id)

        # ── Wave 46: KV lifecycle admission ──
        try:
            estimated_kv_bytes = num_prompt_tokens * 2048
            block_id = hash(req_id) % (10**9)
            self._kv_lifecycle.admit(
                block_id=block_id,
                size_bytes=estimated_kv_bytes,
                prefix_hash="",
            )
            # Register block in migration manager for temperature-based tier management
            from .kv_migration import KVTier
            self._kv_migration.register_block(block_id, tier=KVTier.HOT, byte_size=estimated_kv_bytes)
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
            estimated_bytes = self._memory_aware_scheduler.estimate_kv_memory(num_prompt_tokens)
            if isinstance(estimated_bytes, int) and estimated_bytes > 0:
                admitted = self._memory_aware_scheduler.reserve_memory(
                    request_id=req_id,
                    num_bytes=estimated_bytes,
                    num_tokens=num_prompt_tokens,
                )
                if not admitted:
                    logger.warning(f"Memory-aware scheduler rejected request {req_id}: estimated {estimated_bytes} bytes")
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
                try:
                    self._lifecycle_orchestrator.on_request_finished(req_id)
                except Exception:
                    logger.debug(f"lifecycle cleanup failed in memguard rejection for {req_id}", exc_info=True)
                try:
                    self._kv_lifecycle.release(hash(req_id) % (10**9))
                except Exception:
                    logger.debug(f"kv_lifecycle release failed in memguard rejection for {req_id}", exc_info=True)
                try:
                    self._kv_migration.unregister_block(hash(req_id) % (10**9))
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
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            priority=priority,
            xtc_probability=kwargs.get('xtc_probability', 0.0),
            xtc_threshold=kwargs.get('xtc_threshold', 0.0),
            reasoning_effort=reasoning_effort,
        )

        request = Request(
            request_id=req_id,
            prompt=prompt if isinstance(prompt, str) else token_ids,
            sampling_params=sampling_params,
            prompt_token_ids=token_ids,
            num_prompt_tokens=num_prompt_tokens,
            enable_thinking=enable_thinking,
        )

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

        # Wake engine loop from idle sleep (event-driven scheduling)
        if self._wake_event is not None:
            self._wake_event.set()

        return req_id

    async def abort_request(self, request_id: str) -> None:
        """Deferred abort (oMLX pattern: enqueued, processed at next step)."""
        from .request import RequestOutput
        self.scheduler.abort_request(request_id)
        # Put error output to wake up any waiting consumer
        collector = self._output_collectors.get(request_id)
        if collector is not None:
            collector.put(RequestOutput(
                request_id=request_id,
                finished=True,
                finish_reason="abort",
                error="Request aborted",
            ))
            collector.put(None)  # sentinel
        self._finalize_request(request_id)

    async def abort_all_requests(self) -> None:
        """Abort all active requests (error recovery)."""
        failed_ids = self.scheduler.fail_all_requests()
        for req_id in failed_ids:
            self._finalize_request(req_id)

    async def stream_outputs(self, request_id: str) -> AsyncIterator[Any]:
        """Stream outputs for a request (oMLX/vLLM pattern).

        Fast path: collector.get_nowait() avoids task switch under load.
        Slow path: await collector.get() for efficient waiting when idle.
        Sentinel (None) signals stream end.
        """
        collector = self._output_collectors.get(request_id)
        if collector is None:
            return

        try:
            while True:
                output = collector.get_nowait()
                if output is not None:
                    yield output
                    if output.finished:
                        break
                    continue

                # No buffered output — check if stream already ended
                if collector._sentinel:
                    break

                # Wait for new output from engine loop
                output = await collector.get()
                if output is None:
                    break
                yield output
                if output.finished:
                    break
        except asyncio.CancelledError:
            raise
        finally:
            self._cleanup_request(request_id)

    async def generate(
        self,
        **kwargs,
    ) -> Any:
        """Non-streaming generate: add request, wait for completion, return result."""
        from .request import RequestOutput
        req_id = await self.add_request(**kwargs)

        # Wait for completion
        event = self._finished_events.get(req_id)
        if event:
            await event.wait()

        # Drain collector
        collector = self._output_collectors.get(req_id)
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
            self._cleanup_request(req_id)

        return result

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
                continue

            try:
                _step_start = time.monotonic()

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
                                    self.config.completion_batch_size = rec_batch
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
                              wait_time=time.monotonic() - getattr(r, '_submit_time', time.monotonic()) if hasattr(r, '_submit_time') else 0.0,
                              context_length=r.num_prompt_tokens)
                        for r in self.scheduler.waiting
                    ] if hasattr(self.scheduler.waiting, '__iter__') else []
                    inversions = self._priority_guard.check_inversion(_running, _waiting)
                    for inv in inversions:
                        low_req = next((r for r in _running if r.request_id == inv.low_request_id), None)
                        high_req = next((r for r in _waiting if r.request_id == inv.high_request_id), None)
                        if low_req and high_req:
                            self._priority_guard.apply_inheritance(low_req, high_req)
                            # Boost effective priority on the actual running request
                            actual = self.scheduler.running.get(inv.low_request_id)
                            if actual and actual.sampling_params:
                                actual.sampling_params.priority = low_req.effective_priority
                            logger.debug(f"Priority inheritance: {inv.low_request_id} boosted to {low_req.effective_priority}")
                except Exception:
                    logger.debug("priority inversion guard failed", exc_info=True)

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
                    # Record allocations in fairness tracker for Jain's index computation
                    for alloc in allocations:
                        self._fairness_tracker.record_allocation(
                            alloc.request_id,
                            alloc.prefill_tokens + alloc.decode_tokens,
                        )
                except Exception:
                    logger.debug("token-level scheduling failed", exc_info=True)

                # Run scheduler step on MLX executor thread
                # §14.1: TBO takes priority when enabled; else C18 overlap; else plain
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

                # Wave 43: CompositionScheduler post_step hooks
                if self._composition_scheduler is not None and scheduler_output.outputs:
                    try:
                        self._composition_scheduler.post_step(self.scheduler, scheduler_output)
                    except Exception:
                        logger.debug("composition post_step failed", exc_info=True)

                # AdaptiveBatchSizer: adjust batch size using ACTUAL step wall time
                try:
                    _step_wall_ms = (time.monotonic() - _step_start) * 1000
                    queue_depth = len(self.scheduler.waiting)
                    from .utils.hardware import get_hardware_info as _ghw
                    _hw = _ghw()
                    import mlx.core as _mx
                    _mem_avail = 1.0 - (_mx.get_active_memory() / max(_hw.total_memory_bytes, 1))
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
            except asyncio.CancelledError:
                logger.info("Engine loop cancelled, failing all in-flight requests")
                failed = self.scheduler.fail_all_requests()
                for req_id in failed:
                    self._finalize_request(req_id)
                raise
            except Exception as e:
                logger.error(f"Scheduler step error: {e}", exc_info=True)
                failed = self.scheduler.fail_all_requests()
                from .request import RequestOutput
                for req_id in failed:
                    collector = self._output_collectors.get(req_id)
                    if collector is not None:
                        collector.put(RequestOutput(
                            request_id=req_id,
                            finished=True,
                            finish_reason="error",
                            error=f"Scheduler step error: {e}",
                        ))
                        collector.put(None)  # sentinel
                    self._finalize_request(req_id)
                await asyncio.sleep(0.1)
                continue

            # Distribute outputs to per-request collectors
            # Iterate live dict (abort may insert between steps, we need to see it)
            active_ids = list(self._output_collectors.keys())

            for req_output in scheduler_output.outputs:
                rid = req_output.request_id
                collector = self._output_collectors.get(rid)
                if collector is None:
                    continue

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
                    stream_state = self._stream_states.get(rid)
                    if stream_state and stream_state.should_send(
                        req_output.completion_tokens, req_output.finished
                    ):
                        collector.put(req_output)
                        stream_state.mark_sent(req_output.completion_tokens)

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
                                        new_token_ids=req_output.new_token_ids,
                                        new_text=req_output.new_text,
                                        output_token_ids=req_output.output_token_ids,
                                        output_text=req_output.output_text,
                                        finished=True,
                                        finish_reason=req_output.finish_reason,
                                        prompt_tokens=req_output.prompt_tokens,
                                        completion_tokens=req_output.completion_tokens,
                                        logprobs=req_output.logprobs,
                                        current_state=req_output.current_state,
                                    )
                                    shadow_collector.put(shadow_output)
                                    shadow_collector.put(None)  # sentinel
                                    self._finalize_request(shadow_id)
                    # Finalize: release ALL resources for this request
                    self._finalize_request(rid)

            # Update adaptive batch scheduler metrics
            if scheduler_output.outputs:
                # ── Wave 42: Lifecycle decode tracking for active requests ──
                for req_output in scheduler_output.outputs:
                    rid = req_output.request_id
                    if not req_output.finished and req_output.completion_tokens > 0:
                        state = self._lifecycle_orchestrator.get_state(rid)
                        if state is not None and state.phase.name in ("PREFILLING",):
                            self._lifecycle_orchestrator.on_decode_start(rid)
                        # Budget consumption
                        self._budget_manager.consume(rid, tokens=1)
                        # Sliding window tracking
                        if self._sliding_window_mgr is not None:
                            try:
                                total_pos = req_output.prompt_tokens + req_output.completion_tokens
                                evicted = self._sliding_window_mgr.on_new_token(
                                    token_position=total_pos,
                                    request_id=rid,
                                )
                                # Trim real KV cache for sliding window models
                                if evicted:
                                    req = self.scheduler.running.get(rid)
                                    if req is not None and req.prompt_cache is not None:
                                        self._sliding_window_mgr.trim_kv_cache(
                                            req.prompt_cache, request_id=rid,
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
                    _step_wall_ms = (time.monotonic() - _step_start) * 1000
                    _tokens_gen = sum(
                        o.completion_tokens for o in scheduler_output.outputs if o.completion_tokens
                    )
                    _throughput = _tokens_gen / (_step_wall_ms / 1000) if _step_wall_ms > 0 else 0.0
                    from .auto_tuner import StepMetrics
                    step_metrics = StepMetrics(
                        batch_size=batch_size,
                        tokens_generated=_tokens_gen,
                        wall_time_ms=_step_wall_ms,
                        throughput_tok_s=_throughput,
                    )
                    self._profiler.record_step(step_metrics)
                    # Auto-tune every 100 steps
                    if self._profiler._total_steps % 100 == 0:
                        tuning_decisions = self._auto_tuner.auto_tune()
                        if tuning_decisions:
                            logger.debug(f"AutoTuner applied: {[d.param_name for d in tuning_decisions]}")
                    # SLO checks
                    self._slo_monitor.check_slo("ttft", step_metrics.ttft_ms)
                    self._slo_monitor.check_slo("itl", step_metrics.itl_ms)
                    self._slo_monitor.check_slo("throughput", step_metrics.throughput_tok_s)
                    # Fairness tracker
                    for req_output in scheduler_output.outputs:
                        self._fairness_tracker.record_allocation(
                            req_output.request_id,
                            tokens_allocated=req_output.completion_tokens or 1,
                        )
                except Exception:
                    logger.debug("profiler/auto-tuner failed", exc_info=True)

                # ── Per-request generation timeout enforcement ──
                try:
                    timeout_s = self.config.request_timeout_seconds
                    if timeout_s > 0:
                        now = time.monotonic()
                        for rid in list(self._request_timestamps.keys()):
                            start = self._request_timestamps[rid]
                            if (now - start) > timeout_s:
                                logger.warning(f"Request {rid} timed out ({now - start:.0f}s > {timeout_s}s)")
                                self.scheduler.abort_request(rid)
                                collector = self._output_collectors.get(rid)
                                if collector is not None:
                                    from .request import RequestOutput
                                    collector.put(RequestOutput(
                                        request_id=rid,
                                        finished=True,
                                        finish_reason="timeout",
                                        error=f"Request exceeded timeout ({timeout_s}s)",
                                    ))
                                    collector.put(None)
                                self._signal_finished(rid)
                                self._request_timestamps.pop(rid, None)
                except Exception:
                    logger.debug("timeout enforcement failed", exc_info=True)

                try:
                    import mlx.core as mx
                    active_mem = mx.get_active_memory()
                    from .utils.hardware import get_hardware_info
                    hw = get_hardware_info()
                    mem_usage = active_mem / max(hw.total_memory_bytes, 1)
                    self._adaptive_batch.update_metrics(
                        latency_ms=0.0,  # Latency tracked per-request
                        memory_usage=mem_usage,
                        batch_size=len(scheduler_output.outputs),
                    )
                    # KV compression under memory pressure: compress old blocks
                    # instead of outright eviction when usage > 85%
                    if mem_usage > 0.85 and self._kv_compressor is not None:
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

            await asyncio.sleep(0)
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
        # Clean up prefix hash tracking
        self._kv_prefix_hashes.pop(request_id, None)

    def _finalize_request(self, request_id: str) -> None:
        """Release ALL per-request resources (engine-side + consumer-side).

        Called from:
        - Engine loop finish path (normal completion)
        - abort_request() / abort_all_requests()
        - Engine loop exception handlers
        - Consumer-side _cleanup_request() (idempotent)

        This is idempotent: safe to call multiple times.
        """
        _block_id = hash(request_id) % (10**9)
        # Inflight prefix sharing
        try:
            from .inflight_prefix_sharing import get_inflight_tracker
            get_inflight_tracker().unregister(request_id)
        except Exception:
            logger.debug("inflight prefix unregister failed", exc_info=True)
        # Release LoRA adapter
        lora_id = self._request_lora_adapters.pop(request_id, None)
        if lora_id:
            try:
                from .lora_manager import get_lora_manager
                lora_mgr = get_lora_manager()
                if lora_mgr is not None:
                    lora_mgr.unload_adapter(lora_id)
            except Exception:
                logger.debug("LoRA cleanup failed", exc_info=True)
        # Lifecycle + budget + memory + KV lifecycle
        self._lifecycle_orchestrator.on_request_finished(request_id)
        self._budget_manager.remove(request_id)
        self._memory_aware_scheduler.release_memory(request_id)
        try:
            self._kv_lifecycle.release(_block_id)
        except Exception:
            logger.debug("kv_lifecycle release failed", exc_info=True)
        # KV migration
        try:
            self._kv_migration.unregister_block(_block_id)
        except Exception:
            logger.debug("kv_migration unregister failed", exc_info=True)
        # Dedup
        if self._request_dedup is not None:
            content_hash = self._dedup_hashes.pop(request_id, None)
            self._dedup_shadows.pop(request_id, None)
            if content_hash:
                self._request_dedup.complete(content_hash)
        # Checkpoint
        if self._checkpoint_mgr is not None:
            self._checkpoint_mgr.delete(request_id)
        # Sliding window
        if self._sliding_window_mgr is not None:
            try:
                self._sliding_window_mgr.remove_request(request_id)
            except Exception:
                logger.debug("sliding window cleanup failed", exc_info=True)
        # Consumer-side state
        self._output_collectors.pop(request_id, None)
        self._stream_states.pop(request_id, None)
        self._finished_events.pop(request_id, None)
        self._request_timestamps.pop(request_id, None)
        self._kv_prefix_hashes.pop(request_id, None)
        # Remove from scheduler
        self.scheduler.remove_finished_request(request_id)

    def _cleanup_request(self, request_id: str) -> None:
        """Remove per-request state (consumer-side entry point).

        Delegates to _finalize_request which is idempotent.
        """
        self._finalize_request(request_id)

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
                clean = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in messages
                ]
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
