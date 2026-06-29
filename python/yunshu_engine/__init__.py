"""Yunshu L4 Engine — continuous batching inference."""

__all__ = [
    "BatchedEngine",
    "EngineConfig",
    "GenerationOutput",
    "RequestOutput",
]


def __getattr__(name):
    """Lazy imports for heavy modules — only loaded when accessed."""
    _lazy = {
        "BatchedEngine": ".batched_engine",
        "EngineConfig": ".types",
        "GenerationOutput": ".batched_engine",
        "RequestOutput": ".request",
        "AsyncEngineCore": ".engine_core",
        "EngineCore": ".engine_core",
        "EngineCoreConfig": ".engine_core",
        "ConstrainedSampler": ".json_schema",
        "JsonSchemaConstraint": ".json_schema",
        "JsonState": ".json_schema",
        "apply_json_constraint": ".json_schema",
        "make_constrained_sampler": ".json_schema",
        "ModelManager": ".model_manager",
        "ModelType": ".model_manager",
        "RequestOutputCollector": ".output_collector",
        "RequestStreamState": ".output_collector",
        "Request": ".request",
        "RequestOutputNew": ".request",
        "RequestStatus": ".request",
        "SamplingParams": ".request",
        "Scheduler": ".scheduler",
        "SchedulerConfig": ".scheduler",
        "SchedulerOutput": ".scheduler",
        "SpecHeadInfo": ".speculative_decoder",
        "SpecDecodingConfig": ".speculative_decoder",
        "auto_configure_speculative": ".speculative_decoder",
        "detect_spec_heads": ".speculative_decoder",
        "EncoderCacheManager": ".encoder_cache",
        "EncoderCacheEntry": ".encoder_cache",
        "GrammarBitmaskEngine": ".grammar_bitmask",
        "BitmaskApplicator": ".grammar_bitmask",
        "BitmaskConstrainedSampler": ".grammar_bitmask",
        "TokenStringTable": ".grammar_bitmask",
        "build_bitmask_engine": ".grammar_bitmask",
        "is_bitmask_enabled": ".grammar_bitmask",
        # Streaming Optimizer (GPU/CPU overlap)
        "TokenPipeline": ".streaming_optimizer",
        "PipelineConfig": ".streaming_optimizer",
        "PipelineStage": ".streaming_optimizer",
        "PipelineToken": ".streaming_optimizer",
        "PrefetchSampler": ".streaming_optimizer",
        "SamplingPlan": ".streaming_optimizer",
        "BatchedDetokenizer": ".streaming_optimizer",
        "StreamingBackpressureController": ".streaming_optimizer",
        "BackpressureConfig": ".streaming_optimizer",
        # Diffusion Infrastructure (scheduler + LoRA offloader + distributed coordinator)
        "DiffusionScheduler": ".diffusion_infra",
        "SchedulerType": ".diffusion_infra",
        "NoiseScheduleType": ".diffusion_infra",
        "DiffusionStep": ".diffusion_infra",
        "DiffusionLoRAOffloader": ".diffusion_infra",
        "LoRAAdapter": ".diffusion_infra",
        "MemoryBudget": ".diffusion_infra",
        "DistributedDiffusionCoordinator": ".diffusion_infra",
        "NodeAssignment": ".diffusion_infra",
        "SyncCheckpoint": ".diffusion_infra",
        "StepAssignmentStrategy": ".diffusion_infra",
        # Checkpoint / Restore (fault recovery)
        "InferenceCheckpoint": ".checkpoint",
        "InferenceState": ".checkpoint",
        "AutoCheckpointPolicy": ".checkpoint",
        "FaultRecoveryManager": ".checkpoint",
        "RecoveryStrategy": ".checkpoint",
        "RecoveryResult": ".checkpoint",
        "ProgressEstimator": ".checkpoint",
        "ProgressInfo": ".checkpoint",
        # Mamba / Hybrid KV Cache
        "CacheBlockType": ".mamba_cache",
        "HybridKVCache": ".mamba_cache",
        "MambaSSMState": ".mamba_cache",
        "BlockAlignedCacheSplitter": ".mamba_cache",
        "LayerGroup": ".mamba_cache",
        "CachePoolStats": ".mamba_cache",
        # Scheduler Mixins
        "SchedulerMixin": ".scheduler_mixins",
        "MetricsMixin": ".scheduler_mixins",
        "ProfilingMixin": ".scheduler_mixins",
        "SpecDecodeMixin": ".scheduler_mixins",
        "MemoryPressureMixin": ".scheduler_mixins",
        "CompositionScheduler": ".scheduler_mixins",
        # Forward Batch (3-level batch)
        "RequestSlot": ".forward_batch",
        "ScheduleBatch": ".forward_batch",
        "ForwardBatch": ".forward_batch",
        "BatchResult": ".forward_batch",
        "BatchComposer": ".forward_batch",
        # Request Lifecycle (state machine + adaptive concurrency)
        "RequestPhase": ".request_lifecycle",
        "RequestLifecycleState": ".request_lifecycle",
        "AdaptiveConcurrencyController": ".request_lifecycle",
        "RequestLifecycleOrchestrator": ".request_lifecycle",
        # Model Preprocessors
        "PreprocessorRegistry": ".model_preprocessor",
        "PreprocessedInput": ".model_preprocessor",
        # KV Lifecycle (unified tier management)
        "KVLifecycleManager": ".kv_lifecycle",
        "KVBlock": ".kv_lifecycle",
        "KVTier": ".kv_lifecycle",
        "KVTierConfig": ".kv_lifecycle",
        "CacheWarmingPredictor": ".kv_lifecycle",
        "KVCompactionScheduler": ".kv_lifecycle",
        # Inference Budget (token/time/cost enforcement)
        "InferenceBudget": ".inference_budget",
        "InferenceBudgetManager": ".inference_budget",
        # Request Deduplication
        "RequestDeduplicator": ".request_dedup",
        "DeduplicationEntry": ".request_dedup",
    }
    if name in _lazy:
        import importlib

        module = importlib.import_module(_lazy[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
