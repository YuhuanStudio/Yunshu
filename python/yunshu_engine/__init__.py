"""Yunshu L4 Engine — inference (LLM / VLM / audio / image)."""

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
        "ConstrainedSampler": ".json_schema",
        "JsonSchemaConstraint": ".json_schema",
        "JsonState": ".json_schema",
        "apply_json_constraint": ".json_schema",
        "make_constrained_sampler": ".json_schema",
        "ModelManager": ".model_manager",
        "ModelType": ".model_manager",
        "Request": ".request",
        "RequestStatus": ".request",
        "SamplingParams": ".request",
        "SpecHeadInfo": ".speculative_decoder",
        "SpecDecodingConfig": ".speculative_decoder",
        "auto_configure_speculative": ".speculative_decoder",
        "detect_spec_heads": ".speculative_decoder",
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
        # Mamba / Hybrid KV Cache
        # Scheduler Mixins
        # Forward Batch (3-level batch)
        # Request Lifecycle (state machine + adaptive concurrency)
        # Model Preprocessors
        "PreprocessorRegistry": ".model_preprocessor",
        "PreprocessedInput": ".model_preprocessor",
        # KV Lifecycle (unified tier management)
        # Inference Budget (token/time/cost enforcement)
    }
    if name in _lazy:
        import importlib

        module = importlib.import_module(_lazy[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
