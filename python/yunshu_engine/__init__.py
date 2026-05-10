"""Yunshu L4 Engine — continuous batching inference."""

from .batched_engine import BatchedEngine, GenerationOutput
from .engine import Engine, EngineConfig, RequestOutput, RequestState

__all__ = [
    "BatchedEngine",
    "Engine",
    "EngineConfig",
    "GenerationOutput",
    "RequestOutput",
    "RequestState",
]


def __getattr__(name):
    """Lazy imports for heavy modules — only loaded when accessed."""
    _lazy = {
        "AsyncEngineCore": ".engine_core",
        "EngineCore": ".engine_core",
        "EngineCoreConfig": ".engine_core",
        "ExternalPrefiller": ".external_prefill",
        "PrefillResult": ".external_prefill",
        "PrefillAbortedError": ".external_prefill",
        "check_abort": ".external_prefill",
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
    }
    if name in _lazy:
        import importlib
        module = importlib.import_module(_lazy[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
