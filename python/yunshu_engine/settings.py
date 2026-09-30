"""Every ``YUNSHU_*`` setting, in one registry.

Code never reads ``YUNSHU_*`` environment variables directly: it asks this
module (``settings.get("YUNSHU_MTP")`` or a typed getter), so each setting has
one type, one default, one parsing rule and one line of documentation, and
``docs/CONFIGURATION.md`` is generated from the same table
(``scripts/gen_config_docs.py``).

Sources, highest precedence first:

1. overrides set in-process (``yunshu serve --set KEY=VALUE`` and the typed
   ``serve`` flags),
2. the process environment,
3. a config file (``yunshu serve --config PATH`` or ``YUNSHU_CONFIG``, else the
   user file ``~/.yunshu/config.toml`` that ``yunshu config set`` writes; TOML
   with flat ``KEY = value`` entries, optionally grouped under tables),
4. the registry default.

Values are resolved on every access, so a setting changed in the environment
at runtime (tests, the sleep/wake router) takes effect immediately; there is no
cached copy to go stale.

Stability:

- ``stable`` — a supported deployment setting.
- ``experimental`` — temporary. Each names the measurement that will decide
  it (``decide``) and when it was added; once measured, the winner becomes the
  default and the flag and the losing code path are deleted. At most
  ``MAX_EXPERIMENTAL`` may exist at a time (enforced by a unit test).
- ``internal`` — debugging aids only; kept tiny.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_EXPERIMENTAL = 8

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class SettingError(ValueError):
    """A setting has a value its type cannot parse."""


@dataclass(frozen=True)
class Setting:
    name: str
    type: str  # bool | int | float | gb | str | enum | path | list | json
    default: Any
    description: str
    category: str
    stability: str = "stable"
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    decide: str = ""  # experimental only: what measurement settles it
    added: str = ""  # experimental only: date added
    empty_is_value: bool = False  # "" is a real value, not "unset"
    secret: bool = False


REGISTRY: dict[str, Setting] = {}

CATEGORIES = (
    "model",
    "server",
    "auth",
    "memory",
    "cache",
    "vlm-runner",
    "speculative",
    "text-engine",
    "kernels",
    "voice",
    "image-video",
    "embeddings",
    "mcp",
    "observability",
    "cli",
)


def _add(
    name: str, type_: str, default: Any, description: str, category: str, **kw: Any
) -> None:
    if name in REGISTRY:
        raise RuntimeError(f"duplicate setting {name}")
    if category not in CATEGORIES:
        raise RuntimeError(f"{name}: unknown category {category}")
    REGISTRY[name] = Setting(name, type_, default, description, category, **kw)


# One line per setting, kept as a table.
# fmt: off
# ── model ──────────────────────────────────────────────────────────────
_add("YUNSHU_MODEL", "path", None, "Model path or Hugging Face id served in single-model mode; every requested model name maps to it.", "model")
_add("YUNSHU_MULTI_MODEL", "bool", False, "Multi-model mode: discover models under YUNSHU_MODELS_DIR and load them on demand. Ignored when YUNSHU_MODEL is set.", "model")
_add("YUNSHU_MODELS_DIR", "path", None, "Directory of model folders for multi-model mode (each folder name is a model id). Unset: ~/.yunshu/models (also where `yunshu pull` downloads to).", "model")
_add("YUNSHU_HF_CACHE_MODELS", "bool", True, "Multi-model mode: also offer models already in the Hugging Face cache (the models directory wins on name clashes). `yunshu serve -m org/name` uses a cached copy either way.", "model")
_add("YUNSHU_MODEL_TTL_SECONDS", "float", None, "Multi-model mode: unload a model idle for this many seconds. Unset: never.", "model", minimum=0.0)
_add("YUNSHU_ALLOW_AUTO_LOAD", "bool", False, "Multi-model mode: let audio requests load a model that is not loaded yet (otherwise they are rejected).", "model")
_add("YUNSHU_MAX_LORAS", "int", 4, "Maximum LoRA adapters kept loaded (text models).", "model", minimum=1)
_add("YUNSHU_TRUST_REMOTE_CODE", "bool", False, "Allow Hugging Face tokenizers/processors that need trust_remote_code.", "model")
_add("YUNSHU_WARM_PROMPTS", "str", None, "Prompts prefilled at startup to warm the prefix cache: '||'-separated text or file paths.", "model")

# ── server ─────────────────────────────────────────────────────────────
_add("YUNSHU_CONFIG", "path", None, "TOML config file with YUNSHU_* settings (lower precedence than the environment).", "server")
_add("YUNSHU_MAX_CONCURRENT", "int", None, "Cap on concurrently admitted requests. Unset: adaptive (starts at 8).", "server", minimum=1)
_add("YUNSHU_COMPLETION_BATCH_SIZE", "int", 32, "Text engine: maximum sequences decoded together.", "server", minimum=1)
_add("YUNSHU_DEFAULT_MAX_TOKENS", "int", 512, "Completion length when a request omits max_tokens.", "server", minimum=1)
_add("YUNSHU_MAX_PREFILL_TOKENS", "int", 0, "Reject prompts longer than this many tokens (0: no limit beyond the model context).", "server", minimum=0)
_add("YUNSHU_STARTUP_TIMEOUT", "float", 300.0, "Seconds to wait for the model to load before startup fails.", "server", minimum=0.0)
_add("YUNSHU_DRAIN_TIMEOUT", "float", 30.0, "Seconds to wait for in-flight requests on shutdown.", "server", minimum=0.0)
_add("YUNSHU_KEEP_ALIVE_TIMEOUT", "int", 5, "Seconds an idle HTTP connection stays open.", "server", minimum=0)
_add("YUNSHU_UDS", "path", None, "Serve on this Unix domain socket instead of a TCP port (same app; curl --unix-socket, httpx uds=).", "server")
_add("YUNSHU_WS_MAX_INFLIGHT", "int", 16, "Text WebSocket (/v1/stream, wss /v1/responses): maximum concurrent requests per connection.", "server", minimum=1)
_add("YUNSHU_WS_PING_INTERVAL", "float", 15.0, "Text WebSocket: seconds between server heartbeat pings (0 disables).", "server", minimum=0.0)
_add("YUNSHU_WS_SEND_QUEUE", "int", 256, "Text WebSocket: outbound events buffered per connection before generation is paused (backpressure).", "server", minimum=1)
_add("YUNSHU_MAX_REQUEST_SIZE", "int", 10 * 1024 * 1024, "Maximum request body size in bytes.", "server", minimum=1)
_add("YUNSHU_PROGRESS_INTERVAL_S", "float", 2.0, "Streaming chat/completions: seconds between `: yunshu-progress` SSE comments (queue / prefill progress, ETA) before the first token; 0 turns them off. Strict SSE clients ignore comment lines.", "server", minimum=0.0)
_add("YUNSHU_SLOW_REQUEST_THRESHOLD", "float", 30.0, "Log a warning for requests slower than this many seconds.", "server", minimum=0.0)
_add("YUNSHU_CORS_ORIGINS", "str", "http://localhost:3000,http://localhost:8000", "Comma-separated allowed CORS origins ('*' for any). Also checked for the Realtime WebSocket Origin header.", "server")
_add("YUNSHU_RESPONSE_CACHE", "bool", False, "Cache identical non-streaming responses in memory.", "server")
_add("YUNSHU_BATCH_MAX_ITEMS", "int", 500, "Batch API: maximum requests per batch.", "server", minimum=1)
_add("YUNSHU_BATCH_TIMEOUT", "float", 300.0, "Batch API: default per-batch timeout in seconds.", "server", minimum=0.0)
_add("YUNSHU_ALLOW_LOCAL_FILES", "bool", False, "Allow requests to reference any local file path (default: only under YUNSHU_MEDIA_DIR).", "server")
_add("YUNSHU_MEDIA_DIR", "path", None, "Directory local media paths must live under. Unset: $TMPDIR/yunshu_media.", "server")

# ── auth ───────────────────────────────────────────────────────────────
_add("YUNSHU_AUTH_TOKEN", "str", None, "Bearer token. When set, every request except health/version/docs needs it; unset: inference is open and operational endpoints are denied.", "auth", secret=True)
_add("YUNSHU_AUTH_DISABLED", "bool", False, "Disable auth entirely (operational endpoints open too). Local development only.", "auth")
_add("YUNSHU_DEBUG_ROUTES", "bool", False, "Mount the /debug/* diagnostic routes (engine, system, kv-cache, spec-decode, ...). They need the auth token or YUNSHU_AUTH_DISABLED. /metrics is always mounted.", "observability")
_add("YUNSHU_ACTOR_IDENTITY", "str", "owner", "Identity recorded for authenticated requests in the audit log.", "auth")
_add("YUNSHU_RATE_LIMIT_RPM", "int", 0, "Per-client request rate limit in requests per minute; 0 (default) turns rate limiting off. A local single-user engine has no need for it; set it when the server is exposed to other machines.", "auth", minimum=0)
_add("YUNSHU_TRUSTED_PROXIES", "list", (), "Comma-separated proxy IPs whose X-Forwarded-For header is trusted.", "auth")

# ── memory ─────────────────────────────────────────────────────────────
_add("YUNSHU_MAX_MEMORY_GB", "gb", None, "Multi-model mode memory ceiling in GiB, e.g. '48' or '48GB'; 'disabled' turns the enforcer off. Unset: 80% of unified memory.", "memory")
_add("YUNSHU_PREFILL_STEP_SIZE", "int", 2048, "Text engine: prompt tokens per prefill forward pass; lower it to cap the prefill activation peak on small-memory machines.", "memory", minimum=1)
_add("YUNSHU_MEM_PRESSURE_THRESHOLD", "float", 85.0, "Text engine: evict prefix-cache entries above this memory use (percent, or a fraction <= 1).", "memory", minimum=0.0)

# ── cache (text engine) ────────────────────────────────────────────────
_add("YUNSHU_PREFIX_MAX_ENTRIES", "int", 64, "Text engine: prefix KV cache entries.", "cache", minimum=1)
_add("YUNSHU_PREFIX_HOT_LIMIT", "int", 0, "Text engine: keep only this many prefix KV entries full precision and store older ones 4-bit in RAM (lossy on reuse; memory vs quality). 0 = every entry full precision.", "cache", minimum=0)
_add("YUNSHU_SSD_CACHE", "bool", False, "Text engine: persist prefix KV to SSD.", "cache")
_add("YUNSHU_SSD_CACHE_DIR", "path", "~/.cache/yunshu/kv-ssd", "Text engine: SSD prefix-cache directory.", "cache")
_add("YUNSHU_SSD_CACHE_PRECISION", "enum", "native", "Text engine: SSD prefix-cache storage precision: 'native' (KV and recurrent state stored bit-exact; lossless) or 'int8' (per-tensor int8, about half the disk bytes of bf16; lossy on reuse; memory vs quality).", "cache", choices=("native", "int8"))
_add("YUNSHU_SSD_CACHE_MAX_GB", "float", 10.0, "Text engine: SSD prefix-cache size cap in GiB.", "cache", minimum=0.0)
_add("YUNSHU_KV_QUANT_BITS", "enum", "off", "Text engine KV cache quantization (lossy; memory vs quality): 'off' (lossless), 'auto' (8-bit once the KV cache would exceed ~2 GiB), or 2/3/4/8 bits always.", "cache", choices=("auto", "off", "2", "3", "4", "8"))

# ── VLM runner ─────────────────────────────────────────────────────────
_add("YUNSHU_VLM_APC_MEMORY_GB", "float", 8.0, "VLM runner prefix cache (APC) RAM budget in GiB; 0 disables the prefix cache.", "vlm-runner", minimum=0.0)
_add("YUNSHU_VLM_APC_DISK_DIR", "path", None, "Directory for the APC SSD tier; evicted prefixes reload from disk instead of re-prefilling.", "vlm-runner")
_add("YUNSHU_KV_PRECISION", "enum", "bf16", "KV cache precision of the Qwen3.5-family runner's shared decode batch: 'bf16' (lossless) or 'int8' (int8 codes + one fp16 scale per 32-dim group: ~0.53x the KV memory and read bandwidth for a small attention error; memory vs quality). A lone request and the speculative lane stay bf16. Applies to models with Qwen3.5-family attention (the ragged KV layout).", "vlm-runner", choices=("bf16", "int8"))
_add("YUNSHU_VLM_APC_DISK_GB", "float", 64.0, "Size cap of the APC SSD tier in GiB.", "vlm-runner", minimum=0.0)
_add("YUNSHU_VLM_MAX_IMAGE_BYTES", "int", 25 * 1024 * 1024, "Largest image a request may reference by URL, in bytes.", "vlm-runner", minimum=1)
_add("YUNSHU_VLM_INSECURE_SSL", "bool", False, "Retry image downloads without TLS verification when verification fails.", "vlm-runner")

# ── speculative decoding ───────────────────────────────────────────────
_add("YUNSHU_MTP", "bool", True, "Qwen3.5-family VLMs: draft with the checkpoint's MTP head (batch-invariant, spec on == spec off).", "speculative")
_add("YUNSHU_VLM_DRAFT", "path", None, "Qwen3.5-family VLMs: speculative draft override. A DFlash drafter directory; 'mtp' forces the checkpoint MTP head; 'off' disables drafting. Unset: a DFlash2 drafter matching the model is used automatically when it is in the models dir or the Hugging Face cache, else the MTP head (batch-invariant verify, spec on == spec off).", "speculative")
_add("YUNSHU_MTP_BLOCK_SIZE", "int", None, "Draft block size (DFlash: the ceiling its acceptance-driven depth stays under). Unset: 6 for MTP, the drafter's trained block for DFlash.", "speculative", minimum=2)
_add("YUNSHU_SPEC_TREE", "enum", "off", "Qwen3.5-family single-request speculative lane: 'tree' verifies a draft tree of up to 8 rows (MTP head or DFlash2 lattice, each row with single-step arithmetic); 'off' keeps upstream's rounds. Greedy output equals plain decode in both. Paused: no measured win, long-context attention overhead unresolved.", "speculative", choices=("off", "tree"), stability="experimental", decide="27B server, off vs tree: bench_context_batch novel_en and the default corpus at 1K/8K/32K/131K; delete the tree code unless it wins", added="2026-09-29")
_add("YUNSHU_NGRAM_DEFAULT", "bool", False, "Text models: lossless n-gram speculation on greedy requests by default (per-request spec_decode also enables it). Wins on repetitive output.", "speculative")
_add("YUNSHU_SPEC_PROPOSER", "enum", "ngram", "Text models: speculative proposer family for n-gram speculation.", "speculative", choices=("ngram", "suffix"))
_add("YUNSHU_GEMMA4_ASSISTANT", "path", None, "Text Gemma-4 models: assistant drafter directory (KV-shared speculative drafter).", "speculative")

# ── text engine ────────────────────────────────────────────────────────
_add("YUNSHU_GPU_SAMPLER", "bool", False, "Text models: on-GPU Gumbel-max sampling (no per-token GPU->CPU sync).", "text-engine")
_add("YUNSHU_JUMP_FORWARD", "bool", False, "Text models: emit grammar-forced structural tokens of JSON-schema output without a forward pass.", "text-engine")
_add("YUNSHU_GRAMMAR_BITMASK", "bool", False, "Constrained decoding with the xgrammar-style bitmask engine instead of the allowlist sampler.", "text-engine")
_add("YUNSHU_TOOL_GRAMMAR", "bool", True, "Tool-call constrained decoding (structural tags): free text until the model emits the tool-call start marker, then the call body is masked to the exact call grammar of this request's tools (tool name, that tool's parameter keys, schema-typed values, correct closing); a forced tool_choice starts constrained. Lossless for a call the model would have written validly. Off: decode tool calls unconstrained and repair them after the fact.", "text-engine")
_add("YUNSHU_QUANT_MODE", "enum", "", "Quantize weights in memory at load (lossy; memory vs quality): mxfp4, nvfp4, mxfp8 or affine ('' keeps the checkpoint).", "text-engine", choices=("", "mxfp4", "nvfp4", "mxfp8", "affine"))
_add("YUNSHU_QUANT_CONFIG", "str", None, "Bits/group for affine in-memory quantization: JSON ({\"bits\":4,\"group_size\":64}) or 'bits' / 'bits,group'.", "text-engine")

# ── kernels / experiments ──────────────────────────────────────────────
_add("YUNSHU_ROUND_PREFILL_CHUNK", "int", 512, "Round driver: prompt tokens per prefill span. A decoding request only steps between prefill forwards, so smaller spans keep it running next to a long prompt (Qwen3.8-27B, M5 Max, one MTP row beside an 8K prompt: 512 -> 6 tok/s, 128 -> 24 tok/s, ~20% lower prefill speed). Spans are fixed per prompt, so output stays independent of what else is running; prompts prefilled with different spans are each self-consistent but not bit-identical to each other.", "vlm-runner", minimum=16)
_add("YUNSHU_ROUND_DRIVER", "bool", False, "Dense Qwen3.5-family VLMs: Yunshu's round driver serves text requests (packed forwards over every decoding row's window, alternating with prefill steps that batch several prompts' fixed chunks; row-invariant lane projections; MTP drafts for every greedy row with cost-aware per-row depth; see docs/guides/ROUND_DRIVER.md). Off: upstream BatchGenerator shared batch + single-request speculative lane. APC prefix reuse (exact hybrid checkpoints) runs in the driver; image prompts, int8 KV and MoE stay on the upstream path either way.", "vlm-runner", stability="experimental", decide="27B idle GPU, off vs on: sweep_round_driver.py parity, bench_batch_spec-style rows 1/2/4/8, probe_concurrency, bench_engine_matrix, bench_context_batch b1-b8 + 32K/131K, bench_mixed_load 16K/32K, MMLU-Pro 300 b8 (scripts/research/validate_round_driver.sh); if it wins it becomes the path (with APC / images moved) and this flag, the upstream shared batch and the spec lane are deleted", added="2026-09-29")
_add("YUNSHU_MTP_ROW_EXACT", "bool", False, "Qwen3.5-family runner: oMLX row-exact verify (verify rows bit-identical to one-row decode) instead of batch-invariant kernels.", "kernels", stability="experimental", decide="sweep_mtp_depth parity at long contexts vs decode tok/s (currently 30-50% slower than batch-invariant)", added="2026-09-28")
_add("YUNSHU_ENGINE_LOOP", "bool", False, "Text models: EngineCore continuous-batching loop instead of the single-request fast path.", "text-engine", stability="experimental", decide="unify text-only models onto the batch runner vs keeping this loop (concurrency probe on a text model)", added="2026-06-30")
_add("YUNSHU_OVERLAP", "enum", "", "Text engine loop: overlap CPU and GPU work ('cpu_gpu') or split a batch into two overlapping halves ('two_batch').", "text-engine", stability="experimental", choices=("", "cpu_gpu", "two_batch"), decide="concurrency probe tok/s on a text model with the engine loop; deleted with the loop if text models move to the runner", added="2026-06-30")
_add("YUNSHU_SPEC_UNVERIFIED", "enum", "", "Text models: speculative routes without a lossless-output test: 'eagle' (needs YUNSHU_DRAFT_MODEL), 'mtp' (built-in MTP decoder), 'mlxvlm_mtp' (mlx-vlm MTP backend).", "speculative", stability="experimental", choices=("", "eagle", "mtp", "mlxvlm_mtp"), decide="lossless test against plain greedy + tok/s on a text MTP/EAGLE model", added="2026-09-28")
_add("YUNSHU_DRAFT_MODEL", "path", None, "Draft model for YUNSHU_SPEC_UNVERIFIED=eagle.", "speculative", stability="experimental", decide="goes with YUNSHU_SPEC_UNVERIFIED", added="2026-09-28")

# ── voice ──────────────────────────────────────────────────────────────
_add("YUNSHU_REALTIME_OMNI", "enum", "auto", "Native Qwen3-Omni speech on the Realtime socket: 'auto' when a speakable model is served, 'on', or 'off' (ASR -> LLM -> TTS cascade).", "voice", choices=("auto", "on", "off", "1", "0", "true", "false", "yes", "no"))
_add("YUNSHU_OMNI_MODEL", "path", None, "Voice path model when it differs from the served model (an omni served model is reused automatically).", "voice")
_add("YUNSHU_OMNI_PRELOAD", "bool", True, "Warm the omni model at boot so the first voice request is not cold.", "voice")
_add("YUNSHU_OMNI_THINKER_MAX", "int", 256, "Maximum tokens the omni Thinker writes per voice turn (the spoken reply's length cap).", "voice", minimum=1)
_add("YUNSHU_OMNI_PERSONA", "str", None, "Realtime system persona when the request has none; empty string disables it. Unset: a built-in concise spoken style.", "voice", empty_is_value=True)
_add("YUNSHU_REALTIME_SILENCE_MS", "int", 500, "Server VAD: pause before the model answers, in ms.", "voice", minimum=0)
_add("YUNSHU_REALTIME_BARGE_IN_MS", "int", 120, "Sustained speech needed to interrupt the model mid-reply, in ms.", "voice", minimum=0)
_add("YUNSHU_REALTIME_VAD_THRESHOLD", "float", 0.5, "Server VAD speech-detection threshold.", "voice", minimum=0.0)
_add("YUNSHU_REALTIME_PREFIX_PADDING_MS", "int", 300, "Server VAD: audio kept before detected speech, in ms.", "voice", minimum=0)
_add("YUNSHU_REALTIME_VAD", "enum", "energy", "Server VAD implementation: 'energy' or 'silero'.", "voice", choices=("energy", "silero"))
_add("YUNSHU_REALTIME_VAD_MODEL", "str", "mlx-community/silero-vad", "Silero VAD model id when YUNSHU_REALTIME_VAD=silero.", "voice")
_add("YUNSHU_REALTIME_MAX_INPUT_AUDIO_BYTES", "int", 10 * 1024 * 1024, "Realtime: largest buffered input audio, in bytes.", "voice", minimum=1)
_add("YUNSHU_REALTIME_MAX_CONVERSATION_ITEMS", "int", 1000, "Realtime: conversation items kept per session.", "voice", minimum=1)

# ── image ──────────────────────────────────────────────────────
_add("YUNSHU_DIFFUSION_SCHEDULER", "enum", "", "Image generation sampler override ('' uses the pipeline's own).", "image-video", choices=("", "ddim", "dpm_plus_plus", "euler", "euler_ancestral", "lms"))

# ── embeddings ─────────────────────────────────────────────────────────
_add("YUNSHU_ANE_EMBEDDINGS", "bool", False, "Compute embeddings on the Apple Neural Engine via CoreML when available.", "embeddings")
_add("YUNSHU_ANE_EMBEDDING_MODEL", "str", "intfloat/e5-small-v2", "Embedding model used on the ANE path.", "embeddings")

# ── MCP ────────────────────────────────────────────────────────────────
_add("YUNSHU_MCP_CONFIG", "path", None, "MCP client config file (JSON/YAML) listing tool servers.", "mcp")
_add("YUNSHU_MCP_SERVERS", "json", None, "MCP tool servers as a JSON array (alternative to YUNSHU_MCP_CONFIG).", "mcp")

# ── observability ──────────────────────────────────────────────────────
_add("YUNSHU_LOG_LEVEL", "enum", "INFO", "Log level for Yunshu's loggers (third-party loggers stay at WARNING).", "observability", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
_add("YUNSHU_AUDIT_LOG_FILE", "path", None, "Also write the audit log to this file.", "observability")

# ── CLI ────────────────────────────────────────────────────────────────
_add("YUNSHU_GATEWAY_URL", "str", "http://localhost:8000", "Server URL used by the yunshu CLI client commands.", "cli")
_add("YUNSHU_HF_ENDPOINT", "str", None, "Hugging Face Hub endpoint for `yunshu serve` (exported as HF_ENDPOINT).", "cli")

# fmt: on

# ── resolution ─────────────────────────────────────────────────────────

_overrides: dict[str, str] = {}
_file_values: dict[str, str] | None = None
_file_path_loaded: tuple[str | None, int | None] | None = None


def _setting(name: str) -> Setting:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"{name} is not a registered Yunshu setting") from None


def _to_text(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def set_override(name: str, value: Any) -> None:
    """Highest-precedence value (CLI flags, ``--set``)."""
    _setting(name)
    _overrides[name] = _to_text(value)


def clear_overrides() -> None:
    _overrides.clear()


def user_config_path() -> Path:
    """The per-user config file (``yunshu config set`` writes it)."""
    return Path.home() / ".yunshu" / "config.toml"


def _config_path() -> str | None:
    if "YUNSHU_CONFIG" in _overrides:
        return _overrides["YUNSHU_CONFIG"] or None
    explicit = os.environ.get("YUNSHU_CONFIG")
    if explicit:
        return explicit
    user = user_config_path()
    return str(user) if user.is_file() else None


def _normalize_key(key: str) -> str:
    key = key.strip().upper().replace("-", "_")
    return key if key.startswith("YUNSHU_") else f"YUNSHU_{key}"


def load_config_file(path: str | os.PathLike | None) -> dict[str, str]:
    """Parse a TOML config into ``{YUNSHU_NAME: text}``.

    Keys may be full names (``YUNSHU_MTP``) or short ones (``mtp``); tables only
    group entries (their names are ignored).
    """
    if not path:
        return {}
    data = tomllib.loads(Path(path).expanduser().read_text())
    flat: dict[str, str] = {}

    def walk(table: Mapping[str, Any]) -> None:
        for key, value in table.items():
            if isinstance(value, Mapping):
                walk(value)
            else:
                flat[_normalize_key(key)] = _to_text(value)

    walk(data)
    return flat


def _file() -> dict[str, str]:
    global _file_values, _file_path_loaded
    path = _config_path()
    try:
        key = (path, Path(path).expanduser().stat().st_mtime_ns) if path else None
    except OSError:
        key = (path, None)
    if key != _file_path_loaded or _file_values is None:
        _file_values = load_config_file(path) if path else {}
        _file_path_loaded = key
    return _file_values


def write_config_value(name: str, value: Any | None, path: Path | None = None) -> Path:
    """Set (or with ``value=None`` remove) one setting in a TOML config file,
    validating it against the registry first. Entries are written flat."""
    s = _setting(name)
    target = Path(path).expanduser() if path else user_config_path()
    values = load_config_file(target) if target.is_file() else {}
    if value is None:
        values.pop(name, None)
    else:
        text = _to_text(value)
        _parse(s, text)  # raises SettingError on a bad value
        values[name] = text
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Yunshu settings (`yunshu config set KEY VALUE`); see docs/CONFIGURATION.md"
    ]
    lines += [f"{k} = {json.dumps(v)}" for k, v in sorted(values.items())]
    target.write_text("\n".join(lines) + "\n")
    return target


def raw(name: str) -> tuple[str | None, str]:
    """``(text, source)`` for a setting; source is cli / env / file / default."""
    s = _setting(name)
    if name in _overrides:
        return _overrides[name], "cli"
    env = os.environ.get(name)
    if env is not None and (env.strip() != "" or s.empty_is_value):
        return env, "env"
    file_value = _file().get(name)
    if file_value is not None:
        return file_value, "file"
    return None, "default"


def _parse(s: Setting, text: str) -> Any:
    value = text.strip()
    try:
        if s.type == "bool":
            low = value.lower()
            if low in _TRUE:
                return True
            if low in _FALSE:
                return False
            raise ValueError("expected 1/true/yes/on or 0/false/no/off")
        if s.type == "int":
            out: Any = int(value)
        elif s.type == "float":
            out = float(value)
        elif s.type == "gb":
            # GiB with an optional "GB" suffix; "disabled" = 0 (no limit).
            if value.lower() == "disabled":
                return 0.0
            out = float(value.upper().removesuffix("GB").strip())
            if out <= 0:
                raise ValueError("expected a positive size in GB or 'disabled'")
        elif s.type == "enum":
            # Choices are lower-case except the log level, which is upper-case.
            out = (
                value.upper() if s.choices and s.choices[0].isupper() else value.lower()
            )
            if out not in s.choices:
                raise ValueError(
                    f"expected one of {', '.join(repr(c) for c in s.choices)}"
                )
            return out
        elif s.type == "list":
            return tuple(p.strip() for p in value.split(",") if p.strip())
        elif s.type == "json":
            return json.loads(value)
        elif s.type == "path":
            return value
        else:
            return text if s.empty_is_value else value
    except (ValueError, json.JSONDecodeError) as exc:
        raise SettingError(f"{s.name}={text!r}: {exc}") from None
    if s.minimum is not None and out < s.minimum:
        raise SettingError(f"{s.name}={text!r}: must be >= {s.minimum:g}")
    return out


def get(name: str) -> Any:
    """Typed value of a setting (its registry default when unset)."""
    text, _ = raw(name)
    s = REGISTRY[name]
    if text is None:
        return s.default
    return _parse(s, text)


def get_bool(name: str) -> bool:
    return bool(get(name))


def get_int(name: str) -> int | None:
    return get(name)


def get_float(name: str) -> float | None:
    return get(name)


def get_str(name: str) -> str | None:
    return get(name)


def is_set(name: str) -> bool:
    """True when any source (not the default) provides a value."""
    return raw(name)[1] != "default"


# ── validation and reporting ───────────────────────────────────────────


def close_matches(name: str) -> list[str]:
    """Registered names closest to ``name`` (typo hints)."""
    return difflib.get_close_matches(name, REGISTRY, n=3, cutoff=0.6)


def unknown_names(
    environ: Mapping[str, str] | None = None,
) -> list[tuple[str, list[str]]]:
    """``YUNSHU_*`` names in the environment/config that nothing reads, with
    the closest registered names (typo guard)."""
    env = os.environ if environ is None else environ
    names = {k for k in env if k.startswith("YUNSHU_")} | set(_file())
    out = []
    for name in sorted(names - REGISTRY.keys()):
        out.append((name, close_matches(name)))
    return out


def validate(
    environ: Mapping[str, str] | None = None, *, warn: bool = True
) -> list[str]:
    """Parse every provided value (raise ``SettingError`` listing all bad ones)
    and warn about unregistered ``YUNSHU_*`` names. Returns the warnings."""
    errors = []
    for name in REGISTRY:
        try:
            get(name)
        except SettingError as exc:
            errors.append(str(exc))
    if errors:
        raise SettingError("Invalid Yunshu settings:\n  " + "\n  ".join(errors))
    warnings = []
    for name, close in unknown_names(environ):
        hint = f" (did you mean {', '.join(close)}?)" if close else ""
        msg = f"{name} is not a Yunshu setting and is ignored{hint}"
        warnings.append(msg)
        if warn:
            logger.warning(msg)
    return warnings


def effective(include: tuple[str, ...] = ("stable",)) -> list[dict[str, Any]]:
    """Rows for ``yunshu config``: name, value, source, stability, category."""
    rows = []
    for s in REGISTRY.values():
        if s.stability not in include:
            continue
        text, source = raw(s.name)
        try:
            value = get(s.name)
        except SettingError as exc:
            value = f"<invalid: {exc}>"
        if s.secret and text:
            value = "***"
        rows.append(
            {
                "name": s.name,
                "value": value,
                "source": source,
                "stability": s.stability,
                "category": s.category,
                "description": s.description,
            }
        )
    return rows
