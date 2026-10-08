"""ModelCard: one description of a served model, derived from the checkpoint on disk.

Every field comes from the checkpoint files (``config.json``, the chat template,
``generation_config.json``, safetensors headers, ``model_index.json``, the speech
configs) or from the engine's own routing tables (``spec_select`` for speculative
decoding, the request models for supported parameters). Nothing is guessed from the
model name except where the engine itself does the same (embedding / reranker dirs,
see ``model_manager._detect_model_type``).

The gateway renders the card in several wire formats (OpenAI / Anthropic ``/v1/models``,
OpenRouter, vLLM, LM Studio, Ollama ``/api/show``); see ``yunshu_gateway/model_card_formats.py``.
This module has no MLX import and is cheap to call.
"""

from __future__ import annotations

import json
import logging
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Largest max_tokens / max_completion_tokens the chat routes accept (ChatRequest le=...).
API_MAX_OUTPUT_TOKENS = 131072
# max_tokens applied when a request omits it.
API_DEFAULT_MAX_TOKENS = 512

# card kind -> ModelType.name
_KIND_FOR_TYPE = {
    "LLM": "chat",
    "VLM": "vlm",
    "TTS": "tts",
    "ASR": "asr",
    "IMAGE_GEN": "image",
    "OCR": "ocr",
    "STS": "sts",
    "VIDEO": "video",
    "EMBEDDING": "embedding",
    "RERANKER": "reranker",
    "CLASSIFIER": "classifier",
    "DECISION": "decision",
}

# Sampling / control parameters accepted by the chat routes (names are ChatRequest fields;
# OpenRouter's ``supported_parameters`` uses the same vocabulary).
TEXT_PARAMETERS = [
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "seed",
    "stop",
    "n",
    "stream",
    "logprobs",
    "top_logprobs",
    "min_tokens",
    "response_format",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "reasoning_effort",
    "enable_thinking",
    "thinking_budget",
    "chat_template_kwargs",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "guided_json",
]
# OpenRouter-only spellings of the same features.
_OPENROUTER_ALIASES = {
    "response_format": ["structured_outputs"],
    "reasoning_effort": ["reasoning", "include_reasoning"],
}

_ENDPOINTS: dict[str, list[str]] = {
    "chat": [
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/tokenize",
        "/v1/detokenize",
    ],
    "embedding": ["/v1/embeddings", "/pooling"],
    "reranker": ["/rerank", "/score"],
    "classifier": ["/v1/classify"],
    "decision": ["/v1/decisions", "/v1/systemone"],
    "asr": ["/v1/audio/transcriptions"],
    "tts": ["/v1/audio/speech", "/v1/audio/speech/stream", "/v1/audio/voices"],
    "image": ["/v1/images/generations"],
    "ocr": ["/v1/ocr"],
    "video": [],
    "sts": [],
}
_ENDPOINTS["vlm"] = _ENDPOINTS["chat"]
_ENDPOINTS["omni"] = _ENDPOINTS["chat"] + ["/v1/omni/speech/stream"]

_API_FORMATS = {
    "/v1/chat/completions": "chat_completions",
    "/v1/responses": "responses",
    "/v1/messages": "messages",
    "/v1/completions": "completions",
}


# ── file helpers ─────────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_text(path: Path, limit: int = 4_000_000) -> str:
    try:
        return path.read_text()[:limit]
    except (OSError, UnicodeDecodeError):
        return ""


def _text_config(config: dict) -> dict:
    """The language-model sub-config (Qwen3.5 / GLM-OCR / omni nest it)."""
    for key in ("text_config", "language_config"):
        if isinstance(config.get(key), dict):
            return config[key]
    thinker = config.get("thinker_config")
    if isinstance(thinker, dict) and isinstance(thinker.get("text_config"), dict):
        return thinker["text_config"]
    return config


def chat_template_text(model_path: Path) -> str:
    """The chat template as the tokenizer would load it (file, then tokenizer_config)."""
    for name in ("chat_template.jinja", "chat_template.json"):
        text = _read_text(model_path / name)
        if text:
            if name.endswith(".json"):
                try:
                    data = json.loads(text)
                    tpl = data.get("chat_template") if isinstance(data, dict) else None
                    if isinstance(tpl, str):
                        return tpl
                    if isinstance(tpl, list):
                        return "\n".join(
                            str(t.get("template", ""))
                            for t in tpl
                            if isinstance(t, dict)
                        )
                except ValueError:
                    continue
            return text
    tk = _read_json(model_path / "tokenizer_config.json") or {}
    tpl = tk.get("chat_template")
    if isinstance(tpl, str):
        return tpl
    if isinstance(tpl, list):
        return "\n".join(str(t.get("template", "")) for t in tpl if isinstance(t, dict))
    return ""


# ── template introspection ───────────────────────────────────────────────────

_EFFORT_TUPLE = re.compile(r"reasoning_effort\s+not\s+in\s+\(([^)]*)\)")
_EFFORT_DEFAULT = re.compile(r"reasoning_effort\s*\|\s*default\(\s*['\"]([\w-]+)['\"]")


def reasoning_levels(template: str) -> tuple[list[str], str | None]:
    """Reasoning-effort levels the chat template accepts, and its default.

    Qwen3.8 validates ``reasoning_effort not in ('xhigh', 'medium', 'low')`` and defaults
    to ``xhigh``; both are read straight from the template.
    """
    m = _EFFORT_TUPLE.search(template)
    levels = re.findall(r"['\"]([\w-]+)['\"]", m.group(1)) if m else []
    d = _EFFORT_DEFAULT.search(template)
    return levels, (d.group(1) if d else None)


# OpenAI's ladder mapped onto the template's own levels.
_EFFORT_ALIASES = {
    "minimal": ("low",),
    "none": ("low",),
    "high": ("xhigh", "high", "medium"),
    "max": ("xhigh", "high"),
}


def normalize_effort(effort: str, levels: list[str]) -> str:
    """Map an OpenAI-style ``reasoning_effort`` onto a level the template accepts.

    The Qwen3.8 template raises on anything outside ``levels`` (OpenAI clients send
    ``high``, which it does not know), so ``high`` becomes its ``xhigh``.
    """
    e = str(effort).strip().lower()
    if not levels or e in levels:
        return e
    for cand in _EFFORT_ALIASES.get(e, ()):
        if cand in levels:
            return cand
    return e


def _reasoning(template: str, text_cfg: dict, config: dict) -> dict:
    thinks = "<think>" in template or "enable_thinking" in template
    levels, default = reasoning_levels(template)
    toggle = "enable_thinking" in template
    default_on = None
    if toggle:
        # "enable_thinking is undefined or enable_thinking is true" -> on unless disabled.
        default_on = bool(
            re.search(
                r"enable_thinking\s+is\s+undefined\s+or\s+enable_thinking\s+is\s+true",
                template,
            )
        )
    if not thinks and not levels:
        return {"supported": False}
    out: dict[str, Any] = {
        "supported": True,
        "toggle": "enable_thinking" if toggle else None,
        "default_enabled": default_on,
        "effort_levels": levels,
        "default_effort": default,
        "effort_field": "reasoning_effort" if levels else None,
        "budget_field": "thinking_budget",
        "output_field": "reasoning_content",
    }
    if levels:
        out["effort_aliases"] = {
            a: normalize_effort(a, levels)
            for a in ("minimal", "high")
            if normalize_effort(a, levels) in levels
        }
    return out


# ── safetensors headers → parameter count ────────────────────────────────────

_DTYPE_BITS = {
    "F64": 64,
    "F32": 32,
    "F16": 16,
    "BF16": 16,
    "I64": 64,
    "I32": 32,
    "U32": 32,
    "I16": 16,
    "U16": 16,
    "I8": 8,
    "U8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "BOOL": 8,
}


def _safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 200_000_000:
            raise ValueError("implausible safetensors header")
        return json.loads(f.read(n))


def count_parameters(model_path: Path, quant: dict | None) -> tuple[int | None, int]:
    """(parameter count, weight bytes) from the safetensors headers.

    Quantized linears store ``uint32`` words that hold ``32/bits`` weights each and a
    ``.scales`` / ``.biases`` pair; the count uses the layer's own bit width and skips the
    scale tensors, so a 4-bit 27B checkpoint reports ~27B, not ~3.5B.
    """
    files = sorted(model_path.glob("model*.safetensors"))
    if not files:
        files = sorted(
            p for p in model_path.glob("*.safetensors") if "mtp" not in p.name
        )
    if not files:
        files = sorted(model_path.rglob("*.safetensors"))
    if not files:
        # npz / gguf / bin weights (whisper): size only, no tensor headers to read.
        others = [
            f for ext in ("*.npz", "*.gguf", "*.bin") for f in model_path.glob(ext)
        ]
        return None, sum(f.stat().st_size for f in others)
    total_bytes = sum(f.stat().st_size for f in files)
    default_bits = int((quant or {}).get("bits") or 0)
    per_layer = {
        k: v.get("bits")
        for k, v in (quant or {}).items()
        if isinstance(v, dict) and v.get("bits")
    }
    params = 0
    try:
        for f in files:
            header = _safetensors_header(f)
            for name, meta in header.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                if name.endswith((".scales", ".biases")) and (
                    name.rsplit(".", 1)[0] + ".weight" in header
                ):
                    continue
                n = 1
                for d in meta.get("shape", []):
                    n *= int(d)
                if meta.get("dtype") == "U32" and name.endswith(".weight") and quant:
                    base = name[: -len(".weight")]
                    bits = per_layer.get(base) or default_bits or 4
                    n = n * 32 // bits
                params += n
    except (OSError, ValueError, KeyError, struct.error):
        return None, total_bytes
    return params, total_bytes


def quantization_info(config: dict, model_path: Path | None = None) -> dict | None:
    """Bits per layer group from ``quantization`` / ``quantization_config``, or a component-level
    ``quantize_config.json`` (diffusion checkpoints)."""
    q = config.get("quantization") or config.get("quantization_config")
    if not isinstance(q, dict) and model_path is not None:
        qc = _read_json(model_path / "quantize_config.json")
        q = qc.get("quantization") if qc else None
        if isinstance(q, dict):
            return {
                "bits": q.get("bits"),
                "group_size": q.get("group_size"),
                "mode": q.get("mode", "affine"),
                "layer_groups": {},
                "skip_components": q.get("skip_components", []),
            }
    if not isinstance(q, dict) or not q.get("bits"):
        return None
    groups: dict[str, int] = {}
    for v in q.values():
        if isinstance(v, dict) and v.get("bits"):
            groups[str(v["bits"])] = groups.get(str(v["bits"]), 0) + 1
    return {
        "bits": q.get("bits"),
        "group_size": q.get("group_size"),
        "mode": q.get("mode", "affine"),
        # {bits: number of layers overriding the default}: mixed-precision checkpoints
        # (oQ) keep sensitive projections at higher bits.
        "layer_groups": groups,
    }


# ── the card ─────────────────────────────────────────────────────────────────


@dataclass
class ModelCard:
    id: str
    display_name: str
    kind: str  # chat | vlm | omni | embedding | reranker | asr | tts | sts | image | ocr | video
    family: str | None = None  # config model_type (text model_type for nested configs)
    architecture: str | None = None
    parameters: int | None = None
    quantization: dict | None = None
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    max_output_tokens: int | None = None
    reasoning: dict = field(default_factory=lambda: {"supported": False})
    tools: dict = field(default_factory=dict)
    structured_output: dict = field(default_factory=dict)
    logprobs: dict = field(default_factory=dict)
    embeddings: dict | None = None
    audio: dict | None = None
    image: dict | None = None
    speculative: dict = field(default_factory=lambda: {"available": False})
    prefix_cache: dict = field(default_factory=lambda: {"supported": False})
    state: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    api: dict = field(default_factory=dict)
    contract: dict = field(default_factory=dict)
    supported_parameters: list[str] = field(default_factory=list)
    generation_defaults: dict = field(default_factory=dict)
    created: int = 0
    path: str | None = None

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    def capabilities(self) -> list[str]:
        """Flat capability tags shared by LM Studio / vLLM / Ollama style clients."""
        caps: list[str] = []
        if self.kind in ("chat", "vlm", "omni"):
            caps += ["completion", "chat"]
        if self.tools.get("supported"):
            caps += ["tools", "tool_use", "function_calling"]
        if "image" in self.input_modalities and self.kind in ("chat", "vlm", "omni"):
            caps += ["vision"]
        if "audio" in self.input_modalities and self.kind in ("chat", "vlm", "omni"):
            caps += ["audio"]
        if self.reasoning.get("supported"):
            caps += ["thinking", "reasoning"]
        if self.structured_output.get("json_schema"):
            caps += ["structured_output"]
        if self.kind == "embedding":
            caps += ["embedding"]
        if self.kind == "reranker":
            caps += ["rerank", "score"]
            if (self.architecture or "").endswith("ForSequenceClassification"):
                caps += ["classify"]
        if self.kind == "classifier":
            caps += ["classify"]
        if self.kind == "decision":
            caps += ["decision"]
        if self.kind == "asr":
            caps += ["transcription"]
        if self.kind == "tts":
            caps += ["speech"]
        if self.kind == "image":
            caps += ["image_generation"]
        if self.kind == "ocr":
            caps += ["ocr"]
        return list(dict.fromkeys(caps))


def _kind_for(model_path: Path, config: dict, model_type_name: str | None) -> str:
    if model_type_name is None:
        try:
            from .model_manager import _detect_model_type

            model_type_name = _detect_model_type(str(model_path)).name
        except Exception:
            model_type_name = "LLM"
    kind = _KIND_FOR_TYPE.get(model_type_name, "chat")
    if (
        kind == "classifier"
        and config.get("num_labels", len(config.get("id2label", {})) or 2) == 1
    ):
        kind = "reranker"
    # Text retrieval models (Qwen3-Embedding / -Reranker) load through the LLM engine and
    # serve pooled vectors / scores; the engine's detector already keys the VL variants off
    # the name, so the card does the same for the text ones.
    lowered = str(model_path).lower()
    if kind in ("chat", "vlm"):
        if "reranker" in lowered:
            kind = "reranker"
        elif "embedding" in lowered:
            kind = "embedding"
    thinker = config.get("thinker_config")
    if kind in ("chat", "vlm") and (
        (isinstance(thinker, dict) and thinker.get("audio_config"))
        or "omni" in str(config.get("model_type", "")).lower()
    ):
        kind = "omni"
    return kind


def _modalities(
    kind: str, model_path: Path, config: dict
) -> tuple[list[str], list[str]]:
    thinker = (
        config.get("thinker_config")
        if isinstance(config.get("thinker_config"), dict)
        else {}
    )
    has_video = bool(
        config.get("video_token_id") is not None
        or (model_path / "video_preprocessor_config.json").exists()
        or thinker.get("video_token_id") is not None
    )
    if kind == "chat":
        return ["text"], ["text"]
    if kind == "vlm":
        # Gemma 4 E2B / E4B: a VLM with an audio tower (audio_config + audio_token_id)
        has_audio = (
            isinstance(config.get("audio_config"), dict)
            and config.get("audio_token_id") is not None
        )
        return (
            ["text", "image"]
            + (["audio"] if has_audio else [])
            + (["video"] if has_video else []),
            ["text"],
        )
    if kind == "omni":
        outs = ["text"] + (["audio"] if "talker_config" in config else [])
        ins = ["text", "image", "audio"] + (["video"] if has_video else [])
        return ins, outs
    if kind in ("embedding", "reranker"):
        ins = ["text"] + (["image"] if "vision_config" in config else [])
        return ins, ["embedding"] if kind == "embedding" else ["score"]
    if kind == "classifier":
        return ["text"], ["classification"]
    if kind == "decision":
        return ["text", "image"] if "vision_config" in config else ["text"], [
            "decision"
        ]
    if kind == "asr":
        return ["audio"], ["text"]
    if kind == "tts":
        return ["text"], ["audio"]
    if kind == "sts":
        return ["audio"], ["audio"]
    if kind == "image":
        return ["text", "image"], ["image"]
    if kind == "ocr":
        return ["image"], ["text"]
    if kind == "video":
        return ["text", "image"], ["video"]
    return ["text"], ["text"]


def _context(kind: str, config: dict, text_cfg: dict) -> dict:
    if kind == "asr" and "n_text_ctx" in config:
        return {
            "length": config["n_text_ctx"],
            "native": config["n_text_ctx"],
            "source": "n_text_ctx",
        }
    if kind not in (
        "chat",
        "vlm",
        "omni",
        "embedding",
        "reranker",
        "classifier",
        "ocr",
        "decision",
    ):
        return {}
    length = None
    source = None
    for holder, label in ((text_cfg, "text_config"), (config, "config")):
        for key in (
            "max_position_embeddings",
            "max_seq_len",
            "n_positions",
            "seq_length",
        ):
            if isinstance(holder.get(key), int):
                length, source = holder[key], f"{label}.{key}"
                break
        if length:
            break
    rope = text_cfg.get("rope_scaling") or text_cfg.get("rope_parameters") or {}
    native = length
    if isinstance(rope, dict):
        orig = rope.get("original_max_position_embeddings")
        if isinstance(orig, int) and orig:
            native = orig
        factor = rope.get("factor")
        if (
            isinstance(factor, (int, float))
            and factor > 1
            and native == length
            and length
        ):
            # rope-scaled config that only lists the native window
            length = (
                int(length * factor)
                if rope.get("type") in ("yarn", "linear")
                else length
            )
    if length is None:
        return {}
    return {
        "length": length,
        "native": native,
        # The engine applies no tighter cap than the checkpoint's window; memory is the
        # practical limit (see /debug/memory-guard), not a hard-coded ceiling.
        "effective": length,
        "source": source,
        "rope_scaling": rope
        if isinstance(rope, dict) and rope.get("type") not in (None, "default")
        else None,
    }


def _speculative(kind: str, model_path: Path, config: dict) -> dict:
    if kind not in ("chat", "vlm", "omni"):
        return {"available": False}
    try:
        from . import spec_select
        from .mlxvlm_mtp import is_mtp_capable

        family = config.get("model_type") in ("qwen3_5", "qwen3_6", "qwen3_5_moe")
        mtp = family and is_mtp_capable(str(model_path))
        choice = spec_select.choose(config, spec_family=family, mtp_capable=mtp)
    except Exception:
        logger.debug("speculative probe failed", exc_info=True)
        return {"available": False}
    from . import settings

    return {
        "available": choice.kind != "none",
        "method": {"dflash": "dflash2", "mtp": "mtp"}.get(choice.kind),
        "mtp_head": bool(mtp),
        "drafter": choice.drafter,
        "block_size": settings.get("YUNSHU_MTP_BLOCK_SIZE")
        or (6 if choice.kind == "mtp" else None),
        "lossless": True
        if choice.kind != "none"
        else None,  # batch-invariant verify: spec on == off
        "reason": choice.reason,
    }


def _prefix_cache(kind: str, config: dict, text_cfg: dict) -> dict:
    if kind not in ("chat", "vlm", "omni"):
        return {"supported": False}
    layer_types = text_cfg.get("layer_types") or []
    sliding = "sliding_attention" in layer_types or bool(
        text_cfg.get("use_sliding_window") or config.get("use_sliding_window")
    )
    hybrid = "linear_attention" in layer_types
    return {
        "supported": not sliding,
        "kind": "apc" if kind in ("vlm", "omni") else "kv_prefix",
        "hybrid_checkpoints": hybrid,
        "media_keyed": kind in ("vlm", "omni"),
        "reason": "sliding-window layers cannot be checkpointed at a prefix boundary"
        if sliding
        else None,
    }


def _audio(kind: str, model_path: Path, config: dict) -> dict | None:
    if kind == "asr":
        langs = config.get("support_languages")
        out: dict[str, Any] = {"task": "transcription"}
        if isinstance(langs, list):
            out["languages"] = langs
        if config.get("model_type") == "whisper":
            n_vocab = int(config.get("n_vocab", 0))
            out["multilingual"] = n_vocab >= 51865
            out["translation"] = out["multilingual"]
            out["audio_context_seconds"] = round(
                int(config.get("n_audio_ctx", 0)) * 2 / 100
            )
            out["mel_bins"] = config.get("n_mels")
        return out
    if kind == "tts":
        talker = config.get("talker_config") or {}
        langs = [k for k in (talker.get("codec_language_id") or {}) if k]
        spk = list((talker.get("spk_id") or {}).keys())
        return {
            "task": "speech_synthesis",
            "tts_model_type": config.get("tts_model_type"),
            "languages": langs,
            "voices": spk,
            "voice_design": config.get("tts_model_type") == "voice_design",
        }
    return None


def _image(kind: str, model_path: Path) -> dict | None:
    if kind != "image":
        return None
    idx = _read_json(model_path / "model_index.json") or {}
    return {
        "pipeline": idx.get("_class_name"),
        "components": sorted(k for k in idx if not k.startswith("_")),
    }


def _pooling_mode(model_path: Path) -> str | None:
    """Pooling declared by a sentence-transformers ``1_Pooling/config.json``, when shipped."""
    cfg = _read_json(model_path / "1_Pooling" / "config.json") or {}
    for key, name in (
        ("pooling_mode_lasttoken", "last"),
        ("pooling_mode_mean_tokens", "mean"),
        ("pooling_mode_cls_token", "cls"),
        ("pooling_mode_max_tokens", "max"),
    ):
        if cfg.get(key):
            return name
    return None


def _generation_defaults(model_path: Path) -> dict:
    g = _read_json(model_path / "generation_config.json") or {}
    return {
        k: g[k]
        for k in (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "do_sample",
            "max_new_tokens",
        )
        if k in g
    }


def _display_name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


_STATIC_CACHE: dict[tuple[str, float, str], ModelCard] = {}
_STATIC_LOCK = threading.Lock()


def build_model_card(
    model_path: str | Path,
    *,
    model_id: str | None = None,
    model_type_name: str | None = None,
    loaded: bool = False,
    loading: bool = False,
    pinned: bool = False,
    load_error: str | None = None,
    estimated_bytes: int = 0,
    created: int = 0,
    engine: Any = None,
) -> ModelCard:
    """Derive the card for the checkpoint at ``model_path``.

    ``model_type_name`` is a ``ModelType`` member name (``"LLM"``, ``"VLM"``...); when omitted
    the engine's own detector decides. The static part (everything read from files) is cached
    by path + config mtime; load state is filled in per call.
    """
    p = Path(model_path)
    mid = model_id or p.name
    try:
        mtime = max(
            (p / n).stat().st_mtime
            for n in ("config.json", "model_index.json")
            if (p / n).exists()
        )
    except (OSError, ValueError):
        mtime = 0.0
    key = (str(p), mtime, model_type_name or "")
    with _STATIC_LOCK:
        base = _STATIC_CACHE.get(key)
    if base is None:
        base = _derive(p, mid, model_type_name)
        with _STATIC_LOCK:
            _STATIC_CACHE[key] = base
    from dataclasses import replace

    # Not loaded yet: the checkpoint's own config timestamp keeps `created` stable across calls.
    card = replace(
        base,
        id=mid,
        display_name=_display_name(mid),
        created=created or int(mtime) or int(time.time()),
    )
    card.state = {
        "loaded": loaded,
        "loading": loading,
        "pinned": pinned,
        "error": load_error,
        "status": "loaded"
        if loaded
        else "loading"
        if loading
        else "error"
        if load_error
        else "not-loaded",
    }
    card.memory = {
        "weights_bytes": base.memory.get("weights_bytes"),
        "estimated_bytes": estimated_bytes or base.memory.get("weights_bytes"),
    }
    from .capability_contract import build_contract

    card.contract = build_contract(card)
    if loaded and engine is not None:
        _apply_engine(card, engine)
    return card


def _apply_engine(card: ModelCard, engine: Any) -> None:
    """Runtime facts only a loaded engine knows."""
    try:
        stats = engine.get_stats() if hasattr(engine, "get_stats") else {}
    except Exception:
        return
    if isinstance(stats, dict) and stats.get("apc") is not None:
        card.prefix_cache["active"] = True
        card.prefix_cache["memory_max_bytes"] = stats["apc"].get("memory_max_bytes")


def _derive(p: Path, mid: str, model_type_name: str | None) -> ModelCard:
    config = _read_json(p / "config.json") or {}
    if not config:
        idx = _read_json(p / "model_index.json")
        if idx:
            config = {"_class_name": idx.get("_class_name")}
    text_cfg = _text_config(config)
    kind = _kind_for(p, config, model_type_name)
    template = chat_template_text(p) if kind in ("chat", "vlm", "omni") else ""
    quant = quantization_info(config, p)
    params, weight_bytes = count_parameters(p, quant)
    ins, outs = _modalities(kind, p, config)
    text_like = kind in ("chat", "vlm", "omni")
    tools_ok = text_like and ("tools" in template or "tool_call" in template)
    archs = config.get("architectures") or (
        [config["_class_name"]] if config.get("_class_name") else []
    )
    family = text_cfg.get("model_type") or config.get("model_type")
    ctx = _context(kind, config, text_cfg)
    from .scoring_engine import scoring_kind, scoring_max_length

    if ctx and scoring_kind(config, str(p)):
        tokenizer_config = _read_json(p / "tokenizer_config.json") or {}
        cap = scoring_max_length(config, tokenizer_config.get("model_max_length"))
        ctx["length"] = ctx["effective"] = min(ctx["length"], cap)
        ctx["serving_cap"] = cap

    max_out = None
    if text_like:
        max_out = (
            min(API_MAX_OUTPUT_TOKENS, ctx["length"])
            if ctx.get("length")
            else API_MAX_OUTPUT_TOKENS
        )

    card = ModelCard(
        id=mid,
        display_name=_display_name(mid),
        kind=kind,
        family=family,
        architecture=archs[0] if archs else None,
        parameters=params,
        quantization=quant,
        input_modalities=ins,
        output_modalities=outs,
        context=ctx,
        max_output_tokens=max_out,
        reasoning=_reasoning(template, text_cfg, config)
        if text_like
        else {"supported": False},
        tools={
            "supported": tools_ok,
            "parallel": bool(tools_ok and "tool_calls" in template),
            "tool_choice": ["auto", "none", "required", "function"] if tools_ok else [],
        },
        structured_output={
            "json_object": text_like,
            "json_schema": text_like,
            "regex": text_like,
            "grammar": text_like,
            "choice": text_like,
        },
        logprobs={"supported": text_like, "max_top_logprobs": 20 if text_like else 0},
        embeddings=(
            {
                "dimensions": text_cfg.get("hidden_size") or config.get("hidden_size"),
                "pooling": _pooling_mode(p),
                "normalized": True,
            }
            if kind == "embedding" or text_like
            else None
        ),
        audio=_audio(kind, p, config),
        image=_image(kind, p),
        speculative=_speculative(kind, p, config),
        prefix_cache=_prefix_cache(kind, config, text_cfg),
        supported_parameters=list(TEXT_PARAMETERS) if text_like else [],
        generation_defaults=_generation_defaults(p),
        path=str(p),
    )
    if not tools_ok:
        card.supported_parameters = [
            x
            for x in card.supported_parameters
            if x not in ("tools", "tool_choice", "parallel_tool_calls")
        ]
    if not card.reasoning.get("supported"):
        card.supported_parameters = [
            x
            for x in card.supported_parameters
            if x not in ("reasoning_effort", "enable_thinking", "thinking_budget")
        ]
    elif not card.reasoning.get("effort_levels"):
        card.supported_parameters = [
            x for x in card.supported_parameters if x != "reasoning_effort"
        ]
    if (
        kind == "embedding"
        and card.embeddings is not None
        and not card.embeddings.get("dimensions")
    ):
        card.embeddings["dimensions"] = None
    if kind != "embedding" and not text_like:
        card.embeddings = None
    endpoints = list(_ENDPOINTS.get(kind, []))
    if kind == "reranker" and any(
        a.endswith("ForSequenceClassification") for a in archs
    ):
        endpoints.append("/v1/classify")
    if kind == "asr" and config.get("model_type") == "whisper":
        endpoints.append("/v1/audio/translations")
    card.api = {
        "endpoints": endpoints,
        "formats": [_API_FORMATS[e] for e in endpoints if e in _API_FORMATS],
        "ollama": (["/api/chat", "/api/generate", "/api/show"] if text_like else [])
        + (["/api/embed"] if kind == "embedding" or text_like else []),
    }
    card.memory = {"weights_bytes": weight_bytes or None}
    return card
