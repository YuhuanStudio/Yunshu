"""Render a ``ModelCard`` in the wire formats clients and gateways already understand.

One payload serves all of them because the field names do not collide:

* OpenAI: ``id`` / ``object`` / ``created`` / ``owned_by``.
* Anthropic: ``type: "model"`` / ``display_name`` / ``created_at`` / ``max_input_tokens`` /
  ``max_tokens`` / ``capabilities`` (the object form; flat LM Studio / vLLM booleans ride in it).
* OpenRouter ``/api/v1/models``: ``name``, ``canonical_slug``, ``context_length``,
  ``architecture{modality,input_modalities,output_modalities,tokenizer,instruct_type}``,
  ``pricing``, ``top_provider{context_length,max_completion_tokens}``, ``supported_parameters``,
  ``default_parameters``.
* vLLM: ``root``, ``parent``, ``max_model_len``, ``task``.
* LM Studio ``/api/v0/models``: ``arch``, ``quantization``, ``state``, ``max_context_length``,
  ``publisher``, ``compatibility_type``, ``capabilities``.
* Everything Yunshu knows beyond those, nested under one ``yunshu`` key so strict clients
  ignore it.

Yunxin's vLLM / LM Studio / OpenRouter adapters read exactly these keys
(``max_model_len``, ``max_context_length``, ``context_length``,
``top_provider.max_completion_tokens``, ``architecture.*_modalities``, ``capabilities``,
``model_type`` / ``task``), so a Yunshu server is a provider with no custom adapter.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from yunshu_engine.model_card import ModelCard

# OpenRouter's modality vocabulary: text, image, audio, file, video, embeddings.
_OR_NAMES = {"embedding": "embeddings", "score": "text"}

_TASKS = {
    "chat": "generate",
    "vlm": "generate",
    "omni": "generate",
    "embedding": "embed",
    "reranker": "score",
    "asr": "transcription",
    "tts": "speech",
    "image": "image_generation",
    "ocr": "ocr",
    "video": "video_generation",
    "sts": "speech_to_speech",
}

# LM Studio's own type vocabulary.
_LMSTUDIO_TYPES = {
    "chat": "llm",
    "vlm": "vlm",
    "omni": "vlm",
    "embedding": "embeddings",
}


def _iso(created: int) -> str:
    return datetime.fromtimestamp(created, UTC).isoformat().replace("+00:00", "Z")


def _modality_string(ins: list[str], outs: list[str]) -> str:
    def norm(xs: list[str]) -> list[str]:
        return [_OR_NAMES.get(x, x) for x in xs]

    return "+".join(norm(ins)) + "->" + "+".join(norm(outs))


def _quant_label(card: ModelCard) -> str | None:
    return _quant_label_dict(card.quantization)


def _quant_label_dict(q: dict | None) -> str | None:
    if not q or not q.get("bits"):
        return None
    groups = q.get("layer_groups") or {}
    if groups:
        bits = sorted({int(q["bits"])} | {int(b) for b in groups})
        return "mixed-" + "/".join(str(b) for b in bits) + "bit"
    return f"{q['bits']}bit"


def _wire(card: ModelCard, extra: dict | None) -> dict:
    d = card.to_dict()
    d.pop("path", None)
    d["capability_tags"] = card.capabilities()
    d.update(extra or {})
    return d


def openai_model(card: ModelCard, *, detailed: bool = False) -> dict[str, Any]:
    """The ``/v1/models`` entry: spec fields + Anthropic + OpenRouter + vLLM + LM Studio + card."""
    created = card.created or 0
    ins = [_OR_NAMES.get(x, x) for x in card.input_modalities]
    outs = [_OR_NAMES.get(x, x) for x in card.output_modalities]
    ctx = card.context.get("length")
    text_like = card.kind in ("chat", "vlm", "omni")
    gen = card.generation_defaults
    entry: dict[str, Any] = {
        "id": card.id,
        "object": "model",
        "created": created,
        "owned_by": "yunshu",
        # Anthropic
        "type": "model",
        "display_name": card.display_name,
        "created_at": _iso(created),
        # OpenRouter
        "name": card.display_name,
        "canonical_slug": card.id,
        "description": _description(card),
        "context_length": ctx,
        "architecture": {
            "modality": _modality_string(card.input_modalities, card.output_modalities),
            "input_modalities": ins,
            "output_modalities": outs,
            "tokenizer": card.family,
            "instruct_type": None,
        },
        "pricing": {"prompt": "0", "completion": "0", "request": "0", "image": "0"},
        "top_provider": {
            "context_length": ctx,
            "max_completion_tokens": card.max_output_tokens,
            "is_moderated": False,
        },
        "supported_parameters": _or_parameters(card),
        "default_parameters": {
            k: gen[k]
            for k in ("temperature", "top_p", "top_k", "repetition_penalty")
            if k in gen
        }
        if text_like
        else None,
        # vLLM
        "root": card.id,
        "parent": None,
        "max_model_len": ctx,
        "task": _TASKS.get(card.kind),
        # LM Studio
        "arch": card.family,
        "quantization": _quant_label(card),
        "state": "loaded" if card.state.get("loaded") else "not-loaded",
        "max_context_length": ctx,
        "publisher": "yunshu",
        "compatibility_type": "mlx",
        "model_type": card.kind,
        # Anthropic spec fields (max_input_tokens / max_tokens / capabilities object).
        "max_input_tokens": ctx,
        "max_tokens": card.max_output_tokens,
        "capabilities": anthropic_capabilities(card),
        # Everything else, in one namespace.
        "yunshu": _wire(card, None),
    }
    if text_like:
        # What the gateway itself can run for server tools (web search provider, MCP connector).
        from .server_tools import status as _server_tools_status

        entry["yunshu"]["server_tools"] = _server_tools_status()
    if detailed:
        entry["yunshu"]["path"] = card.path
    return entry


def _description(card: ModelCard) -> str:
    bits = [card.kind, card.architecture or card.family or ""]
    if card.parameters:
        bits.append(f"{card.parameters / 1e9:.1f}B params")
    label = _quant_label(card)
    if label:
        bits.append(label)
    if card.context.get("length"):
        bits.append(f"{card.context['length']} ctx")
    return ", ".join(b for b in bits if b)


def _or_parameters(card: ModelCard) -> list[str]:
    from yunshu_engine.model_card import _OPENROUTER_ALIASES

    out: list[str] = []
    for p in card.supported_parameters:
        out.append(p)
        out.extend(_OPENROUTER_ALIASES.get(p, []))
    return list(dict.fromkeys(out))


def ollama_show(card: ModelCard | dict) -> dict[str, Any]:
    """Ollama ``/api/show``: ``capabilities`` and ``model_info`` (GGUF-style keys).

    Takes a card or its wire dict (the ``yunshu`` block of a ``/v1/models`` item), because the
    Ollama layer reads models over loopback.
    """
    d = _wire(card, None) if isinstance(card, ModelCard) else card
    kind = d.get("kind", "chat")
    text_like = kind in ("chat", "vlm", "omni")
    caps: list[str] = []
    if text_like:
        caps.append("completion")
    if kind == "embedding":
        caps.append("embedding")
    if (d.get("tools") or {}).get("supported"):
        caps.append("tools")
    if text_like and "image" in d.get("input_modalities", []):
        caps.append("vision")
    if (d.get("reasoning") or {}).get("supported"):
        caps.append("thinking")
    if not caps:
        caps.append(_TASKS.get(kind, kind))
    arch = d.get("family") or "unknown"
    params = d.get("parameters")
    ctx = (d.get("context") or {}).get("length")
    info: dict[str, Any] = {"general.architecture": arch}
    if params:
        info["general.parameter_count"] = params
    if ctx:
        info[f"{arch}.context_length"] = ctx
    hidden = (d.get("embeddings") or {}).get("dimensions")
    if hidden:
        info[f"{arch}.embedding_length"] = hidden
    q = d.get("quantization") or {}
    if q.get("bits"):
        info["general.quantization_version"] = 2
    # The loopback wire dict also carries server-tool status; the card itself does not.
    info["yunshu.card"] = {k: v for k, v in d.items() if k != "server_tools"}
    gen = d.get("generation_defaults") or {}
    return {
        "modelfile": f"FROM {d.get('id')}\n",
        "parameters": "\n".join(f"{k} {v}" for k, v in gen.items() if k != "do_sample"),
        "template": "",
        "details": {
            "parent_model": "",
            "format": "mlx",
            "family": arch,
            "families": [arch],
            "parameter_size": f"{params / 1e9:.1f}B" if params else "",
            "quantization_level": _quant_label_dict(q) or "",
        },
        "model_info": info,
        "capabilities": caps,
    }


def anthropic_capabilities(card: ModelCard) -> dict[str, Any]:
    """Anthropic ``ModelCapabilities`` (every key present, as the SDK model requires), plus the
    flat boolean keys LM Studio (``vision``, ``trained_for_tool_use``, ``reasoning``) and vLLM
    (``tools``, ``function_calling``, ``embedding``) style clients read from the same field."""
    text_like = card.kind in ("chat", "vlm", "omni")
    levels = card.reasoning.get("effort_levels") or []
    aliases = card.reasoning.get("effort_aliases") or {}

    def sup(v: bool) -> dict[str, bool]:
        return {"supported": bool(v)}

    def level(name: str) -> dict[str, bool]:
        return sup(name in levels or name in aliases)

    thinking = bool(card.reasoning.get("supported"))
    tools = bool(card.tools.get("supported"))
    vision = text_like and "image" in card.input_modalities
    return {
        "batch": sup(False),
        "citations": sup(False),
        "code_execution": sup(False),
        "context_management": {"supported": False},
        "effort": {
            "supported": bool(levels),
            "low": level("low"),
            "medium": level("medium"),
            "high": level("high"),
            "xhigh": level("xhigh"),
            "max": sup(False),
        },
        "image_input": sup(vision),
        "pdf_input": sup(False),
        "structured_outputs": sup(card.structured_output.get("json_schema")),
        "thinking": {
            "supported": thinking,
            "types": {
                "adaptive": sup(False),
                "enabled": sup(thinking and bool(card.reasoning.get("toggle"))),
            },
        },
        "vision": vision,
        "trained_for_tool_use": tools,
        "tools": tools,
        "function_calling": tools,
        "reasoning": thinking,
        "embedding": card.kind == "embedding",
    }
