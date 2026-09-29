from __future__ import annotations

"""OpenAI Tokenize API compatible router.

Supports encode, decode, and token counting. Model-aware:
resolves tokenizer from loaded engine or model manager with
case-insensitive and provider-prefix fallbacks.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, model_validator

from ..engine import get_engine, get_model_manager

router = APIRouter(tags=["tokenize"])


class TokenizeRequest(BaseModel):
    """vLLM `/tokenize` request: either `prompt` (raw text) or `messages` (chat template)."""

    model: str | None = None
    prompt: str | list[str] | None = None
    messages: list[dict[str, Any]] | None = None
    add_special_tokens: bool | None = None
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    return_token_strs: bool = False

    @model_validator(mode="before")
    @classmethod
    def _legacy_text_alias(cls, data):
        if isinstance(data, dict) and data.get("prompt") is None:
            for alias in ("text", "input"):
                if data.get(alias) is not None:
                    data = dict(data)
                    data["prompt"] = data[alias]
                    break
        return data

    @model_validator(mode="after")
    def _one_of(self):
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("provide exactly one of 'prompt' or 'messages'")
        return self


class DetokenizeRequest(BaseModel):
    model: str | None = None
    tokens: list[int]
    skip_special_tokens: bool = True


def _flatten_content(content) -> str:
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") in ("text", "input_text")
        )
    return (
        content
        if isinstance(content, str)
        else ("" if content is None else str(content))
    )


def _encode(tokenizer, text: str, add_special_tokens: bool) -> list[int]:
    if add_special_tokens:
        return list(tokenizer.encode(text))
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _token_strs(tokenizer, ids: list[int]) -> list[str]:
    out = []
    for i in ids:
        try:
            out.append(tokenizer.decode([i]))
        except Exception:
            out.append("")
    return out


@router.post("/tokenize", response_model=None)
async def tokenize(req: TokenizeRequest, request: Request):
    """Tokenize a prompt or a chat conversation (vLLM `/tokenize` schema)."""
    from .models import _check_model_access, _check_permission

    _check_permission(request, "can_infer")
    model = req.model or ""
    if model:
        _check_model_access(request, model)
    tokenizer = _resolve_tokenizer(model)
    if req.messages is not None:
        msgs = [
            {**m, "content": _flatten_content(m.get("content"))} for m in req.messages
        ]
        kwargs = dict(req.chat_template_kwargs or {})
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tools=req.tools,
                tokenize=False,
                add_generation_prompt=req.add_generation_prompt
                and not req.continue_final_message,
                continue_final_message=req.continue_final_message,
                **kwargs,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"chat template failed: {e}"
            ) from None
        ids = _encode(tokenizer, text, bool(req.add_special_tokens))
        batches = [ids]
        single = True
    else:
        prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        add = True if req.add_special_tokens is None else req.add_special_tokens
        batches = [_encode(tokenizer, p, add) for p in prompts]
        single = not isinstance(req.prompt, list)
    max_len = _resolve_context_limit(model) if model else 0
    if not max_len:
        eng = get_engine()
        max_len = _resolve_context_limit(getattr(eng, "model_name", "") or "")
    resp: dict[str, Any] = {
        "count": sum(len(b) for b in batches),
        "max_model_len": max_len or None,
        "tokens": batches[0] if single else batches,
    }
    if req.return_token_strs:
        resp["token_strs"] = (
            _token_strs(tokenizer, batches[0])
            if single
            else [_token_strs(tokenizer, b) for b in batches]
        )
    return resp


@router.post("/detokenize", response_model=None)
async def detokenize(req: DetokenizeRequest, request: Request):
    """Convert token IDs back to text (vLLM `/detokenize` schema)."""
    from .models import _check_model_access, _check_permission

    _check_permission(request, "can_infer")
    model = req.model or ""
    if model:
        _check_model_access(request, model)
    tokenizer = _resolve_tokenizer(model)
    try:
        try:
            text = tokenizer.decode(
                req.tokens, skip_special_tokens=req.skip_special_tokens
            )
        except TypeError:
            text = tokenizer.decode(req.tokens)
    except (OverflowError, ValueError, IndexError, KeyError) as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid token id in 'tokens': {e}"
        ) from None
    return {"prompt": text}


def _resolve_context_limit(model_id: str) -> int:
    """Resolve the model's max context window from loaded-engine config.

    Returns 0 if no loaded engine matches (caller should treat that as
    "unknown" rather than "no limit").
    """
    manager = get_model_manager()
    if manager is not None:
        entry = manager.get_entry(model_id)
        engines = [entry.engine] if (entry and entry.is_loaded and entry.engine) else []
        if not engines:
            for e in manager.list_entries():
                if e.is_loaded and e.engine and e.model_id.lower() == model_id.lower():
                    engines.append(e.engine)
                    break
        for eng in engines:
            # 1) Direct attrs on engine (LLM path)
            for attr in ("max_position_embeddings", "context_length", "max_seq_len"):
                v = getattr(eng, attr, None)
                if isinstance(v, int) and v > 0:
                    return v
            # 2) engine._model.{max_seq_len, config.max_position_embeddings, args.max_seq_len}
            mdl = getattr(eng, "_model", None) or getattr(eng, "model", None)
            if mdl is not None:
                for attr in ("max_seq_len", "max_position_embeddings"):
                    v = getattr(mdl, attr, None)
                    if isinstance(v, int) and v > 0:
                        return v
                for sub in ("config", "args"):
                    s = getattr(mdl, sub, None)
                    if s is not None:
                        for attr in (
                            "max_position_embeddings",
                            "max_seq_len",
                            "context_length",
                        ):
                            v = getattr(s, attr, None)
                            if isinstance(v, int) and v > 0:
                                return v
            # 3) engine._config / engine.config dict (VLM path)
            for cfg_attr in ("_config", "config"):
                cfg = getattr(eng, cfg_attr, None)
                if isinstance(cfg, dict):
                    for k in (
                        "max_position_embeddings",
                        "context_length",
                        "max_seq_len",
                    ):
                        v = cfg.get(k)
                        if isinstance(v, int) and v > 0:
                            return v
                    tc = cfg.get("text_config") or cfg.get("thinker_config", {}).get(
                        "text_config"
                    )
                    if isinstance(tc, dict):
                        for k in (
                            "max_position_embeddings",
                            "context_length",
                            "max_seq_len",
                        ):
                            v = tc.get(k)
                            if isinstance(v, int) and v > 0:
                                return v
    return 0


def _resolve_tokenizer(model_id: str):
    """Resolve tokenizer by model ID with case-insensitive and prefix-stripping fallbacks."""
    manager = get_model_manager()
    if manager is not None:
        entry = manager.get_entry(model_id)
        if entry and entry.is_loaded and entry.engine:
            tok = getattr(entry.engine, "_tokenizer", None)
            if tok:
                return tok

        # Case-insensitive fallback
        lower = model_id.lower()
        for e in manager.list_entries():
            if e.model_id.lower() == lower and e.is_loaded and e.engine:
                tok = getattr(e.engine, "_tokenizer", None)
                if tok:
                    return tok

        # Provider prefix stripping (e.g. "org/model" -> "model")
        if "/" in model_id:
            stripped = model_id.rsplit("/", 1)[-1]
            for e in manager.list_entries():
                if e.model_id.lower() == stripped.lower() and e.is_loaded and e.engine:
                    tok = getattr(e.engine, "_tokenizer", None)
                    if tok:
                        return tok

    # Default-engine fallback — but ONLY when it actually IS the requested model.
    # The old code returned the global default engine's tokenizer for ANY unloaded
    # model_id → callers got token IDs/counts computed with the WRONG vocabulary while
    # the response echoed the requested model name (silent, no error). Use the fallback
    # only in single-model mode (no manager) or when the engine's identity matches.
    engine = get_engine()
    if engine:
        tok = getattr(engine, "_tokenizer", None)
        if tok:
            _eng_id = str(getattr(engine, "model_name", "") or "")
            _match = (
                manager
                is None  # single-model deployment: the default engine IS the model
                or not model_id
                or model_id == _eng_id
                or model_id.lower() == _eng_id.lower()
                or (
                    ("/" in _eng_id)
                    and model_id.lower() == _eng_id.rsplit("/", 1)[-1].lower()
                )
                or (
                    ("/" in model_id)
                    and model_id.rsplit("/", 1)[-1].lower() == _eng_id.lower()
                )
            )
            if _match:
                return tok

    raise HTTPException(
        status_code=404, detail=f"No tokenizer available for model '{model_id}'"
    )
