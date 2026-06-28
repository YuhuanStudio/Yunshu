from __future__ import annotations

"""OpenAI Tokenize API compatible router.

Supports encode, decode, and token counting. Model-aware:
resolves tokenizer from loaded engine or model manager with
case-insensitive and provider-prefix fallbacks.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, model_validator

from ..engine import get_engine, get_model_manager

router = APIRouter(tags=["tokenize"])


class TokenizeRequest(BaseModel):
    model: str
    text: str | list[str]
    add_special_tokens: bool = True

    @model_validator(mode="after")
    def _check_model(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        return self


class TokenCountRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int = 0
    add_special_tokens: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_input_alias(cls, data):
        """Accept 'input' as an alias for 'prompt' (OpenAI embeddings/tokenize convention).

        If both are provided, 'prompt' wins (explicit canonical name).
        """
        if isinstance(data, dict):
            if data.get("prompt") is None and data.get("input") is not None:
                data = dict(data)
                data["prompt"] = data["input"]
        return data

    @model_validator(mode="after")
    def _check_model(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        return self


class DetokenizeRequest(BaseModel):
    model: str
    tokens: list[int]
    skip_special_tokens: bool = True

    @model_validator(mode="after")
    def _check_model(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        return self


@router.post("/tokenize", response_model=None)
async def tokenize(req: TokenizeRequest, request: Request):
    """Tokenize text into token IDs."""
    from .models import _check_model_access, _check_permission
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    tokenizer = _resolve_tokenizer(req.model)
    texts = req.text if isinstance(req.text, list) else [req.text]
    all_tokens = []
    for text in texts:
        if req.add_special_tokens:
            tokens = tokenizer.encode(text)
        else:
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                tokens = tokenizer.encode(text)
        all_tokens.append(tokens)

    return {
        "tokens": all_tokens if isinstance(req.text, list) else all_tokens[0],
        "count": sum(len(t) for t in all_tokens),
        "model": req.model,
    }


@router.post("/detokenize", response_model=None)
async def detokenize(req: DetokenizeRequest, request: Request):
    """Convert token IDs back to text."""
    from .models import _check_model_access, _check_permission
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    tokenizer = _resolve_tokenizer(req.model)
    # A negative (or otherwise out-of-range) token id makes
    # tokenizer.decode raise OverflowError → a bare 500. Shape it into a clean 422 for a
    # malformed-but-typed input (embeddings.py guards the identical decode; tokenize didn't).
    try:
        if req.skip_special_tokens:
            try:
                text = tokenizer.decode(req.tokens, skip_special_tokens=True)
            except TypeError:
                text = tokenizer.decode(req.tokens)
        else:
            text = tokenizer.decode(req.tokens)
    except (OverflowError, ValueError, IndexError, KeyError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid token id in 'tokens': {e}") from None
    return {"text": text, "model": req.model}


@router.post("/token_count", response_model=None)
async def token_count(req: TokenCountRequest, request: Request):
    """Count tokens for a prompt."""
    from .models import _check_model_access, _check_permission
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    tokenizer = _resolve_tokenizer(req.model)
    texts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    counts = []
    for t in texts:
        if req.add_special_tokens:
            counts.append(len(tokenizer.encode(t)))
        else:
            try:
                counts.append(len(tokenizer.encode(t, add_special_tokens=False)))
            except TypeError:
                counts.append(len(tokenizer.encode(t)))
    total = sum(counts)
    # Resolve model context length: caller override > model config > 0.
    # Previously this returned `req.max_tokens` (the request's max OUTPUT
    # tokens, defaulting to 0) under the field name `max_context_tokens`,
    # which is misleading — callers expect the model's CONTEXT window.
    ctx_limit = req.max_tokens if req.max_tokens > 0 else _resolve_context_limit(req.model)
    # A list `prompt` is a batch of INDEPENDENT prompts, each sent in its
    # own request — so over_context_limit must be "does ANY single prompt overflow",
    # not "does their SUM overflow" (the old `total > ctx_limit` falsely flagged e.g.
    # 5×4k prompts against a 32k window though none individually overflows). For a
    # string prompt, total == counts[0], so `any(...)` is equivalent.
    over = any(c > ctx_limit for c in counts) if ctx_limit > 0 else False
    return {
        "token_count": total if isinstance(req.prompt, str) else counts,
        "max_context_tokens": ctx_limit,
        "over_context_limit": over,
        "model": req.model,
    }


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
                        for attr in ("max_position_embeddings", "max_seq_len", "context_length"):
                            v = getattr(s, attr, None)
                            if isinstance(v, int) and v > 0:
                                return v
            # 3) engine._config / engine.config dict (VLM path)
            for cfg_attr in ("_config", "config"):
                cfg = getattr(eng, cfg_attr, None)
                if isinstance(cfg, dict):
                    for k in ("max_position_embeddings", "context_length", "max_seq_len"):
                        v = cfg.get(k)
                        if isinstance(v, int) and v > 0:
                            return v
                    tc = cfg.get("text_config") or cfg.get("thinker_config", {}).get("text_config")
                    if isinstance(tc, dict):
                        for k in ("max_position_embeddings", "context_length", "max_seq_len"):
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
            tok = getattr(entry.engine, '_tokenizer', None)
            if tok:
                return tok

        # Case-insensitive fallback
        lower = model_id.lower()
        for e in manager.list_entries():
            if e.model_id.lower() == lower and e.is_loaded and e.engine:
                tok = getattr(e.engine, '_tokenizer', None)
                if tok:
                    return tok

        # Provider prefix stripping (e.g. "org/model" -> "model")
        if '/' in model_id:
            stripped = model_id.rsplit('/', 1)[-1]
            for e in manager.list_entries():
                if e.model_id.lower() == stripped.lower() and e.is_loaded and e.engine:
                    tok = getattr(e.engine, '_tokenizer', None)
                    if tok:
                        return tok

    # Default-engine fallback — but ONLY when it actually IS the requested model.
    # The old code returned the global default engine's tokenizer for ANY unloaded
    # model_id → callers got token IDs/counts computed with the WRONG vocabulary while
    # the response echoed the requested model name (silent, no error). Use the fallback
    # only in single-model mode (no manager) or when the engine's identity matches.
    engine = get_engine()
    if engine:
        tok = getattr(engine, '_tokenizer', None)
        if tok:
            _eng_id = str(getattr(engine, "model_name", "") or "")
            _match = (
                manager is None  # single-model deployment: the default engine IS the model
                or not model_id
                or model_id == _eng_id
                or model_id.lower() == _eng_id.lower()
                or (("/" in _eng_id) and model_id.lower() == _eng_id.rsplit("/", 1)[-1].lower())
                or (("/" in model_id) and model_id.rsplit("/", 1)[-1].lower() == _eng_id.lower())
            )
            if _match:
                return tok

    raise HTTPException(status_code=404, detail=f"No tokenizer available for model '{model_id}'")
