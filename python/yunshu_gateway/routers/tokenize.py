from __future__ import annotations
"""OpenAI Tokenize API compatible router.

Supports encode, decode, and token counting. Model-aware:
resolves tokenizer from loaded engine or model manager with
case-insensitive and provider-prefix fallbacks.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..engine import get_engine, get_model_manager

router = APIRouter(tags=["tokenize"])


class TokenizeRequest(BaseModel):
    model: str
    text: str | list[str]
    add_special_tokens: bool = True


class TokenCountRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int = 0
    add_special_tokens: bool = True


class DetokenizeRequest(BaseModel):
    model: str
    tokens: list[int]
    skip_special_tokens: bool = True


@router.post("/tokenize", response_model=None)
async def tokenize(req: TokenizeRequest):
    """Tokenize text into token IDs."""
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
async def detokenize(req: DetokenizeRequest):
    """Convert token IDs back to text."""
    tokenizer = _resolve_tokenizer(req.model)
    if req.skip_special_tokens:
        try:
            text = tokenizer.decode(req.tokens, skip_special_tokens=True)
        except TypeError:
            text = tokenizer.decode(req.tokens)
    else:
        text = tokenizer.decode(req.tokens)
    return {"text": text, "model": req.model}


@router.post("/token_count", response_model=None)
async def token_count(req: TokenCountRequest):
    """Count tokens for a prompt."""
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
    return {
        "token_count": total if isinstance(req.prompt, str) else counts,
        "max_context_tokens": req.max_tokens,
        "over_context_limit": (total > req.max_tokens) if req.max_tokens > 0 else False,
        "model": req.model,
    }


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

    engine = get_engine()
    if engine:
        tok = getattr(engine, '_tokenizer', None)
        if tok:
            return tok

    raise HTTPException(status_code=404, detail=f"No tokenizer available for model '{model_id}'")
