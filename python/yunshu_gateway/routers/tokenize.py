"""OpenAI Tokenize API compatible router."""
from __future__ import annotations

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
            tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.append(tokens)

    return {
        "tokens": all_tokens if isinstance(req.text, list) else all_tokens[0],
        "model": req.model,
    }


@router.post("/detokenize", response_model=None)
async def detokenize(model: str, tokens: list[int]):
    """Convert token IDs back to text."""
    tokenizer = _resolve_tokenizer(model)
    text = tokenizer.decode(tokens)
    return {"text": text, "model": model}


@router.post("/token_count", response_model=None)
async def token_count(req: TokenCountRequest):
    """Count tokens for a prompt."""
    tokenizer = _resolve_tokenizer(req.model)
    texts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    counts = [len(tokenizer.encode(t)) for t in texts]
    total = sum(counts)
    return {
        "token_count": total if isinstance(req.prompt, str) else counts,
        "max_context_tokens": req.max_tokens,
        "over_context_limit": (total > req.max_tokens) if req.max_tokens > 0 else False,
        "model": req.model,
    }


def _resolve_tokenizer(model_id: str):
    from yunshu_engine.batched_engine import BatchedEngine

    manager = get_model_manager()
    if manager is not None:
        entry = manager._entries.get(model_id)
        if entry and entry.is_loaded and entry.engine:
            tok = getattr(entry.engine, '_tokenizer', None)
            if tok:
                return tok

    engine = get_engine()
    if engine:
        tok = getattr(engine, '_tokenizer', None)
        if tok:
            return tok

    raise HTTPException(status_code=404, detail=f"No tokenizer available for model '{model_id}'")
