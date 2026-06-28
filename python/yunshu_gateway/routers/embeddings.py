from __future__ import annotations

"""OpenAI Embeddings API compatible router.

Supports text embedding generation for semantic search, clustering, etc.
Uses MLX-native model inference (BGE, E5, Nomic, etc.).
When YUNSHU_ANE_EMBEDDINGS=1 is set and ANE is available, embeddings are
computed on the Apple Neural Engine via CoreML for lower latency and
reduced GPU contention.

All embeddings are L2-normalized to unit vectors by default, matching the
OpenAI API contract. Matryoshka dimension truncation is supported via the
``dimensions`` parameter.
"""

import logging
import math
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

from ..engine import get_engine, get_model_manager
from .models import _check_model_access, _check_permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["embeddings"])

_VALID_ENCODING_FORMATS = {"float", "base64"}
_MAX_INPUT_TEXT_LENGTH = 8192  # chars per text input
_MAX_TOTAL_INPUTS = 2048


class EmbeddingRequest(BaseModel):
    model: str
    # OpenAI /v1/embeddings spec accepts:
    # - str (single text)
    # - list[str] (batch of texts)
    # - list[int] (single text as token IDs)
    # - list[list[int]] (batch of texts as token IDs)
    # We accept all four; token IDs are decoded to text via the model's
    # tokenizer before embedding so the existing embedding path can reuse
    # its tokenize+pool pipeline.
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: str = "float"  # float, base64
    dimensions: int | None = None
    # explicit pooling override (MEAN / CLS / LAST). When unset, the
    # engine auto-detects from the model's 1_Pooling/config.json.
    pooling_type: str | None = None

    @model_validator(mode="after")
    def validate_request(self):
        if not self.model or not self.model.strip():
            raise ValueError("model: field is required and cannot be empty")
        if self.encoding_format not in _VALID_ENCODING_FORMATS:
            raise ValueError(
                f"encoding_format: must be one of {', '.join(sorted(_VALID_ENCODING_FORMATS))}, "
                f"got '{self.encoding_format}'"
            )
        # Validate input: string must be non-empty, list must have elements
        if isinstance(self.input, str) and not self.input.strip():
            raise ValueError("input: cannot be empty or whitespace-only")
        if isinstance(self.input, str) and len(self.input) > _MAX_INPUT_TEXT_LENGTH:
            raise ValueError(f"input: text length ({len(self.input)}) exceeds maximum ({_MAX_INPUT_TEXT_LENGTH})")
        if isinstance(self.input, list) and not self.input:
            raise ValueError("input: cannot be an empty list")
        if isinstance(self.input, list) and len(self.input) > _MAX_TOTAL_INPUTS:
            raise ValueError(f"input: too many inputs ({len(self.input)} > {_MAX_TOTAL_INPUTS})")
        # reject mixed-type lists. The handler discriminates on
        # input[0] only, so [1, "abc"] would silently mis-route to token-decode
        # and fail confusingly. Require homogeneous element types.
        if isinstance(self.input, list) and self.input:
            _first = type(self.input[0])
            if not all(isinstance(x, _first) for x in self.input):
                raise ValueError(
                    "input: list must be homogeneous (all strings, all token-ids, "
                    "or all token-id lists) — mixed types not allowed"
                )
            # enforce the per-input length cap on list[str] elements too. The cap
            # was only checked for a single-str input, so a list of N (up to 2048) arbitrarily
            # long strings bypassed it — an uncapped forward-pass / memory-pressure DoS seam on
            # a 36GB Mac. (Token-id-list inputs are bounded by the model context downstream.)
            if _first is str:
                for _i, _s in enumerate(self.input):
                    if len(_s) > _MAX_INPUT_TEXT_LENGTH:
                        raise ValueError(
                            f"input[{_i}]: text length ({len(_s)}) exceeds maximum "
                            f"({_MAX_INPUT_TEXT_LENGTH})")
        # Validate dimensions
        if self.dimensions is not None:
            if self.dimensions <= 0:
                raise ValueError("dimensions must be a positive integer")
            if self.dimensions > 8192:
                raise ValueError("dimensions must not exceed 8192")
        # validate the pooling override.
        if self.pooling_type is not None:
            _pt = self.pooling_type.upper()
            if _pt not in ("MEAN", "CLS", "LAST"):
                raise ValueError("pooling_type must be one of MEAN, CLS, LAST")
            self.pooling_type = _pt
        return self


@router.post("/embeddings", response_model=None)
async def create_embedding(req: EmbeddingRequest, request: Request):
    """Generate embeddings for the given input text(s)."""
    _check_permission(request, "can_infer")
    _check_model_access(request, req.model)
    return JSONResponse(await _embed_and_format(req))


async def _embed_and_format(req: EmbeddingRequest) -> dict:
    """Validate → resolve engine → embed → format an OpenAI embeddings dict.

    Auth-free core shared by the /v1/embeddings endpoint (which authorizes
    first) and the /v1/batch executor (authorized at submission time, like the
    other batch executors which call the engine directly). This is what
    un-stubbed batch embeddings — _execute_embedding used to raise
    "not yet supported".
    """
    # Handler-level validation (safety net beyond Pydantic model_validator)
    if req.encoding_format not in _VALID_ENCODING_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"encoding_format must be one of {', '.join(sorted(_VALID_ENCODING_FORMATS))}, got '{req.encoding_format}'",
        )
    if isinstance(req.input, str) and not req.input.strip():
        raise HTTPException(status_code=422, detail="input cannot be empty or whitespace-only")
    if isinstance(req.input, list):
        if not req.input:
            raise HTTPException(status_code=422, detail="input cannot be an empty list")
        if len(req.input) > _MAX_TOTAL_INPUTS:
            raise HTTPException(
                status_code=422,
                detail=f"input: too many inputs ({len(req.input)} > {_MAX_TOTAL_INPUTS})",
            )

    # Normalize input to a list[str] by decoding any token-id forms first.
    # OpenAI spec: list[int] = single text as tokens, list[list[int]] = batch.
    texts: list[str] = []
    if isinstance(req.input, str):
        texts = [req.input]
    elif isinstance(req.input, list) and req.input and isinstance(req.input[0], int):
        # Single sequence of token ids — validate the WHOLE list is ints (not
        # just the first element) so mixed-type inputs like [1, "abc"] are
        # rejected explicitly instead of crashing in tokenizer.decode().
        for i, tok_id in enumerate(req.input):
            if not isinstance(tok_id, int) or isinstance(tok_id, bool):
                raise HTTPException(
                    status_code=422,
                    detail=f"input at index {i} is not an integer token id",
                )
        # distinguish "model not found" (404, matching the string-input path at
        # L203) from "model found but has no tokenizer" (400). The old `... if req.model
        # else None` collapsed a missing/unknown model into a misleading 400.
        _eng = await _resolve_embedding_engine(req.model)
        if _eng is None:
            raise HTTPException(status_code=404, detail=f"Embedding model '{req.model}' not found")
        _tok = getattr(_eng, "_tokenizer", None)
        if _tok is None:
            raise HTTPException(status_code=400, detail="Cannot decode token-id input: model tokenizer unavailable")
        try:
            texts = [_tok.decode(req.input)]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode token-id input: {e}") from None
    elif isinstance(req.input, list) and req.input and isinstance(req.input[0], list):
        # Batch of token-id sequences — validate each sublist contains ints.
        for i, sub in enumerate(req.input):
            if not isinstance(sub, list):
                raise HTTPException(
                    status_code=422,
                    detail=f"input at index {i} must be a list of token ids",
                )
            for j, tok_id in enumerate(sub):
                if not isinstance(tok_id, int) or isinstance(tok_id, bool):
                    raise HTTPException(
                        status_code=422,
                        detail=f"input[{i}][{j}] is not an integer token id",
                    )
        _eng = await _resolve_embedding_engine(req.model)
        if _eng is None:
            raise HTTPException(status_code=404, detail=f"Embedding model '{req.model}' not found")
        _tok = getattr(_eng, "_tokenizer", None)
        if _tok is None:
            raise HTTPException(status_code=400, detail="Cannot decode token-id inputs: model tokenizer unavailable")
        try:
            texts = [_tok.decode(ids) for ids in req.input]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode token-id inputs: {e}") from None
    else:
        # Already list[str]
        for i, t in enumerate(req.input):
            if not isinstance(t, str) or not t.strip():
                raise HTTPException(
                    status_code=422,
                    detail=f"input at index {i} is empty or whitespace-only",
                )
        texts = list(req.input)

    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    # Reject empty strings in the list
    for i, t in enumerate(texts):
        if not t.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Input at index {i} is empty or whitespace-only",
            )

    if len(texts) > 2048:
        raise HTTPException(
            status_code=400,
            detail=f"Too many inputs: {len(texts)} > 2048",
        )

    # Resolve embedding engine
    engine = await _resolve_embedding_engine(req.model)
    if engine is None:
        raise HTTPException(
            status_code=404,
            detail=f"Embedding model '{req.model}' not found",
        )

    try:
        embeddings = await _generate_embeddings(
            engine, texts, model_id=req.model, pooling_type=req.pooling_type,
        )
    except MemoryError:
        logger.error("Embedding generation OOM", exc_info=True)
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except Exception as e:
        logger.error(f"Embedding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Embedding generation failed") from None

    # Resolve tokenizer for token counting (shared across iterations)
    tokenizer = getattr(engine, '_tokenizer', None) if engine else None

    # Format response
    data = []
    total_tokens = 0
    for i, (emb, text) in enumerate(zip(embeddings, texts, strict=False)):
        # Engine.embed() always returns a vector (zero vector for empty input),
        # so empty embeddings should not occur.  Warn and substitute a zero
        # vector to preserve the 1:1 index correspondence required by the
        # OpenAI API.  Skipping entries would create index gaps that break
        # client expectations.
        if not emb:
            logger.warning(
                "Empty embedding at index %d — engine returned no vector. "
                "Substituting zero vector to preserve index alignment.",
                i,
            )
            # Determine a reasonable dimension for the zero vector fallback.
            # Prefer another non-empty vector in this same batch (guarantees the
            # response's data[] are all equal-length, which the OpenAI contract and
            # downstream numpy/faiss loaders require). If the WHOLE batch is empty,
            # ask the engine for its real hidden_size rather than hardcoding 768 —
            # a 768 zero vector silently mismatches a 384/1024-dim model and poisons
            # the client's vector store.
            _native_dim = None
            for _e in embeddings:
                if _e:
                    _native_dim = len(_e)
                    break
            if _native_dim is None:
                try:
                    _native_dim = engine._get_hidden_size()
                except Exception:
                    _native_dim = None
            _dim = req.dimensions or _native_dim or 768
            emb = [0.0] * _dim

        # sanitize non-finite components (NaN/Inf) → 0.0 before serializing. A
        # NaN/Inf would (a) serialize as a bare NaN/Infinity literal in float mode (json
        # default allow_nan=True) → INVALID JSON per RFC 8259, which strict parsers (Go
        # encoding/json, simdjson) reject for the WHOLE response, and (b) pack garbage in
        # base64. This is the NaN-guard keystone (rerank/score/classify), un-propagated
        # to embeddings. embed() adds +1e-12 to the norm so it's rare, but the ANE /
        # fallback-logits paths can still yield a non-finite component.
        # sanitize BEFORE the Matryoshka truncation+renorm below, not after. With
        # the old order a single NaN component made norm = sqrt(...+NaN) = NaN, so `norm > 0`
        # was False and the renorm was SKIPPED — then the NaN was zeroed last, leaving a
        # NON-UNIT truncated vector (e.g. [3,4,NaN]→[3,4,0], L2=5). Cleaning first lets the
        # renorm see finite values and produce a true unit vector ([3,4,0]→[0.6,0.8,0]).
        if any(not math.isfinite(x) for x in emb):
            emb = [x if math.isfinite(x) else 0.0 for x in emb]

        # Truncate to requested dimensions (Matryoshka embedding support)
        if req.dimensions is not None and req.dimensions > 0:
            if len(emb) < req.dimensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Requested dimensions {req.dimensions} exceeds model's "
                           f"native embedding dimension {len(emb)}",
                )
            emb = emb[:req.dimensions]
            # Re-normalize after truncation to maintain unit vector property
            norm = math.sqrt(sum(x * x for x in emb))
            if norm > 0:
                emb = [x / norm for x in emb]

        if req.encoding_format == "base64":
            import base64
            import struct
            packed = struct.pack(f"{len(emb)}f", *emb)
            emb_value = base64.b64encode(packed).decode("ascii")
        else:
            emb_value = emb

        data.append({
            "object": "embedding",
            "index": i,
            "embedding": emb_value,
        })
        # Use tokenizer for accurate token count, fallback to word count.
        # Exclude special tokens (BOS/EOS) from the count to match OpenAI's
        # behavior where prompt_tokens only includes content tokens.
        if tokenizer:
            try:
                total_tokens += len(tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                # Tokenizer doesn't support add_special_tokens — some MLX
                # tokenizers only accept a single positional argument.
                # The encode() result typically includes BOS (+1) and may
                # include EOS (+1), so we subtract up to 2 as a heuristic.
                # Guard against negative count for very short texts.
                raw_count = len(tokenizer.encode(text))
                total_tokens += max(1, raw_count - 2)
        else:
            total_tokens += max(1, len(text) // 4)

    _record_embedding_metrics(total_tokens)

    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


def _record_embedding_metrics(prompt_tokens: int) -> None:
    """Record embedding request metrics."""
    try:
        from ..middleware.metrics import get_metrics
        get_metrics().record_tokens(prompt_tokens, 0)
        get_metrics().record_inference()
    except Exception:
        pass
    # feed the per-request TPM box (embeddings bill prompt tokens only).
    try:
        from ..usage_context import record_billed_tokens
        record_billed_tokens(prompt_tokens or 0)
    except Exception:
        pass


async def _resolve_embedding_engine(model_id: str):
    """Find an embedding engine for the given model."""

    # Try model manager first
    manager = get_model_manager()
    if manager is not None:
        # Direct lookup
        entry = manager.get_entry(model_id)
        if entry is not None and entry.is_loaded and entry.engine is not None:
            return entry.engine

        # Try loading
        try:
            engine = await manager.get_engine(model_id)
            return engine
        except KeyError:
            logger.debug(f"Model not registered: {model_id}")
            # Not registered — fall through to single-engine only if manager
            # has no entries at all (single-engine mode with manager stub)
        except Exception:
            logger.warning(f"Failed to load engine for {model_id}", exc_info=True)
            # Do NOT fall through to single-engine — the user asked for a
            # specific model via the manager, and serving embeddings from a
            # different model would be silently incorrect.
            return None

    # Single engine
    engine = get_engine()
    if engine and engine.is_loaded:
        return engine

    return None


async def _generate_embeddings(engine, texts: list[str], model_id: str = "",
                               pooling_type: str | None = None) -> list[list[float]]:
    """Generate embeddings using the engine.

    Supports:
    1. ANE path: when YUNSHU_ANE_EMBEDDINGS=1 and ANE is available
    2. Engine with embed() method (native embedding model, already normalized)
    3. Fallback: use hidden states from the model with L2 normalization

    ``pooling_type`` (MEAN/CLS/LAST) overrides the model's auto-detected
    pooling via the engine's pool() — for callers who know their model's correct
    pooling or want a specific one.
    """
    # ── Explicit pooling override → engine.pool() (skips ANE/embed auto-detect) ──
    if pooling_type and hasattr(engine, "pool"):
        import asyncio

        from yunshu_engine.mlx_executor import get_mlx_executor
        loop = asyncio.get_running_loop()
        vecs = await loop.run_in_executor(
            get_mlx_executor(), lambda: engine.pool(texts, pooling_type),
        )
        # pool() returns UN-normalized vectors, but embed() (the
        # auto-detect path) returns L2-normalized ones and the OpenAI/vLLM
        # contract is unit vectors. L2-normalize here so the override doesn't
        # silently yield non-unit embeddings (wrong cosine similarities).
        import math
        out = []
        for v in vecs:
            norm = math.sqrt(sum(x * x for x in v))
            out.append([x / norm for x in v] if norm > 0 else v)
        return out
    # ── ANE path: offload to Apple Neural Engine via CoreML ──
    if os.environ.get("YUNSHU_ANE_EMBEDDINGS", "").strip() in ("1", "true", "yes"):
        # the ANE processor is a SINGLETON bound to
        # YUNSHU_ANE_EMBEDDING_MODEL. Only route a request to it when the caller's
        # model matches (or none was specified) — otherwise it would serve a
        # DIFFERENT model's vectors (wrong dimensionality / embedding space) than
        # the request asked for. Mismatched models fall through to the MLX path.
        _ane_model = os.environ.get("YUNSHU_ANE_EMBEDDING_MODEL", "intfloat/e5-small-v2")
        _ane_ok = (not model_id) or model_id == _ane_model \
            or model_id.split("/")[-1].lower() == _ane_model.split("/")[-1].lower()
        try:
            from yunshu_engine.ane_embedding import get_ane_processor
            proc = get_ane_processor() if _ane_ok else None
            if proc is not None:
                logger.debug("Using ANE for embedding inference (model_id=%s)", model_id)
                import asyncio
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, proc.embed, texts)
        except Exception as exc:
            logger.warning(
                "ANE embedding failed, falling back to GPU: %s", exc, exc_info=True,
            )

    # ── Engine with embed() method (BatchedEngine.embed normalizes by default) ──
    if hasattr(engine, 'embed'):
        import asyncio

        from yunshu_engine.mlx_executor import get_mlx_executor
        loop = asyncio.get_running_loop()
        # embed() is synchronous and does GPU work — run on MLX executor thread
        return await loop.run_in_executor(get_mlx_executor(), engine.embed, texts)

    # Fallback: use the model's tokenizer + forward pass for last hidden state
    tokenizer = getattr(engine, '_tokenizer', None)
    model = getattr(engine, '_model', None)
    if tokenizer is None or model is None:
        raise RuntimeError("Engine does not support embedding generation")

    import asyncio

    import mlx.core as mx

    from yunshu_engine.mlx_executor import get_mlx_executor

    loop = asyncio.get_running_loop()

    def _generate_all():
        embeddings = []
        # Resolve the transformer backbone (before LM head projection).
        # MLX models vary: Qwen3.5 uses language_model.model, standard
        # HF uses model, some use transformer, etc.
        backbone = None
        for path in [
            lambda m: getattr(m, 'language_model', None) and getattr(m.language_model, 'model', None),
            lambda m: getattr(m, 'model', None),
            lambda m: getattr(m, 'transformer', None),
        ]:
            candidate = path(model)
            if candidate is not None and callable(candidate):
                backbone = candidate
                break

        for text in texts:
            tokens = tokenizer.encode(text)
            if not tokens:
                embeddings.append([0.0] * 768)
                continue

            input_ids = mx.array([tokens])

            if backbone is not None:
                # Initialize KV caches for each layer if backbone needs them.
                # Use the model's make_cache() if available (handles hybrid
                # architectures like Qwen3.5 with both KVCache and ArraysCache).
                # make_cache() may be on the backbone, the LanguageModel, or
                # the top-level model (qwen3_5.Model → language_model → model).
                caches = None
                _lm = getattr(model, 'language_model', None)
                for _obj in [backbone, _lm, model]:
                    if _obj is not None and hasattr(_obj, 'make_cache'):
                        try:
                            caches = _obj.make_cache()
                            if caches:
                                break
                        except Exception:
                            caches = None

                try:
                    if caches:
                        hidden = backbone(input_ids, cache=caches)
                    else:
                        hidden = backbone(input_ids)
                except TypeError:
                    # backbone doesn't accept cache args
                    hidden = backbone(input_ids)
            else:
                output = model(input_ids)
                hidden = _extract_hidden(output)

            # Mean pooling over sequence dimension
            pooled = mx.mean(hidden, axis=1).squeeze(0)

            # L2 normalize (OpenAI API contract)
            norm = mx.sqrt(mx.sum(pooled * pooled) + 1e-12)
            pooled = pooled / norm

            embeddings.append(pooled.tolist())
        return embeddings

    return await loop.run_in_executor(get_mlx_executor(), _generate_all)


def _extract_hidden(output) -> Any:
    """Extract hidden states from various model output formats."""
    import mlx.core as mx

    if isinstance(output, mx.array):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state
    if hasattr(output, 'hidden_states') and output.hidden_states:
        return output.hidden_states[-1]
    if hasattr(output, 'logits'):
        # For causal LMs, the logits tensor IS the last layer output (before
        # softmax).  This is a reasonable approximation for models that don't
        # expose explicit hidden states, but it is not optimal — the logits
        # have been projected through the vocabulary head which distorts the
        # embedding space.
        logger.warning(
            "Using logits as embedding — model may not produce optimal embeddings. "
            "Consider using a model with explicit hidden state outputs."
        )
        return output.logits
    raise ValueError(
        "Model output has no hidden states or logits — cannot generate embeddings"
    )
