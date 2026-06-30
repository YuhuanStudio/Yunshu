"""OCR endpoint — POST /v1/ocr for image-to-text extraction."""

import contextlib
import logging
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ocr"])


@router.post("/v1/ocr")
async def extract_text_from_image(
    request: Request,
    file: UploadFile = File(...),
    language: str | None = Form(None),
    model: str = Form(""),
    task: str = Form("text"),
):
    """Extract text from an uploaded image using OCR.

    Accepts PNG, JPG, JPEG, WEBP, TIFF image formats. The `language` field is a hint
    passed to the model (it steers output language); the response's `language`/
    `confidence` are echoed/None — no language detection is performed.
    """
    from .models import _check_model_access, _check_permission

    _check_permission(request, "can_infer")
    _check_model_access(request, model)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10 MB
        raise HTTPException(status_code=413, detail="Image file too large (max 10MB)")

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}
    raw_suffix = os.path.splitext(file.filename or "image.png")[1].lower()
    suffix = raw_suffix if raw_suffix in _IMAGE_EXTENSIONS else ".png"

    # wire request-tracker registration + client-disconnect cancellation
    # (the cancel keystone, un-propagated to OCR — audio/images already had it;
    # OCR + video were the missed single-shot-media siblings). The native OCR path uses a
    # blocking mlx_vlm.generate (bounded by max_tokens, not interruptible mid-decode), but
    # registering frees the tracker slot and returns the handler promptly on disconnect
    # (no head-of-line block on the event loop), and the VLM-fallback path
    # (vlm_engine.generate) honors cancel_event between decode steps → real mid-gen stop.
    import uuid as _uuid

    from ..streaming import run_with_disconnect_guard

    _ocr_id = f"ocr-{_uuid.uuid4().hex[:24]}"
    _ocr_tracker = None
    _ocr_cancel = None
    try:
        from yunshu_engine.request_tracker import get_request_tracker

        _ocr_tracker = get_request_tracker()
        _ocr_cancel = _ocr_tracker.register(_ocr_id, model or "ocr").cancel_event
    except Exception:
        _ocr_tracker = None

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)

        # Find OCR engine
        from yunshu_engine.ocr_engine import OCREngine

        from ..engine import get_model_manager

        manager = get_model_manager()

        # Single-model mode: the served model is the global engine (no manager entry).
        ocr_engine = None
        ocr_model_id = None
        try:
            from ..engine import get_engine

            _global = get_engine()
            if isinstance(_global, OCREngine):
                ocr_engine, ocr_model_id = _global, model or "ocr"
        except Exception:
            pass

        # select by model_id (was: first loaded OCREngine, ignoring `model` —
        # the wrong-model keystone, unswept for OCR). Fall back to first-of-type
        # only when `model` is empty (legacy default).
        _first_ocr = None
        _first_ocr_id = None
        _model_lower = model.lower() if model else ""
        _entries = manager.list_entries() if manager is not None else ()
        for entry in _entries:
            if not (
                entry.is_loaded
                and isinstance(getattr(entry, "engine", None), OCREngine)
            ):
                continue
            if _first_ocr is None:
                _first_ocr, _first_ocr_id = entry.engine, entry.model_id
            if _model_lower and (
                entry.model_id == model or entry.model_id.lower() == _model_lower
            ):
                ocr_engine, ocr_model_id = entry.engine, entry.model_id
                break
        if ocr_engine is None and not _model_lower:
            ocr_engine, ocr_model_id = _first_ocr, _first_ocr_id

        if ocr_engine is None:
            # Try loading by model name
            if model:
                try:
                    engine = await manager.get_engine(model)
                    if isinstance(engine, OCREngine):
                        ocr_engine, ocr_model_id = engine, model
                except Exception:
                    logger.debug("failed", exc_info=True)

        if ocr_engine is not None:
            # SECURITY: re-check the RESOLVED model against the key's scope.
            # _check_model_access(model) above is a no-op when model is empty (the
            # default), so without this a model-scoped key could OCR through a model it
            # cannot access simply by omitting `model`. Mirror isolation.
            _check_model_access(request, ocr_model_id)
            # disconnect guard frees the handler promptly on client disconnect
            # (the native blocking generate still runs to completion on the executor, but
            # bounded by max_tokens; None on disconnect → empty result, discarded anyway).
            result = (
                await run_with_disconnect_guard(
                    request,
                    ocr_engine.extract_text(tmp_path, language=language, task=task),
                    cancel_event=_ocr_cancel,
                )
                or {}
            )
            prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
            completion_tokens = int(result.get("completion_tokens", 0) or 0)
            image_tokens = int(result.get("image_tokens", 0) or 0)
            total_tokens = int(
                result.get("total_tokens", prompt_tokens + completion_tokens)
            )
            return {
                "text": result.get("text", ""),
                # include the resolved model id — the VLM-fallback path returns
                # "model" but the native path omitted it, so a client keying on response
                # ["model"] KeyError'd when a real GLM-OCR engine served the request.
                "model": ocr_model_id,
                "confidence": result.get("confidence", 0.0),
                "language": result.get("language", language),
                "task": task,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "image_tokens": image_tokens,
                },
            }

        # VLM fallback: route OCR through any loaded VLM-type model via chat
        # completions semantics (single image + extraction prompt).
        from yunshu_engine.model_manager import ModelType
        from yunshu_engine.vlm_engine import VLMEngine

        vlm_engine = None
        vlm_model_id = None
        if model:
            # Prefer the explicitly requested model when it is a loaded VLM.
            for entry in manager.list_entries():
                if (
                    entry.model_id == model
                    and entry.is_loaded
                    and entry.model_type == ModelType.VLM
                    and isinstance(getattr(entry, "engine", None), VLMEngine)
                ):
                    vlm_engine = entry.engine
                    vlm_model_id = entry.model_id
                    break
        if vlm_engine is None:
            # Any loaded VLM model can serve as an OCR fallback.
            for entry in manager.list_entries():
                if (
                    entry.is_loaded
                    and entry.model_type == ModelType.VLM
                    and isinstance(getattr(entry, "engine", None), VLMEngine)
                ):
                    vlm_engine = entry.engine
                    vlm_model_id = entry.model_id
                    break

        if vlm_engine is None:
            # 503 rather than 404 for parity with /v1/audio/voice-pipeline: the
            # endpoint exists in the OpenAPI surface, it just has no backing
            # engine loaded. 404 (`model_not_found`) would imply the resource
            # itself is missing.
            raise HTTPException(
                status_code=503,
                detail="No OCR engine available — load a VLM model (e.g. GLM-OCR-bf16) first",
            )

        # SECURITY: re-check the RESOLVED fallback VLM against the key's scope
        # (the empty-model default made the top-of-handler _check_model_access a no-op,
        # so a scoped key could OCR through any loaded VLM by omitting `model`).
        _check_model_access(request, vlm_model_id)

        # Build OpenAI-style chat message with single image + extraction prompt.
        # GLM-OCR-family models respond to "Text Recognition:" / "Formula Recognition:"
        # / "Table Recognition:" task prompts (see ocr_engine.py:_TASK_PROMPTS); for
        # generic VLMs, use a plain instruction. We send both for best coverage.
        _OCR_TASK_PROMPTS = {
            "text": "Text Recognition:",
            "formula": "Formula Recognition:",
            "table": "Table Recognition:",
        }
        _task_lc = (task or "text").lower()
        _anchor = _OCR_TASK_PROMPTS.get(_task_lc, "Text Recognition:")
        prompt_text = (
            f"{_anchor} Extract all text from this image verbatim. "
            "Output only the extracted text, no commentary."
        )
        if language:
            prompt_text += f" Respond in {language}."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{tmp_path}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        try:
            # the VLM fallback honors cancel_event between decode steps, so the
            # disconnect guard delivers REAL mid-generation cancellation on this path.
            result = await run_with_disconnect_guard(
                request,
                vlm_engine.generate(
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.0,
                    top_p=1.0,
                    cancel_event=_ocr_cancel,
                ),
                cancel_event=_ocr_cancel,
            )
        except MemoryError:
            raise
        except Exception as e:
            logger.error(
                f"VLM OCR fallback failed for {vlm_model_id}: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500, detail="OCR extraction failed"
            ) from None

        text = (result.get("text") if isinstance(result, dict) else "") or ""

        # Surface usage so callers can meter image-input cost. The VLM engine
        # may already include image-token contributions in its own usage dict;
        # if it doesn't, estimate from the uploaded image and the engine's
        # vision config so prompt_tokens is not artificially tiny.
        # VLMEngine.generate() returns TOP-LEVEL prompt_tokens/
        # completion_tokens, NOT a `usage` sub-dict. The old code read result["usage"]
        # (always None) → both counts came out 0, so the engine's real completion count
        # was discarded and prompt_tokens collapsed to image_tokens only (text prompt
        # lost). Read top-level first, fall back to a usage dict for forward-compat.
        _res = result if isinstance(result, dict) else {}
        usage = _res.get("usage") or {}
        prompt_tokens = int(
            _res.get("prompt_tokens", usage.get("prompt_tokens", 0)) or 0
        )
        completion_tokens = int(
            _res.get("completion_tokens", usage.get("completion_tokens", 0)) or 0
        )
        image_tokens = int(_res.get("image_tokens", usage.get("image_tokens", 0)) or 0)
        if image_tokens == 0:
            try:
                from PIL import Image as _Image

                with _Image.open(tmp_path) as _im:
                    w, h = _im.size
                getattr(getattr(vlm_engine, "_config", None), "get", lambda *_: {})
                vision_cfg: dict = {}
                cfg = getattr(vlm_engine, "_config", None)
                if isinstance(cfg, dict):
                    vision_cfg = cfg.get("vision_config", {}) or {}
                    if not vision_cfg:
                        vision_cfg = (cfg.get("thinker_config", {}) or {}).get(
                            "vision_config", {}
                        ) or {}
                patch = int(vision_cfg.get("patch_size", 14) or 14) or 14
                merge = int(vision_cfg.get("spatial_merge_size", 1) or 1) or 1
                image_tokens = max(
                    1, (max(1, h // patch) * max(1, w // patch)) // (merge * merge)
                )
            except Exception:
                image_tokens = 0
        # If prompt_tokens was returned without an image contribution, add it.
        if prompt_tokens and image_tokens and prompt_tokens < image_tokens:
            prompt_tokens = prompt_tokens + image_tokens
        elif prompt_tokens == 0:
            prompt_tokens = image_tokens
        total_tokens = prompt_tokens + completion_tokens

        return {
            "text": text.strip(),
            # a VLM emits no confidence score — the hardcoded 1.0 was a
            # fabricated "measurement". Report None (honest), matching the native
            # OCR-engine path's fix (ocr_engine.py:201).
            "confidence": None,
            "language": language,
            "task": task,
            "model": vlm_model_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "image_tokens": image_tokens,
            },
        }
    except HTTPException:
        raise
    except MemoryError:
        raise HTTPException(status_code=507, detail="Out of GPU memory") from None
    except Exception as e:
        logger.error(f"OCR extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="OCR extraction failed") from None
    finally:
        # free the request-tracker slot (mirrors audio/images/chat unregister).
        if _ocr_tracker is not None:
            with contextlib.suppress(Exception):
                _ocr_tracker.unregister(_ocr_id)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
