"""OCR endpoint — POST /v1/ocr for image-to-text extraction."""

import os
import tempfile
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ocr"])


@router.post("/v1/ocr")
async def extract_text_from_image(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model: str = Form(""),
    task: str = Form("text"),
):
    """Extract text from an uploaded image using OCR.

    Accepts PNG, JPG, JPEG, WEBP, TIFF image formats.
    Returns extracted text with optional language detection.
    """
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10 MB
        raise HTTPException(status_code=413, detail="Image file too large (max 10MB)")

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}
    raw_suffix = os.path.splitext(file.filename or "image.png")[1].lower()
    suffix = raw_suffix if raw_suffix in _IMAGE_EXTENSIONS else ".png"

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)

        # Find OCR engine
        from ..engine import get_model_manager
        from yunshu_engine.ocr_engine import OCREngine

        manager = get_model_manager()
        if manager is None:
            raise HTTPException(status_code=503, detail="Model manager not initialized")

        ocr_engine = None
        for entry in manager.list_entries():
            if entry.is_loaded and isinstance(getattr(entry, 'engine', None), OCREngine):
                ocr_engine = entry.engine
                break

        if ocr_engine is None:
            # Try loading by model name
            if model:
                try:
                    engine = await manager.get_engine(model)
                    if isinstance(engine, OCREngine):
                        ocr_engine = engine
                except Exception:
                    pass

        if ocr_engine is None:
            raise HTTPException(status_code=404, detail="No OCR engine available")

        result = await ocr_engine.extract_text(tmp_path, language=language, task=task)
        return {
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
            "language": result.get("language", language),
            "task": task,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="OCR extraction failed")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
