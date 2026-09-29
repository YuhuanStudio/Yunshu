"""Request guards shared by the generation routes.

Each guard turns a request the loaded model cannot serve into a clear 400,
instead of silently ignoring input or producing garbage text.
"""

from __future__ import annotations

import contextlib
import os

from fastapi import HTTPException

EMBEDDING_ONLY_MESSAGE = (
    "Model '{model}' is an embedding model and only supports /v1/embeddings "
    "(Ollama: /api/embed); it cannot generate text."
)


def _local_model_dir(name: str) -> str:
    if not name or os.path.isdir(name):
        return name
    try:
        from mlx_lm.utils import hf_repo_to_path

        return str(hf_repo_to_path(name))
    except Exception:
        return name


def is_embedding_only(engine) -> bool:
    """True when the engine serves a sentence-embedding checkpoint.

    A sentence-transformers export ships ``1_Pooling/config.json`` next to the
    weights; chat checkpoints never do. The result is cached on the engine.
    """
    cached = getattr(engine, "_embedding_only", None)
    if isinstance(cached, bool):
        return cached
    name = getattr(engine, "model_name", None)
    result = False
    if isinstance(name, str) and name:
        base = _local_model_dir(name)
        result = os.path.isfile(os.path.join(base, "1_Pooling", "config.json"))
    with contextlib.suppress(Exception):
        engine._embedding_only = result
    return result


def reject_embedding_only(engine, model: str | None = None) -> None:
    """Raise a 400 when a generation route is pointed at an embedding model."""
    if engine is not None and is_embedding_only(engine):
        raise HTTPException(
            status_code=400,
            detail=EMBEDDING_ONLY_MESSAGE.format(
                model=model or getattr(engine, "model_name", "") or ""
            ),
        )


def reject_images_for_text_model(engine, has_images: bool) -> None:
    """Raise a 400 when image input reaches an engine that cannot see images."""
    if not has_images or engine is None:
        return
    from yunshu_engine.vlm_engine import VLMEngine

    try:
        from yunshu_engine.omni_engine import OmniEngine
    except Exception:  # pragma: no cover
        OmniEngine = ()  # type: ignore[assignment,misc]
    if isinstance(engine, (VLMEngine, OmniEngine)):
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "This model does not accept image input (it is not a vision-language "
            "model); remove the image blocks or load a VLM."
        ),
    )
