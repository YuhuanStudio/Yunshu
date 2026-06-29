"""Qwen3-VL multimodal embedding + reranker engine (via mlx_embeddings).

Wraps the `mlx-community/Qwen3-VL-Embedding-*` and `Qwen3-VL-Reranker-*` models —
a unified Qwen3-VL backbone trained for retrieval. One model embeds **text,
images, and cross-modal (text+image)** inputs into a single shared vector space;
the reranker variant scores query↔document relevance as a true cross-encoder.

Why a dedicated engine (not BatchedEngine.embed / VLMEngine):
- BatchedEngine.embed() pools an mlx-lm *text* backbone — it can't ingest images
  and never builds the vision tower, so it only does text.
- VLMEngine does autoregressive *generation*, not pooled embeddings.
- mlx_embeddings' top-level generate()/prepare_inputs() path is broken for this
  model (it pads input_ids to max_length and the image placeholder tokens stop
  matching the vision features). The model's own `model.process(inputs, proc)` —
  which uses the custom Qwen3-VL processor's prepare_embedding_inputs /
  prepare_reranker_inputs — is correct, so that is what we call.

Input contract (mirrors the official Qwen3-VL-Embedding API):
- embed(inputs): inputs is a list where each item is a plain ``str`` (text) or a
  dict ``{"text"?: str, "image"?: str|path|url, "instruction"?: str}``. Returns one
  L2-normalized vector per item (pooled on the EOS/last token, shared space).
- rerank(query, documents, instruction): query/documents are ``str`` or the same
  dict form; returns one relevance score in [0,1] per document (sigmoid of the
  yes/no token logit gap).

All GPU work runs on the shared single-Metal-thread executor, like the other
engines.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default instructions per the official Qwen3-VL-Embedding usage.
_DEFAULT_EMBED_INSTRUCTION = "Represent the user's input"
_DEFAULT_RERANK_INSTRUCTION = "Retrieve documents relevant to the query."


def _normalize_item(item: Any) -> dict:
    """Coerce a str | dict input into the {text?, image?, instruction?} dict form."""
    if isinstance(item, str):
        return {"text": item}
    if isinstance(item, dict):
        return item
    raise ValueError(
        f"embedding input must be a str or a dict with text/image keys, got {type(item).__name__}"
    )


class VLEmbeddingEngine:
    """Multimodal embedding / reranking engine for Qwen3-VL retrieval models."""

    def __init__(self, model_path: str, config: Any | None = None) -> None:
        self._model_path = model_path
        self._config = config
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False
        # Embedder vs reranker: the two share the Qwen3VLForConditionalGeneration
        # architecture and an identical config, so structure can't tell them apart
        # — key off the model name (the mlx-community repos name them explicitly).
        self.is_reranker = "reranker" in model_path.lower()
        from .mlx_executor import get_mlx_executor

        self._executor = get_mlx_executor()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return (
            self._model_path.rsplit("/", 1)[-1]
            if "/" in self._model_path
            else self._model_path
        )

    async def start(self) -> None:
        """Load the model + custom processor on the Metal-owning executor thread."""
        if self._loaded:
            return

        def _load():
            from mlx_embeddings import load

            return load(self._model_path)

        loop = asyncio.get_running_loop()
        self._model, self._processor = await loop.run_in_executor(self._executor, _load)
        # The high-level embed/rerank dispatch needs the custom Qwen3-VL processor.
        if not hasattr(self._processor, "prepare_model_inputs"):
            raise RuntimeError(
                f"{self.model_name}: loaded processor lacks the Qwen3-VL "
                "prepare_model_inputs hook — not a Qwen3-VL embedding/reranker model."
            )
        self._loaded = True
        logger.info(
            "VLEmbeddingEngine ready: %s (%s)",
            self.model_name,
            "reranker" if self.is_reranker else "embedder",
        )

    async def stop(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False

    async def embed(
        self, inputs: list[Any], instruction: str | None = None
    ) -> list[list[float]]:
        """Embed text / image / cross-modal inputs into the shared space.

        inputs: list of ``str`` or ``{"text"?, "image"?, "instruction"?}`` dicts.
        ``instruction`` (if given) is applied to any item that doesn't carry its own.
        Returns one L2-normalized vector per input.
        """
        if not self._loaded:
            raise RuntimeError("Engine not started")
        items = [_normalize_item(x) for x in inputs]
        if instruction:
            for it in items:
                it.setdefault("instruction", instruction)

        def _run():
            import mlx.core as mx

            arr = self._model.process(items, self._processor)
            mx.eval(arr)
            # (N, dim) → list of row vectors
            if arr.ndim == 1:
                return [arr.tolist()]
            return [arr[i].tolist() for i in range(arr.shape[0])]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _run)

    async def rerank(
        self,
        query: Any,
        documents: list[Any],
        instruction: str | None = None,
    ) -> list[float]:
        """Score each document's relevance to the query (true cross-encoder).

        query/documents: ``str`` or ``{"text"?, "image"?}`` dicts. Returns one
        relevance score in [0,1] per document (higher = more relevant).
        """
        if not self._loaded:
            raise RuntimeError("Engine not started")
        if not documents:
            return []
        q = _normalize_item(query)
        docs = [_normalize_item(d) for d in documents]
        payload = {
            "instruction": instruction or _DEFAULT_RERANK_INSTRUCTION,
            "query": q,
            "documents": docs,
        }

        def _run():
            import mlx.core as mx

            scores = self._model.process(payload, self._processor)
            mx.eval(scores)
            return [float(x) for x in scores.reshape(-1).tolist()]

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _run)
