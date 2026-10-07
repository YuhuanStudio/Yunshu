"""EmbeddingGemma 2 engine: text, image, audio and video embeddings in one shared space.

Subclasses VLEmbeddingEngine so the embeddings / scoring routers take their multimodal branch
(async ``embed``). The model lives in ``embedding_gemma2.py``; all MLX work runs on the shared
single-Metal-thread executor.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .vl_embedding_engine import VLEmbeddingEngine

logger = logging.getLogger(__name__)


class GemmaEmbeddingEngine(VLEmbeddingEngine):
    """Serves ``embedding_gemma2`` checkpoints (google/embeddinggemma-2 and conversions)."""

    def __init__(self, model_path: str, config: Any | None = None) -> None:
        super().__init__(model_path, config)
        self.is_reranker = False
        self._gemma: Any = None

    async def start(self) -> None:
        if self._loaded:
            return

        def _load():
            from .embedding_gemma2 import EmbeddingGemma2

            return EmbeddingGemma2(self._model_path)

        loop = asyncio.get_running_loop()
        self._gemma = await loop.run_in_executor(self._executor, _load)
        self._loaded = True
        logger.info("GemmaEmbeddingEngine ready: %s", self.model_name)

    async def stop(self) -> None:
        self._gemma = None
        self._loaded = False

    @property
    def tasks(self) -> list[str]:
        return sorted(self._gemma.prompts) if self._gemma else []

    def decode(self, ids: list[int]) -> str:
        return str(
            self._gemma.processor.tokenizer.decode(ids, skip_special_tokens=True)
        )

    async def embed_with_usage(
        self,
        inputs: list[Any],
        instruction: str | None = None,
        task: str | None = None,
    ) -> tuple[list[list[float]], int]:
        """Unit vectors (native 768) and the total token count. Items: str or
        {"text"?, "image"?, "audio"?, "video"?, "task"?, "instruction"?}; a per-item
        task / instruction overrides the request's."""
        if not self._loaded:
            raise RuntimeError("Engine not started")

        def _run():
            groups: dict[tuple, list[int]] = {}
            norm: list[Any] = []
            for i, it in enumerate(inputs):
                if isinstance(it, dict):
                    it = dict(it)
                    key = (it.pop("task", task), it.pop("instruction", instruction))
                else:
                    key = (task, instruction)
                norm.append(it)
                groups.setdefault(key, []).append(i)
            vecs: list[Any] = [None] * len(inputs)
            total = 0
            for (t, ins), idx in groups.items():
                arr, counts = self._gemma.embed_items([norm[i] for i in idx], t, ins)
                total += sum(counts)
                for r, i in enumerate(idx):
                    vecs[i] = arr[r].tolist()
            return vecs, total

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _run)

    async def embed(
        self, inputs: list[Any], instruction: str | None = None, task: str | None = None
    ) -> list[list[float]]:
        return (await self.embed_with_usage(inputs, instruction, task))[0]

    async def rerank(self, *a: Any, **k: Any) -> list[float]:
        raise NotImplementedError("EmbeddingGemma 2 is an embedder, not a reranker")
