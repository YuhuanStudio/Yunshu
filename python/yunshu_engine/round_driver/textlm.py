"""Text-only mlx-lm checkpoints as round driver targets on the generic layer (T7 of the text-engine plan).

A text checkpoint (``mlx_lm.models.*``) has no vision tower and no mlx-vlm wrapper. The round driver's generic
layer reads three things from its target: ``language_model`` (the call: ``lm(ids, cache=..., inputs_embeds=...)``,
logits back), ``make_cache`` (through ``generic.new_cache``), and ``get_input_embeddings(ids, None, mask=None)``
(the prompt features, ``.to_dict()`` -> ``{"inputs_embeds": [1, L, D]}``). This module supplies exactly that
surface around an unmodified mlx-lm model and nothing else: the arithmetic is the model's own forward.

Not routed yet: no engine selects this adapter (T11 does). Nothing here changes a served request.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, cast

import mlx.core as mx


class TextFeatures:
    """The prompt features of a text prompt: the embedding of every prompt token."""

    def __init__(self, embeds: mx.array):
        self.embeds = embeds  # [1, L, D]

    def to_dict(self) -> dict:
        return {"inputs_embeds": self.embeds}


class TextLM:
    """The language-model call the generic layer makes, over an mlx-lm model.

    ``__call__`` takes the mlx-vlm keyword surface the generic layer uses. ``inputs_embeds`` is handed to the
    model as ``input_embeddings`` when its ``__call__`` takes that keyword (mlx-lm does, for every family that
    shares the embedding with a VLM wrapper), else the model embeds the token ids itself: the same arithmetic.
    ``logits_to_keep`` keeps the last rows of the logits. ``n_to_process`` is mlx-vlm's chunk-size hint and is
    ignored. Any other keyword is an error, never dropped silently.

    Attributes the layer reads that the model defines (``args``, ``layers``, ...) are delegated to the model."""

    _KNOWN = frozenset({"inputs_embeds", "n_to_process", "logits_to_keep"})

    def __init__(self, model: Any):
        self.__dict__["model"] = model
        self.__dict__["_takes_embeddings"] = (
            "input_embeddings" in inspect.signature(type(model).__call__).parameters
        )

    def __getattr__(self, name: str) -> Any:
        model = self.__dict__.get("model")
        if model is None or name.startswith("__"):
            raise AttributeError(name)
        return getattr(model, name)

    def make_cache(self) -> list:
        """The row's prompt cache, built by mlx-lm's own ``make_prompt_cache`` (the model's ``make_cache`` when
        it has one): the cache classes whose ``update_and_fetch`` / ``trim`` the model's attention calls."""
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(self.model)

    def __call__(
        self, inputs: mx.array, cache: list | None = None, **kwargs: Any
    ) -> mx.array:
        unknown = sorted(set(kwargs) - self._KNOWN)
        if unknown:
            raise ValueError(
                f"text round driver row: model call kwargs {unknown} are not carried"
            )
        embeds = kwargs.get("inputs_embeds")
        if embeds is not None and self._takes_embeddings:
            logits = self.model(inputs, cache=cache, input_embeddings=embeds)
        else:
            logits = self.model(inputs, cache=cache)
        keep = kwargs.get("logits_to_keep")
        if keep is not None:
            logits = logits[:, -int(keep) :]
        return cast(mx.array, logits)


class TextModelAdapter:
    """A loaded mlx-lm model in the surface the round driver's generic layer takes from a target."""

    def __init__(self, model: Any):
        if not (
            hasattr(model, "layers") or hasattr(getattr(model, "model", None), "layers")
        ):
            raise ValueError("text round driver: not an mlx-lm language model")
        self.model = model
        self.language_model = TextLM(model)

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: Any = None,
        mask: Any = None,
        **kwargs: Any,
    ) -> TextFeatures:
        """The prompt features of ``input_ids`` ([1, L] ints): the model's token embedding, nothing else."""
        if pixel_values is not None or kwargs:
            raise ValueError("text checkpoints take no media inputs (no vision tower)")
        inner = getattr(self.model, "model", self.model)
        embed = getattr(inner, "embed_tokens", None)
        if embed is None:
            raise ValueError("text round driver: the model has no embed_tokens")
        return TextFeatures(embed(input_ids))


def load_text_model(path: str | Path) -> TextModelAdapter:
    """Load an mlx-lm checkpoint directory (config.json + safetensors) into a ``TextModelAdapter``.

    Weights are materialized (``lazy=False``), as the engine's load path does; quantized checkpoints load as
    quantized. The tokenizer is not loaded here (the engine's tokenizer path owns it)."""
    from mlx_lm.utils import load_model

    model, _config = load_model(Path(path), lazy=False)
    return TextModelAdapter(model)


__all__ = ["TextFeatures", "TextLM", "TextModelAdapter", "load_text_model"]
