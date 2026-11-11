"""Model layers of the round driver (M1): one object per model family, chosen by ``LayerRegistry.select``.

C1 adds the layer objects beside the driver without moving any call site: each adapter delegates to the
function the driver already uses (``forward.supports``, ``flashnext.is_flash_next``, ``generic.new_cache``).
The driver keeps its ``self.flash`` / ``self.generic`` tests until C2-C8 wire the layer in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import mlx.core as mx

from . import flashnext, forward, generic


@dataclass(frozen=True)
class LayerCaps:
    """What the driver may rely on for one layer (census section C; consumed from C2 on)."""

    split_invariant: bool
    replay: bool
    ngram: bool
    lazy_head: bool
    lone_pipelining: bool
    slot_decode: bool
    identity_logits: bool
    per_token_rows: bool


class ModelLayer(Protocol):
    """A model family's forward on the round driver; an instance is bound to one language model."""

    name: ClassVar[str]
    caps: ClassVar[LayerCaps]

    @classmethod
    def supports(cls, lm: Any) -> bool:
        """True when this layer serves the language model ``lm``."""
        ...

    def __init__(self, lm: Any) -> None: ...

    def new_cache(self) -> list:
        """One row's own prompt cache."""
        ...

    def logits(self, hidden: mx.array) -> mx.array:
        """Logits of hidden rows ``[R, D]`` (generic: the rows already are logits)."""
        ...


class PackedLayer:
    """Qwen3.5 family (dense and MoE): ``forward.py`` packed rows, ``DecodeBatch`` slot buffers."""

    name = "qwen3_5"
    caps = LayerCaps(
        split_invariant=True,
        replay=True,
        ngram=False,
        lazy_head=False,
        lone_pipelining=False,
        slot_decode=True,
        identity_logits=False,
        per_token_rows=True,
    )

    @classmethod
    def supports(cls, lm: Any) -> bool:
        return forward.supports(
            lm
        )  # the packed arm of driver.py:466 (the else branch there)

    def __init__(self, lm: Any) -> None:
        self.lm = lm

    def new_cache(self) -> list:
        return self.lm.make_cache()  # driver.py:585 (_new_cache)

    def logits(self, hidden: mx.array) -> mx.array:
        return forward.logits(self.lm, hidden)  # driver.py:576-581


class FlashNextLayer:
    """Flash-Next (Qwen4): ``flashnext.py`` HC/QSA/GDN/MoE forward, ``FlashNextBatch``."""

    name = "flash_next"
    caps = LayerCaps(
        split_invariant=False,
        replay=True,
        ngram=False,
        lazy_head=True,
        lone_pipelining=False,
        slot_decode=False,
        identity_logits=False,
        per_token_rows=True,
    )

    @classmethod
    def supports(cls, lm: Any) -> bool:
        return flashnext.is_flash_next(lm)  # driver.py:466

    def __init__(self, lm: Any) -> None:
        self.lm = lm

    def new_cache(self) -> list:
        return self.lm.make_cache()  # driver.py:585 (_new_cache)

    def logits(self, hidden: mx.array) -> mx.array:
        return flashnext.logits(self.lm, hidden)  # driver.py:576-581


class GenericLayer:
    """Every other mlx-vlm family: the model's own upstream forward per row, no drafter (``generic.py``)."""

    name = "generic"
    caps = LayerCaps(
        split_invariant=False,
        replay=False,
        ngram=True,
        lazy_head=False,
        lone_pipelining=True,
        slot_decode=False,
        identity_logits=True,
        per_token_rows=False,
    )

    @classmethod
    def supports(cls, lm: Any) -> bool:
        # the fallback: the driver's generic arm is "neither Flash-Next nor packed" (engine: vlm_engine.py:2570)
        return True

    def __init__(self, lm: Any) -> None:
        self.lm = lm

    def new_cache(self) -> list:
        return generic.new_cache(self.lm)  # driver.py:585 (_new_cache)

    def logits(self, hidden: mx.array) -> mx.array:
        return hidden  # driver.py:576-581 (identity)


class LayerRegistry:
    """Layers in selection order: the first whose ``supports`` holds serves the model."""

    layers: tuple[type, ...] = (FlashNextLayer, PackedLayer, GenericLayer)

    def select(self, model: Any) -> Any:
        # the driver's own rule for the language model (driver.py: ``model.language_model`` if present)
        lm = model.language_model if hasattr(model, "language_model") else model
        for cls in self.layers:
            if cls.supports(lm):
                return cls(lm)
        raise RuntimeError(
            "no round-driver layer supports this model"
        )  # unreachable: GenericLayer


REGISTRY = LayerRegistry()
