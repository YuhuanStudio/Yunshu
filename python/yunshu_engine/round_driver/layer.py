"""Model layers of the round driver (M1): one object per model family, bound to one language model.

``RoundDriver`` takes a layer (``layer=``) and asks it for its drafters (head, DFlash batch, default draft
width) and its decode set (the batch rows join). The layer objects delegate to the functions the driver used
before, so the arithmetic and the construction order are unchanged.

Selection: the driver's ``generic_layer=`` flag keeps the old rule (generic if set, else Flash-Next by
``flashnext.is_flash_next``, else packed). ``LayerRegistry.select`` is NOT used for selection yet: the engine's
rule is wider (``vlm_engine._round_driver_wanted``: spec family, lane self-test), and ``PackedLayer.supports``
is structural only (``forward.supports``). The registry becomes the selector only when the engine passes a
layer object (C8).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast

import mlx.core as mx

from . import flashnext, forward, generic
from .batch import MAX_WINDOW, DecodeBatch

SCORING_SPAN = (
    32  # prompt_logprobs (generic): one prompt span scores at most this many rows
)


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


@dataclass
class Drafters:
    """The drafter objects of one driver: a row head, a DFlash batch, and the default drafts per round."""

    head: Any
    dflash: Any
    max_draft: int


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

    def drafters(
        self,
        model: Any,
        *,
        drafter: Any,
        dflash_drafter: Any,
        plain_verify: bool,
        kv_format: str,
    ) -> Drafters:
        """The layer's head and DFlash batch for these drafter arguments, with its default draft width."""
        ...

    def decode_set(self, kv_format: str, dflash: Any = None) -> Any:
        """The batch rows join: the DFlash batch when there is one, else the layer's own decode set."""
        ...

    def admit(self, row: Any, req: Any) -> None:
        """Per-row layer state, set in ``RoundDriver.add`` before the APC restore."""
        ...

    def span_end(
        self,
        row: Any,
        start: int,
        budget: int,
        *,
        atom: Callable[[int], int],
        scoring: bool = False,
    ) -> int:
        """End of the prompt segment that starts at ``start`` (the layer's own plan)."""
        ...

    def prefill(
        self,
        row: Any,
        seg: Any,
        start: int,
        end: int,
        *,
        capture_ids: Any = (),
        captured: list | None = None,
    ) -> mx.array | None:
        """One prompt span of ``row`` on its caches: hidden rows (one per token) or, generic, the logits of
        the prompt-ending span (None otherwise)."""
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
        # structural only (dense SwiGLU / Qwen3.5-MoE MLPs); the engine adds the spec family and the lane self-test
        return forward.supports(lm)

    def __init__(self, lm: Any) -> None:
        self.lm = lm

    def new_cache(self) -> list:
        return self.lm.make_cache()  # driver.py:585 (_new_cache)

    def logits(self, hidden: mx.array) -> mx.array:
        return forward.logits(self.lm, hidden)  # driver.py:576-581

    def drafters(
        self,
        model: Any,
        *,
        drafter: Any,
        dflash_drafter: Any,
        plain_verify: bool,
        kv_format: str,
    ) -> Drafters:
        from .mtp import MTPHead

        head = MTPHead(drafter, self.lm, self.logits) if drafter is not None else None
        dflash = None
        if dflash_drafter is not None or (plain_verify and drafter is None):
            from .. import tree_verify
            from .dflash import DFlashBatch

            if kv_format == "int8":
                from ..kernels import int8_tile

                int8_tile.install()
                # fail loudly at load: no int8 tree-attention kernel installed means no int8 serve
                # (the in-graph reference is test-only and never a fallback)
                tree_verify.require_int8_kernels()
            dflash = DFlashBatch(model, dflash_drafter, kv_format)
        # MAX_DECODE_TOKENS - 1 (driver.py:71)
        return Drafters(head, dflash, MAX_WINDOW - 1)

    def decode_set(self, kv_format: str, dflash: Any = None) -> Any:
        return dflash if dflash is not None else DecodeBatch(self.lm, kv_format)

    def admit(self, row: Any, req: Any) -> None:
        return None  # no per-row layer state (driver.py add)

    def cut(self, row: Any, start: int, atom: Callable[[int], int]) -> int:
        """End of the one prompt atom that starts at ``start``: the next multiple of the atom size, cut at the
        row's next APC checkpoint and at the prompt's end. Split-invariant prefill keeps the plan a function of
        the prompt alone (grid plus checkpoints, so a restored row continues with a cold prefill's spans)."""
        a = atom(start)
        n = len(row.req.ids)
        end = min((start // a + 1) * a, n)
        # The prompt's tail (the rows after its last 64-aligned boundary) always runs apart, as in
        # main's absolute prompt steps (boundary = min(grid, (end - 1) // 64 * 64)): a cell that
        # contains the boundary is cut there even when no checkpoint is planned (a short prompt, a
        # request that stores nothing). Without this cut a 99-row prompt ran as ONE span on the
        # long kernels where main runs 64 rows + a 35-row tail on the tail kernels.
        limit = (n - 1) // 64 * 64
        if start < limit < end:
            end = limit
        for c in row.ckpts:
            if start < c < end:
                end = c
                break
        return end

    def span_end(
        self,
        row: Any,
        start: int,
        budget: int,
        *,
        atom: Callable[[int], int],
        scoring: bool = False,
    ) -> int:
        """End of the segment a prefill step runs for ``row`` from ``start``: consecutive atoms (as many as
        ``budget`` takes) become ONE segment, so a step runs large matmuls and a restored prompt's remainder
        is one forward (not one per grid cell). A run stops at a checkpoint (its state is stored there) and
        before the prompt's tail (the tail runs apart, on the tail kernels, as in a restored run). Prefill
        arithmetic is split-invariant at 64-aligned boundaries (nax matmul, BQ32 attention, chunked GDN), so
        the segment size never changes a prompt's output. ``scoring`` (prompt_logprobs) is generic-only."""
        end = self.cut(row, start, atom)
        limit = (len(row.req.ids) - 1) // 64 * 64
        while end not in row.ckpts and end < limit:
            nxt = self.cut(row, end, atom)
            # a run keeps one atom size (the long-position atoms read the KV in
            # fewer, larger passes)
            if nxt > limit or nxt - start > budget or atom(end) != atom(start):
                break
            end = nxt
        return end

    def prefill(
        self,
        row: Any,
        seg: Any,
        start: int,
        end: int,
        *,
        capture_ids: Any = (),
        captured: list | None = None,
    ) -> mx.array | None:
        return forward.lane_forward(
            self.lm, seg, capture_ids=capture_ids, captured=captured
        )


def _grid_cut(row: Any, start: int, atom: Callable[[int], int]) -> int:
    """Flash-Next and generic prompt span: the absolute grid plus the row's checkpoints and nothing else.
    Flash-Next prefill is not split-invariant (chunked GDN, stock matmuls), so a cold run and a hit run the
    same spans (no 64-aligned tail cut, no run of several atoms)."""
    a = atom(start)
    n = len(row.req.ids)
    end = min((start // a + 1) * a, n)
    for c in row.ckpts:
        if start < c < end:
            return cast(int, c)
    return end


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

    def drafters(
        self,
        model: Any,
        *,
        drafter: Any,
        dflash_drafter: Any,
        plain_verify: bool,
        kv_format: str,
    ) -> Drafters:
        from .flashnext_dflash import FlashNextDFlash

        head = (
            flashnext.FlashNextHead(drafter, self.lm, self.logits)
            if drafter is not None
            else None
        )
        dflash = (
            FlashNextDFlash(model, dflash_drafter, kv_format)
            if dflash_drafter is not None
            else None
        )
        return Drafters(head, dflash, flashnext.MAX_CHAIN - 1)

    def decode_set(self, kv_format: str, dflash: Any = None) -> Any:
        return (
            dflash
            if dflash is not None
            else flashnext.FlashNextBatch(self.lm, kv_format)
        )

    def admit(self, row: Any, req: Any) -> None:
        return None  # no per-row layer state (driver.py add)

    def span_end(
        self,
        row: Any,
        start: int,
        budget: int,
        *,
        atom: Callable[[int], int],
        scoring: bool = False,
    ) -> int:
        return _grid_cut(row, start, atom)  # one span per grid cell, no run merge

    def prefill(
        self,
        row: Any,
        seg: Any,
        start: int,
        end: int,
        *,
        capture_ids: Any = (),
        captured: list | None = None,
    ) -> mx.array | None:
        return flashnext.prefill(
            self.lm, seg, capture_ids=capture_ids, captured=captured
        )


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

    def drafters(
        self,
        model: Any,
        *,
        drafter: Any,
        dflash_drafter: Any,
        plain_verify: bool,
        kv_format: str,
    ) -> Drafters:
        if (
            drafter is not None
            or dflash_drafter is not None
            or flashnext.is_flash_next(self.lm)
        ):
            raise ValueError(
                "round driver: the generic layer takes no drafter and no Flash-Next model"
            )
        return Drafters(None, None, 0)

    def decode_set(self, kv_format: str, dflash: Any = None) -> Any:
        return generic.GenericBatch(self.lm, kv_format)

    def admit(self, row: Any, req: Any) -> None:
        if req.prompt_kwargs is None:
            raise ValueError("generic round driver row: prompt_kwargs are required")
        row.features = generic.PromptFeatures(req.prompt_kwargs, len(req.ids))
        row.state = generic.RowState()

    def span_end(
        self,
        row: Any,
        start: int,
        budget: int,
        *,
        atom: Callable[[int], int],
        scoring: bool = False,
    ) -> int:
        end = _grid_cut(row, start, atom)
        if scoring:
            end = min(end, start + SCORING_SPAN)
        return end

    def prefill(
        self,
        row: Any,
        seg: Any,
        start: int,
        end: int,
        *,
        capture_ids: Any = (),
        captured: list | None = None,
    ) -> mx.array | None:
        # the model's own call: logits only for the span that ends the prompt
        return generic.prefill(self.lm, row, start, end)


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
