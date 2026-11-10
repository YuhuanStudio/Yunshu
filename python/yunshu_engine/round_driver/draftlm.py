"""Model drafters of the round driver's generic layer: a separate draft LM and the Gemma-4 assistant.

Both plug in exactly where the model-free n-gram drafter does (``ngram.NgramDrafter``): a per-row
object whose ``propose(context, room)`` returns up to ``room`` draft tokens after the row's pending
token; the driver feeds ``[pending, *drafts]`` as one window of the model's own forward, keeps the
prefix whose top-1 equals the draft and trims the rejected tail (lone row, trimmable cache; see
``RoundDriver._ngram_window``). The back-off (cool-down after misses, plain decode while windows are
slower than plain steps) is the base class's. Greedy rows only, the same guarantee as the n-gram
drafter: every kept token is the model's own top-1 over the window.

* ``DraftLMDrafter``: an ordinary mlx-lm checkpoint (``YUNSHU_DRAFT_MODEL``) drafting greedily. Per row
  it keeps its own KV cache over the row's context and re-feeds what the row accepted since its last
  proposal (one batched forward), then chains ``room - 1`` single-token steps and trims them again.
* ``AssistantDrafter``: the Gemma-4 assistant (``YUNSHU_GEMMA4_ASSISTANT``), which reads the target's
  last hidden state and the target's own shared K/V (``Gemma4AssistantProposer.propose_chain``). The
  driver captures the hidden state of every step of a lone row (``GenericBatch.hidden``) and hands the
  kept position to ``carry``.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from .ngram import NgramDrafter

logger = logging.getLogger(__name__)

PREFILL_CHUNK = 2048  # draft-model prompt chunk (bounds its prefill workspace)
_NO_PROPOSER = object()


class DraftLM:
    """The loaded draft model, shared by every row."""

    def __init__(self, model: Any, path: str = "") -> None:
        self.model = model
        self.path = path

    @classmethod
    def load(cls, path: str, target_vocab: int | None = None) -> DraftLM:
        from mlx_lm.utils import load

        loaded = load(path)
        model = loaded[0]
        args = getattr(model, "args", None)
        vocab = getattr(args, "vocab_size", None)
        if target_vocab and vocab and int(vocab) != int(target_vocab):
            raise ValueError(
                f"draft model vocab {vocab} != target vocab {target_vocab}: a draft needs the target's tokenizer"
            )
        draft = cls(model, path)
        draft.new_cache()  # raises when the draft's cache cannot trim
        return draft

    def new_cache(self) -> list:
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)
        if not all(c.is_trimmable() for c in cache):
            raise ValueError(
                "the draft model's cache cannot trim (rotating / recurrent): it cannot serve as a drafter"
            )
        return cache


class DraftLMDrafter(NgramDrafter):
    name = mode = "draft-lm"

    def __init__(self, draft: DraftLM) -> None:
        super().__init__("draft-lm", proposer=_NO_PROPOSER)
        self.draft = draft
        self.cache: list | None = None
        self.fed = 0  # context tokens in the draft cache

    def _forward(self, tokens: list[int]) -> int:
        out = self.draft.model(mx.array(tokens, dtype=mx.int32)[None], cache=self.cache)
        return int(mx.argmax(out[0, -1]).item())

    def _trim(self, n: int) -> None:
        if n > 0 and self.cache is not None:
            for c in self.cache:
                c.trim(n)

    def _propose(self, context: list[int], room: int) -> list[int]:
        if self.cache is None:
            self.cache = self.draft.new_cache()
        n = len(context)
        from .generic import _trimmable

        if not all(_trimmable(c, n - self.fed + room) for c in self.cache):
            return []  # a rotating draft cache that would wrap cannot cut its chain back
        if (
            self.fed >= n
        ):  # the context is shorter than the draft's (never in a plain row): resync
            self._trim(self.fed - (n - 1))
            self.fed = n - 1
        feed = context[self.fed :]
        d = 0
        for at in range(0, len(feed), PREFILL_CHUNK):
            d = self._forward(feed[at : at + PREFILL_CHUNK])
        drafts = [d]
        for _ in range(room - 1):
            d = self._forward([d])
            drafts.append(d)
        self._trim(len(drafts) - 1)
        self.fed = n
        return drafts


class AssistantDrafter(NgramDrafter):
    name = mode = "gemma4-assistant"
    needs_hidden = True
    needs_target_cache = True

    def __init__(self, proposer: Any) -> None:
        super().__init__("gemma4-assistant", proposer=_NO_PROPOSER)
        self.assistant = proposer
        self.hidden: mx.array | None = None  # target hidden at the last cached position
        self.cache: list | None = (
            None  # the row's target cache (set by the driver each step)
        )

    def carry(self, hidden: mx.array | None) -> None:
        if hidden is not None and hidden.ndim == 1:
            hidden = hidden[None]
        self.hidden = hidden

    def _propose(self, context: list[int], room: int) -> list[int]:
        cache, hidden = self.cache, self.hidden
        if cache is None or hidden is None:
            return []
        sl, fl = self.assistant.sliding_kv_layer, self.assistant.full_kv_layer
        kv = {sl: tuple(cache[sl].state), fl: tuple(cache[fl].state)}
        offset = int(cache[fl].offset) - 1
        return self.assistant.propose_chain(context[-1], hidden, kv, offset, k=room)  # type: ignore[return-value]


class RowDrafters:
    """What the driver draws a row's drafter from (built once per engine; ``None`` rows use n-gram)."""

    def __init__(self, draft_lm: DraftLM | None = None, assistant: Any = None) -> None:
        self.draft_lm = draft_lm
        self.assistant = assistant

    def make(self) -> NgramDrafter | None:
        if self.assistant is not None:
            return AssistantDrafter(self.assistant)
        if self.draft_lm is not None:
            return DraftLMDrafter(self.draft_lm)
        return None


def build_row_drafters(
    lm: Any, model_path: str, draft_model: str | None, assistant_dir: str | None
) -> RowDrafters | None:
    """The configured drafters for ``lm`` (a mlx-vlm language model); a failing load logs and is skipped."""
    draft_lm = assistant = None
    cfg = getattr(lm, "config", None)
    if assistant_dir:
        try:
            import json
            from pathlib import Path

            from ..gemma4_assistant import Gemma4AssistantProposer

            tm = lm.model
            tcfg = json.loads((Path(model_path) / "config.json").read_text())
            embed = tm.embed_tokens
            weight = embed.weight
            if hasattr(
                embed, "scales"
            ):  # quantized embedding: the drafter needs the dense table
                weight = mx.dequantize(
                    embed.weight,
                    embed.scales,
                    getattr(embed, "biases", None),
                    embed.group_size,
                    embed.bits,
                )
            assistant = Gemma4AssistantProposer.from_paths(
                assistant_dir, weight, float(getattr(tm, "embed_scale", 1.0)), tcfg
            )
            logger.info(
                "Round driver: Gemma-4 assistant drafter loaded from %s", assistant_dir
            )
        except Exception:
            logger.warning(
                "Round driver: Gemma-4 assistant drafter not loaded", exc_info=True
            )
    if draft_model and assistant is None:
        try:
            draft_lm = DraftLM.load(draft_model, getattr(cfg, "vocab_size", None))
            logger.info("Round driver: draft LM loaded from %s", draft_model)
        except Exception:
            logger.warning("Round driver: draft LM not loaded", exc_info=True)
    if draft_lm is None and assistant is None:
        return None
    return RowDrafters(draft_lm, assistant)
