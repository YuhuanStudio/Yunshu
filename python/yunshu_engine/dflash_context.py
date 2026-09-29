# Upstream (derived): Blaizzy/mlx-vlm (MIT) mlx_vlm/speculative/utils.py, mlx_vlm/models/qwen3_5/language.py, mlx_vlm/speculative/drafters/qwen3_dflash/dflash.py @ v0.7.3
# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""DFlash speculative decoding on Qwen3.5: lossless prefill, verify head, and
the drafter's context window.

1. Prefill with captured layers. A single request's speculative prompt cache is
   a one-row ``BatchKVCache``; upstream ``Qwen3_5Model`` runs such a request on
   an extracted per-row ``KVCache`` -- the plain-decode arithmetic -- but only
   when no hidden states are captured (``hidden_sink is None``). DFlash
   captures ``target_layer_ids`` during prefill, so its prompt ran the batched
   attention path instead: different bits in every KV entry after the first
   attention layer, and spec on != spec off from the first near-tie on (MTP
   captures nothing and was unaffected). ``install`` gives the capturing call
   the same per-row path.
2. Greedy verify head. mlx-vlm's greedy DFlash round asks the target for
   ``speculative_verify_dflash_hidden`` (captured layers + final hidden) and
   takes the argmax with ``speculative_argmax_from_hidden`` -- the head MTP
   verify uses. Qwen3.5 lacks the hook, so DFlash computed T x vocab verify
   logits through the verifier's own head matmul. ``install`` adds it (as
   upstream Nemotron-H has).
3. Drafter context window. When every drafter layer is a sliding-window layer,
   the drafter attends to at most ``sliding_window - 1`` context positions.
   Upstream trims to that window inside each attention layer, but only after
   the drafter's input projection (``fc``: 5 x hidden -> hidden) and the fused
   context K/V projection ran over the whole prompt: ~48 TFLOP in the first
   speculative round at 131K tokens on Qwen3.8-27B, plus ~6.7 GiB of captured
   hidden states (5 layers x 131K x 5120 x bf16) held through prefill.
   - ``DFlashDraftModel._hidden`` trims the target hidden states to the window
     before the projections and advances each drafter cache by the skipped
     count (what the attention layers do after projecting), so draft keys keep
     their absolute RoPE positions.
   - ``SpeculativePrefill`` retains only the chunks covering the last window of
     captured hidden states. The dropped positions shift every drafter position
     by the same constant, which RoPE attention does not see.
   Drafts can differ from upstream's only by rounding (matmuls over fewer
   rows); the target verifies every draft, so the output does not depend on it.
"""

from __future__ import annotations

from typing import Any

_STATE: dict = {"installed": False}


def context_window(drafter: Any) -> int | None:
    """Context positions a DFlash drafter can attend to, or None when some
    layer is full attention (then it reads the whole prompt)."""
    config = getattr(drafter, "config", None)
    layer_types = list(getattr(config, "layer_types", None) or [])
    window = getattr(config, "sliding_window", None)
    if not layer_types or window is None:
        return None
    if any(t != "sliding_attention" for t in layer_types):
        return None
    return max(1, int(window) - 1)


def _speculative_verify_dflash_hidden(self, inputs, cache, capture_layer_ids):
    out = self(
        inputs,
        cache=cache,
        capture_layer_ids=capture_layer_ids,
        speculative_verify=True,
        return_hidden=True,
        skip_logits=True,
    )
    return out.hidden_states[:-1], out.hidden_states[-1], out.gdn_states


def _install_row_capture() -> None:
    """Route a capturing one-row call through upstream's per-row path."""
    from mlx_vlm.models.qwen3_5 import language as lang

    cls = lang.Qwen3_5Model
    orig = cls.__call__

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        position_ids=None,
        capture_layer_ids=None,
        hidden_sink=None,
    ):
        batch = (inputs_embeds if inputs_embeds is not None else inputs).shape[0]
        fa_cache = cache[self.fa_idx] if cache is not None else None
        if (
            hidden_sink is None
            or batch != 1
            or fa_cache is None
            or not lang._is_single_row_batch_cache(fa_cache)
        ):
            return orig(
                self,
                inputs,
                inputs_embeds=inputs_embeds,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                capture_layer_ids=capture_layer_ids,
                hidden_sink=hidden_sink,
            )
        # Same as upstream's hidden_sink-free branch, keeping the capture.
        row_cache = [
            None
            if entry is None
            else lang._extract_row_cache(entry, 0)
            if lang._is_single_row_batch_cache(entry)
            else entry
            for entry in cache
        ]
        out = orig(
            self,
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=row_cache,
            position_ids=position_ids,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
        )
        for i, entry in enumerate(row_cache):
            if cache[i] is None or entry is None:
                continue
            if hasattr(cache[i].__class__, "merge"):
                cache[i] = cache[i].__class__.merge([entry])
        return out

    cls.__call__ = __call__


def _install_context_window() -> None:
    from mlx_vlm.speculative import utils as spec_utils
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

    orig_hidden = DFlashDraftModel._hidden

    def _hidden(self, inputs, target_hidden, cache):
        keep = context_window(self)
        length = int(target_hidden.shape[1])
        if keep is not None and length > keep:
            skip = length - keep
            target_hidden = target_hidden[:, skip:]
            for c in cache:
                c.offset += skip
        return orig_hidden(self, inputs, target_hidden, cache)

    DFlashDraftModel._hidden = _hidden

    prefill_cls = spec_utils.SpeculativePrefill
    orig_init = prefill_cls.__init__
    orig_append = prefill_cls.append

    def __init__(self, draft_kind, drafter):
        orig_init(self, draft_kind, drafter)
        self._yunshu_keep = (
            context_window(drafter)
            if draft_kind == "dflash" and drafter is not None
            else None
        )

    def append(self, output):
        orig_append(self, output)
        keep = getattr(self, "_yunshu_keep", None)
        if keep is None or not self.chunks:
            return
        # The final chunk (``finish``) adds its own positions, so the window
        # is covered as long as the retained chunks hold ``keep`` positions.
        retained = 0
        for index in range(len(self.chunks) - 1, -1, -1):
            retained += int(self.chunks[index][0].shape[1])
            if retained >= keep:
                if index:
                    del self.chunks[:index]
                break

    prefill_cls.__init__ = __init__
    prefill_cls.append = append


def install(language_model: Any = None) -> bool:
    """Install the three patches (idempotent). ``language_model``'s class gets
    the greedy DFlash verify hook when it verifies through
    ``speculative_argmax_from_hidden`` but lacks it."""
    cls = type(language_model) if language_model is not None else None
    if (
        cls is not None
        and callable(getattr(cls, "speculative_argmax_from_hidden", None))
        and not hasattr(cls, "speculative_verify_dflash_hidden")
    ):
        cls.speculative_verify_dflash_hidden = _speculative_verify_dflash_hidden
    if _STATE["installed"]:
        return True
    _install_row_capture()
    _install_context_window()
    _STATE["installed"] = True
    return True


__all__ = ["context_window", "install"]
