# Patches upstream (MIT): mlx-vlm qwen4_exp.LanguageModel.__call__; see vendor.json.
"""DFlash layer features on the Qwen4 invariant forward (MTP keeps HC states)."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_PREFILL_CAPTURE: ContextVar[Any] = ContextVar("qwen4_dflash_capture", default=None)


def stream_mean(hidden, hc_count, hidden_size):
    """Official DFlash2 features: completed layer outputs averaged over HC streams."""
    return hidden.reshape(*hidden.shape[:-1], hc_count, hidden_size).mean(axis=-2)


def capture_forward(
    lm,
    inputs,
    cache,
    capture_layer_ids,
    *,
    inputs_embeds=None,
    position_ids=None,
    skip_logits=False,
):
    import mlx.core as mx
    from mlx_vlm.models.base import LanguageModelOutput
    from mlx_vlm.models.qwen4_exp.language import _QWEN4_BATCH_INVARIANT_FORWARD

    from .kernels.tf_hc_forward import _tf_model

    forward = _QWEN4_BATCH_INVARIANT_FORWARD
    if position_ids is None and cache is not None:
        offset = cache[lm.model.fa_idx].offset
        offsets = mx.array(offset).astype(mx.int64).reshape(-1)
        offsets = mx.broadcast_to(offsets, (inputs.shape[0],))
        if lm._rope_deltas is not None:
            offsets = offsets + lm._rope_deltas[: inputs.shape[0]].reshape(-1)
        position_ids = offsets[:, None] + mx.arange(inputs.shape[1])[None]
        if lm._position_ids is not None and lm._position_ids.ndim == 3:
            position_ids = mx.broadcast_to(position_ids[None], (3, *position_ids.shape))
    captured = []
    hidden = _tf_model(
        forward,
        lm.model,
        inputs,
        cache,
        inputs_embeds,
        position_ids,
        capture_layer_ids=set(capture_layer_ids),
        hidden_sink=captured,
    )
    if len(captured) != len(set(capture_layer_ids)):
        raise ValueError("Qwen4 DFlash capture layer is outside the target")
    logits = None
    if not skip_logits:
        mixed = forward._hyper_connection(lm.model.hyper_connection_mixer, hidden)
        logits = (
            forward._embedding_as_linear(lm.model.embed_tokens, mixed)
            if lm.args.tie_word_embeddings
            else forward._linear(lm.lm_head, mixed)
        )
    return LanguageModelOutput(logits=logits, hidden_states=captured), hidden


def install(lm: Any) -> None:
    """Preserve five mixed layer features during decode and transactional verify."""
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpDecoderLayer
    from mlx_vlm.speculative.cache_state import start_speculative_cache

    cls = type(lm)
    if cls.__dict__.get("_yunshu_dflash_capture"):
        return
    original = cls.__call__
    layer_call = Qwen4ExpDecoderLayer.__call__

    def completed_layer(self, *args, **kwargs):
        hidden = layer_call(self, *args, **kwargs)
        active = _PREFILL_CAPTURE.get()
        if active is not None:
            selected, captured, streams, width = active
            if id(self) in selected:
                captured.append(stream_mean(hidden, streams, width))
        return hidden

    Qwen4ExpDecoderLayer.__call__ = completed_layer

    def call(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        layers = kwargs.get("capture_layer_ids")
        if layers and (inputs.shape[1] == 1 or kwargs.get("speculative_verify")):
            transaction = None
            if kwargs.get("speculative_verify"):
                transaction = start_speculative_cache(cache, inputs.shape[1])
            try:
                output, _ = capture_forward(
                    self,
                    inputs,
                    cache,
                    layers,
                    inputs_embeds=inputs_embeds,
                    position_ids=kwargs.get("position_ids"),
                    skip_logits=kwargs.get("skip_logits", False),
                )
                output.gdn_states = transaction
                return output
            except BaseException:
                if transaction is not None:
                    transaction.abort()
                raise
        if layers:
            # The upstream wrapper truncates return_hidden to one MTP HC state.
            # Public layer captures already contain all DFlash features.
            kwargs["return_hidden"] = False
            captured = []
            selected = {
                id(layer)
                for index, layer in enumerate(self.model.layers)
                if index in layers
            }
            token = _PREFILL_CAPTURE.set(
                (selected, captured, self.args.hc_count, self.args.hidden_size)
            )
            try:
                output = original(self, inputs, inputs_embeds, mask, cache, **kwargs)
                if len(captured) != len(set(layers)):
                    raise ValueError("Qwen4 DFlash capture layer is outside the target")
                output.hidden_states = captured
                return output
            finally:
                _PREFILL_CAPTURE.reset(token)
        return original(self, inputs, inputs_embeds, mask, cache, **kwargs)

    def verify(self, inputs, cache, capture_layer_ids):
        transaction = start_speculative_cache(cache, inputs.shape[1])
        try:
            output, hidden = capture_forward(
                self,
                inputs,
                cache,
                capture_layer_ids,
                skip_logits=True,
            )
            return output.hidden_states, hidden, transaction
        except BaseException:
            transaction.abort()
            raise

    cls.__call__ = call
    cls.speculative_verify_dflash_hidden = verify
    cls._yunshu_dflash_capture = True
