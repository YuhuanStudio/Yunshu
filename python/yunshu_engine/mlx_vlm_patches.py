"""Runtime monkeypatches for the Qwen3-Omni audio path.

STATUS: the upstream PR (yuhuanowo → Blaizzy/mlx-vlm, commit
`11e9d9f fix(qwen3_omni_moe): enable audio input`) is MERGED. When the active
mlx-vlm already carries the fix (e.g. reference/mlx-vlm) each patch DETECTS it via
source inspection and SKIPS — so this module is now a no-op there, and only still
applies on an older mlx-vlm release that predates the merge (defence in depth).
Remove the module entirely once the pinned/installed mlx-vlm is past the merge.

Carries two fixes for the Qwen3-Omni audio path. Without them, any Omni call with
audio input crashes in the audio tower / masked_scatter. Both patches are
idempotent and no-ops for non-audio / non-Omni requests, so applying them globally
is safe.

Bug #1 — qwen3_omni_moe.Model.get_input_embeddings reads only
`input_features_mask`, but the shared prepare_inputs path emits the audio mask
as `feature_attention_mask`. The mask is dropped, get_audio_features skips its
2-D reshape, and the audio tower gets a 4-D input it can't broadcast.

Bug #2 — qwen3_omni_moe uses MessageFormat.LIST_WITH_IMAGE_FIRST, whose
formatter (`MessageFormatter._format_list_with_image`) inserts image tokens but
not audio ones. `{"type": "audio"}` is dropped, the prompt has zero audio
tokens, and masked_scatter fails ((N) vs (0)). The sibling
`_format_list_with_image_type` already does this insertion; we mirror it.

Remove this module once the upstream fix ships in a released mlx-vlm.
"""

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_mlx_vlm_patches() -> list[str]:
    """Apply the Omni audio fixes once. Returns the list of patch names applied."""
    global _APPLIED
    if _APPLIED:
        return []

    applied: list[str] = []
    applied += _patch_qwen3_omni_audio_mask()
    applied += _patch_formatter_audio_token()
    applied += _patch_qwen3_5_empty_chunk()
    _APPLIED = True

    if applied:
        logger.info("Applied mlx-vlm Omni audio patches: %s", ", ".join(applied))
    return applied


def _patch_qwen3_5_empty_chunk() -> list[str]:
    """Make the qwen3_5 hybrid decoder survive an empty (S=0) chunk.

    Both layer types reshape with `-1` (`z.reshape(B,S,-1,head_v_dim)` in the
    GatedDeltaNet; `q.reshape(B,L,heads,-1)` in the attention), and `-1` cannot be
    inferred from a 0-element array → "Cannot infer the shape of an empty array".
    This fires on the A2 hybrid-VLM text-prefix reuse path (VLMEngine, Qwen3.5/3.6)
    when a boundary-snapshot resume leaves an empty suffix. Guarding at the DECODER
    LAYER (one place, covers linear-attn + full-attn + mlp) is robust: a transformer
    layer over 0 tokens is a no-op on the residual stream and leaves the KV/recurrent
    state unchanged, so we return `x` untouched. Non-empty chunks run unmodified.
    """
    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5DecoderLayer as Layer
    except Exception:
        return []
    if getattr(Layer, "_yunshu_empty_chunk_patched", False):
        return []
    _orig = Layer.__call__

    def patched(self, x, *args, **kwargs):
        if getattr(x, "ndim", 0) >= 2 and x.shape[1] == 0:  # S == 0: nothing to do
            return x
        return _orig(self, x, *args, **kwargs)

    Layer.__call__ = patched
    Layer._yunshu_empty_chunk_patched = True
    return ["qwen3_5 empty-chunk (S=0) decoder-layer guard"]


def _patch_qwen3_omni_audio_mask() -> list[str]:
    """Bug #1: accept the audio mask under either param name."""
    try:
        from mlx_vlm.models.qwen3_omni_moe import qwen3_omni_moe as omni
    except Exception:
        return []

    Model = omni.Model
    if getattr(Model, "_yunshu_audio_mask_patched", False):
        return []

    # Upstream merged (commit 11e9d9f): get_input_embeddings already reads
    # feature_attention_mask → skip, don't shadow the upstream code.
    try:
        import inspect

        if "feature_attention_mask" in inspect.getsource(Model.get_input_embeddings):
            Model._yunshu_audio_mask_patched = True
            return []
    except Exception:
        pass

    orig = Model.get_input_embeddings

    def patched(self, *args, input_features_mask=None, **kwargs):
        # prepare_inputs passes the mask as `feature_attention_mask`; only
        # gemma4/gemma3n use `input_features_mask`. Accept both.
        if input_features_mask is None:
            input_features_mask = kwargs.pop("feature_attention_mask", None)
        else:
            kwargs.pop("feature_attention_mask", None)
        return orig(self, *args, input_features_mask=input_features_mask, **kwargs)

    Model.get_input_embeddings = patched
    Model._yunshu_audio_mask_patched = True
    return ["qwen3_omni audio-mask key"]


def _patch_formatter_audio_token() -> list[str]:
    """Bug #2: insert audio placeholder in the LIST_WITH_IMAGE_FIRST formatter."""
    try:
        from mlx_vlm import prompt_utils as pu
    except Exception:
        return []

    MF = pu.MessageFormatter
    if getattr(MF, "_yunshu_audio_token_patched", False):
        return []

    # Upstream merged: the formatter already inserts the audio placeholder. Detect
    # BEHAVIOURALLY (source contains "audio" via param names even when unfixed) —
    # call it with one audio and check the output. If already inserted → skip, so
    # we don't double-insert over the upstream code.
    try:
        _probe = MF("qwen3_omni_moe")._format_list_with_image(
            "x", "user", False, False, num_images=0, num_audios=1, image_first=True
        )
        if "audio" in [
            c.get("type") for c in _probe.get("content", []) if isinstance(c, dict)
        ]:
            MF._yunshu_audio_token_patched = True
            return []
    except Exception:
        pass

    orig = MF._format_list_with_image

    def patched(
        self,
        prompt,
        role,
        skip_image_token,
        skip_audio_token,
        num_images,
        num_audios,
        image_first=False,
        use_image_url=False,
        **kwargs,
    ):
        msg = orig(
            self,
            prompt,
            role,
            skip_image_token,
            skip_audio_token,
            num_images,
            num_audios,
            image_first=image_first,
            use_image_url=use_image_url,
            **kwargs,
        )
        if role == "user" and not skip_audio_token and num_audios > 0:
            msg["content"] = (
                msg["content"] + [pu.MessageBuilder.audio_message()] * num_audios
            )
        return msg

    MF._format_list_with_image = patched
    MF._yunshu_audio_token_patched = True
    return ["formatter audio token"]
