"""mlx-vlm native MTP backend for Qwen3.5 / Qwen3.6.

Speculative decoding via mlx-vlm's CORRECT MTP path (MTP-aware GatedDeltaNet that
captures intermediate SSM states + rollback_speculative_cache + qwen3_5_mtp
drafter + run_speculative_rounds). The served VLM path now drafts through
``VLMBatchRunner`` instead; this
standalone backend honors only temperature and remains EXPERIMENTAL. Our mlx-lm-based MTP patch could not do this — it
lacked the SSM intermediate-state capture (so 27B gave garbage); mlx-vlm has it.

Opt-in: requires the installed mlx-vlm to ship ``mlx_vlm.speculative`` (>=0.7.3,
the locked version). Greedy/pure-temperature
requests get the MTP speedup (lossless by construction — the verify is exact);
everything else falls back to plain autoregressive generation on the same model.

This standalone research backend owns its own mlx-vlm model and drafter.
It has no serving dispatch. The checkpoint detection and in-memory drafter
helpers below remain shared with the VLM batch runner.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def _mlxvlm_has_mtp() -> bool:
    """True when the installed mlx-vlm ships the speculative MTP runtime (>=0.7)."""
    import importlib.util as u

    try:
        return u.find_spec("mlx_vlm.speculative.mtp") is not None
    except ModuleNotFoundError:
        return False


def is_mtp_capable(model_path: str) -> bool:
    """True only when a supported checkpoint has indexed native MTP weights."""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return False
    tc = cfg.get("text_config") or cfg
    if int(tc.get("mtp_num_hidden_layers", 0) or 0) <= 0:
        return False
    if cfg.get("model_type") not in ("qwen3_5", "qwen3_6"):
        return False
    # A config can advertise MTP even when the downloaded checkpoint omits
    # the head. Do not turn an unreadable/missing index into a positive match.
    try:
        idx = json.loads(
            (Path(model_path) / "model.safetensors.index.json").read_text()
        )
        wm = idx.get("weight_map", idx)
        if not isinstance(wm, dict):
            return False
        shards = {
            shard
            for key, shard in wm.items()
            if key.startswith("mtp.") or key.startswith("language_model.mtp.")
        }
        return bool(shards) and all(
            isinstance(shard, str) and (Path(model_path) / shard).is_file()
            for shard in shards
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def unindexed_mtp_warning(model_path: str) -> str | None:
    """Explain an ignored standalone head without claiming it is loadable."""
    heads = sorted(Path(model_path).glob("mtp-weights*.safetensors"))
    if not heads or is_mtp_capable(model_path):
        return None
    return (
        "MTP weights found outside a usable model.safetensors.index.json: "
        + ", ".join(p.name for p in heads)
        + "; native MTP is unavailable and may start with draft=off. "
        "Use a checkpoint whose index lists its mtp.* tensors."
    )


@contextlib.contextmanager
def _tolerant_target_load():
    """Scoped: let the TARGET ignore the embedded mtp.* keys it doesn't use
    (they belong to the drafter). Restored immediately after — the drafter and
    everything else still load strictly. A target parameter without a tensor still fails."""
    from .checkpoint_keys import lenient_extras_load

    with lenient_extras_load():
        yield


def _load_mtp_head_tensors(model_path: str) -> dict:
    """Read indexed MTP tensors without copying a drafter onto disk."""
    import mlx.core as mx
    from safetensors import safe_open

    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map", index)
    if not isinstance(weight_map, dict):
        raise ValueError("Invalid safetensors weight map")
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith("language_model.mtp.") or key.startswith("mtp."):
            by_shard.setdefault(shard, []).append(key)
    sel = {}
    for shard, keys in by_shard.items():
        path = Path(model_path) / shard
        try:
            # Match mlx-vlm's selective load where the safetensors dtype permits.
            with safe_open(path, framework="mlx") as source:
                tensors = {key: mx.array(source.get_tensor(key)) for key in keys}
        except (AttributeError, RuntimeError, TypeError):
            # safetensors' MLX bridge currently rejects bf16 in the Qwen3.8
            # head. Limit mx.load to its indexed shard, not every model shard.
            full_shard = mx.load(str(path))
            tensors = {key: full_shard[key] for key in keys}
        for key, value in tensors.items():
            name = key.removeprefix("language_model.mtp.").removeprefix("mtp.")
            sel[name] = value
    if not sel:
        raise ValueError(f"No MTP tensors found in {model_path}")
    return sel


def _build_drafter(model_path: str, out_dir: str) -> str:
    """Optional explicit export; serving loads the head in memory instead."""
    import mlx.core as mx

    out = Path(out_dir)
    if (out / "model.safetensors").exists() and (out / "config.json").exists():
        return str(out)
    out.mkdir(parents=True, exist_ok=True)
    sel = _load_mtp_head_tensors(model_path)
    src_cfg = json.loads((Path(model_path) / "config.json").read_text())
    tc = dict(src_cfg.get("text_config") or {})
    mx.save_safetensors(str(out / "model.safetensors"), sel, metadata={"format": "mlx"})
    dcfg = {
        "model_type": "qwen3_5_mtp",
        "text_config": tc,
        "block_size": int(tc.get("mtp_num_hidden_layers", 1) + 2),
        "tie_word_embeddings": bool(tc.get("tie_word_embeddings", True)),
    }
    if any(k.endswith(".scales") for k in sel):
        q = src_cfg.get("quantization")
        if q is not None:
            dcfg["quantization"] = q
            dcfg["quantization_config"] = q
    (out / "config.json").write_text(json.dumps(dict(sorted(dcfg.items())), indent=2))
    for n in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "chat_template.jinja",
    ):
        src = Path(model_path) / n
        if src.exists():
            shutil.copy(src, out / n)
    return str(out)


def _load_drafter_in_memory(model_path: str):
    """Instantiate mlx-vlm's Qwen MTP head from the existing target shards."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import (
        Qwen3_5MTPDraftModel,
    )

    src_cfg = json.loads((Path(model_path) / "config.json").read_text())
    text_cfg = src_cfg.get("text_config") or {}
    weights = _load_mtp_head_tensors(model_path)
    config = Qwen3_5MTPConfig.from_dict(
        {
            "model_type": "qwen3_5_mtp",
            "text_config": text_cfg,
            "block_size": int(text_cfg.get("mtp_num_hidden_layers", 1)) + 2,
            "tie_word_embeddings": bool(text_cfg.get("tie_word_embeddings", True)),
        }
    )
    drafter = Qwen3_5MTPDraftModel(config)
    if any(key.endswith(".scales") for key in weights):
        quant = src_cfg.get("mtplx_mtp_quantization") or src_cfg.get("quantization")
        if not isinstance(quant, dict):
            raise ValueError("Quantized MTP head has no quantization config")
        nn.quantize(
            drafter,
            group_size=int(quant["group_size"]),
            bits=int(quant["bits"]),
            mode=quant.get("mode", "affine"),
            class_predicate=lambda path, _module: f"{path}.scales" in weights,
        )
    drafter.load_weights(list(weights.items()), strict=True)
    mx.eval(drafter.parameters())
    return drafter


class MLXVLMMtp:
    """Loads an mlx-vlm Qwen3.5/3.6 target + its MTP drafter and serves greedy
    requests with lossless MTP speculative decoding."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.drafter = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        if not _mlxvlm_has_mtp():
            raise RuntimeError(
                "installed mlx-vlm lacks speculative MTP (need mlx-vlm>=0.7.3)"
            )
        from mlx_lm.tokenizer_utils import load as load_tokenizer
        from mlx_vlm.speculative.drafters import validate_drafter_compatibility
        from mlx_vlm.utils import load_model as vlm_load

        with _tolerant_target_load():
            self.model = vlm_load(Path(self.model_path))
        self.drafter = _load_drafter_in_memory(self.model_path)
        validate_drafter_compatibility(self.model, self.drafter, "mtp")
        # Per-request streaming detokenizers are available through the MLX
        # wrapper; it forwards ordinary HF tokenizer methods unchanged.
        self.tokenizer = load_tokenizer(Path(self.model_path))
        self._loaded = True
        logger.info("MLXVLMMtp loaded: %s (+ MTP drafter)", self.model_path)

    def _encode_text(self, text: str) -> list[int]:
        """Encode a pre-templated prompt string with a double-BOS guard.

        When the caller supplies a prompt already produced by the
        engine's _apply_chat_template (which has done role normalization +
        family-adapter + dangling-think close), encode it without re-adding BOS
        if it already starts with the literal bos_token. Mirrors
        BatchedEngine._encode_prompt.
        """
        bos = getattr(self.tokenizer, "bos_token", None)
        add_special = not (isinstance(bos, str) and bos and text.startswith(bos))
        try:
            return self.tokenizer.encode(text, add_special_tokens=add_special)
        except TypeError:
            return self.tokenizer.encode(text)

    def _encode(self, messages: list[dict]) -> list[int]:
        # Fallback when no pre-templated prompt is supplied. Normalize
        # OpenAI's `developer`→system and legacy `function`→tool roles BEFORE the
        # template, else apply_chat_template raises "Unknown role" and crashes the
        # request (developer is sent routinely). Mirrors the engine's remap.
        norm = []
        for m in messages:
            role = m.get("role")
            if role == "developer":
                m = {**m, "role": "system"}
            elif role == "function":
                m = {**m, "role": "tool"}
            norm.append(m)
        txt = self.tokenizer.apply_chat_template(
            norm, add_generation_prompt=True, tokenize=False
        )
        return self._encode_text(txt)

    def iter_token_ids(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        prompt: str | list[int] | None = None,
        use_mtp: bool = True,
        cancel_event=None,
    ) -> Iterator[int]:
        """Yield verified tokens on the executor thread as soon as they exist.

        If ``prompt`` (a pre-templated string from the engine's
        _apply_chat_template) is given it is used directly — preferred, since it
        carries the full role normalization + family adapter + BOS guard."""
        import mlx.core as mx
        from mlx_lm.models import cache as kvcache
        from mlx_lm.sample_utils import make_sampler
        from mlx_vlm.generate.ar import _make_cache
        from mlx_vlm.speculative.utils import (
            make_speculative_prompt_cache,
            run_speculative_rounds,
            speculative_prefill_kwargs,
        )

        if not self._loaded:
            self.load()
        ids = (
            list(prompt)
            if isinstance(prompt, list)
            else self._encode_text(prompt)
            if prompt is not None
            else self._encode(messages)
        )
        input_mx = mx.array([ids], dtype=mx.int32)
        lm = self.model.language_model
        greedy = temperature <= 0.0
        # temp>0 → per-request sampler (mlx-lm's make_sampler routes through
        # the @mx.compile-cached categorical_sampling whose trapped global PRNG state
        # collapses sequential/concurrent temp>0 requests to identical streams). No
        # seed is plumbed here, but a fresh numpy Generator per call already breaks the
        # shared-cache collapse. Greedy stays on argmax make_sampler.
        if not greedy:
            from .batched_engine import _build_temp_sampler

            sampler = _build_temp_sampler(
                temperature=temperature, top_p=1.0, top_k=0, min_p=0.0, seed=None
            )
        else:
            sampler = make_sampler(temp=0.0)

        def sample(logits):
            return sampler(logits.reshape(-1, logits.shape[-1]))

        eos = set()
        for a in ("eos_token_id",):
            e = getattr(self.tokenizer, a, None)
            if isinstance(e, int):
                eos.add(e)

        if greedy and use_mtp:
            pk = speculative_prefill_kwargs("mtp", self.drafter)
            cache_ = make_speculative_prompt_cache(
                lm,
                draft_kind="mtp",
                batch_size=1,
                left_padding=[0],
                make_cache=lambda model, left_padding: _make_cache(model, left_padding),
            )
            out = lm(input_mx, cache=cache_, **pk)
            first_tok = sample(out.logits[:, -1:])
            for emitted, (tk, _lp) in enumerate(
                run_speculative_rounds(
                    self.model,
                    self.drafter,
                    cache_,
                    input_mx,
                    first_tok,
                    out.logits[:, -1:],
                    out,
                    draft_kind="mtp",
                    max_tokens=max_tokens,
                    sampler=sample,
                    sampler_is_greedy=True,
                ),
                start=1,
            ):
                if cancel_event is not None and cancel_event.is_set():
                    break
                t = int(tk) if not isinstance(tk, list) else int(tk[0])
                if t in eos:
                    break
                yield t
                if emitted >= max_tokens:
                    break
            return

        # Non-greedy fallback: plain autoregressive on the same model.
        c = kvcache.make_prompt_cache(lm)
        o = lm(input_mx, cache=c)
        t = int(sample(o.logits[:, -1:]).item())
        emitted = 0
        if t in eos or (cancel_event is not None and cancel_event.is_set()):
            return
        yield t
        emitted = 1
        for _ in range(max_tokens - 1):
            if cancel_event is not None and cancel_event.is_set():
                break
            if emitted >= max_tokens:
                break
            o = lm(mx.array([[t]]), cache=c)
            t = int(sample(o.logits[:, -1:]).item())
            if t in eos:
                break
            yield t
            emitted += 1

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        prompt: str | list[int] | None = None,
        use_mtp: bool = True,
        cancel_event=None,
    ) -> dict:
        """Collect token iterator for non-streaming callers."""
        toks = list(
            self.iter_token_ids(
                messages, max_tokens, temperature, prompt, use_mtp, cancel_event
            )
        )
        return {
            "text": self.tokenizer.decode(toks),
            "token_ids": toks,
            "completion_tokens": len(toks),
            "used_mtp": temperature <= 0.0 and use_mtp,
        }
