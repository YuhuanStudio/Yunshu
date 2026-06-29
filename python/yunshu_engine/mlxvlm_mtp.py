"""mlx-vlm native MTP backend for Qwen3.5 / Qwen3.6.

Speculative decoding via mlx-vlm's CORRECT MTP path (MTP-aware GatedDeltaNet that
captures intermediate SSM states + rollback_speculative_cache + qwen3_5_mtp
drafter + run_speculative_rounds). HONESTY: a standalone proof script
(scripts/bench/bench_mtp_vlm_27b.py, self-marked "INTEGRATION TODO") measured ~1.82x
on Qwen3.6-27B (M3 Max) — that figure is NOT served/regression-gated, and the wired
backend honors only temperature + is non-streaming + single-backend. Treat as
EXPERIMENTAL, not a shipped prod win. Our mlx-lm-based MTP patch could not do this — it
lacked the SSM intermediate-state capture (so 27B gave garbage); mlx-vlm has it.

Opt-in: requires reference/mlx-vlm on the path (it imports cleanly in our env;
the installed mlx_vlm 0.5.0 lacks the speculative module). Greedy/pure-temperature
requests get the MTP speedup (lossless by construction — the verify is exact);
everything else falls back to plain autoregressive generation on the same model.

This is a standalone backend (its own mlx-vlm model + drafter), used only when
YUNSHU_MTP=1 and the model is MTP-capable. It does NOT touch the default
mlx-lm fast path.
"""

from __future__ import annotations

import contextlib
import glob
import json
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REF_MLXVLM = "reference/mlx-vlm"


def _ensure_mlxvlm_on_path() -> bool:
    """Put reference/mlx-vlm first on sys.path (has the MTP the installed 0.5.0 lacks)."""
    ref = os.path.abspath(_REF_MLXVLM)
    if not os.path.isdir(ref):
        return False
    if ref not in sys.path:
        sys.path.insert(0, ref)
    # Drop a pre-imported MTP-less mlx_vlm so the reference one is picked up.
    import importlib.util as u

    spec = u.find_spec("mlx_vlm.speculative.mtp")
    return spec is not None


def is_mtp_capable(model_path: str) -> bool:
    """True if the checkpoint is a Qwen3.5/3.6 with native MTP weights."""
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
    # MTP weights present (bare mtp.* or VLM-nested language_model.mtp.*)?
    try:
        idx = json.loads(
            (Path(model_path) / "model.safetensors.index.json").read_text()
        )
        wm = idx.get("weight_map", idx)
        return any(".mtp." in k or k.startswith("mtp.") for k in wm)
    except Exception:
        return True  # config says MTP; assume embedded


@contextlib.contextmanager
def _tolerant_target_load():
    """Scoped: let the TARGET ignore the embedded mtp.* keys it doesn't use
    (they belong to the drafter). Restored immediately after — the drafter and
    everything else still load strictly."""
    import mlx.nn as nn

    orig = nn.Module.load_weights

    def _lw(self, weights, strict=True):
        try:
            return orig(self, weights, strict=strict)
        except ValueError:
            return orig(self, weights, strict=False)

    nn.Module.load_weights = _lw
    try:
        yield
    finally:
        nn.Module.load_weights = orig


def _build_drafter(model_path: str, out_dir: str) -> str:
    """Split the native MTP head into a standalone drafter folder. Handles BOTH
    the bare ``mtp.*`` layout and this checkpoint's VLM-nested
    ``language_model.mtp.*`` layout (mlx-vlm's own splitter only handles bare)."""
    import mlx.core as mx

    out = Path(out_dir)
    if (out / "model.safetensors").exists() and (out / "config.json").exists():
        return str(out)
    out.mkdir(parents=True, exist_ok=True)
    sel = {}
    for sh in glob.glob(str(Path(model_path) / "*.safetensors")):
        for k, v in mx.load(sh).items():
            if k.startswith("language_model.mtp."):
                sel[k[len("language_model.mtp.") :]] = v
            elif k.startswith("mtp."):
                sel[k[len("mtp.") :]] = v
    if not sel:
        raise ValueError(f"No MTP tensors found in {model_path}")
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


class MLXVLMMtp:
    """Loads an mlx-vlm Qwen3.5/3.6 target + its MTP drafter and serves greedy
    requests with lossless MTP speculative decoding."""

    def __init__(self, model_path: str, drafter_dir: str | None = None):
        self.model_path = model_path
        self.drafter_dir = drafter_dir or (model_path.rstrip("/") + "-mtp-drafter")
        self.model = None
        self.drafter = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        if not _ensure_mlxvlm_on_path():
            raise RuntimeError("reference/mlx-vlm with speculative MTP not available")
        from mlx_vlm.speculative.drafters import load_drafter
        from mlx_vlm.utils import load_model as vlm_load
        from transformers import AutoTokenizer

        drafter_path = _build_drafter(self.model_path, self.drafter_dir)
        with _tolerant_target_load():
            self.model = vlm_load(Path(self.model_path))
        self.drafter, kind = load_drafter(drafter_path)
        if kind != "mtp":
            raise RuntimeError(f"expected mtp drafter, got {kind}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
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

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.0,
        prompt: str | None = None,
    ) -> dict:
        """Greedy → MTP (lossless ~1.8x); temperature>0 → plain generation.
        Returns {text, token_ids, completion_tokens, used_mtp}.

        If ``prompt`` (a pre-templated string from the engine's
        _apply_chat_template) is given it is used directly — preferred, since it
        carries the full role normalization + family adapter + BOS guard."""
        import mlx.core as mx
        from mlx_lm.models import cache as kvcache
        from mlx_lm.sample_utils import make_sampler
        from mlx_vlm.speculative.utils import (
            make_speculative_prompt_cache,
            run_speculative_rounds,
            speculative_prefill_kwargs,
        )

        if not self._loaded:
            self.load()
        ids = (
            self._encode_text(prompt) if prompt is not None else self._encode(messages)
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

        if greedy:
            pk = speculative_prefill_kwargs("mtp", self.drafter)
            cache_ = make_speculative_prompt_cache(
                lm, draft_kind="mtp", batch_size=1, left_padding=[0], make_cache=None
            )
            out = lm(input_mx, cache=cache_, **pk)
            first_tok = sample(out.logits[:, -1:])
            toks: list[int] = []
            for tk, _lp in run_speculative_rounds(
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
            ):
                t = int(tk) if not isinstance(tk, list) else int(tk[0])
                if t in eos:
                    break
                toks.append(t)
                if len(toks) >= max_tokens:
                    break
            text = self.tokenizer.decode(toks)
            return {
                "text": text,
                "token_ids": toks,
                "completion_tokens": len(toks),
                "used_mtp": True,
            }

        # Non-greedy fallback: plain autoregressive on the same model.
        c = kvcache.make_prompt_cache(lm)
        o = lm(input_mx, cache=c)
        t = int(sample(o.logits[:, -1:]).item())
        toks = []
        if t not in eos:
            toks.append(t)
        for _ in range(max_tokens - 1):
            o = lm(mx.array([[t]]), cache=c)
            t = int(sample(o.logits[:, -1:]).item())
            if t in eos:
                break
            toks.append(t)
        return {
            "text": self.tokenizer.decode(toks),
            "token_ids": toks,
            "completion_tokens": len(toks),
            "used_mtp": False,
        }
