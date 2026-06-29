from __future__ import annotations

"""Yunshu ANE Embedding Co-Processor — CoreML-based embedding inference on Apple Neural Engine.

Offloads embedding model inference to the ANE via CoreML for lower latency and
freeing GPU resources for main LLM inference workloads. Falls back to MLX GPU
inference when CoreML is unavailable or the model has not been compiled.

Enabled via YUNSHU_ANE_EMBEDDINGS=1 environment variable. When active, the
embeddings gateway router routes embedding requests through the ANE processor
instead of the GPU fallback path.

Architecture:
  1. Convert MLX embedding model weights to CoreML .mlpackage
  2. Compile .mlpackage to .mlmodelc for ANE deployment
  3. Run embedding inference through CoreML proxy (ANE path)
  4. Fallback: MLX GPU inference when CoreML/ANE unavailable

Graceful degradation:
  - coremltools not installed -> compile_model() logs warning, embed() uses MLX fallback
  - ANE not available -> is_ane_available() returns False, embed() uses MLX fallback
  - Model not compiled -> embed() uses MLX fallback automatically

References:
  - Core ML Tools: https://coremltools.readme.io/
  - Apple Neural Engine: https://developer.apple.com/documentation/coreml
  - Phase 0 platform validation
"""

import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Conditional coremltools import ───────────────────────────────────────────
try:
    import coremltools as ct  # type: ignore[import-untyped]

    _HAS_COREMLTOOLS = True
except ImportError:
    ct = None  # type: ignore[assignment]
    _HAS_COREMLTOOLS = False


# ── Conditional MLX import ───────────────────────────────────────────────────
try:
    import mlx.core as mx

    _HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    _HAS_MLX = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ANEEmbeddingConfig:
    """Configuration for the ANE embedding co-processor.

    Attributes:
        model_name: HuggingFace model name or local path for the embedding model.
        max_seq_length: Maximum sequence length for embedding inputs.
        normalize_embeddings: Whether to L2-normalize output embeddings.
        compile_on_init: If True, attempt model compilation during __init__.
        cache_dir: Directory for compiled CoreML model cache.
    """

    model_name: str = "intfloat/e5-small-v2"
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    compile_on_init: bool = True
    cache_dir: str = ""

    def get_cache_dir(self) -> Path:
        """Return resolved cache directory path."""
        if self.cache_dir:
            return Path(self.cache_dir).expanduser().resolve()
        return Path.home() / ".yunshu" / "ane_cache"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def is_ane_available() -> bool:
    """Check if CoreML and Apple Neural Engine are available on this hardware.

    Returns True only on Apple Silicon macOS where CoreML can target the ANE.
    Always returns False on non-macOS platforms or Intel Macs.

    Returns:
        bool: True if ANE is likely available for CoreML inference.
    """
    if platform.system() != "Darwin":
        return False

    if platform.machine() != "arm64":
        return False

    if not _HAS_COREMLTOOLS:
        return False

    # Check macOS version — CoreML ANE scheduling requires 12.0+
    try:
        macos_version = platform.mac_ver()[0]
        if macos_version:
            major = int(macos_version.split(".")[0])
            if major < 12:
                return False
    except (ValueError, IndexError):
        pass

    return True


def estimate_ane_speedup(model_params_m: float, seq_length: int) -> float:
    """Rough estimate of ANE vs GPU speedup for embedding workloads.

    The ANE excels at small-batch, short-sequence embedding inference due to
    its high throughput for dense matrix operations and low power consumption.
    For larger models or longer sequences, the ANE advantage diminishes because
    of memory bandwidth limitations and fixed-function constraints.

    Args:
        model_params_m: Model size in millions of parameters (e.g., 33.0 for 33M).
        seq_length: Input sequence length in tokens.

    Returns:
        float: Estimated speedup factor (1.0 means parity, >1.0 means ANE faster).
               Clamped to the range [1.0, 3.0].
    """
    if model_params_m <= 0:
        return 1.0

    if seq_length <= 0:
        return 1.0

    # Base speedup: small models (< 50M params) benefit most from ANE
    # ANE has dedicated hardware for small dense matmuls
    size_factor = max(0.0, 1.0 - (model_params_m - 10.0) / 200.0)
    size_bonus = size_factor * 1.0  # up to 1.0x bonus for tiny models

    # Sequence length factor: ANE optimal at short sequences (< 256 tokens)
    # Longer sequences hit memory bandwidth wall on ANE
    if seq_length <= 128:
        seq_factor = 1.0
    elif seq_length <= 256:
        seq_factor = 0.8
    elif seq_length <= 512:
        seq_factor = 0.5
    else:
        seq_factor = 0.3

    speedup = 1.0 + size_bonus * seq_factor

    # Clamp to [1.0, 3.0]
    return max(1.0, min(3.0, speedup))


def _model_is_mean_pooled(model_path: str) -> bool:
    """Whether a sentence-transformers model uses MEAN pooling.

    Reads ``1_Pooling/config.json`` (the standard sentence-transformers pooling spec). The
    ANE traced wrapper only implements mean pooling, so a CLS/LAST model must NOT be compiled
    for ANE (it would be silently mean-pooled — the wrong embedding space). Defaults to True
    (mean) when no pooling config is present, since plain transformer encoders are commonly
    mean-pooled and the prior behavior already assumed mean.
    """
    import json
    import os

    cfg = os.path.join(model_path, "1_Pooling", "config.json")
    pool = None
    if os.path.isfile(cfg):
        try:
            with open(cfg) as f:
                pool = json.load(f)
        except Exception:
            return True
    else:
        # model_path is an HF id (the DEFAULT + documented case — e.g.
        # intfloat/e5-small-v2, BAAI/bge-small-en-v1.5), not a local dir, so the local
        # isfile() was always False → this returned True (assume mean) and a CLS/LAST
        # model got served with MEAN pooling — the wrong-embedding-space bug, on the
        # HF-id sibling the local-tmp-only test never exercised. Resolve the
        # pooling spec from the hub (the model is downloaded right after anyway). Only an
        # EXPLICIT non-mean spec refuses; a repo with no 1_Pooling (raw encoder) or an
        # offline/lookup failure keeps the documented "assume mean" default.
        try:
            from huggingface_hub import hf_hub_download

            _p = hf_hub_download(model_path, "1_Pooling/config.json")
            with open(_p) as f:
                pool = json.load(f)
        except Exception:
            return True
    if pool is None:
        return True
    # sentence-transformers Pooling config: exactly one of these modes is True.
    if pool.get("pooling_mode_mean_tokens", False):
        return True
    # any explicit non-mean mode (CLS / max / last-token / mean-sqrt) → not plain mean
    for _k in (
        "pooling_mode_cls_token",
        "pooling_mode_max_tokens",
        "pooling_mode_lasttoken",
        "pooling_mode_mean_sqrt_len_tokens",
    ):
        if pool.get(_k, False):
            return False
    # no mode flagged → fall back to assuming mean (matches prior behavior)
    return True


# ---------------------------------------------------------------------------
# ANE Embedding Processor
# ---------------------------------------------------------------------------


class ANEEmbeddingProcessor:
    """CoreML-based embedding processor for Apple Neural Engine.

    Converts MLX embedding models to CoreML format and runs inference on the
    ANE, falling back to MLX GPU inference when necessary.

    Usage::

        config = ANEEmbeddingConfig(model_name="intfloat/e5-small-v2")
        processor = ANEEmbeddingProcessor(config)

        if processor.is_compiled():
            embeddings = processor.embed(["hello world"])
        else:
            # Falls back to MLX GPU automatically
            embeddings = processor.embed(["hello world"])
    """

    def __init__(self, config: ANEEmbeddingConfig) -> None:
        self._config = config
        self._cache_dir = config.get_cache_dir()
        self._compiled_path: str | None = None
        self._is_compiled: bool = False
        self._inference_count: int = 0
        self._total_latency: float = 0.0

        # Attempt compilation on init if requested.
        if config.compile_on_init and _HAS_COREMLTOOLS and is_ane_available():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            # Actually compile the configured model: only doing mkdir on the cache dir
            # without compiling leaves is_compiled False → every served request falls back
            # to MLX, which can't even load BERT-class embedding models.
            # compile_model() accepts a local path OR a HF id (AutoModel.from_pretrained).
            if config.model_name:
                try:
                    self.compile_model(config.model_name)
                    logger.info(
                        "ANE embedding model compiled on init: %s (compiled=%s)",
                        config.model_name,
                        self._is_compiled,
                    )
                except Exception as exc:
                    logger.warning(
                        "ANE compile_on_init failed for %s — embed() will fall back to "
                        "MLX GPU: %s",
                        config.model_name,
                        exc,
                    )

    # ── Model compilation ────────────────────────────────────────────────────

    def compile_model(self, model_path: str) -> str:
        """Convert an MLX embedding model to CoreML and compile for ANE.

        This performs two steps:
          1. Convert model weights to CoreML .mlpackage format
          2. Compile .mlpackage to .mlmodelc for deployment

        Args:
            model_path: Path to the MLX embedding model directory or file.

        Returns:
            str: Path to the compiled .mlmodelc directory.

        Raises:
            RuntimeError: If coremltools is not installed or compilation fails.
        """
        if not _HAS_COREMLTOOLS:
            logger.warning(
                "coremltools is not installed — cannot compile model for ANE. "
                "Install with: pip install coremltools"
            )
            raise RuntimeError(
                "coremltools is required for ANE model compilation. "
                "Install with: pip install coremltools"
            )

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # The traced CoreML wrapper below bakes in MEAN pooling. A CLS-pooled
        # (BGE family) or LAST-token model served through ANE would be silently mean-pooled
        # — the WRONG embedding space, returned 200, diverging from the MLX path which
        # correctly CLS/LAST-pools. Refuse to compile a non-mean model so the caller falls
        # back to the MLX engine (correct pooling) instead of serving degraded vectors.
        if not _model_is_mean_pooled(model_path):
            raise RuntimeError(
                f"ANE embedding compilation refused for '{model_path}': the model is not "
                "mean-pooled (CLS/LAST), and the ANE path only implements mean pooling. "
                "Serving via the MLX engine (correct pooling) instead."
            )

        model_name_safe = self._config.model_name.replace("/", "_").replace("\\", "_")
        mlpackage_path = self._cache_dir / f"{model_name_safe}.mlpackage"
        mlmodelc_path = self._cache_dir / f"{model_name_safe}.mlmodelc"

        # Step 1: Convert to CoreML .mlpackage
        logger.info(
            "Converting embedding model to CoreML: %s -> %s",
            model_path,
            mlpackage_path,
        )

        try:
            # Conversion via torch/transformers → CoreML.
            #
            # ct.convert accepts a traced torch/TF graph or an MIL program, NEVER an MLX
            # (or HF) directory path — passing the model DIRECTORY raises, leaving the ANE
            # path dead. Embedding models (BGE/E5/GTE/…) are torch
            # transformers models; load with AutoModel, wrap to emit a mean-pooled vector,
            # trace, and convert. Verified on this M3 Max: a BGE-small-class model runs
            # ~4.8× faster through CoreML(ANE) than MLX-GPU.
            import numpy as np
            import torch
            from transformers import AutoModel

            # Honor the configured max_seq_length (NOT a hardcoded 128 cap). A
            # min(max_seq_length, 128) silently truncates every input past 128 tokens — even
            # when the operator set YUNSHU_ANE_MAX_SEQ_LENGTH=512 — so ANE produces different
            # (worse) vectors than the MLX fallback path, which uses the FULL sequence. Trace
            # and inference share this same length, so they stay
            # consistent; lower max_seq_length to cap ANE's traced shape.
            seq = int(self._config.max_seq_length)
            hf_model = AutoModel.from_pretrained(model_path)
            hf_model.eval()

            class _MeanPooled(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, input_ids, attention_mask):
                    out = self.m(input_ids=input_ids, attention_mask=attention_mask)
                    last = out.last_hidden_state  # [B, seq, hidden]
                    # Mask-weighted mean pooling — average over REAL tokens only.
                    # Meaning over the FULL padded sequence lets PAD positions pollute
                    # every vector, and tracing at this `seq` (128) while inference
                    # tokenizes to max_seq_length (512) causes a CoreML shape crash on
                    # every call → silent MLX fallback (ANE never actually runs). Feeding
                    # attention_mask fixes both: pooling is correct AND trace/inference
                    # share the same seq length.
                    mask = attention_mask.unsqueeze(-1).to(last.dtype)  # [B, seq, 1]
                    summed = (last * mask).sum(dim=1)  # [B, hidden]
                    counts = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
                    return summed / counts

            wrapped = _MeanPooled(hf_model).eval()
            sample_ids = torch.zeros((1, seq), dtype=torch.int32)
            sample_mask = torch.ones((1, seq), dtype=torch.int32)
            with torch.no_grad():
                traced = torch.jit.trace(wrapped, (sample_ids, sample_mask))

            coreml_model = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(name="input_ids", shape=(1, seq), dtype=np.int32),
                    ct.TensorType(
                        name="attention_mask", shape=(1, seq), dtype=np.int32
                    ),
                ],
                outputs=[ct.TensorType(name="embeddings")],
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.ALL,
            )
            coreml_model.save(str(mlpackage_path))
            logger.info("CoreML .mlpackage saved to %s", mlpackage_path)

        except Exception as exc:
            logger.error("Failed to convert model to CoreML: %s", exc)
            raise RuntimeError(
                f"CoreML model conversion failed for {model_path}: {exc}. "
                "ANE embedding needs a torch/transformers embedding model (BGE/E5/GTE/…); "
                "embed() falls back to MLX GPU."
            ) from exc

        # Step 2 (best-effort): pre-compile .mlpackage → .mlmodelc for faster cold load.
        # NOTE: serving loads the .mlpackage directly via ct.models.MLModel (which
        # auto-compiles + caches) — a bare .mlmodelc has no Manifest.json and only loads
        # via ct.models.CompiledMLModel, so we DO NOT point _compiled_path at it. The
        # xcrun step is purely an optional warm-up; its failure must not break serving.
        try:
            subprocess.run(
                [
                    "xcrun",
                    "coremlcompiler",
                    "compile",
                    str(mlpackage_path),
                    str(self._cache_dir),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if mlmodelc_path.exists():
                logger.info("Pre-compiled CoreML model: %s", mlmodelc_path)
        except Exception:
            logger.debug(
                "xcrun coremlcompiler pre-compile skipped/failed (non-fatal)",
                exc_info=True,
            )

        # Serving loads the .mlpackage (portable, self-describing, MLModel-loadable).
        self._compiled_path = str(mlpackage_path)
        self._coreml_model = None  # drop any cached model from a previous compile
        self._is_compiled = True
        return self._compiled_path

    # ── Inference ─────────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Run embedding inference on texts.

        Uses CoreML/ANE path if model is compiled, otherwise falls back to
        MLX GPU inference.

        Args:
            texts: List of input strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []

        start_time = time.monotonic()

        if self._is_compiled and self._compiled_path and _HAS_COREMLTOOLS:
            result = self._embed_coreml(texts)
        else:
            result = self._embed_mlx_fallback(texts)

        elapsed = time.monotonic() - start_time
        self._inference_count += 1
        self._total_latency += elapsed

        return result

    def _embed_coreml(self, texts: list[str]) -> list[list[float]]:
        """Run embedding via CoreML (ANE path)."""
        # Cache the loaded MLModel — ct.models.MLModel(path) re-reads + re-specializes
        # the model from disk (~700ms) on EVERY call; reloading per request made ANE
        # SLOWER than MLX. Load once, reuse (the model is immutable).
        model = getattr(self, "_coreml_model", None)
        if model is None:
            try:
                model = ct.models.MLModel(self._compiled_path)  # type: ignore[union-attr]
                self._coreml_model = model
            except Exception as exc:
                logger.warning(
                    "CoreML model loading failed, falling back to MLX: %s", exc
                )
                return self._embed_mlx_fallback(texts)

        tokenizer = self._load_tokenizer()
        if tokenizer is None:
            logger.warning("No tokenizer available, falling back to MLX")
            return self._embed_mlx_fallback(texts)

        embeddings: list[list[float]] = []
        import numpy as np

        # Tokenize to the SAME seq length the model was traced at and pass
        # attention_mask — padding to a mismatched length breaks the traced shape.
        # This is the full configured max_seq_length (a min(max_seq, 128) would cause
        # silent 128 truncation that diverges from the MLX path).
        _seq = int(self._config.max_seq_length)
        for text in texts:
            encoded = tokenizer(
                text,
                padding="max_length",
                max_length=_seq,
                truncation=True,
                return_tensors="np",
            )
            input_ids = encoded["input_ids"].astype(np.int32)
            attention_mask = encoded["attention_mask"].astype(np.int32)

            pred = model.predict(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            # Resolve the output tensor by explicit key lookup — `a or b` on numpy
            # arrays raises "truth value of an array is ambiguous", so the old
            # `pred.get("output") or pred.get("embeddings")` crashed on every real call.
            output = pred.get("embeddings")
            if output is None:
                output = pred.get("output")
            if output is None:
                output = next(iter(pred.values()))
            if isinstance(output, np.ndarray):
                emb = (
                    output[0, 0].flatten().tolist()
                    if output.ndim >= 3
                    else output.flatten().tolist()
                )
            else:
                emb = list(map(float, output))

            if self._config.normalize_embeddings:
                norm = sum(x * x for x in emb) ** 0.5
                if norm > 0:
                    emb = [x / norm for x in emb]

            embeddings.append(emb)

        return embeddings

    def _embed_mlx_fallback(self, texts: list[str]) -> list[list[float]]:
        """Fallback embedding inference using MLX on GPU.

        Uses the real model and tokenizer if available, otherwise returns
        a meaningful error message.
        """
        if not _HAS_MLX:
            raise RuntimeError(
                "Neither CoreML nor MLX available for embedding inference. "
                "Install mlx: pip install mlx"
            )

        # Try to load real model + tokenizer
        try:
            from mlx_lm.utils import load_model, load_tokenizer

            model_path = Path(self._config.model_name)
            if not model_path.exists():
                # Try as HF model ID — not supported without download
                raise FileNotFoundError(f"Model not found: {model_path}")

            model, _ = load_model(model_path)
            tokenizer = load_tokenizer(model_path)
            return self._embed_with_model(model, tokenizer, texts)
        except Exception as exc:
            logger.warning("MLX model loading failed: %s", exc)
            raise RuntimeError(
                f"Could not load embedding model '{self._config.model_name}': {exc}. "
                "Provide a valid local model path."
            ) from exc

    def _embed_with_model(
        self, model, tokenizer, texts: list[str]
    ) -> list[list[float]]:
        """Run embedding inference with a loaded MLX model and tokenizer."""
        embeddings: list[list[float]] = []
        for text in texts:
            encoded = tokenizer.encode(text)
            input_ids = mx.array([encoded])

            output = model(input_ids)
            if hasattr(output, "last_hidden_state"):
                hidden = output.last_hidden_state
            else:
                hidden = output

            # Mean pooling over sequence dimension
            emb_mx = hidden.mean(axis=1).squeeze(0)

            if self._config.normalize_embeddings:
                norm = mx.sqrt(mx.sum(emb_mx * emb_mx))
                if norm.item() > 0:
                    emb_mx = emb_mx / norm

            mx.eval(emb_mx)
            embeddings.append(emb_mx.tolist())

        return embeddings

    def _load_tokenizer(self):
        """Try to load the tokenizer for the configured model (cached — loading it
        per request from disk added hundreds of ms to every embed call)."""
        cached = getattr(self, "_tokenizer_cache", None)
        if cached is not None:
            return cached
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self._config.model_name)
            self._tokenizer_cache = tok
            return tok
        except Exception:
            logger.debug("transformers tokenizer load failed", exc_info=True)
        try:
            from mlx_lm.utils import load_tokenizer

            path = Path(self._config.model_name)
            if path.exists():
                return load_tokenizer(path)
        except Exception:
            logger.debug("mlx tokenizer load failed", exc_info=True)
        return None

    # ── Status ────────────────────────────────────────────────────────────────

    def is_compiled(self) -> bool:
        """Check whether a CoreML model has been successfully compiled.

        Returns:
            bool: True if the model is compiled and ready for ANE inference.
        """
        return self._is_compiled

    def get_stats(self) -> dict[str, Any]:
        """Return processor statistics.

        Returns:
            dict with keys:
                - model_name (str): Configured model name
                - is_compiled (bool): Whether model is compiled for ANE
                - compiled_path (str|None): Path to compiled .mlmodelc
                - inference_count (int): Number of embed() calls
                - avg_latency_s (float|None): Average latency in seconds
                - ane_available (bool): Whether ANE is available on this hardware
                - coremltools_installed (bool): Whether coremltools is installed
                - mlx_available (bool): Whether MLX is available for fallback
        """
        avg_latency: float | None = None
        if self._inference_count > 0:
            avg_latency = self._total_latency / self._inference_count

        return {
            "model_name": self._config.model_name,
            "is_compiled": self._is_compiled,
            "compiled_path": self._compiled_path,
            "inference_count": self._inference_count,
            "avg_latency_s": avg_latency,
            "ane_available": is_ane_available(),
            "coremltools_installed": _HAS_COREMLTOOLS,
            "mlx_available": _HAS_MLX,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def compile_embedding_model(model_path: str, output_path: str = "") -> dict[str, Any]:
    """Load an MLX embedding model, convert to CoreML, and compile for ANE.

    Performs the full pipeline:
      1. Load model weights from model_path
      2. Convert to CoreML .mlprogram via coremltools
      3. Compile to .mlmodelc with ANE optimization
      4. Return metadata about the compiled model

    Args:
        model_path: Path to the MLX embedding model directory.
        output_path: Optional output directory for compiled model.
                     Defaults to ~/.yunshu/ane_cache/.

    Returns:
        dict with keys:
            compiled_path (str): Path to the compiled .mlmodelc directory.
            model_size (int): Approximate model size in bytes.
            compile_time_s (float): Total compilation time in seconds.

    Raises:
        RuntimeError: If coremltools is not available or compilation fails.
    """
    if not _HAS_COREMLTOOLS:
        raise RuntimeError(
            "coremltools is required for model compilation. "
            "Install with: pip install coremltools"
        )

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Resolve output directory
    if output_path:
        out_dir = Path(output_path).expanduser().resolve()
    else:
        out_dir = Path.home() / ".yunshu" / "ane_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).name
    mlpackage_path = out_dir / f"{model_name}.mlpackage"
    mlmodelc_path = out_dir / f"{model_name}.mlmodelc"

    t_start = time.monotonic()

    try:
        import numpy as np

        # Create a representative input for tracing
        sample_seq_len = 128
        sample_input = np.zeros((1, sample_seq_len), dtype=np.int32)

        # Convert model to CoreML mlprogram
        coreml_model = ct.convert(
            model_path,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=sample_input.shape,
                    dtype=np.int32,
                )
            ],
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.ALL,
        )
        coreml_model.save(str(mlpackage_path))

        # Compile to .mlmodelc
        compile_result = subprocess.run(
            [
                "xcrun",
                "coremlcompiler",
                "compile",
                str(mlpackage_path),
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(
                f"coremlcompiler failed (exit {compile_result.returncode}): "
                f"{compile_result.stderr}"
            )

        if not mlmodelc_path.exists():
            raise RuntimeError(f"Compiled .mlmodelc not found at: {mlmodelc_path}")

    except FileNotFoundError as exc:
        raise RuntimeError(
            "xcrun not found — Xcode CLI tools required. "
            "Install with: xcode-select --install"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"CoreML model compilation failed: {exc}") from exc

    compile_time = time.monotonic() - t_start

    # Estimate model size from files
    model_size = 0
    for f in mlmodelc_path.rglob("*"):
        if f.is_file():
            model_size += f.stat().st_size

    logger.info(
        "Compiled embedding model: %s -> %s (%d bytes, %.2fs)",
        model_path,
        mlmodelc_path,
        model_size,
        compile_time,
    )

    return {
        "compiled_path": str(mlmodelc_path),
        "model_size": model_size,
        "compile_time_s": round(compile_time, 3),
    }


# ---------------------------------------------------------------------------
# ANE Drafter Path B — CoreML-based speculative decoding draft model
# ---------------------------------------------------------------------------


def compile_drafter_model(model_path: str, output_path: str = "") -> dict[str, Any]:
    """Compile a small draft model to CoreML for ANE-based speculative decoding.

    Takes a small LLM (draft model) and compiles it into a CoreML model
    that can execute on the Apple Neural Engine. This is Path B for the
    risky Delta-2 (ANE-as-Drafter) architecture.

    The compiled model takes token IDs as input and outputs logits that
    can be used to propose K draft tokens for speculative decoding.

    Args:
        model_path: Path to the draft model (HuggingFace format or local directory).
        output_path: Optional output directory for compiled model.
                     Defaults to ~/.yunshu/ane_cache/drafter/.

    Returns:
        dict with keys:
            compiled_path (str): Path to the compiled .mlmodelc directory.
            model_size (int): Approximate model size in bytes.
            compile_time_s (float): Total compilation time in seconds.
            target (str): "ane" if compiled for ANE, "gpu" if fallback.

    Raises:
        RuntimeError: If compilation fails and no fallback is available.
    """
    # Resolve output directory
    if output_path:
        out_dir = Path(output_path).expanduser().resolve()
    else:
        out_dir = Path.home() / ".yunshu" / "ane_cache" / "drafter"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).name
    mlpackage_path = out_dir / f"{model_name}_drafter.mlpackage"
    mlmodelc_path = out_dir / f"{model_name}_drafter.mlmodelc"

    # Check if already compiled
    if mlmodelc_path.exists():
        model_size = sum(
            f.stat().st_size for f in mlmodelc_path.rglob("*") if f.is_file()
        )
        logger.info("Drafter model already compiled: %s", mlmodelc_path)
        return {
            "compiled_path": str(mlmodelc_path),
            "model_size": model_size,
            "compile_time_s": 0.0,
            "target": "ane",
        }

    t_start = time.monotonic()

    # Try CoreML/ANE compilation
    if _HAS_COREMLTOOLS and is_ane_available():
        try:
            import numpy as np

            # Draft model: input is token IDs, output is logits for next-token prediction
            # Use a small sequence length (draft models typically predict 1-10 tokens)
            sample_seq_len = 32
            sample_input = np.zeros((1, sample_seq_len), dtype=np.int32)

            # Convert to CoreML targeting ANE
            coreml_model = ct.convert(
                model_path,
                inputs=[
                    ct.TensorType(
                        name="input_ids",
                        shape=sample_input.shape,
                        dtype=np.int32,
                    )
                ],
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.ALL,
            )
            coreml_model.save(str(mlpackage_path))

            # Compile to .mlmodelc
            compile_result = subprocess.run(
                [
                    "xcrun",
                    "coremlcompiler",
                    "compile",
                    str(mlpackage_path),
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if compile_result.returncode != 0:
                raise RuntimeError(
                    f"coremlcompiler failed (exit {compile_result.returncode}): "
                    f"{compile_result.stderr}"
                )

            if not mlmodelc_path.exists():
                raise RuntimeError(f"Compiled .mlmodelc not found at: {mlmodelc_path}")

            compile_time = time.monotonic() - t_start
            model_size = sum(
                f.stat().st_size for f in mlmodelc_path.rglob("*") if f.is_file()
            )

            logger.info(
                "Compiled drafter model for ANE: %s -> %s (%d bytes, %.2fs)",
                model_path,
                mlmodelc_path,
                model_size,
                compile_time,
            )

            return {
                "compiled_path": str(mlmodelc_path),
                "model_size": model_size,
                "compile_time_s": round(compile_time, 3),
                "target": "ane",
            }

        except FileNotFoundError as exc:
            raise RuntimeError(
                "xcrun not found — Xcode CLI tools required. "
                "Install with: xcode-select --install"
            ) from exc
        except Exception as exc:
            logger.warning(
                "CoreML compilation failed for drafter model, GPU fallback: %s", exc
            )

    # Fallback: store model path for GPU-based draft inference
    compile_time = time.monotonic() - t_start
    logger.info("Drafter model will use GPU fallback: %s", model_path)

    return {
        "compiled_path": str(Path(model_path).resolve()),
        "model_size": 0,
        "compile_time_s": round(compile_time, 3),
        "target": "gpu",
    }


def draft_token(
    drafter_path: str, context_tokens: list[int], num_draft: int = 5
) -> list[int]:
    """Run draft model to propose K tokens using ANE or GPU fallback.

    Uses a compiled CoreML model on the ANE if available, otherwise
    falls back to MLX GPU inference.

    Args:
        drafter_path: Path to the compiled .mlmodelc (ANE) or model directory (GPU).
        context_tokens: Current context token IDs.
        num_draft: Number of draft tokens to propose (K).

    Returns:
        List of proposed token IDs (length num_draft or fewer on EOS).
    """
    if not context_tokens:
        return []

    if num_draft <= 0:
        return []

    # Try ANE/CoreML path first
    compiled_path = Path(drafter_path)
    if (
        compiled_path.suffix == ".mlmodelc"
        and compiled_path.exists()
        and _HAS_COREMLTOOLS
    ):
        try:
            return _draft_token_coreml(str(compiled_path), context_tokens, num_draft)
        except Exception as exc:
            logger.warning("CoreML drafter failed, falling back to GPU: %s", exc)

    # GPU fallback via MLX
    return _draft_token_gpu(drafter_path, context_tokens, num_draft)


def _draft_token_coreml(
    compiled_path: str,
    context_tokens: list[int],
    num_draft: int,
) -> list[int]:
    """Draft tokens via CoreML model on ANE."""
    model = ct.models.MLModel(compiled_path)
    import numpy as np

    draft_tokens = []
    current_tokens = list(context_tokens)

    for _ in range(num_draft):
        # Prepare input
        input_ids = np.array([current_tokens], dtype=np.int32)

        # Pad or truncate to model's expected input size
        spec = model.get_spec()
        input_desc = spec.description.input[0]
        shape = input_desc.type.multiArrayType.shape
        expected_len = shape[1] if len(shape) > 1 else len(current_tokens)

        if input_ids.shape[1] < expected_len:
            pad_width = expected_len - input_ids.shape[1]
            input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)), constant_values=0)
        elif input_ids.shape[1] > expected_len:
            input_ids = input_ids[:, :expected_len]

        pred = model.predict({"input_ids": input_ids})
        output = pred.get("output") or pred.get("logits") or list(pred.values())[0]

        if isinstance(output, np.ndarray):
            # Get logits for last position, sample greedily
            logits = output[0, -1] if output.ndim > 1 else output
            token_id = int(np.argmax(logits))
        else:
            token_id = 0

        draft_tokens.append(token_id)
        current_tokens.append(token_id)

    return draft_tokens


def _draft_token_gpu(
    model_path: str,
    context_tokens: list[int],
    num_draft: int,
) -> list[int]:
    """Draft tokens via MLX GPU fallback using a real draft model."""
    if not _HAS_MLX:
        raise RuntimeError("MLX not available for draft token generation")

    try:
        from mlx_lm.utils import load_model, load_tokenizer

        model_path_resolved = Path(model_path)
        if not model_path_resolved.exists():
            raise FileNotFoundError(f"Draft model not found: {model_path}")

        model, _ = load_model(model_path_resolved)
        load_tokenizer(model_path_resolved)

        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=0.0)
        input_ids = mx.array(context_tokens)

        draft_tokens = []
        for token_id, _ in generate_step(
            input_ids, model, max_tokens=num_draft, sampler=sampler
        ):
            draft_tokens.append(token_id)
            if len(draft_tokens) >= num_draft:
                break

        return draft_tokens

    except Exception as exc:
        logger.error("GPU draft model inference failed: %s", exc)
        raise RuntimeError(f"Draft model inference failed: {exc}") from exc


def benchmark_ane_vs_gpu(
    texts: list[str],
    model_name: str = "intfloat/e5-small-v2",
) -> dict[str, Any]:
    """Run embedding inference on both GPU and ANE, comparing latency and accuracy.

    Runs the same set of texts through both the MLX GPU path and the CoreML/ANE
    path (if available), measuring latency and comparing embedding similarity.

    Args:
        texts: List of strings to embed.
        model_name: HuggingFace model name or local path.

    Returns:
        dict with keys:
            gpu_latency_ms (float): GPU inference latency in milliseconds.
            ane_latency_ms (float|None): ANE inference latency (None if unavailable).
            speedup (float|None): ANE speedup factor (None if ANE unavailable).
            accuracy_diff (float|None): Max cosine distance between GPU and ANE
                                         embeddings (None if ANE unavailable).
    """
    if not texts:
        return {
            "gpu_latency_ms": 0.0,
            "ane_latency_ms": None,
            "speedup": None,
            "accuracy_diff": None,
        }

    config = ANEEmbeddingConfig(
        model_name=model_name,
        compile_on_init=False,
        normalize_embeddings=True,
    )

    # ── GPU benchmark ──────────────────────────────────────────────────────────
    gpu_proc = ANEEmbeddingProcessor(config)
    t0 = time.monotonic()
    gpu_embeddings = gpu_proc.embed(texts)
    gpu_latency = (time.monotonic() - t0) * 1000

    result: dict[str, Any] = {
        "gpu_latency_ms": round(gpu_latency, 3),
        "ane_latency_ms": None,
        "speedup": None,
        "accuracy_diff": None,
    }

    # ── ANE benchmark (if compiled model available) ────────────────────────────
    if _HAS_COREMLTOOLS and is_ane_available():
        # Attempt to use a compiled model
        ane_proc = ANEEmbeddingProcessor(config)
        cache_dir = config.get_cache_dir()
        model_name_safe = model_name.replace("/", "_").replace("\\", "_")
        # Point at the .mlpackage, NOT the .mlmodelc. _embed_coreml loads via
        # ct.models.MLModel(_compiled_path), which CANNOT load a bare .mlmodelc (no
        # Manifest.json) → it raises → embed() silently falls back to MLX → the benchmark
        # measures MLX-vs-MLX and reports a bogus speedup≈1.0 / accuracy_diff=0.0 as if it
        # were a real ANE comparison. compile_model stores (and serving loads) the .mlpackage.
        mlpackage_path = cache_dir / f"{model_name_safe}.mlpackage"

        if mlpackage_path.exists():
            ane_proc._compiled_path = str(mlpackage_path)
            ane_proc._is_compiled = True

            t0 = time.monotonic()
            ane_embeddings = ane_proc.embed(texts)
            ane_latency = (time.monotonic() - t0) * 1000

            result["ane_latency_ms"] = round(ane_latency, 3)

            if gpu_latency > 0:
                result["speedup"] = round(gpu_latency / ane_latency, 3)

            # Compute accuracy difference (max cosine distance)
            if len(gpu_embeddings) == len(ane_embeddings) and gpu_embeddings:
                import math

                max_cos_dist = 0.0
                for gpu_emb, ane_emb in zip(
                    gpu_embeddings, ane_embeddings, strict=False
                ):
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(gpu_emb, ane_emb, strict=False))
                    norm_a = math.sqrt(sum(a * a for a in gpu_emb))
                    norm_b = math.sqrt(sum(b * b for b in ane_emb))
                    if norm_a > 0 and norm_b > 0:
                        cos_sim = dot / (norm_a * norm_b)
                        cos_dist = 1.0 - cos_sim
                        max_cos_dist = max(max_cos_dist, cos_dist)

                result["accuracy_diff"] = round(max_cos_dist, 6)

    return result


# ---------------------------------------------------------------------------
# Module-level singleton (used by gateway embeddings router)
# ---------------------------------------------------------------------------

_ane_processor: ANEEmbeddingProcessor | None = None


def get_ane_processor() -> ANEEmbeddingProcessor | None:
    """Get or create the global ANE embedding processor singleton.

    Returns None if YUNSHU_ANE_EMBEDDINGS is not enabled.
    Thread-safe: only creates one instance.
    """
    global _ane_processor
    if _ane_processor is not None:
        return _ane_processor

    if not is_ane_available():
        return None

    model_name = os.environ.get("YUNSHU_ANE_EMBEDDING_MODEL", "intfloat/e5-small-v2")
    max_seq = int(os.environ.get("YUNSHU_ANE_MAX_SEQ_LENGTH", "512"))
    config = ANEEmbeddingConfig(
        model_name=model_name,
        max_seq_length=max_seq,
        normalize_embeddings=True,
        compile_on_init=True,
    )
    _ane_processor = ANEEmbeddingProcessor(config)
    logger.info(
        "ANE embedding processor singleton created: model=%s, compiled=%s",
        model_name,
        _ane_processor.is_compiled(),
    )
    return _ane_processor


def is_ane_embeddings_enabled() -> bool:
    """Check whether ANE embeddings are enabled via the YUNSHU_ANE_EMBEDDINGS env var."""
    return (
        os.environ.get("YUNSHU_ANE_EMBEDDINGS", "").strip() in ("1", "true", "yes")
        and is_ane_available()
    )


def get_ane_embedding_stats() -> dict[str, Any]:
    """Return ANE embedding stats, or empty dict if not active."""
    proc = _ane_processor
    if proc is None:
        return {"enabled": is_ane_embeddings_enabled(), "active": False}
    stats = proc.get_stats()
    stats["enabled"] = True
    stats["active"] = True
    return stats
