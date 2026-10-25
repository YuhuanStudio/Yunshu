"""Serving hook of the Splash-style ANE FFN split (YUNSHU_ANE_FFN_SPLIT=splash).

The split itself lives in ``yunshu_engine.ane`` (derived from Splash 1.4.0, Apache-2.0). This
module wires it into a Qwen3.5-family prefill: it patches the dense MLP's ``__call__`` (a class
patch, registered in vendor.json as an mlx-vlm patch), keeps the per-layer order of one prompt
step, validates the step after its last layer and, when the Neural Engine's work failed, restores
the caches and runs the step again on the GPU alone. Only single-row prompt steps of at least
the calibrated rows take the split; decode, speculative verify and short tails run as before.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RUNTIME: Runtime | None = None
_PATCHED = False
_ORIGINAL: Callable[..., Any] | None = None
_LOCK = threading.Lock()


class AneChunkFailed(RuntimeError):  # noqa: N818
    """The Neural Engine's part of a prompt step failed; the caller restores and reruns on the GPU."""


class Runtime:
    """One engine's split: the AneFfn, the per-step state and counters."""

    def __init__(self, split, reason: str):
        self.split = split
        self.reason = reason
        self.layers = len(split.layers)
        self.engaged = False  # inside a guarded prompt step
        self._chunk_ok = False
        self._expect = 0
        self.chunks_split = 0
        self.chunks_gpu = 0
        self.failures = 0

    # -- per layer ---------------------------------------------------------------------
    def run(self, module, x):
        """The split FFN of `module` for `x` ([1, rows, hidden] bf16) or None for the GPU path."""
        index = getattr(module, "_ane_index", None)
        if index is None or x.ndim != 3 or x.shape[0] != 1:
            return None
        rows = int(x.shape[1])
        if index == 0:
            self._expect = 0
            self._chunk_ok = (
                bool(self.split.splits(rows)) and x.dtype.name == "bfloat16"
            )
            if self._chunk_ok and not self.split.begin():
                self._chunk_ok = False
            if not self._chunk_ok:
                self.chunks_gpu += 1
        if not self._chunk_ok or index != self._expect:
            return None
        from yunshu_engine.ane.ffn_split import AneRetired

        try:
            out = self.split.add(index, x.reshape(rows, -1), None, rows)
        except AneRetired as exc:
            self.failures += 1
            raise AneChunkFailed(str(exc)) from exc
        self._expect += 1
        if index == self.layers - 1:
            self._chunk_ok = False
            if not self.split.finish():
                self.failures += 1
                raise AneChunkFailed(self.split.reason)
            self.chunks_split += 1
        return out.reshape(1, rows, -1)

    def abandon(self) -> None:
        """A step that ended (or raised) before its last layer: settle the split's tickets."""
        if self._chunk_ok and self._expect:
            with contextlib.suppress(Exception):
                self.split.finish()
        self._chunk_ok = False
        self._expect = 0

    @property
    def retired(self) -> bool:
        return bool(self.split.retired())

    def summary(self) -> dict[str, Any]:
        return {
            "share": round(self.split.share, 3),
            "minimum_rows": self.split.minimum_rows,
            "chunks_split": self.chunks_split,
            "chunks_gpu": self.chunks_gpu,
            "failures": self.failures,
            "retired": self.retired,
            "ane_ms": round(self.split.served_ms, 1),
        }


def current() -> Runtime | None:
    return _RUNTIME


def _patched_call(self, x):
    rt = _RUNTIME
    if rt is not None and rt.engaged:
        out = rt.run(self, x)
        if out is not None:
            return out
    assert _ORIGINAL is not None
    return _ORIGINAL(self, x)


def _patch_mlp_class(cls) -> None:
    global _PATCHED, _ORIGINAL
    with _LOCK:
        if _PATCHED:
            return
        _ORIGINAL = cls.__call__
        cls.__call__ = _patched_call
        _PATCHED = True


def install(
    language_model, cache_dir: Path, model_id: str, *, share=None, minimum_rows=None
):
    """Calibrate (or recall) and start the split on `language_model`'s dense MLPs.

    Returns the Runtime, or None with the reason logged when the GPU runs the FFN alone.
    """
    global _RUNTIME
    from yunshu_engine.ane import ffn_startup
    from yunshu_engine.ane.ffn_split import layer_from_module

    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
    except ImportError:
        logger.info(
            "ANE FFN split: this mlx-vlm has no Qwen3.5 MLP; the GPU runs the FFN alone"
        )
        return None
    layers = getattr(getattr(language_model, "model", None), "layers", None)
    if not layers or not all(
        isinstance(getattr(layer, "mlp", None), Qwen3_5MLP) for layer in layers
    ):
        logger.info(
            "ANE FFN split: not a dense Qwen3.5-family model; the GPU runs the FFN alone"
        )
        return None
    mlps = [layer.mlp for layer in layers]
    try:
        projections = [layer_from_module(m) for m in mlps]
    except (AttributeError, TypeError, ValueError) as exc:
        logger.info(
            "ANE FFN split: unsupported projections (%s); the GPU runs the FFN alone",
            exc,
        )
        return None
    outcome = ffn_startup.start(
        projections,
        cache_dir,
        model_id,
        given_share=share,
        given_minimum_rows=minimum_rows,
    )
    if outcome.kind != "split" or outcome.split is None:
        logger.info("The GPU runs the prefill FFN alone: %s", outcome.reason)
        return None
    for index, mlp in enumerate(mlps):
        mlp._ane_index = index
    _patch_mlp_class(Qwen3_5MLP)
    _RUNTIME = Runtime(outcome.split, outcome.reason)
    logger.info("Neural Engine FFN split %s", outcome.reason)
    return _RUNTIME


def uninstall() -> None:
    global _RUNTIME
    rt, _RUNTIME = _RUNTIME, None
    if rt is not None:
        with contextlib.suppress(Exception):
            rt.split.close()


# ---- prompt-step guard ----------------------------------------------------------------
def _snapshot(prompt_cache) -> list[tuple[Any, dict[str, Any]]]:
    saved = []
    for entry in prompt_cache or ():
        state: dict[str, Any] = {}
        for name in ("offset", "_idx"):
            if hasattr(entry, name):
                state[name] = getattr(entry, name)
        cache = getattr(entry, "cache", None)
        if isinstance(cache, list):
            state["cache"] = list(cache)
        saved.append((entry, state))
    return saved


def _restore(saved) -> None:
    for entry, state in saved:
        for name, value in state.items():
            setattr(entry, name, list(value) if name == "cache" else value)


def guarded(prompt_cache, step: Callable[[], Any]):
    """Run one prompt step with the split engaged; on an ANE failure restore the caches and
    run the step again on the GPU alone."""
    rt = _RUNTIME
    if rt is None or rt.retired:
        return step()
    saved = _snapshot(prompt_cache)
    rt.engaged = True
    try:
        return step()
    except AneChunkFailed as exc:
        logger.warning(
            "ANE FFN split failed in a prompt step (%s); rerunning it on the GPU", exc
        )
        rt.engaged = False
        _restore(saved)
        return step()
    finally:
        rt.engaged = False
        rt.abandon()


def kernel_suffix() -> str:
    """Part of the APC kernel id: states computed with the split differ from the GPU's alone."""
    return "+ane-ffn-splash-v1" if _RUNTIME is not None else ""
