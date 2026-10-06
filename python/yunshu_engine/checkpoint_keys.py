"""Align a checkpoint's tensor names with the model class after a lenient (strict=False) load.

A lenient load silently drops every tensor whose name the model class does not know. For Gemma-4
that is right (vestigial k/v of KV-shared layers). For embedding checkpoints exported from the
bare backbone (Qwen3-Embedding: ``embed_tokens.weight`` / ``layers.N...`` / ``norm.weight``, no
``model.`` prefix) it dropped ALL 310 tensors and served a randomly initialised network. The
helpers here are pure (name sets only) so the policy is unit-tested on CPU.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable


def key_rename(ckpt_keys: Iterable[str], model_keys: Iterable[str]) -> dict[str, str]:
    """Map checkpoint names -> model names so every model parameter is covered.

    Returns {} when the names already line up (extras alone are fine). Tries the identity, then a
    ``model.`` prefix on the names the model does not know. Raises ValueError when some model
    parameter has no tensor even after that: the load must fail rather than serve random weights.
    """
    ck, mk = set(ckpt_keys), set(model_keys)
    if mk <= ck:
        return {}
    ren = {k: "model." + k for k in ck if k not in mk and ("model." + k) in mk}
    covered = (ck & mk) | set(ren.values())
    missing = sorted(mk - covered)
    if missing:
        raise ValueError(
            f"checkpoint does not cover {len(missing)} model parameters "
            f"(e.g. {missing[:3]}); refusing a lenient load that would leave them random"
        )
    return ren


def extras_only(ckpt_keys: Iterable[str], model_keys: Iterable[str]) -> None:
    """Raise ValueError when some model parameter has no checkpoint tensor; extra tensors are fine."""
    missing = sorted(set(model_keys) - set(ckpt_keys))
    if missing:
        raise ValueError(
            f"checkpoint does not cover {len(missing)} model parameters "
            f"(e.g. {missing[:3]}); refusing a lenient load that would leave them random"
        )


@contextlib.contextmanager
def lenient_extras_load():
    """Scoped: a strict ``load_weights`` that fails only because the checkpoint carries tensors
    the model does not use (an embedded drafter's mtp.*, vestigial weights) retries leniently.
    A checkpoint that leaves any model parameter uncovered still fails: a lenient retry there
    would serve randomly initialised weights."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    orig = nn.Module.load_weights

    def _lw(self, file_or_weights, strict=True):
        try:
            return orig(self, file_or_weights, strict=strict)
        except ValueError:
            if not strict or not isinstance(file_or_weights, list):
                raise
            extras_only(
                (k for k, _ in file_or_weights),
                (k for k, _ in tree_flatten(self.parameters())),
            )
            return orig(self, file_or_weights, strict=False)

    nn.Module.load_weights = _lw
    try:
        yield
    finally:
        nn.Module.load_weights = orig
