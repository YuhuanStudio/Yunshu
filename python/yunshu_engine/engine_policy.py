from __future__ import annotations

"""Engine policy extracted from batched_engine.

Runtime dependencies stay on the compatibility facade so existing patches apply.
"""

from typing import Any


def _is_cancelled(event: Any) -> bool:
    """Thread-safe cancel check. Works from the MLX executor thread.

    asyncio.Event.is_set() reads ._value (GIL-protected bool), which is
    safe from any thread in CPython.  Using the explicit attribute avoids
    the thread-safety warning from calling asyncio APIs off the event loop.
    """
    if event is None:
        return False
    if isinstance(event, _engine.asyncio.Event):
        return event._value
    return event.is_set()


def _read_config_eos_ids(model_path: str) -> frozenset:
    """Read eos_token_id from a model's generation_config.json / config.json.

    Models like Gemma-4 declare MULTIPLE eos ids there (e.g. [1, 106, 50] — the
    <eos>, the turn-end <turn|>, and a channel token) but the tokenizer often
    exposes only the single primary eos (1). The model then ends its turn with
    106, generation doesn't stop, and it rambles (repeated answers + channel
    reasoning). Merging the config eos ids into the engine stop set fixes that.
    generation_config wins over config; empty for HF ids / unreadable configs."""
    if model_path in _engine._CONFIG_EOS_CACHE:
        return _engine._CONFIG_EOS_CACHE[model_path]
    import json as _json
    from pathlib import Path as _P

    ids: set[int] = set()
    try:
        d = _P(model_path)
        if d.is_dir():
            for fn in ("generation_config.json", "config.json"):
                fp = d / fn
                if not fp.exists():
                    continue
                e = _json.loads(fp.read_text()).get("eos_token_id")
                if isinstance(e, bool):
                    continue
                if isinstance(e, int):
                    ids.add(e)
                elif isinstance(e, (list, tuple)):
                    ids.update(int(x) for x in e if isinstance(x, int))
                if ids:
                    break  # generation_config.json wins
    except Exception:
        _engine.logger.debug("config eos_token_id read failed", exc_info=True)
    result = frozenset(ids)
    _engine._CONFIG_EOS_CACHE[model_path] = result
    return result


def _parse_quant_config_env(s: str) -> dict | None:
    """Parse YUNSHU_QUANT_CONFIG into an mlx-lm quantization dict.

    Accepts JSON ({"group_size":64,"bits":4}), a bare int (bits, group_size=64), or a
    compact "bits" / "bits,group_size" / "bits:group_size" string. Returns None if it
    can't be parsed (caller ignores it with a warning rather than crashing the load).
    This is passed to mlx_lm.utils.load via `model_config={"quantization": <dict>}` — NOT
    as a bogus `quantization=` kwarg (which load() doesn't accept → TypeError → dead load).
    """
    if not s:
        return None
    s = s.strip()
    try:
        import json as _json

        parsed = _json.loads(s)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, bool):
            return None
        if isinstance(parsed, int):
            return {"bits": parsed, "group_size": 64}
    except Exception:
        pass
    parts = [p for p in s.replace(":", ",").split(",") if p.strip()]
    try:
        if len(parts) == 1:
            return {"bits": int(parts[0]), "group_size": 64}
        if len(parts) >= 2:
            return {"bits": int(parts[0]), "group_size": int(parts[1])}
    except ValueError:
        pass
    return None


def _prefill_step_size() -> int:
    """Chunk size for prompt prefill (tokens processed per forward pass).

    mlx-lm's generate_step chunks the prefill at this size so the activation peak
    stays bounded regardless of prompt length (a 200k-token prefill peaks ~12GB on
    a 36GB M3 Max). Default 2048 (mlx-lm's default). Lower it to reduce the prefill
    peak on very memory-constrained hardware; raise it to speed up long prefills
    when memory is ample. Tuned via YUNSHU_PREFILL_STEP_SIZE.
    """
    return _engine.settings.get("YUNSHU_PREFILL_STEP_SIZE")


def _resolve_model_max_ctx(model) -> int:
    """Resolve a model's context window. mlx-lm Model objects store
    config in `.args` (Qwen/Llama use max_position_embeddings), NOT `.config` /
    a bare max_seq_len — so the old single-attr lookups were dead for most
    models. Check the model, its .config, and its .args for the usual keys.
    Returns 0 when undeterminable (caller treats that as 'no clamp')."""
    for src in (model, getattr(model, "config", None), getattr(model, "args", None)):
        if src is None:
            continue
        for attr in ("max_position_embeddings", "max_seq_len", "n_positions"):
            v = getattr(src, attr, None)
            if isinstance(v, int) and v > 0:
                return v
    return 0


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
