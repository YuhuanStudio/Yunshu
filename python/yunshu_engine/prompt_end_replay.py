"""Zero-forward exact repeat: replay the final prompt block of a request that was served before.

The canonical prefill leaves the last 1-64 prompt rows (the tail after the last 64-aligned
boundary) for ``PromptProcessingBatch.generate``. An APC hit restores the aligned checkpoint
and still runs that tail forward (about 65 ms fixed + 1.7 ms per row at 32K). When the same
prompt is sent again, the result of that forward is already known, so it is stored once and
written back instead of recomputed: the tail's K/V rows, the recurrent state after the prompt,
the last-row logits and the drafter's tail features. The stored arrays are the outputs of the
same forward a hit would run, so everything downstream (sampling, decode, harvest) sees
identical values. Anything outside the plain text, one-row, no-logits-processor case runs the
forward as before.
"""

from __future__ import annotations

import collections
import logging
import types
from typing import Any

from . import dflash_apc_window

logger = logging.getLogger(__name__)
MAX_ENTRIES = 3
MAX_BYTES = 768 << 20
_STORE: collections.OrderedDict = collections.OrderedDict()
_STATS = {"stored": 0, "replayed": 0, "evicted": 0, "skipped": 0}
_NOT_TEXT = (
    "pixel_values",
    "pixel_values_videos",
    "input_features",
    "image_grid_thw",
    "video_grid_thw",
    "inputs_embeds",
)
# position_ids / attention_mask of a text prompt are functions of its token ids, which the replay key hashes
# (with the media hash), so they do not keep a block from replaying.


def stats() -> dict[str, int]:
    return {**_STATS, "entries": len(_STORE), "bytes": resident_bytes()}


def resident_bytes() -> int:
    return sum(e.nbytes for e in _STORE.values())


def clear() -> None:
    _STORE.clear()
    for k in _STATS:
        _STATS[k] = 0


class _Entry:
    """slots: per layer ``("a", ArraysCache.__dict__ copy)`` or ``("k", (keys, values))``."""

    def __init__(self, base, rows, slots, logits, hidden, nbytes):
        self.base, self.rows, self.slots = base, rows, slots
        self.logits, self.hidden, self.nbytes = logits, hidden, nbytes


class _OneShot:
    """Stands in for the model for exactly one call and hands back the stored output."""

    def __init__(self, batch, real, output):
        self._batch, self._real, self._output = batch, real, output

    def __call__(self, *args, **kwargs):
        self._batch.model = self._real
        return self._output

    def __getattr__(self, name):
        return getattr(self._real, name)


def _classes():
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache

    return ArraysCache, BatchKVCache


_SKIPPED: set[str] = set()


def _skip(reason: str) -> None:
    """Log each skip reason once per process, so a run shows why replay did not engage."""
    if reason not in _SKIPPED:
        _SKIPPED.add(reason)
        logger.info("prompt-end replay: skipped (%s)", reason)
    return None


def skip_reason(reason: str) -> None:
    _skip(reason)


def _plan(batch) -> tuple[Any, int, int] | None:
    """``(key, base, rows)`` when this batch's final block may be replayed or recorded."""
    meta = getattr(batch, "_apc_meta", None) or []
    if len(meta) != 1 or not meta[0] or len(getattr(batch, "uids", ())) != 1:
        return _skip("not one row/uid")
    # logits processors (penalties, grammar guides) run on the returned logits after the forward, in
    # PromptProcessingBatch.generate; replay only substitutes the forward's output, so they see the same input.
    if any(getattr(batch, "_right_pad_per_row", None) or []):
        return _skip("right padding")
    draft = getattr(batch, "draft_model", None)
    kind = getattr(batch, "draft_kind", None) if draft is not None else None
    if kind not in (None, "dflash"):
        return _skip("draft kind")
    kwargs = getattr(batch, "_prompt_kwargs", None) or {}
    media = [k for k in _NOT_TEXT if kwargs.get(k) is not None]
    if media:
        return _skip(f"media kwargs {media}")
    embeds = getattr(batch, "_inputs_embeds", None)
    ids = meta[0].get("full_input_ids")
    cache = getattr(batch, "prompt_cache", None) or []
    arrays_cls, kv_cls = _classes()
    if embeds is None or embeds.shape[0] != 1 or ids is None or not cache:
        return _skip("no embeds/ids/cache")
    bases = set()
    for c in cache:
        if type(c) is kv_cls:
            if c.keys is None or c.keys.shape[0] != 1 or c._right_padding is not None:
                return _skip("kv shape/padding")
            bases.add(int(c._idx))
        elif type(c) is not arrays_cls:
            return _skip("unknown cache type")
    if len(bases) != 1:
        return _skip("mixed kv bases")
    key = (
        id(batch.model),
        kind,
        dflash_apc_window.key_for(meta[0].get("extra_hash", 0), ids),
    )
    return key, bases.pop(), int(embeds.shape[1])


class Session:
    """One ``generate`` call: replays a stored block, or records this one for the next."""

    def __init__(self, batch, key, base, rows):
        self.batch, self.key, self.base, self.rows = batch, key, base, rows
        self.replayed = False
        self._layout = [type(c) for c in batch.prompt_cache]
        self._recorded: dict[str, Any] = {}
        self._patched: tuple[Any, Any] | None = None
        self._stub: _OneShot | None = None

    def _fits(self, entry: _Entry) -> bool:
        arrays_cls, _ = _classes()
        lm = getattr(self.batch.model, "language_model", self.batch.model)
        # a missing rope state makes the real forward set model state; keep that path
        if getattr(lm, "_rope_deltas", None) is None:
            return False
        cache = self.batch.prompt_cache
        if (entry.base, entry.rows, len(entry.slots)) != (
            self.base,
            self.rows,
            len(cache),
        ):
            return False
        for c, (tag, payload) in zip(cache, entry.slots, strict=True):
            if type(c) is arrays_cls:
                if tag != "a" or len(payload["cache"]) != len(c.cache):
                    return False
            elif tag != "k" or c.keys.shape[1] != payload[0].shape[1]:
                return False
        return True

    def begin(self) -> None:
        entry = _STORE.get(self.key)
        try:
            fits = entry is not None and self._fits(entry)
        except Exception:  # a pure check: any surprise means run the real forward
            fits = False
        if entry is None:
            _skip("nothing stored for this prompt yet (recording)")
        elif not fits:
            _skip("stored entry does not fit this batch")
        if entry is not None and fits:
            # writes the cache: a failure here must not be swallowed
            self._replay(entry)
            return
        sp = getattr(self.batch, "_speculative_prefill", None)
        if sp is None:
            return
        original = sp.finish

        def record(output):
            hidden = getattr(output, "hidden_states", None)
            self._recorded = {
                "logits": output.logits,
                "hidden": list(hidden) if hidden is not None else None,
            }
            return original(output)

        self._patched = (sp, original)
        sp.finish = record

    def _replay(self, entry: _Entry) -> None:
        arrays_cls, _ = _classes()
        batch = self.batch
        for c, (_, payload) in zip(batch.prompt_cache, entry.slots, strict=True):
            if type(c) is arrays_cls:
                c.__dict__.update(payload)
                c.cache = list(payload["cache"])
            else:
                c.update_and_fetch(*payload)
        output = types.SimpleNamespace(
            logits=entry.logits,
            hidden_states=list(entry.hidden) if entry.hidden is not None else None,
            shared_kv_states=None,
        )
        self._stub = _OneShot(batch, batch.model, output)
        batch.model = self._stub
        _STORE.move_to_end(self.key)
        _STATS["replayed"] += 1
        self.replayed = True
        logger.info("prompt-end replay: %d tail rows after %d", self.rows, self.base)

    def end(self) -> None:
        """Undo any patch; runs when ``generate`` returns or raises."""
        if self._stub is not None and self.batch.model is self._stub:
            self.batch.model = self._stub._real
        if self._patched is not None:
            self._patched[0].__dict__.pop("finish", None)
            self._patched = None

    def store(self, gen_batch) -> None:
        """Keep this block for the next identical request (only after a real forward)."""
        if self.replayed or "logits" not in self._recorded:
            return
        import mlx.core as mx

        arrays_cls, kv_cls = _classes()
        cache = list(getattr(gen_batch, "prompt_cache", None) or [])
        slots: list[tuple[str, Any]] = []
        copies: list[Any] = []
        nbytes = 0
        for c in cache:
            if type(c) is arrays_cls:
                meta = dict(c.__dict__)
                slots.append(("a", meta))
                nbytes += sum(int(a.nbytes) for a in c.cache if a is not None)
            elif type(c) is kv_cls and int(c._idx) == self.base + self.rows:
                # copies: a view would pin the whole context buffer and force every
                # decode step's in-place K/V write to copy it
                rows = (
                    mx.contiguous(mx.array(c.keys[..., self.base : c._idx, :])),
                    mx.contiguous(mx.array(c.values[..., self.base : c._idx, :])),
                )
                slots.append(("k", rows))
                copies.extend(rows)
                nbytes += int(rows[0].nbytes) + int(rows[1].nbytes)
            else:
                _STATS["skipped"] += 1
                return
        logits, hidden = self._recorded["logits"], self._recorded["hidden"]
        nbytes += int(logits.nbytes) + sum(int(h.nbytes) for h in hidden or [])
        if [type(c) for c in cache] != self._layout or nbytes > MAX_BYTES:
            _STATS["skipped"] += 1
            return
        mx.eval(copies)
        _STORE[self.key] = _Entry(self.base, self.rows, slots, logits, hidden, nbytes)
        _STORE.move_to_end(self.key)
        _STATS["stored"] += 1
        while len(_STORE) > MAX_ENTRIES or (
            resident_bytes() > MAX_BYTES and len(_STORE) > 1
        ):
            _STORE.popitem(last=False)
            _STATS["evicted"] += 1


def open_session(batch) -> Session | None:
    try:
        plan = _plan(batch)
    except Exception as exc:
        return _skip(f"plan error: {exc!r}")
    if plan is None:
        return None
    session = Session(batch, *plan)
    session.begin()
    return session
