"""Scratch probe appended to a throw-away copy of yunshu_engine/apc_manager.py (never committed
into the engine): logs, at the end of every prefill (flush start) and after it, the MLX gauges
and a per-entry byte breakdown of the APC, so the memory at the peak instant can be attributed.

Appended by ``apc_probe_install.py``; writes JSON lines to the path in APC_PROBE_LOG.
"""

import json as _json
import os as _os
import time as _time


def _probe_rows(manager):
    from mlx_vlm.models.cache import ArraysCache, KVCache, cache_nbytes

    share = getattr(manager, "_kv_share", {})
    anchors = getattr(manager, "_anchors", {})
    rows = []
    with manager.lock:
        for key, entry in manager._exact_cache.items():
            state = kv = 0
            for c in entry.prompt_cache:
                if type(c) is KVCache:
                    if c.keys is not None:
                        kv += c.keys.nbytes + c.values.nbytes
                elif type(c) is ArraysCache:
                    state += cache_nbytes(c)
                else:
                    state += cache_nbytes(c)
            viewed = share.get(key, (0, 0))[1]
            rows.append(
                dict(
                    n=len(entry.token_ids),
                    state_mib=round(state / 2**20, 1),
                    kv_mib=round(kv / 2**20, 1),
                    kv_viewed_mib=round(viewed / 2**20, 1),
                    anchor=key in anchors,
                )
            )
    return rows


def _probe_log(manager, event):
    import mlx.core as mx

    path = _os.environ.get("APC_PROBE_LOG")
    if not path:
        return
    rows = _probe_rows(manager)
    row = dict(
        t=round(_time.time(), 3),
        event=event,
        active_gib=round(mx.get_active_memory() / 2**30, 3),
        cache_gib=round(mx.get_cache_memory() / 2**30, 3),
        peak_gib=round(mx.get_peak_memory() / 2**30, 3),
        resident_gib=round(manager.resident_bytes() / 2**30, 3),
        anchor_gib=round(getattr(manager, "anchor_bytes", lambda: 0)() / 2**30, 3),
        entries=rows,
    )
    with open(path, "a") as f:
        f.write(_json.dumps(row) + "\n")


_orig_flush = _Coordinator.flush_deferred_checkpoints  # noqa: F821


def _probe_flush(self):
    _probe_log(self.manager, "flush_start")
    try:
        return _orig_flush(self)
    finally:
        _probe_log(self.manager, "flush_end")
        import mlx.core as mx

        mx.reset_peak_memory()


_Coordinator.flush_deferred_checkpoints = _probe_flush  # noqa: F821

_orig_lookup = YunshuAPCManager.lookup_exact_cache  # noqa: F821


def _probe_lookup(self, token_ids, *args, **kwargs):
    out = _orig_lookup(self, token_ids, *args, **kwargs)
    _probe_log(self, f"lookup_hit_{out[1]}_of_{len(token_ids)}")
    return out


YunshuAPCManager.lookup_exact_cache = _probe_lookup  # noqa: F821
