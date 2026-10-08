"""The unified-memory ledger behind ``GET /v1/yunshu/memory``.

Every number comes from a counter the process already keeps (MLX allocator counters, the model
manager, the prefix cache's own occupancy, ``proc_pid_rusage``, ``sysctl``). A figure with no
source is ``None``, never a guess. Owners that are attributed by subtraction say so
(``estimated: true``). Nothing here runs GPU work, walks the garbage collector or starts a
subprocess; a call reads counters only (the host block is cached for ``HOST_TTL_S``).
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import threading
import time
from typing import Any

from yunshu_engine import units

logger = logging.getLogger(__name__)

HOST_TTL_S = 15.0

_PRESSURE_NAMES = {1: "normal", 2: "warn", 4: "critical"}


def gb(value: float | int | None, digits: int = 3) -> float | None:
    """Bytes -> binary GB (GB = 1024**3, as macOS and the YUNSHU_*_GB settings); None stays None."""
    return units.gb(value, digits)


def put(
    out: dict[str, Any], key: str, n_bytes: float | int | None, digits: int = 3
) -> None:
    """``out[key_gb]`` (binary GB) and ``out[key_bytes]`` (exact int); both None when unknown."""
    if n_bytes is None:
        out[f"{key}_gb"] = None
        out[f"{key}_bytes"] = None
    else:
        units.put_gb(out, key, n_bytes, digits)


# ── host (OS) ───────────────────────────────────────────────────────────

_libc: Any = None


def _sysctl_raw(name: str, size: int) -> bytes | None:
    """``sysctlbyname`` without a subprocess; None where the name does not exist."""
    global _libc
    try:
        if _libc is None:
            _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        buf = ctypes.create_string_buffer(size)
        n = ctypes.c_size_t(size)
        if _libc.sysctlbyname(name.encode(), buf, ctypes.byref(n), None, 0) != 0:
            return None
        return buf.raw[: n.value]
    except (OSError, AttributeError):
        return None


def _sysctl_int(name: str) -> int | None:
    raw = _sysctl_raw(name, 8)
    if raw is None or len(raw) not in (4, 8):
        return None
    return int.from_bytes(raw, "little", signed=True)


def _host_uncached() -> dict[str, Any]:
    out: dict[str, Any] = {"pressure_level": None}
    for key in ("total", "swap_used", "swap_total", "wired_limit", "available"):
        put(out, key, None)
    total = _sysctl_int("hw.memsize")
    if total is None:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            total = None
    put(out, "total", total, 1)
    level = _sysctl_int("kern.memorystatus_vm_pressure_level")
    if level is not None:
        out["pressure_level"] = _PRESSURE_NAMES.get(level, str(level))
    swap = _sysctl_raw("vm.swapusage", 32)
    if swap is not None and len(swap) >= 24:
        s_total, _avail, s_used = struct.unpack_from("<QQQ", swap, 0)
        put(out, "swap_total", s_total, 2)
        put(out, "swap_used", s_used, 2)
    wired_mb = _sysctl_int("iogpu.wired_limit_mb")
    if wired_mb is not None:
        # 0 means "the system default", which is not a number: report it as unknown.
        put(out, "wired_limit", wired_mb * 1048576 if wired_mb else None, 2)
    try:
        import psutil

        put(out, "available", psutil.virtual_memory().available, 2)
    except Exception:
        logger.debug("psutil unavailable", exc_info=True)
    return out


_host_lock = threading.Lock()
_host_cache: tuple[float, dict[str, Any]] | None = None


def host(now: float | None = None) -> dict[str, Any]:
    """OS memory state, cached for ``HOST_TTL_S`` so a 3 s poll does not re-read sysctl."""
    global _host_cache
    t = time.monotonic() if now is None else now
    with _host_lock:
        if _host_cache is not None and t - _host_cache[0] < HOST_TTL_S:
            return dict(_host_cache[1])
    value = _host_uncached()
    with _host_lock:
        _host_cache = (t, value)
    return dict(value)


def reset_host_cache() -> None:
    global _host_cache
    with _host_lock:
        _host_cache = None


# ── MLX / process ───────────────────────────────────────────────────────

_recommended: list[int | None] = []


def recommended_working_set() -> int | None:
    """Metal's recommended working-set size (cached: it is a device constant)."""
    if _recommended:
        return _recommended[0]
    value: int | None = None
    try:
        import mlx.core as mx

        fn = getattr(mx, "device_info", None) or getattr(
            getattr(mx, "metal", None), "device_info", None
        )
        info = fn() if callable(fn) else {}
        v = info.get("max_recommended_working_set_size")
        value = int(v) if v else None
    except Exception:
        logger.debug("device_info unavailable", exc_info=True)
    _recommended.append(value)
    return value


def mlx_counters() -> dict[str, int | None]:
    out: dict[str, int | None] = {"active": None, "cache": None, "peak": None}
    try:
        import mlx.core as mx

        out["active"] = int(mx.get_active_memory())
        out["cache"] = int(mx.get_cache_memory())
        out["peak"] = int(mx.get_peak_memory())
    except Exception:
        logger.debug("mlx memory unavailable", exc_info=True)
    return out


def _bytes_block(values: dict[str, float | int | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, n in values.items():
        put(out, key, n)
    return out


def _mlx_block(mlx: dict[str, Any]) -> dict[str, Any]:
    return _bytes_block(
        {
            "active": mlx["active"],
            "cache": mlx["cache"],
            "peak": mlx["peak"],
            "recommended_working_set": recommended_working_set(),
        }
    )


def _overshoot_block(overshoot: float | int | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    put(out, "attribution_overshoot", overshoot if overshoot else None)
    return out


def process_footprint() -> dict[str, int | None]:
    """phys_footprint of this process, and its peak when the in-process sampler runs."""
    from yunshu_engine import footprint_sampler

    cur = footprint_sampler.self_footprint_bytes() or None
    sampler = footprint_sampler.current()
    peak = sampler.peak if sampler is not None and sampler.peak else None
    return {"footprint": cur, "footprint_peak": peak}


# ── owners ──────────────────────────────────────────────────────────────

_weight_cache: dict[int, int] = {}


def _tree_nbytes(tree: Any) -> int:
    """Sum ``nbytes`` over the arrays of a nested dict / list / tuple of parameters."""
    if isinstance(tree, dict):
        return sum(_tree_nbytes(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(_tree_nbytes(v) for v in tree)
    return int(getattr(tree, "nbytes", 0) or 0)


def measured_weight_bytes(engine: Any) -> int | None:
    """Bytes of the engine's parameter arrays (array metadata only, no evaluation); cached per
    engine object because the weights do not change after load."""
    key = id(engine)
    if key in _weight_cache:
        return _weight_cache[key] or None
    total = 0
    try:
        for attr in ("_model", "model"):
            model = getattr(engine, attr, None)
            if model is not None and hasattr(model, "parameters"):
                total = _tree_nbytes(model.parameters())
                break
    except Exception:
        logger.debug("weight walk failed", exc_info=True)
        total = 0
    _weight_cache[key] = total
    return total or None


def _loaded_models(manager: Any, engine: Any, display_id: str | None) -> list[dict]:
    rows: list[dict] = []
    if manager is not None:
        for e in manager.list_entries():
            if not e.is_loaded:
                continue
            rows.append(
                {
                    "id": e.model_id,
                    "engine": e.engine,
                    "estimated_bytes": int(e.estimated_bytes or 0) or None,
                }
            )
    elif engine is not None and getattr(engine, "is_loaded", False):
        rows.append(
            {
                "id": display_id or getattr(engine, "model_name", None) or "default",
                "engine": engine,
                "estimated_bytes": None,
            }
        )
    return rows


def _apc_snapshot(engine: Any) -> dict | None:
    fn = getattr(engine, "apc_snapshot", None)
    if not callable(fn):
        return None
    try:
        snap = fn()
    except Exception:
        logger.debug("apc snapshot failed", exc_info=True)
        return None
    return snap or None


def _guard_margin_pct(engine: Any) -> float | None:
    guard = getattr(engine, "_memory_guard", None)
    if guard is None:
        core = getattr(engine, "_engine_core", None)
        guard = getattr(core, "_memory_guard", None) if core is not None else None
    margin = getattr(guard, "_safety_margin_pct", None)
    return float(margin) if isinstance(margin, (int, float)) else None


def collect(manager: Any, engine: Any, display_id: str | None = None) -> dict[str, Any]:
    """The ledger. ``manager`` / ``engine`` are the gateway's model manager and default
    engine; either may be None."""
    mlx = mlx_counters()
    proc = process_footprint()
    hst = host()
    owners: list[dict[str, Any]] = []
    attributed = 0
    apc_max = None
    warm_max = None
    guard_margin = None
    storage: list[dict[str, Any]] = []

    for m in _loaded_models(manager, engine, display_id):
        eng = m["engine"]
        measured = measured_weight_bytes(eng) if eng is not None else None
        if measured is not None:
            w_bytes, w_est, w_src = measured, False, "model parameters"
        else:
            w_bytes, w_est, w_src = m["estimated_bytes"], True, "model manager estimate"
        owners.append(
            {
                "kind": "weights",
                "id": m["id"],
                "bytes": w_bytes,
                "reclaimable": manager is not None,
                "estimated": w_est,
                "source": w_src if w_bytes is not None else None,
            }
        )
        attributed += w_bytes or 0
        snap = _apc_snapshot(eng) if eng is not None else None
        if snap is None:
            continue
        if guard_margin is None:
            guard_margin = _guard_margin_pct(eng)
        apc_max = snap.get("memory_max_bytes", apc_max)
        warm_max = snap.get("warm_max_bytes", warm_max)
        ram = snap.get("resident_bytes")
        owners.append(
            {
                "kind": "apc_ram",
                "id": m["id"],
                "bytes": ram,
                "reclaimable": True,
                "estimated": False,
                "source": "prefix cache occupancy" if ram is not None else None,
            }
        )
        attributed += ram or 0
        if snap.get("warm_bytes") is not None:
            owners.append(
                {
                    "kind": "apc_warm",
                    "id": m["id"],
                    "bytes": snap["warm_bytes"],
                    "reclaimable": True,
                    "estimated": False,
                    "source": "warm tier occupancy",
                }
            )
            attributed += snap["warm_bytes"] or 0
        for t in snap.get("storage_tiers") or []:
            storage.append(
                {
                    "model": m["id"],
                    "tier": t.get("name"),
                    "used_bytes": t.get("used_bytes"),
                    "cap_bytes": t.get("cap_bytes"),
                    "available": t.get("available"),
                }
            )

    # Live KV of in-flight requests is not counted anywhere yet: unknown, not zero.
    owners.append(
        {
            "kind": "live_kv",
            "id": None,
            "bytes": None,
            "reclaimable": False,
            "estimated": False,
            "source": None,
        }
    )
    owners.append(
        {
            "kind": "mlx_cache",
            "id": None,
            "bytes": mlx["cache"],
            "reclaimable": True,
            "estimated": False,
            "source": "mlx allocator cache" if mlx["cache"] is not None else None,
        }
    )
    other: int | None = None
    overshoot = 0
    if mlx["active"] is not None:
        other = mlx["active"] - attributed
        if other < 0:
            overshoot, other = -other, 0
    owners.append(
        {
            "kind": "other",
            "id": None,
            "bytes": other,
            "reclaimable": False,
            "estimated": True,
            "source": "mlx active minus attributed owners"
            if other is not None
            else None,
        }
    )

    for o in owners:
        o["gb"] = gb(o["bytes"])

    total_gb = hst["total_gb"]
    free_bytes = hst.get("available_bytes")
    return {
        "object": "yunshu.memory",
        "t": round(time.time(), 3),
        "total_gb": total_gb,
        "total_bytes": hst.get("total_bytes"),
        "free_gb": hst["available_gb"],
        "free_bytes": free_bytes,
        "host": {
            k: v
            for k, v in hst.items()
            if k not in ("total_gb", "total_bytes", "available_gb", "available_bytes")
        },
        "mlx": _mlx_block(mlx),
        "process": _bytes_block(
            {"footprint": proc["footprint"], "footprint_peak": proc["footprint_peak"]}
        ),
        "owners": owners,
        **_overshoot_block(overshoot),
        "storage_tiers": storage,
        "limits": {
            **_bytes_block({"apc_max": apc_max, "apc_warm_max": warm_max}),
            "guard_margin_pct": guard_margin,
        },
    }
