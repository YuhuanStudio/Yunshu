"""Memory-aware admission: never admit work into swap.

Three pieces, all scheduling only (no output ever changes, only when and where work runs):

* :func:`plan_load` - the load-time budget: weights + minimum KV + OS reserve against physical
  memory; the APC RAM budget shrinks to what is left, and the load is refused with a clear
  message only when even the minimum does not fit.
* :class:`MemoryGate` - per-request admission against the *live* host headroom (macOS free +
  inactive + purgeable pages, MLX's reclaimable buffer cache, the kernel pressure level and
  swap growth). A request that does not fit waits in the queue with a reason; the first request
  of an idle server is always admitted (nothing to wait for).
* the pressure signal that tells the runner to shed APC RAM entries (to SSD where configured)
  before the kernel starts swapping.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

GIB = 1 << 30
LEVEL_NORMAL, LEVEL_WARN, LEVEL_CRITICAL = 1, 2, 4
# Minimum room for the KV of a short request plus the prefill workspace.
MIN_WORKING_GIB = 1.0
MIN_CONTEXT_TOKENS = 8192
# Prefill workspace per token of a chunk: hidden size x bf16 x an intermediate expansion
# (MLP up/gate, attention scores, MoE gathers). Conservative; the live headroom is the truth.
ACT_EXPANSION = 24
PREFILL_STEP = 2048
# Swap growth (above the lowest value seen) that counts as "already swapping".
SWAP_GROWTH_BYTES = 512 << 20


@dataclass(frozen=True)
class HostMemory:
    total: int
    available: int
    swap_used: int
    level: int = LEVEL_NORMAL


def default_reserve_bytes(total: int) -> int:
    """Memory kept free for macOS (page cache, compressor, other apps): 5% of RAM, at least
    4 GiB (128 GiB machine: 6.4 GiB; 16 GiB machine: 4 GiB)."""
    return int(max(4 * GIB, 0.05 * total))


# ---------------------------------------------------------------- host readings
_libc = None


def _sysctl_bytes(name: str, size: int) -> bytes | None:
    global _libc
    try:
        if _libc is None:
            _libc = ctypes.CDLL("libc.dylib", use_errno=True)
        buf = ctypes.create_string_buffer(size)
        n = ctypes.c_size_t(size)
        if _libc.sysctlbyname(name.encode(), buf, ctypes.byref(n), None, 0) != 0:
            return None
        return buf.raw[: n.value]
    except Exception:
        return None


def _sysctl_int(name: str) -> int | None:
    raw = _sysctl_bytes(name, 4)
    return int.from_bytes(raw, "little") if raw else None


def parse_vm_stat(text: str) -> int | None:
    """Available bytes from ``vm_stat`` output: free + inactive + speculative + purgeable."""
    page = 16384
    pages = {}
    for line in text.splitlines():
        if "page size of" in line:
            digits = "".join(
                c
                for c in line.split("page size of")[1].split("bytes")[0]
                if c.isdigit()
            )
            if digits:
                page = int(digits)
            continue
        key, _, val = line.partition(":")
        val = val.strip().rstrip(".")
        if val.isdigit():
            pages[key.strip()] = int(val)
    need = ("Pages free", "Pages inactive")
    if any(k not in pages for k in need):
        return None
    n = (
        pages["Pages free"]
        + pages["Pages inactive"]
        + pages.get("Pages speculative", 0)
        + pages.get("Pages purgeable", 0)
    )
    return n * page


def read_host() -> HostMemory | None:
    """Live host memory, or None when it cannot be read (callers then admit as before)."""
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=1.0, check=True
        ).stdout
        available = parse_vm_stat(out)
        total = _sysctl_bytes("hw.memsize", 8)
        if available is None or not total:
            return None
        swap = _sysctl_bytes(
            "vm.swapusage", 32
        )  # xsw_usage: total, avail, used (u64 x3)
        used = int.from_bytes(swap[16:24], "little") if swap and len(swap) >= 24 else 0
        level = _sysctl_int("kern.memorystatus_vm_pressure_level") or LEVEL_NORMAL
        return HostMemory(int.from_bytes(total, "little"), available, used, level)
    except Exception:
        logger.debug("host memory unreadable", exc_info=True)
        return None


# ---------------------------------------------------------------- sizing
def kv_bytes_per_token(config: dict, kv_bytes: int = 2) -> int:
    """KV bytes one token adds across the attention layers (linear-attention layers keep a
    fixed state and add nothing per token). 0 when the geometry cannot be read."""
    tc = config.get("text_config") or config
    layers = int(tc.get("num_hidden_layers") or tc.get("n_layers") or 0)
    if not layers:
        return 0
    types = tc.get("layer_types")
    if isinstance(types, list) and types:
        full = sum(1 for t in types if "full" in str(t))
    elif tc.get("full_attention_interval"):
        full = layers // int(tc["full_attention_interval"])
    else:
        full = layers
    heads = int(tc.get("num_attention_heads") or 0)
    kvh = int(tc.get("num_key_value_heads") or heads or 0)
    hd = int(
        tc.get("head_dim") or ((tc.get("hidden_size") or 0) // heads if heads else 0)
    )
    return full * 2 * kvh * hd * kv_bytes


def activation_bytes_per_token(config: dict) -> int:
    tc = config.get("text_config") or config
    return int(tc.get("hidden_size") or 0) * 2 * ACT_EXPANSION


def weights_bytes_on_disk(model_path: str | Path) -> int:
    try:
        return sum(p.stat().st_size for p in Path(model_path).glob("*.safetensors"))
    except OSError:
        return 0


def sibling_view(model_path: str | Path) -> Path | None:
    """A smaller view of the same checkpoint next to it (``<name>-ple-ssd``), if present."""
    p = Path(model_path)
    cand = p.with_name(p.name + "-ple-ssd")
    return cand if cand.is_dir() and cand != p else None


# ---------------------------------------------------------------- load-time plan
@dataclass
class LoadPlan:
    apc_gb: float
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def plan_load(
    total_bytes: int,
    weights_bytes: int,
    apc_gb: float,
    *,
    kv_per_token: int = 0,
    reserve_bytes: int | None = None,
    model_path: str | Path | None = None,
) -> LoadPlan:
    """Fit weights + minimum KV/working set + OS reserve + APC budget into ``total_bytes``.

    The APC budget shrinks to the room left (0 = prefix cache off); the plan carries an error
    only when even the minimum (APC off, MIN_CONTEXT_TOKENS of KV, 1 GiB working set) does not
    fit. When the checkpoint has a ``-ple-ssd`` view the message names it."""
    reserve = (
        default_reserve_bytes(total_bytes) if reserve_bytes is None else reserve_bytes
    )
    minimum_kv = kv_per_token * MIN_CONTEXT_TOKENS + int(MIN_WORKING_GIB * GIB)
    base = weights_bytes + reserve + minimum_kv
    plan = LoadPlan(apc_gb=float(apc_gb))
    if base > total_bytes:
        hint = ""
        view = sibling_view(model_path) if model_path else None
        if view is not None:
            hint = f"; load the smaller view {view} instead"
        plan.apc_gb = 0.0
        plan.error = (
            f"model needs {base / GIB:.1f} GiB (weights {weights_bytes / GIB:.1f} + OS "
            f"reserve {reserve / GIB:.1f} + minimum KV and working set "
            f"{minimum_kv / GIB:.1f}) but only {total_bytes / GIB:.1f} GiB of memory is "
            f"available to the engine{hint}"
        )
        return plan
    room = (total_bytes - base) / GIB
    if plan.apc_gb > room:
        plan.notes.append(
            f"APC RAM budget lowered {plan.apc_gb:.1f} -> {max(0.0, room):.1f} GiB: weights "
            f"{weights_bytes / GIB:.1f} GiB leave {room:.1f} GiB after the "
            f"{reserve / GIB:.1f} GiB OS reserve and the minimum KV"
        )
        plan.apc_gb = max(
            0.0, float(int(room * 4) / 4)
        )  # quarter-GiB steps, round down
        if plan.apc_gb < 1.0:
            plan.apc_gb = 0.0
    view = sibling_view(model_path) if model_path else None
    if view is not None and weights_bytes / total_bytes > 0.6:
        plan.notes.append(f"a smaller view of this checkpoint exists: {view}")
    return plan


# ---------------------------------------------------------------- admission gate
MEMORY_LIMIT_MARK = "[memory_limit] "


def memory_limit_error(text: str) -> MemoryError | None:
    """The refusal rebuilt from an engine error string that carries the mark (the engine
    reports failures as text through its output queue)."""
    if text.startswith(MEMORY_LIMIT_MARK):
        return MemoryAdmissionError(text[len(MEMORY_LIMIT_MARK) :])
    return None


class MemoryAdmissionError(MemoryError):
    """A request that cannot fit even on an idle server (it would run the process out of
    Metal memory and take every running request with it): refused with the numbers."""


@dataclass(frozen=True)
class Decision:
    admit: bool
    reason: str | None = None  # "memory" when held, "memory_limit" when refused
    detail: str = ""
    refuse: bool = False


class MemoryGate:
    def __init__(
        self,
        *,
        kv_per_token: int,
        act_per_token: int,
        attn_heads: int = 0,
        reserve_bytes: int = 0,
        reclaimable: Callable[[], int] = lambda: 0,
        ceiling_free: Callable[[], int | None] = lambda: None,
        read: Callable[[], HostMemory | None] = read_host,
        ttl_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.kv_per_token = kv_per_token
        self.act_per_token = act_per_token
        self.attn_heads = attn_heads
        self.reserve = reserve_bytes
        self._reclaimable = reclaimable
        self._ceiling_free = ceiling_free
        self._read = read
        self._ttl = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._cached: HostMemory | None = None
        self._cached_at = -1e9
        self._swap_floor: int | None = None
        self.holds = 0  # distinct requests held at least once
        self.hold_checks = 0
        self.sheds = 0
        self.shed_bytes = 0
        self.last: dict = {}

    def host(self) -> HostMemory | None:
        now = self._clock()
        with self._lock:
            if now - self._cached_at >= self._ttl:
                self._cached = self._read()
                self._cached_at = now
                if self._cached is not None:
                    floor = self._swap_floor
                    self._swap_floor = (
                        self._cached.swap_used
                        if floor is None
                        else min(floor, self._cached.swap_used)
                    )
            return self._cached

    def estimate(
        self, prompt_tokens: int, max_tokens: int, cached_tokens: int = 0
    ) -> int:
        """KV for the prompt and the output plus the prefill workspace of one chunk: the
        activations (``act_per_token``) and the attention scores of the chunk against the
        whole prompt (chunk x prompt x heads x 2 bytes; measured on the 27B: 52K-token prompt,
        6.2 GiB over the resident state against 5.1 GiB from this formula)."""
        kv = (max(0, prompt_tokens) + max(0, max_tokens)) * self.kv_per_token
        chunk = min(max(0, prompt_tokens - cached_tokens), PREFILL_STEP)
        return (
            kv
            + chunk * self.act_per_token
            + chunk * max(0, prompt_tokens) * self.attn_heads * 2
        )

    def available(self) -> int | None:
        """Bytes the engine can still grow into: host available + reclaimable buffers, bounded
        by the configured ceiling when one is set."""
        host = self.host()
        limits = []
        if host is not None:
            limits.append(host.available + self._reclaimable())
        cap = self._ceiling_free()
        if cap is not None:
            limits.append(cap)
        return min(limits) if limits else None

    def decide(self, estimate: int, committed: int, any_active: bool) -> Decision:
        """Admit the request or hold it. ``committed`` is the not-yet-materialized part of the
        requests already running (their KV will still grow into the headroom)."""
        host = self.host()
        avail = self.available()
        hard = self._ceiling_free()
        if not any_active and hard is not None and estimate > hard:
            self.last = {"available": avail, "estimate": estimate, "refused": True}
            return Decision(
                False,
                "memory_limit",
                f"needs {estimate / GIB:.2f} GiB but only {max(0, hard) / GIB:.2f} GiB "
                "of the Metal working set is free after the weights; shorten the prompt "
                "or max_tokens",
                refuse=True,
            )
        if avail is None or (not any_active):
            self.last = {"available": avail, "estimate": estimate, "idle_admit": True}
            return Decision(True)
        live = avail - committed
        why = None
        if host is not None and host.level >= LEVEL_CRITICAL:
            why = "host memory pressure is critical"
        elif (
            host is not None
            and self._swap_floor is not None
            and host.swap_used - self._swap_floor > SWAP_GROWTH_BYTES
        ):
            why = f"swap grew {(host.swap_used - self._swap_floor) / GIB:.1f} GiB"
        elif live - estimate < self.reserve:
            why = (
                f"needs {estimate / GIB:.2f} GiB, headroom {max(0, live) / GIB:.2f} GiB "
                f"(keeping {self.reserve / GIB:.1f} GiB free)"
            )
        self.last = {"available": avail, "committed": committed, "estimate": estimate}
        if why is None:
            return Decision(True)
        self.hold_checks += 1
        return Decision(False, "memory", why)

    def pressure_shortfall(self) -> int:
        """Bytes to free from reclaimable caches so the host is back above its reserve (0 when
        it is, or when the host cannot be read)."""
        host = self.host()
        if host is None:
            return 0
        short = max(0, self.reserve - (host.available + self._reclaimable()))
        if host.level >= LEVEL_WARN:
            short = max(short, self.reserve // 2)
        return short

    def snapshot(self) -> dict:
        host = self._cached
        return {
            "holds": self.holds,
            "hold_checks": self.hold_checks,
            "sheds": self.sheds,
            "shed_bytes": self.shed_bytes,
            "reserve_bytes": self.reserve,
            "available_bytes": self.last.get("available"),
            "swap_used_bytes": host.swap_used if host else None,
            "pressure_level": host.level if host else None,
        }


# ---------------------------------------------------------------- engine glue
def effective_total_bytes(physical: int, ceiling_gb: float | None) -> int:
    """Physical memory, lowered to ``YUNSHU_MAX_MEMORY_GB`` when that is set (> 0)."""
    if ceiling_gb:
        return min(physical, int(float(ceiling_gb) * GIB))
    return physical


def build_gate(
    config: dict,
    *,
    total_bytes: int,
    ceiling_gb: float | None,
    reserve_gb: float | None,
    kv_bytes: int = 2,
) -> MemoryGate | None:
    """The runner's admission gate (None when the model geometry is unknown)."""
    kv = kv_bytes_per_token(config, kv_bytes)
    if kv <= 0:
        return None
    reserve = (
        int(float(reserve_gb) * GIB)
        if reserve_gb is not None
        else default_reserve_bytes(total_bytes)
    )

    def reclaimable() -> int:
        import mlx.core as mx

        return int(mx.get_cache_memory())

    def ceiling_free() -> int | None:
        """Bytes left under the hard limit: the Metal recommended working set (running past
        it ends in "Insufficient Memory" for every request) and the user's ceiling."""
        import mlx.core as mx

        limits = []
        if ceiling_gb:
            limits.append(int(float(ceiling_gb) * GIB))
        working_set = mx.device_info().get("max_recommended_working_set_size")
        if working_set:
            limits.append(int(working_set))
        if not limits:
            return None
        return min(limits) - int(mx.get_active_memory())

    return MemoryGate(
        kv_per_token=kv,
        act_per_token=activation_bytes_per_token(config),
        attn_heads=int(
            (config.get("text_config") or config).get("num_attention_heads") or 0
        ),
        reserve_bytes=reserve,
        reclaimable=reclaimable,
        ceiling_free=ceiling_free,
    )
