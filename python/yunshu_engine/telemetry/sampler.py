"""Bounded host-only sampling. Native APIs are called exclusively by the worker.

Request receipts integrate GPU+DRAM host power over phase windows. They are
estimates including other processes/idle power, not isolated device accounting.
The last unfinished interval uses the previous reading, explicitly reported.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from typing import Any

from . import apple


def unknown(reason: str) -> dict:
    return {"state": "unknown", "reason": reason}


def frequency(states: list, table: list[int]) -> tuple[float | None, str | None]:
    if any(
        not name or residency < 0 or not math.isfinite(residency)
        for name, residency in states
    ):
        return None, "invalid GPU state residency"
    indices = sorted(
        int(n[1:]) for n, _ in states if n and n.startswith("P") and n[1:].isdigit()
    )
    mapped = set(range(1, len(table) + 1))
    if (
        not table
        or table != sorted(table)
        or any(mhz <= 0 for mhz in table)
        or len(indices) != len(set(indices))
        or not mapped.issubset(indices)
    ):
        return None, "GPU DVFS table does not match residency states"
    # M5 exposes reserved P14/P15 entries even with only 13 physical clocks.
    # Zero-residency entries contribute no time; never guess an occupied clock.
    if any(
        int(name[1:]) not in mapped and residency > 0
        for name, residency in states
        if name.startswith("P") and name[1:].isdigit()
    ):
        return None, "active GPU state has no qualified frequency"
    active = [
        (int(n[1:]), r)
        for n, r in states
        if n and n.startswith("P") and n[1:].isdigit() and r > 0
    ]
    total = sum(r for _, r in active)
    return (sum(table[i - 1] * r for i, r in active) / total if total else None), (
        None if total else "no active GPU residency"
    )


class HostSampler:
    def __init__(
        self,
        interval: float = 1.0,
        energy_factory=apple.EnergySampler,
        temperature_factory=apple.TemperatureSampler,
        table_factory=apple.gpu_frequency_table_mhz,
    ):
        if not math.isfinite(interval) or interval < 0.1:
            raise ValueError("interval must be finite and >= 0.1")
        self.interval = interval
        self._energy_factory = energy_factory
        self._temperature_factory = temperature_factory
        self._table_factory = table_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot: dict = unknown("sampler has not produced a reading")
        self._history: deque = deque(maxlen=7200)
        self._total = {"prefill": 0.0, "decode": 0.0}

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self._thread = threading.Thread(
                target=self._run, name="yunshu-host-telemetry", daemon=True
            )
            self._thread.start()
        except Exception as exc:
            self._thread = None
            with self._lock:
                self._snapshot = unknown(f"sampler worker could not start: {exc}")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=12)

    def _run(self) -> None:
        energy = temperature = None
        try:
            energy = self._energy_factory()
            try:
                temperature = self._temperature_factory()
                temp_reason = None
            except Exception as exc:
                temp_reason = str(exc)
            table = self._table_factory()
            while not self._stop.wait(self.interval):
                try:
                    reading = energy.read()
                    end = getattr(reading, "t_end", 0.0) or time.perf_counter()
                    temps: dict = unknown(temp_reason or "no die sensors")
                    if temperature is not None:
                        try:
                            summary = temperature.die_summary()
                            temps = (
                                {"state": "ok", **summary}
                                if summary.get("die_max_c") is not None
                                else unknown("no HID die sensors")
                            )
                        except Exception as exc:
                            temps = unknown(str(exc))
                    self.publish(reading, end, temps, table)
                except Exception as exc:
                    with self._lock:
                        self._snapshot = unknown(str(exc))
                        self._history.append((time.perf_counter(), 0.0, None, None))
        except Exception as exc:
            with self._lock:
                self._snapshot = unknown(str(exc))
        finally:
            for resource in (temperature, energy):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        with self._lock:
                            self._snapshot = unknown(f"native cleanup failed: {exc}")

    def publish(self, reading: Any, end: float, temps: dict, table: list[int]) -> None:
        """CPU-test seam; no native operations."""
        watts = {}
        reasons = {}
        for domain, channel in (
            ("cpu", "CPU Energy"),
            ("gpu", "GPU Energy"),
            ("ane", "ANE"),
            ("dram", "DRAM"),
        ):
            value = reading.watts.get(channel)
            if value is None or not math.isfinite(value) or value < 0:
                value = None
                reasons[domain] = getattr(reading, "reasons", {}).get(
                    channel, f"missing or invalid {channel} counter"
                )
            watts[domain] = value
        watts["package"] = (
            sum(watts.values()) if all(v is not None for v in watts.values()) else None
        )
        mhz, reason = frequency(reading.gpu_states, table)
        ratio = reading.gpu_active_fraction()
        if ratio is not None and (not math.isfinite(ratio) or not 0 <= ratio <= 1):
            ratio = None
            reason = "invalid GPU active residency ratio"
        if reason == "invalid GPU state residency":
            ratio = None
        gpu = {"frequency_mhz": mhz, "active_ratio": ratio}
        if reason:
            gpu["reason"] = reason
        snap = {
            "state": "ok" if not reasons else "partial",
            "sampled_at": time.time(),
            "interval_s": reading.seconds,
            "watts": watts,
            "gpu": gpu,
            "temperature": temps,
            "reasons": reasons,
        }
        governed = (
            watts["gpu"] + watts["dram"]
            if watts["gpu"] is not None and watts["dram"] is not None
            else None
        )
        with self._lock:
            self._snapshot = snap
            self._history.append((end, reading.seconds, governed, watts["gpu"]))

    def snapshot(self) -> dict:
        with self._lock:
            snap = copy.deepcopy(self._snapshot)
        if (
            snap.get("sampled_at")
            and time.time() - snap["sampled_at"] > self.interval * 3
        ):
            return unknown("last telemetry reading is stale")
        return snap

    def window(self, start: float, end: float, tokens: int) -> dict:
        if not start or end <= start:
            return {
                **unknown("phase timing unavailable"),
                "joules": None,
                "joules_per_token": None,
                "gpu_watts_mean": None,
            }
        with self._lock:
            rows = list(self._history)
        joules = gpu_j = covered = extrapolated = 0.0
        cursor = start
        for stop, dt, power, gpu in rows:
            a, b = max(cursor, start, stop - dt), min(end, stop)
            if b <= a:
                continue
            if power is None or a > cursor + 0.01:
                break
            joules += power * (b - a)
            gpu_j += gpu * (b - a)
            covered += b - a
            cursor = max(cursor, b)
        if rows and cursor < end:
            stop, _, power, gpu = rows[-1]
            if (
                power is not None
                and cursor >= stop - 0.01
                and end - stop <= self.interval * 2
            ):
                extrapolated = end - cursor
                joules += power * extrapolated
                gpu_j += gpu * extrapolated
                covered += extrapolated
        duration = end - start
        ok = covered > 0 and abs(covered - duration) <= max(1e-9, duration * 1e-9)
        return {
            "state": "estimated" if ok else "unknown",
            "reason": None if ok else "phase not fully covered by valid samples",
            "joules": joules if ok else None,
            "joules_per_token": joules / tokens if ok and tokens > 0 else None,
            "gpu_watts_mean": gpu_j / duration if ok else None,
            "coverage_ratio": min(covered / duration, 1.0),
            "extrapolated_s": extrapolated,
        }

    def receipt(self, stats: Any) -> dict:
        first = stats.t_first
        boundary = getattr(stats, "t_prefill_end", 0) or first
        return {
            "schema": "yunshu.energy.v1",
            "method": "host_window_gpu_plus_dram",
            "includes_other_processes": True,
            "overlapping_requests_share_host_energy": True,
            "prefill": self.window(
                stats.t_admit,
                boundary,
                max(stats.prompt_tokens - stats.cached_tokens, 0),
            ),
            "decode": self.window(boundary, stats.t_last, stats.generated),
        }

    def record(self, receipt: dict) -> None:
        with self._lock:
            for phase in self._total:
                value = receipt.get(phase, {}).get("joules")
                if value is not None:
                    self._total[phase] += value

    def prometheus(self) -> str:
        snap = self.snapshot()
        watts, gpu, temp = (
            snap.get("watts", {}),
            snap.get("gpu", {}),
            snap.get("temperature", {}),
        )
        lines: list[str] = []
        for name, value in (
            ("gpu_watts", watts.get("gpu")),
            ("package_watts", watts.get("package")),
            ("gpu_frequency_mhz", gpu.get("frequency_mhz")),
            ("gpu_active_ratio", gpu.get("active_ratio")),
            ("die_temperature_celsius", temp.get("die_max_c")),
        ):
            if value is not None:
                lines.extend((f"# TYPE yunshu_{name} gauge", f"yunshu_{name} {value}"))
        with self._lock:
            totals = dict(self._total)
        lines.append("# TYPE yunshu_request_energy_joules_total counter")
        lines.extend(
            f'yunshu_request_energy_joules_total{{phase="{p}"}} {v}'
            for p, v in totals.items()
        )
        return "\n".join(lines) + "\n"


_SERVICE: HostSampler | None = None


def start() -> None:
    global _SERVICE
    from yunshu_engine import settings

    if settings.get("YUNSHU_TELEMETRY") == "on" and _SERVICE is None:
        _SERVICE = HostSampler(settings.get("YUNSHU_TELEMETRY_INTERVAL_S"))
        _SERVICE.start()


def stop() -> None:
    global _SERVICE
    if _SERVICE is not None:
        _SERVICE.close()
        _SERVICE = None


def get() -> HostSampler | None:
    return _SERVICE


def snapshot() -> dict:
    return _SERVICE.snapshot() if _SERVICE else unknown("telemetry disabled")
