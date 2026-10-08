"""Bounded, CPU-only host probes; failures retain an explicit unknown reason."""

from __future__ import annotations

import re
import subprocess
import threading
import time

_LOCK = threading.Lock()
_CACHED: dict | None = None
_EXPIRES = 0.0


def parse_thermal(text: str) -> dict:
    values = dict(
        re.findall(
            r"(CPU_Speed_Limit|CPU_Scheduler_Limit|CPU_Available_CPUs)\s*=\s*(\d+)",
            text,
        )
    )
    if not values:
        if (
            "No thermal warning level has been recorded" in text
            and "No performance warning level has been recorded" in text
        ):
            return {
                "state": "normal",
                "cpu_speed_limit_percent": None,
                "cpu_scheduler_limit_percent": None,
                "available_cpus": None,
            }
        return {"state": "unknown", "reason": "pmset did not report thermal limits"}
    speed = int(values.get("CPU_Speed_Limit", "100"))
    return {
        "state": "throttled"
        if speed < 100 or int(values.get("CPU_Scheduler_Limit", "100")) < 100
        else "normal",
        "cpu_speed_limit_percent": int(values["CPU_Speed_Limit"])
        if "CPU_Speed_Limit" in values
        else None,
        "cpu_scheduler_limit_percent": int(values["CPU_Scheduler_Limit"])
        if "CPU_Scheduler_Limit" in values
        else None,
        "available_cpus": int(values["CPU_Available_CPUs"])
        if "CPU_Available_CPUs" in values
        else None,
    }


def parse_power(text: str) -> dict:
    source = re.search(r"Now drawing from '([^']+)'", text)
    if not source:
        return {"state": "unknown", "reason": "pmset did not report power source"}
    battery = re.search(r"(\d+)%;\s*([^;]+)", text)
    return {
        "state": "battery" if source[1] == "Battery Power" else "ac",
        "source": source[1],
        "battery_percent": int(battery[1]) if battery else None,
        "battery_status": battery[2].strip() if battery else None,
    }


def _run(args: list[str]) -> str:
    return subprocess.run(
        args, capture_output=True, text=True, check=True, timeout=2
    ).stdout


def snapshot() -> dict:
    global _CACHED, _EXPIRES
    with _LOCK:
        if _CACHED is not None and time.monotonic() < _EXPIRES:
            return dict(_CACHED)
        out = {"object": "yunshu.host", "sampled_at": time.time(), "cache_ttl_s": 15}
        for name, args, parser in (
            ("thermal", ["pmset", "-g", "therm"], parse_thermal),
            ("power", ["pmset", "-g", "batt"], parse_power),
        ):
            try:
                out[name] = parser(_run(args))
            except (OSError, subprocess.SubprocessError) as exc:
                out[name] = {"state": "unknown", "reason": str(exc)}
        try:
            import psutil

            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            out["memory"] = {
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "used_percent": memory.percent,
                "swap_used_bytes": swap.used,
                "swap_total_bytes": swap.total,
            }
        except Exception as exc:
            out["memory"] = {"state": "unknown", "reason": str(exc)}
        try:
            text = _run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
            level = int(text.strip())
            out["memory_pressure"] = {
                "state": {1: "normal", 2: "warning", 4: "critical"}.get(
                    level, "unknown"
                ),
                "level": level,
            }
            if level not in (1, 2, 4):
                out["memory_pressure"]["reason"] = "unrecognized OS pressure level"
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            out["memory_pressure"] = {"state": "unknown", "reason": str(exc)}
        _CACHED, _EXPIRES = out, time.monotonic() + 15
        return dict(out)
