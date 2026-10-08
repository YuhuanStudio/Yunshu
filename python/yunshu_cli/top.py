"""Cached server host telemetry view; never starts a local sampler."""

from __future__ import annotations

import time

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table


def render(host: dict) -> Table:
    table = Table(title="Yunshu host telemetry")
    table.add_column("Metric")
    table.add_column("Value")
    telemetry = host.get("telemetry", {})
    for key, value in telemetry.get("watts", {}).items():
        table.add_row(
            f"{key.upper()} watts", "unknown" if value is None else f"{value:.2f} W"
        )
    gpu = telemetry.get("gpu", {})
    for key, unit, scale in (("frequency_mhz", "MHz", 1), ("active_ratio", "%", 100)):
        value = gpu.get(key)
        table.add_row(
            key, "unknown" if value is None else f"{value * scale:.1f} {unit}"
        )
    temp = telemetry.get("temperature", {})
    for key in ("die_max_c", "die_mean_c"):
        value = temp.get(key)
        table.add_row(key, "unknown" if value is None else f"{value:.1f} °C")
    for key in ("thermal", "memory_pressure"):
        table.add_row(key, host.get(key, {}).get("state", "unknown"))
    memory = host.get("memory", {})
    swap = memory.get("swap_used_bytes")
    table.add_row("Swap used", "unknown" if swap is None else f"{swap / 2**30:.2f} GiB")
    if telemetry.get("reason"):
        table.add_row("Telemetry", telemetry["reason"])
    return table


def top(
    ctx: typer.Context,
    once: bool = typer.Option(False, "--once"),
    interval: float = typer.Option(1.0, "--interval", min=0.1),
) -> None:
    """Watch GPU power, frequency, temperature and OS pressure (admin key)."""
    from ._output import auth_headers, emit, fail, is_json

    url = (ctx.obj or {}).get("url", "http://localhost:8000")

    def fetch():
        try:
            response = httpx.get(
                url + "/v1/yunshu/host", headers=auth_headers(), timeout=10
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            fail(str(exc), code=2)

    if once or is_json():
        host = fetch()
        if is_json():
            emit(host)
        else:
            Console().print(render(host))
        return
    try:
        with Live(render(fetch()), refresh_per_second=2) as live:
            while True:
                time.sleep(interval)
                live.update(render(fetch()))
    except KeyboardInterrupt:
        return
