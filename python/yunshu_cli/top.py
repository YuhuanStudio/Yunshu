"""Live status using the currently merged server monitoring API."""

import time

import typer
from rich.table import Table

from ._output import is_json
from .status import console, status


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
    url: str = typer.Option(
        "http://localhost:8000", "--url", "-u", envvar="YUNSHU_GATEWAY_URL"
    ),
    once: bool = typer.Option(False, "--once", help="Print one snapshot and exit."),
    interval: float = typer.Option(2.0, "--interval", min=0.5, help="Refresh seconds."),
) -> None:
    """Refresh server health, memory, engine and models. --json returns one snapshot."""
    while True:
        if not once and not is_json():
            console.clear()
        status(url=url)
        if once or is_json():
            return
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            return
