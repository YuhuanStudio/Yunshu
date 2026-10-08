"""Live status using the currently merged server monitoring API."""

import time

import typer

from ._output import is_json
from .status import console, status


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
