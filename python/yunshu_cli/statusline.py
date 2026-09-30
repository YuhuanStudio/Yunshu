"""``yunshu statusline``: a one-line live view of the local engine for coding-agent status lines.

Claude Code runs a ``statusLine`` command on every refresh, hands it a JSON session object on stdin and
shows the first line it prints. A hosted API has nothing to say about the engine behind it; a local one
does: whether the request in flight is still prefilling (and how far along), the decode speed, and how much
of the last prompt came from the prefix cache. The numbers come from ``GET /yunshu/status`` (the same live
state the ``: yunshu-progress`` SSE comments carry); nothing is invented, and an unreachable server prints a
short marker instead of failing (a status line must never break the agent).
"""

from __future__ import annotations

import json
import sys
from typing import Any

import typer

statusline_app = typer.Typer(
    help="One-line engine status for coding-agent status lines."
)


def _k(n: float | int | None) -> str:
    if n is None:
        return "?"
    n = float(n)
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{int(n)}"


def _secs(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.0f}s" if v >= 10 else f"{v:.1f}s"


def render(status: dict[str, Any] | None, session: dict[str, Any] | None = None) -> str:
    """Format the status line from a ``/yunshu/status`` payload and (optionally) the agent's session JSON."""
    session = session or {}
    model = (session.get("model") or {}).get("display_name") or ""
    parts: list[str] = [model or "yunshu"]
    if status is None:
        return " | ".join([*parts, "engine unreachable"])
    reqs = (status.get("requests") or {}).get("items") or []
    live = next(
        (r for r in reqs if r.get("phase") in ("prefill", "decode", "queued")), None
    )
    if live is None and reqs:
        live = reqs[0]
    if live is not None:
        phase = live.get("phase")
        if phase == "prefill":
            seg = (
                f"prefill {live.get('percent', 0):.0f}% "
                f"{_k(live.get('processed_tokens'))}/{_k(live.get('prompt_tokens'))}"
            )
            if live.get("tokens_per_second"):
                seg += f" @{_k(live['tokens_per_second'])}/s"
            if live.get("eta_s") is not None:
                seg += f" eta {_secs(live['eta_s'])}"
            if live.get("cached_tokens"):
                seg += f" (cache {_k(live['cached_tokens'])})"
            parts.append(seg)
        elif phase == "queued":
            seg = f"queued #{live.get('queue_position', 0)}"
            if live.get("queue_est_wait_ms"):
                seg += f" ~{_secs(live['queue_est_wait_ms'] / 1000)}"
            parts.append(seg)
        elif phase == "decode":
            tps = (status.get("throughput") or {}).get("live_decode_tps")
            parts.append(f"decode {tps:.0f} tok/s" if tps else "decode")
        else:
            parts.append(str(phase))
    else:
        last = status.get("last")
        if last:
            bits = []
            if last.get("decode_tps"):
                bits.append(f"{last['decode_tps']:.0f} tok/s")
            p = last.get("prompt_tokens") or 0
            if p:
                bits.append(f"cache {100 * (last.get('cached_tokens') or 0) / p:.0f}%")
            if last.get("ttft_ms") is not None:
                bits.append(f"ttft {_secs(last['ttft_ms'] / 1000)}")
            if bits:
                parts.append("last " + " ".join(bits))
        else:
            parts.append("idle")
    pct = (session.get("context_window") or {}).get("used_percentage")
    if isinstance(pct, int | float):
        parts.append(f"ctx {pct:.0f}%")
    return " | ".join(parts)


def fetch_status(
    url: str, api_key: str = "", timeout: float = 1.5
) -> dict[str, Any] | None:
    import httpx

    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = httpx.get(
            url.rstrip("/") + "/v1/yunshu/status", headers=headers, timeout=timeout
        )
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001 - a status line never raises
        return None


def _read_session() -> dict[str, Any]:
    if sys.stdin is None or sys.stdin.isatty():
        return {}
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


@statusline_app.callback(invoke_without_command=True)
def statusline(
    url: str = typer.Option(
        "http://localhost:8000",
        "--url",
        "-u",
        envvar="YUNSHU_GATEWAY_URL",
        help="Server URL.",
    ),
    api_key: str = typer.Option("", "--api-key", "-k", envvar="YUNSHU_API_KEY"),
) -> None:
    """Print one status line (reads Claude Code's session JSON from stdin when piped)."""
    typer.echo(render(fetch_status(url, api_key), _read_session()))
