"""Replay synthetic same-shape requests against base and candidate; compare metrics; lossless => tokens identical.

Fails closed: an incomplete run, a missing token-id list, a length mismatch or any differing token fails the
comparison. ``send`` is injected (HTTP in production, a fake in tests).
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from .shapes import Shape, synthetic_messages

REQUIRED = ("ids", "finish", "ttft_ms", "decode_tps")


class ReplayError(RuntimeError):
    pass


def run_arm(
    send: Callable[[dict], dict], shapes: list[Shape], seed: int = 1
) -> list[dict]:
    rows = []
    for i, s in enumerate(shapes):
        body = {
            "messages": synthetic_messages(s, seed + i),
            "max_tokens": s.completion_tokens,
            "temperature": 0,
        }
        r = send(body)
        missing = [k for k in REQUIRED if r.get(k) in (None, [])]
        if missing:
            raise ReplayError(
                f"incomplete run {i} (missing {missing}): measurement fails closed"
            )
        rows.append({**r, "bucket": s.bucket})
    return rows


def _med(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals) if vals else float("nan")


def compare(base: list[dict], cand: list[dict], lossless: bool = True) -> dict:
    if len(base) != len(cand) or not base:
        return {"ok": False, "reason": "unequal or empty runs", "identical": False}
    diff = [
        i
        for i, (b, c) in enumerate(zip(base, cand, strict=True))
        if b["ids"] != c["ids"] or b["finish"] != c["finish"]
    ]
    out = {
        "n": len(base),
        "identical": not diff,
        "first_diff_row": diff[0] if diff else None,
        "ttft_ms": {"base": _med(base, "ttft_ms"), "cand": _med(cand, "ttft_ms")},
        "decode_tps": {
            "base": _med(base, "decode_tps"),
            "cand": _med(cand, "decode_tps"),
        },
        "mem_gb": {"base": _med(base, "mem_gb"), "cand": _med(cand, "mem_gb")},
        "content_level": False,
    }
    out["ttft_delta_pct"] = 100 * (out["ttft_ms"]["cand"] / out["ttft_ms"]["base"] - 1)
    out["decode_delta_pct"] = 100 * (
        out["decode_tps"]["cand"] / out["decode_tps"]["base"] - 1
    )
    out["ok"] = not diff or not lossless
    out["reason"] = (
        "" if out["ok"] else f"lossless candidate changed tokens at row {diff[0]}"
    )
    return out


def http_sender(url: str, model: str, timeout: float = 600.0) -> Callable[[dict], dict]:
    """Streaming chat sender; needs the server started with YUNSHU_REPLAY_TOKEN_IDS=1 for exact token ids."""
    import httpx

    def send(body: dict) -> dict:
        t0, first, ids, finish, stats, done = (
            time.perf_counter(),
            None,
            [],
            None,
            {},
            False,
        )
        payload = {
            **body,
            "model": model,
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_usage": True},
        }
        with httpx.stream(
            "POST", url + "/v1/chat/completions", json=payload, timeout=timeout
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line == "data: [DONE]":
                    done = True
                elif line.startswith("data: "):
                    o = json.loads(line[6:])
                    if o.get("usage"):
                        stats = o["usage"].get("x_yunshu", stats) or stats
                    for c in o.get("choices", []):
                        d = c.get("delta", {})
                        if first is None and any(
                            d.get(k)
                            for k in ("content", "reasoning_content", "tool_calls")
                        ):
                            first = time.perf_counter()
                        for lp in (c.get("logprobs") or {}).get("content", []):
                            if "token_id" not in lp:
                                raise ReplayError("server did not return token ids")
                            ids.append(lp["token_id"])
                        finish = c.get("finish_reason") or finish
        if not done or first is None:
            raise ReplayError("incomplete stream")
        return {
            "ids": ids,
            "finish": finish,
            "ttft_ms": (first - t0) * 1000,
            "decode_tps": stats.get("decode_tps"),
            "mem_gb": stats.get("peak_memory_gb"),
        }

    return send


def write(path: Path, result: dict) -> None:
    path.write_text(json.dumps({**result, "complete": True}) + "\n")
