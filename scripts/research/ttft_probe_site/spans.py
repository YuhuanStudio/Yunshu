"""Diagnostic call spans, with no MLX imports or extra GPU synchronization.

Profiling perturbs Python costs: use this to locate waits, never for A/B speed.
"""

import json
import threading
import time
from collections import deque

NAMES = {
    "_tokenize_with_cache": "template_tokenize",
    "_format_prompt": "template",
    "lookup_exact_cache": "apc_lookup_clone",
    "clone": "apc_clone",
    "prompt_step": "new_token_prefill",
    "generate": "first_round",
    "_sample": "sampling",
    "sample": "sampling",
    "_sample_with_positions": "sampling",
    "sample_target": "sampling",
}
C_NAMES = {"eval", "async_eval", "clear_cache", "synchronize"}


class CallSpans:
    def __init__(self, capacity=400000):
        self.rows = deque(maxlen=capacity)
        self.active = {}
        self.http_seen = set()
        self.sse_seen = set()

    def profile(self, frame, event, arg):
        tid = threading.get_ident()
        if event in ("call", "return"):
            name = frame.f_code.co_name
            if (
                event == "call"
                and name in ("run_asgi", "send")
                and "uvicorn" in frame.f_code.co_filename
            ):
                self.http_event(frame, name)
            if name not in NAMES or not NAMES[name]:
                return
            filename = frame.f_code.co_filename
            if "yunshu_engine" not in filename and "mlx_vlm" not in filename:
                return
            key = (tid, id(frame), "py")
            label = NAMES[name]
            detail = f"{filename}:{name}:{frame.f_code.co_firstlineno}"
        elif event in ("c_call", "c_return", "c_exception"):
            name = getattr(arg, "__name__", "")
            if name not in C_NAMES or not str(
                getattr(arg, "__module__", "")
            ).startswith("mlx"):
                return
            key = (tid, id(frame), name)
            label = "gpu_wait"
            detail = f"{frame.f_code.co_filename}:{frame.f_code.co_name}:{frame.f_lineno}:{name}"
        else:
            return
        if event in ("call", "c_call"):
            self.active[key] = (time.perf_counter(), label, detail)
        else:
            start = self.active.pop(key, None)
            if start is not None:
                t, label, detail = start
                self.rows.append(
                    (
                        t,
                        time.perf_counter(),
                        threading.current_thread().name,
                        label,
                        detail,
                    )
                )

    def http_event(self, frame, name):
        cycle = frame.f_locals.get("self")
        scope = getattr(cycle, "scope", {})
        if scope.get("path") != "/v1/chat/completions":
            return
        key = id(cycle)
        label = None
        if name == "run_asgi" and key not in self.http_seen:
            self.http_seen.add(key)
            label = "http_receive"
        if name == "send":
            msg = frame.f_locals.get("message", {})
            if msg.get("type") == "http.response.body":
                if key not in self.sse_seen and sse_has_token(msg.get("body", b"")):
                    self.sse_seen.add(key)
                    label = "http_first_sse"
                if not msg.get("more_body", False):
                    self.http_seen.discard(key)
                    self.sse_seen.discard(key)
        if label:
            now = time.perf_counter()
            self.rows.append(
                (
                    now,
                    now,
                    threading.current_thread().name,
                    label,
                    f"{frame.f_code.co_filename}:{name}",
                )
            )


def sse_has_token(body):
    for line in body.splitlines():
        if not line.startswith(b"data:"):
            continue
        try:
            payload = json.loads(line[5:].strip())
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            if (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("tool_calls")
            ):
                return True
    return False
