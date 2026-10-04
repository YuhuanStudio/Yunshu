"""Process-local HTTP first-visible-token tracing, never installed in production.

Call install(output, sync_layers=False) in a server launcher. The ordinary trace
adds no MLX evaluation. sync_layers is a diagnostic pass: per-layer barriers
attribute GPU time but perturb graph fusion and must not be called HTTP timing.
Sequential benchmark requests only; overlapping requests fail closed.
"""

import functools
import importlib
import json
import os
import threading
import time
from pathlib import Path


class Trace:
    def __init__(self, output, sync_layers=False):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.sync_layers = sync_layers
        self.events = []
        self.started = None
        self.first_visible = None
        self.lock = threading.Lock()

    def record(self, name, begin, **extra):
        with self.lock:
            if self.started is not None and self.first_visible is None:
                self.events.append(
                    dict(
                        name=name,
                        start_ms=(begin - self.started) * 1000,
                        duration_ms=(time.perf_counter() - begin) * 1000,
                        **extra,
                    )
                )

    def wrap(self, obj, name, label=None, details=None, sync=False):
        original = getattr(obj, name)

        @functools.wraps(original)
        def run(*args, **kwargs):
            active = self.started is not None and self.first_visible is None
            if not active:
                return original(*args, **kwargs)
            extra = details(*args, **kwargs) if details else {}
            if label == "tokenizer.encode":
                import traceback

                extra["caller"] = [
                    f"{f.name}:{f.lineno}" for f in traceback.extract_stack()[-7:-1]
                ]
            begin = time.perf_counter()
            if sync:
                import mlx.core as mx

                mx.eval(args[1])
                begin = time.perf_counter()
            result = original(*args, **kwargs)
            if sync:
                mx.eval(result)
            self.record(label or name, begin, **extra)
            return result

        setattr(obj, name, run)

    def begin(self):
        if self.started is not None:
            raise RuntimeError("HTTP trace requires sequential requests")
        self.events = []
        self.first_visible = None
        self.started = time.perf_counter()

    def finish(self):
        row = dict(
            phase="request_complete",
            success=self.first_visible is not None,
            synchronized_layers=self.sync_layers,
            first_visible_ms=self.first_visible,
            events=self.events,
        )
        with self.output.open("a") as output:
            output.write(json.dumps(row) + "\n")
        self.started = None
        return row


def install(output, sync_layers=False):
    from fastapi import FastAPI

    from yunshu_engine.vlm_engine import VLMEngine

    trace = Trace(output, sync_layers)
    original_asgi = FastAPI.__call__

    async def asgi(self, scope, receive, send):
        if scope.get("path") != "/v1/chat/completions":
            return await original_asgi(self, scope, receive, send)
        trace.begin()

        async def traced_send(message):
            begin = time.perf_counter()
            visible = False
            if message.get("type") == "http.response.body":
                for line in message.get("body", b"").splitlines():
                    if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                        event = json.loads(line[6:])
                        visible |= any(
                            c.get("delta", {}).get("content")
                            for c in event.get("choices", [])
                        )
            await send(message)
            trace.record("ASGI.send", begin, type=message["type"])
            if visible and trace.first_visible is None:
                trace.first_visible = (time.perf_counter() - trace.started) * 1000

        try:
            return await original_asgi(self, scope, receive, traced_send)
        finally:
            trace.finish()

    FastAPI.__call__ = asgi
    for name in (
        "_runner_input",
        "_tokenize_with_cache",
        "_format_prompt",
        "_resolve_prompt_cache_plan",
    ):
        trace.wrap(VLMEngine, name)

    # Install after load so wrappers see the actual shipped kernel/cache hooks.
    original_load = VLMEngine.load

    def load(self):
        original_load(self)
        tokenizer = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        # Wrap encode on the class: an instance attribute would disqualify the
        # production tokenizer prefix cache and change the measured path.
        trace.wrap(type(tokenizer), "encode", "tokenizer.encode")
        trace.wrap(tokenizer, "apply_chat_template", "tokenizer.apply_chat_template")
        detok = self._tokenizer.detokenizer
        trace.wrap(type(detok), "add_token", "detokenizer.add_token")
        targets = [
            ("yunshu_engine.vlm_batch_runner", "VLMBatchRunner", "_admit"),
            ("yunshu_engine.apc_manager", "_Coordinator", "lookup"),
            ("yunshu_engine.apc_manager", "_Coordinator", "merge_rows"),
            ("yunshu_engine.apc_manager", "_Coordinator", "flush_deferred_checkpoints"),
            ("mlx_vlm.generate.ar", "PromptProcessingBatch", "prompt_step"),
            ("mlx_vlm.generate.ar", "PromptProcessingBatch", "generate"),
        ]
        prof_dir = os.environ.get("YUNSHU_TRACE_LOOKUP_PROF") or (
            str(trace.output.with_name("lookup-prof"))
            if trace.output.with_name("lookup-prof.enable").exists()
            else None
        )
        if prof_dir:
            import cProfile
            import io
            import pstats

            from yunshu_engine.apc_manager import _Coordinator

            inner = _Coordinator.lookup
            counter = iter(range(10**6))

            def profiled(self, *args, **kwargs):
                prof = cProfile.Profile()
                try:
                    return prof.runcall(inner, self, *args, **kwargs)
                finally:
                    buf = io.StringIO()
                    pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(18)
                    Path(prof_dir).mkdir(parents=True, exist_ok=True)
                    name = f"lookup-{next(counter)}.txt"
                    (Path(prof_dir) / name).write_text(buf.getvalue())

            _Coordinator.lookup = profiled
        for module, cls, method in targets:
            trace.wrap(
                getattr(importlib.import_module(module), cls),
                method,
                cls + "." + method,
            )
        from yunshu_engine.vlm_batch_runner import VLMBatchRunner

        emit = VLMBatchRunner._emit

        def emitted(runner, job, item):
            begin = time.perf_counter()
            result = emit(runner, job, item)
            if isinstance(item, tuple):
                trace.record("raw_token.emit", begin, token=item[0])
            return result

        VLMBatchRunner._emit = emitted
        from mlx_vlm.models.qwen3_5 import language as q

        def layer_details(layer, x, *args, **kwargs):
            return dict(rows=int(x.shape[-2]), kind="gdn" if layer.is_linear else "fa")

        trace.wrap(
            q.Qwen3_5DecoderLayer, "__call__", "layer", layer_details, sync=sync_layers
        )
        trace.wrap(
            q.Qwen3_5Model,
            "__call__",
            "model",
            lambda model, x, *args, **kwargs: dict(rows=int(x.shape[-1])),
        )

    VLMEngine.load = load
    return trace
