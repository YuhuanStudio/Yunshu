"""Run tfbench with synchronized prefill/restore/layout spans (diagnostic only).

Use the ordinary tfbench arguments. No engine setting or arithmetic is changed.
The server wrapper adds synchronization, so use plain tfbench for speed verdicts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def server_command(command, mode="--serve"):
    if command and Path(command[0]).name == "yunshu":
        return [sys.executable, str(Path(__file__).resolve()), mode, *command[1:]]
    return command


def measured_call(original, name, mx, emit, active):
    """Synchronize only prompt components; decode timing stays uninstrumented."""

    def measured(self, x, *args, **kwargs):
        if not active() or not 1 < x.shape[1] <= 2048:
            return original(self, x, *args, **kwargs)
        mx.synchronize()
        t = time.perf_counter()
        result = original(self, x, *args, **kwargs)
        mx.eval(result)
        mx.synchronize()
        emit(
            span=name,
            rows=x.shape[1],
            module=id(self),
            elapsed_s=time.perf_counter() - t,
        )
        return result

    return measured


def install_profile():
    import mlx.core as mx
    from mlx_vlm.generate.ar import PromptProcessingBatch

    from yunshu_engine.apc_manager import _Coordinator
    from yunshu_engine.kernels.lane_linear import LaneLinear

    totals = {"stock_s": 0.0, "stock_calls": 0}
    layer_hooks = []
    state = {"prompt": False}

    def install_layer_hooks():
        # Install after engine startup patches have selected the served methods.
        if layer_hooks:
            return
        from mlx_vlm.models.qwen3_5 import language

        for name in ("Qwen3_5Attention", "Qwen3_5GatedDeltaNet", "Qwen3_5MLP"):
            cls = getattr(language, name)
            original = cls.__call__

            cls.__call__ = measured_call(
                original, name, mx, emit, lambda: state["prompt"]
            )
            layer_hooks.append(cls)

    def emit(**record):
        print("TURN2PROFILE " + json.dumps(record), flush=True)

    stock = LaneLinear.stock

    def timed_stock(self):
        t = time.perf_counter()
        result = stock(self)
        mx.eval(result)
        totals["stock_s"] += time.perf_counter() - t
        totals["stock_calls"] += 1
        return result

    LaneLinear.stock = timed_stock

    step = PromptProcessingBatch.prompt_step

    def timed_step(self):
        install_layer_hooks()
        mx.synchronize()
        start = self._processed_prompt_columns
        totals.update(stock_s=0.0, stock_calls=0)
        t = time.perf_counter()
        state["prompt"] = True
        try:
            n = step(self)
        finally:
            state["prompt"] = False
        mx.synchronize()
        emit(
            span="prompt_step",
            start=start,
            cached=list(self._cached_tokens_per_row),
            rows=n,
            elapsed_s=time.perf_counter() - t,
            **totals,
        )
        return n

    PromptProcessingBatch.prompt_step = timed_step

    generate = PromptProcessingBatch.generate

    def timed_generate(self, *args, **kwargs):
        mx.synchronize()
        t = time.perf_counter()
        result = generate(self, *args, **kwargs)
        mx.synchronize()
        emit(span="first_token", elapsed_s=time.perf_counter() - t)
        return result

    PromptProcessingBatch.generate = timed_generate

    lookup = _Coordinator.lookup

    def timed_lookup(self, ids, **kwargs):
        mx.synchronize()
        t = time.perf_counter()
        result = lookup(self, ids, **kwargs)
        mx.synchronize()
        emit(
            span="apc_restore",
            tokens=len(ids),
            cached=result.get("prefix_len", 0) if result else 0,
            elapsed_s=time.perf_counter() - t,
        )
        return result

    _Coordinator.lookup = timed_lookup


def main():
    if sys.argv[1:2] in (["--serve"], ["--serve-tail"]):
        mode = sys.argv.pop(1)
        if mode == "--serve":
            install_profile()
        else:
            from yunshu_engine.kernels import nax_prefill

            original = nax_prefill.tail_matmul
            seen = set()

            def tail(x, weight, sbt, *, bits):
                rows = x.shape[0]
                if 512 < rows < 640 and rows not in seen:
                    seen.add(rows)
                    print("TURN2TAIL " + json.dumps(dict(rows=rows)), flush=True)
                return original(x, weight, sbt, bits=bits)

            nax_prefill.tail_matmul = tail
        from yunshu_cli import main as cli

        cli()
        return
    import subprocess

    import tfbench

    original = subprocess.Popen

    def launch(command, *args, **kwargs):
        return original(server_command(command), *args, **kwargs)

    subprocess.Popen = launch
    tfbench.main()


if __name__ == "__main__":
    main()
