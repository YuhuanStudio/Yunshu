"""Run tfbench with synchronized prefill/restore/layout spans (diagnostic only).

Use the ordinary tfbench arguments. No engine setting or arithmetic is changed.
The server wrapper adds synchronization, so use plain tfbench for speed verdicts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def server_command(command):
    if command and Path(command[0]).name == "yunshu":
        return [sys.executable, str(Path(__file__).resolve()), "--serve", *command[1:]]
    return command


def install_profile():
    import mlx.core as mx
    from mlx_vlm.generate.ar import PromptProcessingBatch

    from yunshu_engine.apc_manager import _Coordinator
    from yunshu_engine.kernels.lane_linear import LaneLinear

    totals = {"stock_s": 0.0, "stock_calls": 0}

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
        mx.synchronize()
        start = self._processed_prompt_columns
        totals.update(stock_s=0.0, stock_calls=0)
        t = time.perf_counter()
        n = step(self)
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
    if sys.argv[1:2] == ["--serve"]:
        sys.argv.pop(1)
        install_profile()
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
