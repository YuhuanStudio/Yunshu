"""Attribute the real server prompt path, never replay isolated operator inputs.

Uses synchronized, nested per-layer spans. This deliberately perturbs scheduling;
compare its prompt wall with plain tfbench before interpreting absolute costs.
No model inputs, outputs, or arithmetic are changed. Aggregates contain shapes only.
"""

from __future__ import annotations

import collections
import functools
import json
import sys
import time
from pathlib import Path


def exclusive(elapsed, children):
    return max(0.0, elapsed - children)


def projection_key(module, x):
    rows = 1
    for n in x.shape[:-1]:
        rows *= n
    return f"gemm:M{rows}:K{module.input_dims}:N{module.output_dims}:b{module.bits}"


def install(*, plain=False):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_vlm.generate.ar import PromptProcessingBatch

    from yunshu_engine.kernels.lane_linear import LaneLinear

    active = False
    installed = False
    stack = []
    totals = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])

    def evaluate(value):
        arrays = [v for _, v in tree_flatten(value) if isinstance(v, mx.array)]
        if arrays:
            mx.eval(*arrays)
        mx.synchronize()

    def tensor_bytes(value):
        return sum(v.nbytes for _, v in tree_flatten(value) if isinstance(v, mx.array))

    def wrap(fn, label):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            if not active:
                return fn(*args, **kwargs)
            # Resolve producers before the span; nested spans remain exclusive.
            try:
                evaluate(
                    (tuple(a for a in args if not isinstance(a, nn.Module)), kwargs)
                )
            except ValueError:  # traced inside a compiled function: run unprofiled
                return fn(*args, **kwargs)
            key = label(*args, **kwargs) if callable(label) else label
            frame = [0.0]
            stack.append(frame)
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                evaluate(result)
            finally:
                elapsed = time.perf_counter() - start
                stack.pop()
                if stack:
                    stack[-1][0] += elapsed
                row = totals[key]
                row[0] += exclusive(elapsed, frame[0])
                row[1] += 1
                row[2] += elapsed
                if "result" in locals():
                    row[3] += tensor_bytes(
                        (
                            tuple(a for a in args if not isinstance(a, nn.Module)),
                            kwargs,
                            result,
                        )
                    )
            return result

        return call

    def hooks():
        nonlocal installed
        if installed:
            return
        installed = True
        from mlx_vlm.models.qwen3_5 import language as lang

        from yunshu_engine.kernels import attn_prefill, gdn_prefill

        LaneLinear.__call__ = wrap(LaneLinear.__call__, projection_key)
        mx.fast.rms_norm = wrap(mx.fast.rms_norm, "norm.rms")
        lang.swiglu = wrap(lang.swiglu, "elementwise.swiglu")
        for name in ("contiguous", "concatenate", "pad"):
            setattr(mx, name, wrap(getattr(mx, name), "copy." + name))
        nn.Conv1d.__call__ = wrap(nn.Conv1d.__call__, "gdn.conv")
        lang.Qwen3_5RMSNormGated.__call__ = wrap(
            lang.Qwen3_5RMSNormGated.__call__, "gdn.gated_norm.remainder"
        )
        for name in ("Qwen3_5Attention", "Qwen3_5GatedDeltaNet", "Qwen3_5MLP"):
            cls = getattr(lang, name)
            cls.__call__ = wrap(cls.__call__, name + ".remainder")
        attn_prefill.attend = wrap(
            attn_prefill.attend,
            lambda q, k, v: (
                f"attention:M{q.shape[2]}:K{k.shape[2]}:H{q.shape[1]}:D{q.shape[3]}"
            ),
        )
        gdn_prefill.chunked_padded = wrap(gdn_prefill.chunked_padded, "gdn.core")

    import ctypes

    metal = ctypes.CDLL("/Volumes/P5Plus/yunshu-build/metalrt/libmetalrt_trace2.dylib")
    metal.metalrt_end.argtypes = [ctypes.c_char_p, ctypes.c_int]
    original = PromptProcessingBatch.prompt_step

    def step(self):
        nonlocal active
        if not plain:
            hooks()
        mx.synchronize()
        totals.clear()
        metal.metalrt_begin()
        start = time.perf_counter()
        processed = self._processed_prompt_columns
        active = not plain
        try:
            result = original(self)
            mx.synchronize()
        finally:
            active = False
        wall = time.perf_counter() - start
        buffer = ctypes.create_string_buffer(16384)
        size = metal.metalrt_end(buffer, len(buffer))
        if size < 0 or size >= len(buffer):
            raise RuntimeError("Metal timeline overflow")
        timeline = json.loads(buffer.value)
        print(
            "COLDTTFT_PROFILE "
            + json.dumps(
                dict(
                    start=processed,
                    rows=result,
                    cached=list(self._cached_tokens_per_row),
                    wall_s=wall,
                    attributed_s=sum(v[0] for v in totals.values()),
                    spans={
                        k: dict(
                            exclusive_s=v[0],
                            count=v[1],
                            inclusive_s=v[2],
                            minimum_tensor_bytes=v[3],
                        )
                        for k, v in totals.items()
                    },
                    timeline=timeline,
                    method="plain-real-prompt"
                    if plain
                    else "synchronized-real-prompt-nested-spans",
                )
            ),
            flush=True,
        )
        return result

    PromptProcessingBatch.prompt_step = step
    gen_original = PromptProcessingBatch.generate

    def gen(self, *args, **kwargs):
        nonlocal active
        if not plain:
            hooks()
        mx.synchronize()
        totals.clear()
        metal.metalrt_begin()
        start = time.perf_counter()
        active = not plain
        host_s = 0.0
        try:
            result = gen_original(self, *args, **kwargs)
            host_s = time.perf_counter() - start
            mx.synchronize()
        finally:
            active = False
        tbuf = ctypes.create_string_buffer(16384)
        tsize = metal.metalrt_end(tbuf, len(tbuf))
        gen_timeline = json.loads(tbuf.value) if 0 <= tsize < len(tbuf) else {}
        gen_timeline["host_return_ms"] = host_s * 1e3
        print(
            "COLDTTFT_PROFILE "
            + json.dumps(
                dict(
                    start=-1,
                    rows=1,
                    cached=list(self._cached_tokens_per_row),
                    wall_s=time.perf_counter() - start,
                    attributed_s=sum(v[0] for v in totals.values()),
                    spans={
                        k: dict(
                            exclusive_s=v[0],
                            count=v[1],
                            inclusive_s=v[2],
                            minimum_tensor_bytes=v[3],
                        )
                        for k, v in totals.items()
                    },
                    timeline=gen_timeline,
                    method="generate-final-row",
                )
            ),
            flush=True,
        )
        return result

    PromptProcessingBatch.generate = gen


def main():
    if sys.argv[1:2] in (["--serve"], ["--serve-plain"]):
        mode = sys.argv.pop(1)
        install(plain=mode == "--serve-plain")
        from yunshu_cli import main as cli

        cli()
        return
    plain = "--plain" in sys.argv
    if plain:
        sys.argv.remove("--plain")
    import subprocess

    import tfbench

    popen = subprocess.Popen

    def launch(command, *args, **kwargs):
        if command and Path(command[0]).name == "yunshu":
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--serve-plain" if plain else "--serve",
                *command[1:],
            ]
        return popen(command, *args, **kwargs)

    subprocess.Popen = launch
    tfbench.main()


if __name__ == "__main__":
    main()
