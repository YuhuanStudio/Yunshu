"""Research-only direct-capacity restore A/B and synchronized phase profile.

--profile adds barriers and eager target-logit evaluation: use it to attribute
work, never as serving TTFT evidence. Unprofiled arms retain coarse timers and
allocator counters; confirm a winning arm through the ordinary HTTP launcher.
"""

import argparse
import asyncio
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def direct_clone(c, *, min_capacity_tokens, eval_targets, clone):
    import mlx.core as mx
    from mlx_vlm.models.cache import KVCache

    if (
        type(c) is not KVCache
        or c.keys is None
        or c.values is None
        or min_capacity_tokens is None
        or min_capacity_tokens <= c.offset
    ):
        return clone(
            c, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
        )
    capacity = max(c.offset, int(min_capacity_tokens))
    if c.step > 0:
        capacity = ((capacity + c.step - 1) // c.step) * c.step
    out = KVCache()
    out.offset = c.offset
    out.meta_state = c.meta_state
    arrays = []
    for source in (c.keys, c.values):
        target = mx.zeros(
            (*source.shape[:2], capacity, source.shape[3]), dtype=source.dtype
        )
        target[..., : c.offset, :] = source[..., : c.offset, :]
        arrays.append(target)
    out.keys, out.values = arrays
    eval_targets.extend(arrays)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "shipped",
            "async",
            "spans",
            "cow",
            "cowasync",
            "barrier",
            "paired32",
            "baseline",
            "direct",
            "keep",
            "wide",
            "native",
            "pool",
            "nativepool",
            "combo",
        ),
        default=["baseline", "direct", "keep"],
    )
    a = ap.parse_args()
    modern = {"shipped", "async", "cow", "cowasync", "spans", "barrier", "paired32"}
    if set(a.modes) & modern and set(a.modes) - modern:
        ap.error("shipped modes must not be mixed with legacy experiment modes")
    if a.dry_run:
        print(
            json.dumps(
                dict(phase="complete", success=True, dry_run=True, modes=a.modes)
            )
        )
        return

    from yunshu_engine import settings
    from yunshu_engine.apc_manager import _Coordinator
    from yunshu_engine.vlm_batch_runner import RunStats
    from yunshu_engine.vlm_engine import VLMEngine

    if settings.get_bool("YUNSHU_VLM_APC_DISK"):
        raise RuntimeError("restore A/B requires explicit RAM-only APC for every model")
    if "27B" in a.model and settings.get_str("YUNSHU_VLM_DRAFT") != "mtp":
        raise RuntimeError("27B probe requires explicit MTP")
    engine = VLMEngine(a.model)
    asyncio.run(engine.start())
    runner = engine._batch_runner
    import mlx.core as mx
    from mlx_vlm import apc_adapters

    from yunshu_engine.kernels import buffer_cache

    span_counts = {}

    def uninstall_spans():
        pass

    if "spans" in a.modes:
        from span_forward import install as install_spans

        span_counts, uninstall_spans = install_spans()
    cow_counts = {}

    def uninstall_cow():
        pass

    if set(a.modes) & {"cow", "cowasync"}:
        from cow_restore import install as install_cow

        cow_counts, uninstall_cow = install_cow()
    if set(a.modes) & {"async", "cowasync"}:
        from async_restore import install

        async_counts, uninstall_async = install()
    else:
        async_counts = {}

        def uninstall_async():
            pass

    kernel_uninstall = None
    clone = apc_adapters.clone_cache_entry
    from mlx_vlm.models.qwen3_5.language import Qwen3_5Model
    from native_model_cache import wrap as native_wrap

    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.kernels.tensorfold import lane_qmm

    original_model_forward = Qwen3_5Model.__call__
    shipped_native_trace = original_model_forward
    while hasattr(shipped_native_trace, "_span_forward_original"):
        shipped_native_trace = shipped_native_trace._span_forward_original
    native_model_forward = native_wrap(original_model_forward)
    clear = buffer_cache.clear_if_over
    piece, max_rows = lane_linear.PIECE, lane_qmm.MAX_ROWS
    original_pool_limit = mx.set_cache_limit(buffer_cache._STATE["limit"])
    mx.set_cache_limit(original_pool_limit)

    def direct(c, **kwargs):
        return direct_clone(c, clone=clone, **kwargs)

    flush = _Coordinator.flush_deferred_checkpoints
    merge = _Coordinator.merge_rows
    events, original = [], []
    phase_stack = []
    for modname, symbol in [
        ("mlx_vlm.apc", "APCManager.lookup_exact_cache"),
        ("mlx_vlm.apc", "_clone_prompt_cache_for_apc"),
        ("yunshu_engine.kernels.lane_linear", "LaneLinear.stock"),
        ("yunshu_engine.apc_manager", "_Coordinator.merge_rows"),
        ("mlx_vlm.generate.ar", "PromptProcessingBatch.prompt_step"),
        ("mlx_vlm.generate.ar", "PromptProcessingBatch.generate"),
        ("yunshu_engine.vlm_batch_runner", "VLMBatchRunner._admit"),
        ("yunshu_engine.vlm_batch_runner", "VLMBatchRunner._step_generator"),
    ]:
        if symbol == "LaneLinear.stock" and not a.profile:
            continue
        obj = importlib.import_module(modname)
        *parents, name = symbol.split(".")
        for parent in parents:
            obj = getattr(obj, parent)
        fn = getattr(obj, name)

        def wrap(*args, _fn=fn, _name=symbol, **kwargs):
            if a.profile:
                mx.synchronize()
            start = time.perf_counter()
            begin = start
            active, pool = mx.get_active_memory(), mx.get_cache_memory()
            details = {}
            if _name == "_clone_prompt_cache_for_apc":
                from mlx_vlm.apc import _cache_nbytes

                if a.profile:
                    mx.reset_peak_memory()
                details["source_bytes"] = _cache_nbytes(args[0])
                details["min_capacity_tokens"] = kwargs.get("min_capacity_tokens")
            phase_stack.append(_name)
            try:
                result = _fn(*args, **kwargs)
                if _name == "_clone_prompt_cache_for_apc" and result:
                    details["restored_bytes"] = _cache_nbytes(result)
                    details["kv_layouts"] = [
                        dict(
                            keys=list(c.keys.shape),
                            values=list(c.values.shape),
                            dtype=str(c.keys.dtype),
                            offset=int(c.offset),
                        )
                        for c in result
                        if getattr(c, "keys", None) is not None
                    ]
                return result
            finally:
                phase_stack.pop()
                if a.profile:
                    mx.synchronize()
                    if _name == "_clone_prompt_cache_for_apc":
                        details["peak_active_bytes"] = mx.get_peak_memory()
                events.append(
                    dict(
                        name=_name,
                        begin=begin,
                        end=time.perf_counter(),
                        ms=1000 * (time.perf_counter() - start),
                        active_before=active,
                        pool_before=pool,
                        active_after=mx.get_active_memory(),
                        pool_after=mx.get_cache_memory(),
                        **details,
                    )
                )

        original.append((obj, name, fn))
        setattr(obj, name, wrap)
    if a.profile:
        model_class = type(engine._model.language_model)
        forward = model_class.__call__

        def profiled_forward(self, *args, **kwargs):
            phase = phase_stack[-1] if phase_stack else None
            if phase not in (
                "PromptProcessingBatch.prompt_step",
                "PromptProcessingBatch.generate",
            ):
                return forward(self, *args, **kwargs)
            mx.synchronize()
            begin = time.perf_counter()
            output = forward(self, *args, **kwargs)
            targets = [c.state for c in kwargs.get("cache", [])]
            # prompt_step discards logits; forcing them executes an otherwise
            # unused full-span LM head and corrupts the prefill attribution.
            if "n_to_process" not in kwargs:
                targets.append(output.logits if hasattr(output, "logits") else output)
            mx.eval(targets)
            mx.synchronize()
            events.append(
                dict(
                    name="target_forward",
                    phase=phase,
                    begin=begin,
                    end=time.perf_counter(),
                    ms=1000 * (time.perf_counter() - begin),
                    rows=int(args[0].shape[-1]),
                )
            )
            return output

        original.append((model_class, "__call__", forward))
        model_class.__call__ = profiled_forward
    if a.profile:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5DecoderLayer

        layer_forward = Qwen3_5DecoderLayer.__call__

        def profiled_layer(self, x, *args, **kwargs):
            phase = phase_stack[-1] if phase_stack else None
            if phase not in (
                "PromptProcessingBatch.prompt_step",
                "PromptProcessingBatch.generate",
            ):
                return layer_forward(self, x, *args, **kwargs)
            mx.synchronize()
            begin = time.perf_counter()
            result = layer_forward(self, x, *args, **kwargs)
            mx.eval(result)
            mx.synchronize()
            events.append(
                dict(
                    name="decoder_layer",
                    begin=begin,
                    end=time.perf_counter(),
                    phase=phase,
                    kind="gdn" if self.is_linear else "fa",
                    rows=int(x.shape[-2]),
                    ms=1000 * (time.perf_counter() - begin),
                )
            )
            return result

        original.append((Qwen3_5DecoderLayer, "__call__", layer_forward))
        Qwen3_5DecoderLayer.__call__ = profiled_layer
    a.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = (
            Path("/Volumes/P5Plus/yunshu-build/tfnew/prompts")
            / f"prose-{max(8192, a.ctx)}.txt"
        ).read_text()
        ids, _, salt = engine._executor.submit(
            engine._runner_input,
            [{"role": "user", "content": text}],
            [],
            [],
            False,
            {"enable_thinking": False},
        ).result()
        if a.ctx < 8192:
            ids = ids[: a.ctx]
        chat_ids = None
        if a.chat:
            reply = json.loads(
                (
                    Path("/Volumes/P5Plus/yunshu-build/codex/prefill2-http-final-0055")
                    / f"turn2-reply-prose-{a.ctx}.json"
                ).read_text()
            )["reply"]
            chat_ids, _, _ = engine._executor.submit(
                engine._runner_input,
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": "Continue with the next part, at the same length.",
                    },
                ],
                [],
                [],
                False,
                {"enable_thinking": False},
            ).result()
        with a.out.open("w") as out:
            out.write(
                json.dumps(
                    dict(
                        phase="metadata",
                        model=a.model,
                        profiled=a.profile,
                        native_restore_installed=bool(
                            getattr(
                                apc_adapters.clone_cache_entry,
                                "_yunshu_restore_views",
                                False,
                            )
                        ),
                        source_sha256=hashlib.sha256(
                            Path(__file__).read_bytes()
                        ).hexdigest(),
                        direct_source_sha256=hashlib.sha256(
                            Path(direct_clone.__code__.co_filename).read_bytes()
                        ).hexdigest(),
                    )
                )
                + "\n"
            )
            out.flush()

            def run(tokens, label, mode, rep, n=16):
                events.clear()
                native_before = native_model_forward.native_calls
                shipped_native_before = getattr(
                    original_model_forward, "native_calls", 0
                )
                stats = RunStats()
                start = time.perf_counter()
                first, generated = None, []
                for tok in runner.iter_tokens(
                    tokens,
                    max_tokens=n,
                    temperature=0,
                    seed=1234,
                    apc_semantic_hash=salt,
                    stats=stats,
                ):
                    if first is None:
                        first_at = time.perf_counter()
                        first = first_at - start
                    generated.append(tok)
                if not generated or not stats.finish_reason:
                    raise RuntimeError("incomplete fixed-cost request")
                if label == "prime" and stats.cached_tokens:
                    raise RuntimeError("prime unexpectedly restored a checkpoint")
                if (
                    label.startswith("revisit-")
                    and stats.cached_tokens != len(tokens) - 1
                ):
                    raise RuntimeError(
                        "revisit did not restore the expected full prefix"
                    )
                if label == "chat-turn2" and stats.cached_tokens != len(ids) - 1:
                    raise RuntimeError(
                        "chat did not restore the original prompt checkpoint"
                    )
                if "27B" in a.model and (
                    not stats.used_draft or stats.spec_mode != "mtp"
                ):
                    raise RuntimeError("fixed-cost probe did not engage the drafter")
                # DONE can reach the consumer before its executor slice finishes.
                # Settle deferred publication / cleanup before changing APC or
                # attributing the next request's component events.
                engine._executor.submit(lambda: None).result()
                row = dict(
                    label=label,
                    mode=mode,
                    rep=rep,
                    ctx=a.ctx,
                    ttft_ms=first * 1000,
                    cached=stats.cached_tokens,
                    fresh=len(tokens) - stats.cached_tokens,
                    digest=hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
                    used_draft=stats.used_draft,
                    spec_mode=stats.spec_mode,
                    events=list(events),
                    profiled=a.profile,
                    async_restore_counts=dict(async_counts),
                    cow_restore_counts=dict(cow_counts),
                    span_forward_counts=dict(span_counts),
                    shipped_native_forward_calls=getattr(
                        original_model_forward, "native_calls", 0
                    )
                    - shipped_native_before,
                    shipped_native_installed=bool(
                        getattr(
                            shipped_native_trace, "_yunshu_singleton_capacity", False
                        )
                    ),
                    native_forward_calls=native_model_forward.native_calls
                    - native_before,
                    runner_first_ms=stats.first_token_s * 1000,
                    first_at=first_at,
                )
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps(row), flush=True)

            run(ids[:16], "compile", "deferred", -1)
            for rep in range(a.reps):
                for mode in a.modes if rep % 2 == 0 else list(reversed(a.modes)):
                    if span_counts:
                        span_counts["enabled"] = mode == "spans"
                    if cow_counts:
                        cow_counts["enabled"] = mode in ("cow", "cowasync")
                    if kernel_uninstall:
                        kernel_uninstall()
                        kernel_uninstall = None
                    if mode == "barrier":
                        from lane_final_barrier import install as install_kernel

                        kernel_uninstall = install_kernel()
                    elif mode == "paired32":
                        from lane_paired32 import install as install_kernel

                        kernel_uninstall = install_kernel()
                    if async_counts:
                        async_counts["enabled"] = mode in ("async", "cowasync")
                    if mode not in (
                        "shipped",
                        "async",
                        "cow",
                        "cowasync",
                        "spans",
                        "barrier",
                        "paired32",
                    ):
                        Qwen3_5Model.__call__ = (
                            native_model_forward
                            if mode in ("native", "nativepool", "combo")
                            else original_model_forward
                        )
                        apc_adapters.clone_cache_entry = (
                            direct if mode in ("direct", "keep") else clone
                        )
                        lane_qmm.MAX_ROWS = (
                            512 if mode in ("wide", "combo") else max_rows
                        )
                        lane_linear.PIECE = 512 if mode in ("wide", "combo") else piece
                        mx.set_cache_limit(
                            buffer_cache._STATE["limit"]
                            if mode in ("keep", "pool", "nativepool", "combo")
                            else original_pool_limit
                        )
                        buffer_cache.clear_if_over = (
                            (lambda: None)
                            if mode in ("keep", "pool", "nativepool", "combo")
                            else clear
                        )
                    runner.apc_manager.clear()
                    run(ids, "prime", mode, rep)
                    for revisit in range(3):
                        run(ids, f"revisit-{revisit}", mode, rep)
                    if chat_ids is not None:
                        run(chat_ids, "chat-turn2", mode, rep)
                    else:
                        # A chat or this synthetic suffix supersedes the original
                        # hybrid checkpoint. Measure only one branch per prime.
                        run(ids[:-1] + ids[-65:] + ids[-1:], "suffix-65", mode, rep)
            out.write(json.dumps(dict(phase="complete", success=True)) + "\n")
    finally:
        Qwen3_5Model.__call__ = original_model_forward
        apc_adapters.clone_cache_entry = clone
        buffer_cache.clear_if_over = clear
        lane_linear.PIECE, lane_qmm.MAX_ROWS = piece, max_rows
        mx.set_cache_limit(original_pool_limit)
        _Coordinator.flush_deferred_checkpoints = flush
        _Coordinator.merge_rows = merge
        for obj, name, fn in original:
            setattr(obj, name, fn)
        if kernel_uninstall:
            kernel_uninstall()
        uninstall_async()
        uninstall_cow()
        uninstall_spans()
        asyncio.run(engine.stop())


if __name__ == "__main__":
    main()
