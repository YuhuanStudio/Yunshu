"""Benchmark-only raw token receipt; activated by CSPEC_TOKEN_TRACE."""

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

if os.environ.get("CSPEC_TOKEN_TRACE"):
    from yunshu_engine.vlm_batch_runner import VLMBatchRunner

    _original = VLMBatchRunner.iter_tokens
    _trace = Path(os.environ["CSPEC_TOKEN_TRACE"])
    _lock = threading.Lock()

    def _capture(self, input_ids, **kwargs):
        tokens = []
        prompt = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        disabled = os.environ.get("CSPEC_DISABLE_DRAFT") == "1"
        if (
            os.environ.get("CSPEC_DISABLE_PLAIN") == "1"
            and kwargs.get("guide") is None
            and not kwargs.get("logprobs")
        ):
            disabled = True
        if disabled:
            kwargs["allow_draft"] = False
            logging.getLogger("yunshu_engine.vlm_batch_runner").info(
                "Speculation disabled for parity: allow_draft=False"
            )
        try:
            for token in _original(self, input_ids, **kwargs):
                tokens.append(int(token))
                yield token
        finally:
            record = {
                "prompt_digest": hashlib.sha256(
                    json.dumps(prompt).encode()
                ).hexdigest(),
                "token_digest": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                "token_ids": tokens,
            }
            with _lock, _trace.open("a") as out:
                out.write(json.dumps(record) + "\n")

    VLMBatchRunner.iter_tokens = _capture

    if os.environ.get("CSPEC_CACHE_STATE_CHECK") == "1":
        import mlx.core as mx
        import numpy as np

        _step = VLMBatchRunner._step_generator

        def _cache_capture(self, group):
            _step(self, group)
            batch = getattr(group.gen, "_generation_batch", None)
            if batch is None or getattr(group.gen, "_prompt_batch", None) is not None:
                return
            for job in group.jobs.values():
                if job.stats.generated or job.uid not in getattr(batch, "uids", []):
                    continue
                digest = hashlib.sha256()
                details = []
                for index, cache in enumerate(batch.prompt_cache):
                    if cache is None:
                        continue
                    if hasattr(cache, "keys") and cache.keys is not None:
                        arrays = [
                            cache.keys[:, :, : len(job.ids)],
                            cache.values[:, :, : len(job.ids)],
                        ]
                    else:
                        state = getattr(cache, "state", [])
                        arrays = state if isinstance(state, (list, tuple)) else [state]
                    for value in arrays:
                        if not isinstance(value, mx.array):
                            continue
                        digest.update(
                            json.dumps(
                                [index, list(value.shape), str(value.dtype)]
                            ).encode()
                        )
                        # Preserve bits; numpy cannot represent bfloat16 directly.
                        bits = (
                            value.view(mx.uint16)
                            if value.dtype == mx.bfloat16
                            else value
                        )
                        payload = np.asarray(bits).tobytes()
                        digest.update(payload)
                        details.append(
                            [
                                index,
                                type(cache).__name__,
                                list(value.shape),
                                str(value.dtype),
                                hashlib.sha256(payload).hexdigest(),
                            ]
                        )
                with _lock, Path(str(_trace) + ".states").open("a") as out:
                    out.write(
                        json.dumps({"digest": digest.hexdigest(), "details": details})
                        + "\n"
                    )

        VLMBatchRunner._step_generator = _cache_capture

        # Low-volume numerical diagnostic: actual prefill cuts and cache masks.
        from mlx_vlm.models.qwen3_5.language import LanguageModel

        _model_call = LanguageModel.__call__

        def _model_trace(self, inputs, *args, **kwargs):
            cache = kwargs.get("cache") or []
            if not kwargs.get("speculative_verify"):

                def scalar(x):
                    return x.tolist() if isinstance(x, mx.array) else x

                from mlx_vlm.generate import ar
                from yunshu_engine.constrained_spec import current_request

                note = {
                    "request_present": current_request() is not None,
                    "ctor": ar.PromptProcessingBatch.__init__.__code__.co_filename,
                    "selected_ctor": ar._generate_module_override(
                        "PromptProcessingBatch", ar.PromptProcessingBatch
                    ).__init__.__code__.co_filename,
                    "shape": list(inputs.shape),
                    "n": kwargs.get("n_to_process"),
                    "hidden": kwargs.get("return_hidden"),
                    "shared": kwargs.get("return_shared_kv"),
                    "capture": kwargs.get("capture_layer_ids"),
                    "cache": [
                        [
                            type(c).__name__,
                            scalar(getattr(c, "left_padding", None)),
                            scalar(getattr(c, "lengths", None)),
                            scalar(getattr(c, "offset", None)),
                        ]
                        for c in cache[:4]
                    ],
                }
                with Path(str(_trace) + ".calls").open("a") as out:
                    out.write(json.dumps(note) + "\n")
            return _model_call(self, inputs, *args, **kwargs)

        LanguageModel.__call__ = _model_trace
