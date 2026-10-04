"""Research-only joint projection/MLP dispatch with unchanged mixer spans.

REJECTED as lossless: three 32K HTTP pairs diverged on 256-token outputs.
Short hidden/cache and 200-item gates passed but did not detect that failure.
Never install this experiment in serving; retained for failure diagnosis.
The preserve_descriptors variant also failed the 256-token HTTP digest gate
(prose turn-2 and code cold/warm), despite a 64-layer hidden/cache bit gate.

The scheduler joins a short non-stored boundary, but FA and GDN still execute
exactly the old spans. Quantized lane projections and MLPs share their dispatch;
the final one-token generate remains untouched.
"""

import contextvars
import logging

_INSTALLATION = None

_PLAN = contextvars.ContextVar("span_forward_plan", default=())
_SPANS = contextvars.ContextVar("span_forward_spans", default=())
_PROJECTED = contextvars.ContextVar("span_forward_projected", default=None)
_RANGE = contextvars.ContextVar("span_forward_range", default=None)


def joined_boundaries(points, prefix, protected, *, limit=512, tail=8):
    kept, removed, previous = [], [], prefix
    for index, point in enumerate(points):
        following = points[index + 1] if index + 1 < len(points) else point
        if point <= prefix:
            kept.append(point)
        elif (
            point not in protected
            and 0 < following - point <= tail
            and following - previous <= limit
        ):
            removed.append(point)
        else:
            kept.append(point)
            previous = point
    return kept, removed


def install(*, preserve_descriptors=False):
    global _INSTALLATION

    if _INSTALLATION is not None:
        if _INSTALLATION[0]["preserve_descriptors"] != preserve_descriptors:
            raise RuntimeError("uninstall the previous span experiment first")
        return _INSTALLATION
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache
    from mlx_vlm.models.qwen3_5 import language as q

    from yunshu_engine.kernels.lane_linear import LaneLinear

    counts = {
        "enabled": True,
        "joins": 0,
        "forwards": 0,
        "projections": 0,
        "preserve_descriptors": preserve_descriptors,
    }
    originals = []

    def patch(cls, name, fn):
        originals.append((cls, name, getattr(cls, name)))
        setattr(cls, name, fn)

    init = PromptProcessingBatch.__init__

    def initialize(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._joined_span_points = ()
        coordinator = getattr(self, "_apc_coordinator", None)
        if not (
            counts["enabled"]
            and coordinator
            and coordinator.is_checkpoint
            and coordinator.manager.prefill_stride
            and len(self.uids) == 1
            and not any(self._right_pad_per_row or [])
            and len(self._apc_meta) == 1
            and all(type(c) in (ArraysCache, BatchKVCache) for c in self.prompt_cache)
        ):
            return
        meta = self._apc_meta[0]
        if not meta or coordinator.request(meta["full_input_ids"]) is not None:
            return
        for cache in self.prompt_cache:
            if type(cache) is BatchKVCache and (
                cache.left_padding.size != 1
                or int(cache.left_padding.item()) != 0
                or cache._right_padding is not None
                or (cache.keys is not None and cache.keys.dtype != mx.bfloat16)
            ):
                return
        points = meta.get("checkpoint_lengths") or []
        kept, removed = joined_boundaries(
            points, int(meta.get("prefix_len", 0)), coordinator._store_lengths
        )
        if removed:
            meta["checkpoint_lengths"] = kept
            self._joined_span_points = tuple(removed)
            counts["joins"] += len(removed)

    patch(PromptProcessingBatch, "__init__", initialize)
    step = PromptProcessingBatch.prompt_step

    def prompt_step(self):
        token = _PLAN.set(getattr(self, "_joined_span_points", ()))
        try:
            return step(self)
        finally:
            _PLAN.reset(token)

    patch(PromptProcessingBatch, "prompt_step", prompt_step)
    model_forward = q.Qwen3_5Model.__call__

    def model(self, inputs, *args, **kwargs):
        cache = kwargs.get("cache")
        plan = _PLAN.get()
        spans = ()
        if plan and inputs.shape[0] == 1 and inputs.shape[1] <= 512 and cache:
            source = cache[self.fa_idx]
            if type(source) in (KVCache, BatchKVCache):
                offset = source._idx if type(source) is BatchKVCache else source.offset
                inner = [
                    point - offset
                    for point in plan
                    if offset < point < offset + inputs.shape[1]
                ]
                if inner:
                    bounds = [0, *inner, int(inputs.shape[1])]
                    spans = tuple(zip(bounds[:-1], bounds[1:], strict=True))
        token = _SPANS.set(spans)
        try:
            if spans:
                if preserve_descriptors and counts["forwards"] == 0:
                    logging.getLogger("yunshu_engine.research.span_forward").info(
                        "Original-descriptor projection interleave engaged: spans=%s",
                        spans,
                    )
                counts["forwards"] += 1
            return model_forward(self, inputs, *args, **kwargs)
        finally:
            _SPANS.reset(token)

    model._span_forward_original = model_forward
    model._yunshu_singleton_capacity = bool(
        getattr(model_forward, "_yunshu_singleton_capacity", False)
    )
    patch(q.Qwen3_5Model, "__call__", model)
    norm = nn.RMSNorm.__call__

    def rms_norm(self, x):
        spans = _SPANS.get()
        if spans and x.ndim == 3 and x.shape[1] == spans[-1][1]:
            return mx.concatenate(
                [norm(self, x[:, begin:end]) for begin, end in spans], axis=1
            )
        return norm(self, x)

    patch(nn.RMSNorm, "__call__", rms_norm)
    linear = LaneLinear.__call__

    def projected(self, x):
        values, region = _PROJECTED.get(), _RANGE.get()
        if values is not None and region is not None and id(self) in values:
            begin, end = region
            return values[id(self)][:, begin:end]
        return linear(self, x)

    patch(LaneLinear, "__call__", projected)

    def projections(inputs, modules):
        if inputs.dtype != mx.bfloat16 or not all(
            type(module) is LaneLinear for module in modules
        ):
            return {}
        counts["projections"] += len(modules)
        if preserve_descriptors:
            values = {}
            for module in modules:
                pieces = [
                    linear(module, inputs[:, begin:end]) for begin, end in _SPANS.get()
                ]
                # Publish both original-shaped calls together before moving to
                # the next weight; lazy graph traversal alone can reorder them.
                mx.async_eval(pieces)
                values[id(module)] = mx.concatenate(pieces, axis=1)
            return values
        return {id(module): linear(module, inputs) for module in modules}

    delta = q.Qwen3_5GatedDeltaNet.__call__

    def gated_delta(self, inputs, mask=None, cache=None):
        spans = _SPANS.get()
        if not spans:
            return delta(self, inputs, mask, cache)
        values = projections(
            inputs, (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a)
        )
        token = _PROJECTED.set(values)
        outputs = []
        try:
            for begin, end in spans:
                region = _RANGE.set((begin, end))
                try:
                    x = inputs[:, begin:end]
                    piece_mask = q._create_qwen3_5_ssm_mask(x, cache)
                    outputs.append(delta(self, x, piece_mask, cache))
                finally:
                    _RANGE.reset(region)
            return mx.concatenate(outputs, axis=1)
        finally:
            _PROJECTED.reset(token)

    patch(q.Qwen3_5GatedDeltaNet, "__call__", gated_delta)
    attention = q.Qwen3_5Attention.__call__

    def full_attention(
        self, x, mask=None, cache=None, position_ids=None, position_embeddings=None
    ):
        spans = _SPANS.get()
        if not spans:
            return attention(self, x, mask, cache, position_ids, position_embeddings)
        token = _PROJECTED.set(projections(x, (self.q_proj, self.k_proj, self.v_proj)))
        outputs = []
        try:
            for begin, end in spans:
                region = _RANGE.set((begin, end))
                try:
                    piece = x[:, begin:end]
                    ids = (
                        position_ids[..., begin:end]
                        if position_ids is not None
                        else None
                    )
                    embeddings = (
                        tuple(v[..., begin:end, :] for v in position_embeddings)
                        if position_embeddings
                        else None
                    )
                    outputs.append(
                        attention(
                            self,
                            piece,
                            q._create_qwen3_5_attention_mask(piece, cache),
                            cache,
                            ids,
                            embeddings,
                        )
                    )
                finally:
                    _RANGE.reset(region)
            return mx.concatenate(outputs, axis=1)
        finally:
            _PROJECTED.reset(token)

    patch(q.Qwen3_5Attention, "__call__", full_attention)
    mlp = q.Qwen3_5MLP.__call__

    def feed_forward(self, x):
        spans = _SPANS.get()
        if (
            spans
            and preserve_descriptors
            and x.dtype == mx.bfloat16
            and all(
                type(m) is LaneLinear
                for m in (self.gate_proj, self.up_proj, self.down_proj)
            )
        ):
            pieces = [x[:, begin:end] for begin, end in spans]
            gates = [linear(self.gate_proj, piece) for piece in pieces]
            mx.async_eval(gates)
            ups = [linear(self.up_proj, piece) for piece in pieces]
            mx.async_eval(ups)
            hidden = [q.swiglu(gate, up) for gate, up in zip(gates, ups, strict=True)]
            outputs = [linear(self.down_proj, value) for value in hidden]
            mx.async_eval(outputs)
            return mx.concatenate(outputs, axis=1)
        if spans and (
            x.dtype != mx.bfloat16
            or not all(
                type(m) is LaneLinear
                for m in (self.gate_proj, self.up_proj, self.down_proj)
            )
        ):
            return mx.concatenate(
                [mlp(self, x[:, begin:end]) for begin, end in spans], axis=1
            )
        return mlp(self, x)

    patch(q.Qwen3_5MLP, "__call__", feed_forward)

    def uninstall():
        global _INSTALLATION

        for cls, name, fn in reversed(originals):
            setattr(cls, name, fn)
        _INSTALLATION = None

    _INSTALLATION = (counts, uninstall)
    return _INSTALLATION
