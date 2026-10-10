# Patches upstream (MIT): mlx-vlm qwen4_exp Qwen4ExpBatchInvariantForward._model (tracked in vendor.json;
# Blaizzy/mlx-vlm). The kernels it calls are derived from ashhart/TensorFold (Apache-2.0), see
# kernels/tf_flash_next/.
"""Qwen4-Exp layer loop on TensorFold's fused hyper-connection kernels.

Upstream runs each of the 97 hyper-connection (HC) calls of a decode step as about 14 small kernels
(inject, norm, down projection, SiLU, up projection, sigmoid mix, inject gates).  TensorFold's chain does
the same work as three: ``hc_norm`` (write the previous block's branch back into the four residual streams
and take per-stream sums of squares), ``hc_project`` (down projection, SiLU, up projection, sigmoid mix and
the inject gates, split over the matrix units).  On the real oQ4e weights a chain of 97 dependent calls took
7.95 ms with the upstream path and 2.79 ms with these kernels (flashnext80-hcbench).

Each row's result is independent of the other rows and of how many rows are in flight (TensorFold's row and
tile paths are bit-equal), so a single decode step, a verify window, a prefill chunk and a batched step all
agree with each other; they differ from the upstream HC arithmetic in the last bf16 bit of the mixed input
(about 1% of the maximum), which is the reason this path is an A/B-gated option.

Everything else in the layer (attention, gated delta net, MoE, PLE) is upstream's code, called unchanged.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .tf_flash_next.base import MAX_ROWS

_STATE: dict[str, Any] = {
    "original": None,
    "installed": False,
    "modules": {},
    "moe": True,
    "gdn": True,
    "gated_delta": None,
}


def _plan(module: Any) -> Any:
    """Cached kernel inputs of one HyperConnection module, or None when its weights are not affine."""
    from .tf_flash_next.base import AFFINE_BITS, QWeights
    from .tf_flash_next.inputs import floats

    key = id(module)
    plan = _STATE["modules"].get(key)
    if plan is not None:
        return plan if plan != "ineligible" else None
    try:
        linears = [module.input_mix_weight_down, module.input_mix_weight_up]
        inject = (
            getattr(module, "block_inject_weight", None)
            if "block_inject_weight" in module
            else None
        )
        for lin in linears + ([inject] if inject is not None else []):
            if (
                getattr(lin, "mode", "affine") != "affine"
                or lin.bits not in AFFINE_BITS
            ):
                raise ValueError("not affine")
        down = QWeights.of(
            *([linears[0], inject] if inject is not None else [linears[0]])
        )
        up = QWeights.of(linears[1])
        norm_scale = (1.0 + module.hc_norm.weight.astype(mx.float32)).reshape(-1)
        plan = {
            "down": down,
            "up": up,
            "scale": norm_scale,
            "eps": floats([module.hc_norm.eps]),
            "low": int(linears[0].weight.shape[0]),
            "streams": int(module.hc_count),
        }
        mx.eval(
            down.weight,
            down.scales,
            down.biases,
            up.weight,
            up.scales,
            up.biases,
            norm_scale,
        )
    except Exception:  # noqa: BLE001 - any odd pack keeps the upstream path
        _STATE["modules"][key] = "ineligible"
        return None
    _STATE["modules"][key] = plan
    return plan


def _dense_rows(linear: Any) -> Any:
    """A router linear's rows in bf16 (a quantized one dequantized): one matvec takes them all."""
    if hasattr(linear, "scales"):
        return mx.dequantize(
            linear.weight,
            linear.scales,
            linear.biases,
            group_size=linear.group_size,
            bits=linear.bits,
        ).astype(mx.bfloat16)
    return linear.weight.astype(mx.bfloat16)


def _moe_plan(mlp: Any) -> Any:
    """Router rows (experts + shared gate) and projections for the fused MoE kernels, or None."""
    key = ("moe", id(mlp))
    plan = _STATE["modules"].get(key)
    if plan is not None:
        return plan if plan != "ineligible" else None
    try:
        from .tf_flash_next.experts import _format

        sw, se = mlp.switch_mlp, mlp.shared_expert
        _format(sw.gate_proj, sw.up_proj)
        _format(sw.down_proj)
        _format(se.gate_proj, se.up_proj)
        _format(se.down_proj)
        router_rows = mx.concatenate(
            [_dense_rows(mlp.gate), _dense_rows(mlp.shared_expert_gate)]
        )
        mx.eval(router_rows)
        plan = {
            "router": router_rows,
            "top_k": int(mlp.top_k),
            "experts": int(mlp.num_experts),
            "sw": sw,
            "se": se,
        }
    except Exception:  # noqa: BLE001 - any odd format keeps the upstream MoE for that layer
        _STATE["modules"][key] = "ineligible"
        return None
    _STATE["modules"][key] = plan
    return plan


def _moe(mlp: Any, x2d: Any) -> Any:
    """Fused router + experts for <= MAX_ROWS rows: ("grouped", (Y, weights, logits)) for the next hc_norm."""
    from .tf_flash_next import experts

    plan = _moe_plan(mlp)
    if not _STATE["moe"] or plan is None or x2d.shape[0] > MAX_ROWS:
        return None
    logits = experts.router(x2d, plan["router"])
    sw, se = plan["sw"], plan["se"]
    act, picks, weights = experts.expert_gateup(
        x2d,
        logits,
        plan["top_k"],
        plan["experts"],
        sw.gate_proj,
        sw.up_proj,
        shared=(se.gate_proj, se.up_proj),
    )
    y = experts.expert_down_y(act, picks, sw.down_proj, se.down_proj)
    return "grouped", (y, weights, logits)


def _gdn_plan(layer_attn: Any) -> Any:
    """Constants of one Gated DeltaNet layer for TensorFold's gdn_step, or None for shapes it does not take."""
    key = ("gdn", id(layer_attn))
    plan = _STATE["modules"].get(key)
    if plan is not None:
        return plan if plan != "ineligible" else None
    from .tf_flash_next.inputs import floats

    try:
        if layer_attn.head_k_dim != 128 or layer_attn.head_v_dim != 128:
            raise ValueError("gdn_step is written for 128-dim heads")
        conv_w = mx.contiguous(layer_attn.conv1d.weight[:, :, 0])
        mx.eval(conv_w)
        plan = {
            "conv_w": conv_w,
            "eps": floats([layer_attn.norm.eps]),
            "nk": int(layer_attn.num_k_heads),
            "nv": int(layer_attn.num_v_heads),
            "dk": int(layer_attn.head_k_dim),
            "dv": int(layer_attn.head_v_dim),
            "taps": int(layer_attn.conv_kernel_size),
        }
    except Exception:  # noqa: BLE001 - any odd shape keeps the upstream layer
        _STATE["modules"][key] = "ineligible"
        return None
    _STATE["modules"][key] = plan
    return plan


def _gated_delta(self: Any, layer: Any, inputs: Any, mask: Any, cache: Any) -> Any:
    """Gated DeltaNet after the projections as one TensorFold kernel (conv, norms, gates, delta rule, gated norm).

    Rows are processed in order inside the kernel and every row's recurrent state and conv window comes back,
    so the cache keeps its speculative history exactly as upstream's ``update_recurrent`` / ``update_window``
    contract expects (a partial accept needs no recompute).  One stream only; padded or batched calls keep
    upstream's layer."""
    from .tf_flash_next import gdn

    plan = _gdn_plan(layer) if _STATE["gdn"] else None
    batch, length, _ = inputs.shape
    if (
        plan is None
        or batch != 1
        or mask is not None
        or cache is None
        or getattr(cache, "lengths", None) is not None
        or not hasattr(cache, "update_recurrent")
    ):
        return _STATE["gated_delta"](self, layer, inputs, mask, cache)
    mixed_qkv, z, b, a = self._linears(
        (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a), inputs
    )
    projected = mx.concatenate(
        [
            mixed_qkv.reshape(length, -1),
            z.reshape(length, -1),
            b.reshape(length, -1),
            a.reshape(length, -1),
        ],
        axis=-1,
    )
    channels = mixed_qkv.shape[-1]
    conv_state = (
        cache[0][0]
        if cache[0] is not None
        else mx.zeros((plan["taps"] - 1, channels), dtype=inputs.dtype)
    )
    ssm_state = cache[1][0] if cache[1] is not None else None
    out, conv_rows, ssm_rows = gdn.gdn_step(
        projected,
        conv_state,
        ssm_state,
        plan["conv_w"],
        layer.A_log,
        layer.dt_bias,
        layer.norm.weight,
        plan["eps"],
        nk=plan["nk"],
        nv=plan["nv"],
        dk=plan["dk"],
        dv=plan["dv"],
    )
    if cache.history_capacity:
        conv_input = mx.concatenate([conv_state[None], mixed_qkv], axis=1)
        cache.update_window(0, conv_input, plan["taps"] - 1, lengths=None)
    else:
        cache.cache[0] = conv_rows[length - 1 : length]

    def update(initial, steps):
        del initial
        history = ssm_rows[: length - 1][None] if steps is not None else None
        return out, ssm_rows[length - 1 : length], history

    cache.update_recurrent(1, length, update)
    if hasattr(cache, "advance"):
        cache.advance(length)
        self._helpers()._qwen3_5_advance_lengths_info(cache, length)
    return self._linear(layer.out_proj, out.reshape(1, length, -1))


def eligible(model: Any) -> bool:
    """Every layer's two HC modules have affine weights the kernels read (decided once per model)."""
    key = ("model", id(model))
    found = _STATE["modules"].get(key)
    if found is None:
        found = all(
            _plan(getattr(layer, name)) is not None
            for layer in model.layers
            for name in ("attn_hyper_connection", "mlp_hyper_connection")
        )
        _STATE["modules"][key] = found
    return found


def _tf_model(self, model, inputs, cache, inputs_embeds, position_ids):
    from mlx_vlm.models.qwen4_exp import language as ql

    from .tf_flash_next import hc as tf_hc

    hidden = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
    if hidden.dtype != mx.bfloat16 or not eligible(model):
        return _STATE["original"](
            self, model, inputs, cache, inputs_embeds, position_ids
        )
    hidden = mx.tile(hidden, (1, 1, model.args.hc_count))
    if cache is None:
        cache = [None] * len(model.layers)
    fa_mask = ql._create_qwen3_5_attention_mask(hidden, cache[model.fa_idx])
    ssm_mask = ql._create_qwen3_5_ssm_mask(hidden, cache[model.ssm_idx])
    batch, length, width = hidden.shape
    rows = batch * length
    dims = width // model.args.hc_count
    streams = model.args.hc_count

    def norm(h, pending=None):
        if pending is None:
            return tf_hc.hc_norm(h, streams=streams, write_back="none")
        kind, branch, inject = pending
        return tf_hc.hc_norm(
            h, streams=streams, write_back=kind, branch=branch, inject=inject
        )

    def project(hn, ssp, module):
        plan = _plan(module)
        return tf_hc.hc_project(
            hn,
            ssp,
            plan["down"],
            plan["up"],
            plan["scale"],
            eps=plan["eps"],
            streams=plan["streams"],
            low=plan["low"],
        )

    h = hidden.reshape(rows, width)
    pending = (
        None  # (branch [rows, dims], inject gates) of the previous layer's MoE block
    )
    for layer, layer_cache in zip(model.layers, cache, strict=True):
        layer_mask = ssm_mask if layer.is_linear else fa_mask
        hn, ssp = norm(h, pending)
        if "ple" in layer:
            stream = hn.reshape(batch, length, width)
            stream = stream + self._ple(
                layer.ple, stream, inputs, layer_cache, layer_mask
            )
            hn = stream.reshape(rows, width)
            _, ssp = norm(hn)
        mixed, inject = project(hn, ssp, layer.attn_hyper_connection)
        mixed = mixed.reshape(batch, length, dims)
        if layer.is_linear:
            branch = _gated_delta(
                self, layer.linear_attn, mixed, layer_mask, layer_cache
            )
        else:
            branch = self._qsa_attention(
                layer.self_attn, mixed, layer_cache, position_ids, layer_mask
            )
        hn2, ssp2 = norm(hn, ("plain", (branch.reshape(rows, dims),), inject))
        mixed, inject2 = project(hn2, ssp2, layer.mlp_hyper_connection)
        fused = _moe(layer.mlp, mixed)
        if fused is None:
            branch2 = self._feed_forward(layer.mlp, mixed.reshape(batch, length, dims))
            fused = ("plain", (branch2.reshape(rows, dims),))
        h = hn2
        pending = (fused[0], fused[1], inject2)
        mx.async_eval(h, *fused[1])
    hn, _ = norm(h, pending)
    return hn.reshape(batch, length, width)


def set_gdn(enabled: bool) -> None:
    """Route Gated DeltaNet layers through TensorFold's gdn_step (default) or keep upstream's."""
    _STATE["gdn"] = bool(enabled)


def set_moe(enabled: bool) -> None:
    """Route the MoE blocks through the fused router + expert kernels (default) or keep upstream's."""
    _STATE["moe"] = bool(enabled)


def install() -> bool:
    """Run the qwen4_exp layer loop on the fused HC kernels (idempotent)."""
    try:
        from mlx_vlm.models.qwen4_exp import language
    except Exception:  # noqa: BLE001 - mlx_vlm without qwen4_exp
        return False
    cls = language.Qwen4ExpBatchInvariantForward
    if _STATE["installed"]:
        return True
    _STATE["original"] = cls._model
    _STATE["gated_delta"] = cls._gated_delta
    cls._model = _tf_model
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    from mlx_vlm.models.qwen4_exp import language

    language.Qwen4ExpBatchInvariantForward._model = _STATE["original"]
    _STATE["installed"] = False
    _STATE["modules"].clear()
