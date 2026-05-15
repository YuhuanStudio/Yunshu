from __future__ import annotations
"""Yunshu Mesh — Model sharding for distributed inference.

Handles both tensor parallel and pipeline parallel sharding of MLX models.

Tensor parallel: splits attention/MLP weights across nodes using
MLX's shard_linear/shard_inplace. Each node holds 1/N of each layer's
weights and uses all_sum/all_gather to coordinate.

Pipeline parallel: splits layers across nodes. Each node runs a contiguous
subset of layers and passes activations via send/recv. Uses exo's
PipelineFirstLayer/PipelineLastLayer pattern for clean inter-node handoff.

API:
  shard_model(model, group, strategy="auto") -> sharded model
  load_sharded(model_name, group, strategy) -> (model, tokenizer)
"""


import logging
from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def mx_barrier(group: Optional[mx.distributed.Group] = None) -> None:
    """Synchronize all processes using all_sum on CPU stream."""
    if group is None:
        return
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
        )
    )


def get_inner_model(model: nn.Module) -> nn.Module:
    """Extract the inner transformer model from a wrapper."""
    for attr in ("model", "transformer", "language_model", "backbone"):
        inner = getattr(model, attr, None)
        if isinstance(inner, nn.Module):
            # Handle language_model.model pattern
            inner_inner = getattr(inner, "model", None)
            if isinstance(inner_inner, nn.Module):
                return inner_inner
            return inner
    raise ValueError("Model must have 'model', 'transformer', or 'backbone' attribute")


def get_layers(inner_model: nn.Module) -> list:
    """Get the list of transformer layers from the inner model."""
    if hasattr(inner_model, "layers"):
        return list(inner_model.layers)
    if hasattr(inner_model, "h"):
        return list(inner_model.h)
    raise ValueError("Model must have 'layers' or 'h' attribute")


# ── Pipeline Parallel Wrappers ──


class PipelineFirstLayer(nn.Module):
    """Wraps the first layer of a pipeline stage.

    On non-zero ranks, receives activations from the previous stage
    before running the layer. On rank 0, runs the layer normally.
    """

    def __init__(self, original_layer, rank: int, group: mx.distributed.Group):
        super().__init__()
        dict.__setitem__(self, "_original_layer", original_layer)
        self.r = rank
        self.group = group
        self.is_prefill = False

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        if self.r != 0:
            mx.eval(x)
            x = mx.distributed.recv_like(x, self.r - 1, group=self.group)
            mx.eval(x)
        return self._original_layer(x, *args, **kwargs)


class PipelineLastLayer(nn.Module):
    """Wraps the last layer of a pipeline stage.

    On non-last ranks, sends activations to the next stage.
    On the last rank during decode, uses all_gather to combine outputs.
    """

    def __init__(self, original_layer, rank: int, world_size: int,
                 group: mx.distributed.Group):
        super().__init__()
        dict.__setitem__(self, "_original_layer", original_layer)
        self.r = rank
        self.s = world_size
        self.group = group
        self.is_prefill = False

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        output = self._original_layer(x, *args, **kwargs)
        mx.eval(output)

        if self.r != self.s - 1:
            output = mx.distributed.send(
                output, (self.r + 1) % self.s, group=self.group
            )
            mx.eval(output)

        if not self.is_prefill:
            output = mx.distributed.all_gather(output, group=self.group)[-output.shape[0]:]
            mx.eval(output)

        return output


# ── ShardedMoE Wrapper ──


class ShardedMoE(nn.Module):
    """Wraps an MoE layer with distributed sum_gradients / all_sum.

    Before the MoE forward, sum_gradients ensures all ranks see the
    same input. After, all_sum combines the partial expert outputs.
    """

    def __init__(self, original_layer, group: mx.distributed.Group):
        super().__init__()
        dict.__setitem__(self, "_original_layer", original_layer)
        self.group = group

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        from mlx.nn.layers.distributed import sum_gradients
        x = sum_gradients(self.group)(x)
        y = self._original_layer(x, *args, **kwargs)
        y = mx.distributed.all_sum(y, group=self.group)
        return y


# ── Tensor Parallel Strategies ──


def _shard_llama_like(model, group: mx.distributed.Group) -> None:
    """Shard Llama-like models (Llama, Qwen2, Mistral, etc.).

    Pattern:
    - Q/K/V projections: all-to-sharded (split output dim)
    - O projection: sharded-to-all (reduce across ranks)
    - MLP gate/up: all-to-sharded
    - MLP down: sharded-to-all
    """
    from mlx.nn.layers.distributed import shard_linear

    all_to_sharded = partial(shard_linear, sharding="all-to-sharded", group=group)
    sharded_to_all = partial(shard_linear, sharding="sharded-to-all", group=group)
    n = group.size()

    inner = get_inner_model(model)
    layers = get_layers(inner)

    for layer in layers:
        mx.eval(layer.parameters())

        # Attention
        attn = layer.self_attn
        attn.q_proj = all_to_sharded(attn.q_proj)
        attn.k_proj = all_to_sharded(attn.k_proj)
        attn.v_proj = all_to_sharded(attn.v_proj)
        attn.o_proj = sharded_to_all(attn.o_proj)
        attn.n_heads //= n
        if attn.n_kv_heads is not None:
            attn.n_kv_heads //= n

        # MLP
        layer.mlp.gate_proj = all_to_sharded(layer.mlp.gate_proj)
        layer.mlp.down_proj = sharded_to_all(layer.mlp.down_proj)
        layer.mlp.up_proj = all_to_sharded(layer.mlp.up_proj)

        mx.eval(layer)
        mx.clear_cache()


def _shard_with_model_shard(model, group: mx.distributed.Group) -> None:
    """Use the model's built-in shard() method (mlx-lm standard)."""
    model.shard(group)


def _detect_model_family(model) -> str:
    """Detect which model family for sharding strategy selection."""
    # Check for built-in shard support first
    if hasattr(model, "shard") and callable(model.shard):
        return "builtin"

    model_type = type(model).__name__.lower()
    try:
        inner = get_inner_model(model)
        inner_type = type(inner).__name__.lower()
    except ValueError:
        inner_type = ""

    # Check model type names for known families
    for name in ("llama", "mistral", "phi", "gemma"):
        if name in model_type or name in inner_type:
            return "llama_like"

    return "unknown"


# ── Public API ──


def shard_tensor_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
) -> nn.Module:
    """Apply tensor parallel sharding to a model.

    Tries built-in model.shard() first, then falls back to
    model-family-specific custom sharding.

    Args:
        model: The loaded (lazy) MLX model.
        group: mx.distributed Group for this shard.

    Returns:
        The sharded model (modified in place).
    """
    family = _detect_model_family(model)
    logger.info(f"Tensor parallel sharding model family: {family}")

    if family == "builtin":
        logger.info("Using built-in model.shard()")
        _shard_with_model_shard(model, group)
    elif family == "llama_like":
        logger.info("Using Llama-like custom sharding")
        _shard_llama_like(model, group)
    else:
        logger.info(
            f"No custom sharding for {family}, trying built-in model.shard()"
        )
        if hasattr(model, "shard"):
            _shard_with_model_shard(model, group)
        else:
            raise ValueError(
                f"Model type {type(model).__name__} has no shard() method "
                f"and no custom sharding strategy available"
            )

    return model


def shard_pipeline_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    start_layer: int,
    end_layer: int,
) -> nn.Module:
    """Apply pipeline parallel sharding to a model.

    Slices the model to only include layers [start_layer, end_layer)
    and wraps the first/last layers for inter-node communication.

    Args:
        model: The loaded (lazy) MLX model.
        group: mx.distributed Group.
        start_layer: First layer index for this shard.
        end_layer: Exclusive end layer index.

    Returns:
        The pipeline-sharded model.
    """
    rank = group.rank()
    world_size = group.size()

    inner = get_inner_model(model)
    layers = get_layers(inner)

    # Evaluate and trim layers to this shard's range
    shard_layers = layers[start_layer:end_layer]
    for i, layer in enumerate(shard_layers):
        mx.eval(layer)
        mx.clear_cache()
        logger.debug(f"Pipeline rank {rank}: loaded layer {start_layer + i}")

    # Wrap first layer for recv from previous stage
    if shard_layers:
        shard_layers[0] = PipelineFirstLayer(shard_layers[0], rank, group=group)

    # Wrap last layer for send to next stage + all_gather
    if shard_layers:
        shard_layers[-1] = PipelineLastLayer(
            shard_layers[-1], rank, world_size, group=group
        )

    # Update the model's layer list
    if hasattr(inner, "layers"):
        inner.layers = shard_layers
    elif hasattr(inner, "h"):
        inner.h = shard_layers

    return model


def load_sharded_model(
    model_name: str,
    group: Optional[mx.distributed.Group] = None,
    strategy: str = "auto",
    start_layer: int = 0,
    end_layer: int = -1,
) -> tuple[nn.Module, object]:
    """Load a model with optional distributed sharding.

    Args:
        model_name: HuggingFace model ID or local path.
        group: mx.distributed Group (None = single-node).
        strategy: "auto", "tensor", "pipeline", or "none".
        start_layer: For pipeline: first layer index.
        end_layer: For pipeline: exclusive end (-1 = all).

    Returns:
        (model, tokenizer) tuple.
    """
    from mlx_lm.utils import load as load_model

    if group is None:
        logger.info(f"Loading model single-node: {model_name}")
        model, tokenizer = load_model(model_name)
        return model, tokenizer

    world_size = group.size()
    rank = group.rank()
    logger.info(f"Loading model distributed: rank={rank}, world_size={world_size}")

    # Lazy load to allow sharding before weight materialization
    model, tokenizer = load_model(model_name, lazy=True)

    if strategy == "auto":
        # Prefer tensor parallel (better throughput for small clusters)
        if world_size <= 4 and hasattr(model, "shard"):
            strategy = "tensor"
        else:
            strategy = "tensor"  # Default to tensor for now

    if strategy == "tensor":
        logger.info(f"Rank {rank}: Applying tensor parallel sharding")
        shard_tensor_parallel(model, group)
    elif strategy == "pipeline":
        if end_layer == -1:
            inner = get_inner_model(model)
            layers = get_layers(inner)
            end_layer = len(layers)
        total_layers = end_layer - start_layer
        layers_per_rank = total_layers // world_size
        remainder = total_layers % world_size

        my_start = start_layer + rank * layers_per_rank + min(rank, remainder)
        my_layers = layers_per_rank + (1 if rank < remainder else 0)
        my_end = my_start + my_layers

        logger.info(f"Rank {rank}: Pipeline layers [{my_start}, {my_end})")
        shard_pipeline_parallel(model, group, my_start, my_end)

    mx.eval(model)
    mx_barrier(group)

    logger.info(f"Rank {rank}: Model loaded and sharded successfully")
    return model, tokenizer
