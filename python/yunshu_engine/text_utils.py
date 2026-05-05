"""Output text cleaning utilities.

Adapted from oMLX's api/utils.py — removes special tokens from model output
that may leak through tokenization edge cases.
"""

import re

# Pattern matching common special tokens that should be removed from output.
# oMLX pattern: these tokens sometimes appear in output due to tokenizer quirks.
SPECIAL_TOKENS_PATTERN = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|"
    r"<\|end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|"
    r"</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]"
)


def clean_special_tokens(text: str) -> str:
    """Remove special tokens from model output."""
    if not text:
        return text
    return SPECIAL_TOKENS_PATTERN.sub("", text).strip()


def chunk_prompt_tokens(
    token_ids: list[int],
    chunk_size: int = 2048,
) -> list[list[int]]:
    """Split prompt tokens into chunks for chunked prefill.

    Chunked prefill breaks long prompts into manageable pieces that
    fit within GPU memory constraints. Each chunk is processed
    sequentially, accumulating KV cache across chunks.

    Args:
        token_ids: Full prompt token IDs.
        chunk_size: Maximum tokens per chunk (default from prefill_step_size).

    Returns:
        List of token ID chunks.
    """
    if not token_ids:
        return []

    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunks.append(token_ids[i : i + chunk_size])
    return chunks


def estimate_prefill_memory(
    num_tokens: int,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype_size: int = 2,
    num_attention_heads: int = 32,
) -> int:
    """Estimate peak memory usage during prefill (bytes).

    For head_dim <= 128 (most models): O(n) tiled SDPA
    For head_dim > 128: O(n^2) full attention matrix

    Returns estimated bytes needed for the prefill operation.
    """
    kv_memory = num_tokens * num_layers * num_kv_heads * head_dim * dtype_size * 2

    if head_dim > 128:
        # Full attention matrix materialized in float32
        attention = num_attention_heads * num_tokens * num_tokens * 4
        attention += num_attention_heads * num_tokens * head_dim * 4
    else:
        # Tiled/fused — O(n) memory
        attention = num_attention_heads * num_tokens * head_dim * 4

    return kv_memory + attention


def should_chunk_prefill(
    num_tokens: int,
    available_memory: int,
    model_config: dict | None = None,
    safety_factor: float = 0.8,
) -> bool:
    """Determine if a prompt should be chunked for prefill.

    Returns True if the prompt is too large for single-pass prefill.
    """
    if model_config is None:
        model_config = {}

    estimate = estimate_prefill_memory(
        num_tokens,
        num_layers=model_config.get("num_layers", 32),
        num_kv_heads=model_config.get("num_kv_heads", 8),
        head_dim=model_config.get("head_dim", 128),
        num_attention_heads=model_config.get("num_attention_heads", 32),
    )

    return estimate * safety_factor > available_memory


def get_eos_token_ids(tokenizer) -> list[int]:
    """Extract EOS token IDs from a tokenizer."""
    eos_ids = set()
    if hasattr(tokenizer, "eos_token_id"):
        eid = tokenizer.eos_token_id
        if isinstance(eid, (list, tuple)):
            eos_ids.update(eid)
        elif eid is not None:
            eos_ids.add(eid)
    if hasattr(tokenizer, "eos_token_ids"):
        eids = tokenizer.eos_token_ids
        if isinstance(eids, (list, tuple)):
            eos_ids.update(eids)
        elif isinstance(eids, int):
            eos_ids.add(eids)
    return list(eos_ids)
