"""Random-weight models built from real architecture configs, for end-to-end CPU tests.

No Metal and no Hugging Face needed: the Qwen tokenizer is built from llama.cpp's vocab-only
GGUF (the Apache-2.0 Qwen vocabulary, 5.9 MB, too big to commit), the weights are random. Outputs
are meaningless; identity properties (A == B bit for bit) are fully testable.

Fetch the vocabulary once with ``python -m tests.unit.cpu_models`` (it lands in
``tests/fixtures/.cache/``, gitignored), or point ``QWEN_VOCAB_GGUF`` at a copy. Tests that need
it skip when it is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

GGUF_URL = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models/"
    "ggml-vocab-qwen2.gguf"
)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DEFAULT_GGUF = FIXTURES / ".cache" / "ggml-vocab-qwen2.gguf"
TEMPLATES = FIXTURES / "glmspark_templates"


def qwen_vocab_gguf() -> Path | None:
    # transformers builds a tokenizer from a GGUF only through torch, which Yunshu does not install
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return None
    for cand in (os.environ.get("QWEN_VOCAB_GGUF"), str(DEFAULT_GGUF)):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def fetch_qwen_vocab() -> Path:
    import urllib.request

    DEFAULT_GGUF.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(GGUF_URL, DEFAULT_GGUF)
    return DEFAULT_GGUF


def build_tokenizer(dst: Path, template: str = "qwen38"):
    """Write a Qwen2 fast tokenizer (+ the repo's qwen chat template) into ``dst``."""
    from transformers import AutoTokenizer

    gguf = qwen_vocab_gguf()
    assert gguf is not None, "Qwen vocab GGUF not found"
    tok = AutoTokenizer.from_pretrained(str(gguf.parent), gguf_file=gguf.name)
    dst.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(dst))
    if template:
        (dst / "chat_template.jinja").write_text(
            (TEMPLATES / f"{template}.jinja").read_text()
        )
    # the template's stop token is the eos, like the real Qwen3 checkpoints
    path = dst / "tokenizer_config.json"
    cfg = json.loads(path.read_text())
    cfg["eos_token"] = "<|im_end|>"
    path.write_text(json.dumps(cfg))
    return AutoTokenizer.from_pretrained(str(dst))


_SESSION_DIR: dict[str, Path] = {}


def session_dir(name: str) -> Path:
    """A per-process scratch directory for expensive builds (the tokenizer takes ~15 s)."""
    import atexit
    import shutil
    import tempfile

    if "root" not in _SESSION_DIR:
        root = Path(tempfile.mkdtemp(prefix="yunshu-cpu-models-"))
        atexit.register(shutil.rmtree, root, True)
        _SESSION_DIR["root"] = root
    return _SESSION_DIR["root"] / name


def save_weights(model, dst: Path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    mx.eval(model.parameters())
    mx.save_safetensors(
        str(dst / "model.safetensors"),
        dict(tree_flatten(model.parameters())),
        metadata={"format": "mlx"},
    )


def build_qwen3_dir(dst: Path, vocab_size: int = 151936, seed: int = 0) -> Path:
    """A tiny random ``qwen3`` (mlx-lm) checkpoint directory with the real Qwen tokenizer.

    Shape follows Qwen3-0.6B (GQA, head_dim decoupled from hidden, tied embeddings), shrunk."""
    import mlx.core as mx
    from mlx_lm.models import qwen3

    cfg = dict(
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        rms_norm_eps=1e-6,
        vocab_size=vocab_size,
        max_position_embeddings=4096,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        torch_dtype="float32",
    )
    mx.random.seed(seed)
    model = qwen3.Model(qwen3.ModelArgs.from_dict(cfg))
    dst.mkdir(parents=True, exist_ok=True)
    build_tokenizer(dst)
    (dst / "config.json").write_text(json.dumps(cfg))
    save_weights(model, dst)
    return dst


def qwen35_config(vocab_size: int = 151936, layers: int = 4) -> dict:
    """The Qwen3.5 layout (gated-delta-net layers + a full attention layer every 4th, M-RoPE,
    vision tower with spatial merge) at toy width. Ids follow the Qwen tokenizer's vocabulary."""
    return dict(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        eos_token_id=151645,
        vocab_size=vocab_size,
        tie_word_embeddings=False,
        text_config=dict(
            model_type="qwen3_5_text",
            hidden_size=256,
            intermediate_size=512,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_conv_kernel_dim=4,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=256,
            rms_norm_eps=1e-6,
            vocab_size=vocab_size,
            max_position_embeddings=32768,
            eos_token_id=151645,
            tie_word_embeddings=False,
            full_attention_interval=4,
            rope_parameters=dict(
                type="default",
                mrope_section=[11, 11, 10],
                rope_theta=100000,
                partial_rotary_factor=0.25,
            ),
        ),
        vision_config=dict(
            model_type="qwen3_5",
            depth=1,
            hidden_size=64,
            intermediate_size=128,
            out_hidden_size=256,
            num_heads=2,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
            num_position_embeddings=64,
            in_channels=3,
            deepstack_visual_indexes=[],
        ),
    )


def _make_repeater(model) -> None:
    import mlx.core as mx

    lm = model.language_model
    for layer in lm.model.layers:
        for name in ("self_attn", "linear_attn"):
            block = getattr(layer, name, None)
            if block is not None:
                proj = block.o_proj if name == "self_attn" else block.out_proj
                proj.weight = mx.zeros_like(proj.weight)
        layer.mlp.down_proj.weight = mx.zeros_like(layer.mlp.down_proj.weight)
    lm.lm_head.weight = lm.model.embed_tokens.weight


def build_qwen35_dir(
    dst: Path,
    dtype=None,
    seed: int = 3,
    layers: int = 4,
    template: str = "qwen35",
    mtp: bool = False,
    repeater: bool = False,
) -> Path:
    """A tiny random hybrid ``qwen3_5`` VLM checkpoint dir (mlx-vlm layout) that VLMEngine loads:
    fp32 by default, because bf16 prefill is span-dependent on the CPU stack. ``mtp=True`` adds
    the native MTP head (``mtp.*`` tensors + an index listing them), as Qwen3.5 checkpoints ship.
    ``repeater=True`` zeroes every block's output projections and ties the head to the embeddings,
    so greedy decode repeats the last token: drafts get accepted, which random weights never do."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen3_5 import qwen3_5

    cfg = qwen35_config(layers=layers)
    if mtp:
        cfg["text_config"]["mtp_num_hidden_layers"] = 1
    mx.random.seed(seed)
    model = qwen3_5.Model(qwen3_5.ModelConfig.from_dict(json.loads(json.dumps(cfg))))
    model.set_dtype(dtype or mx.float32)
    if repeater:
        _make_repeater(model)
    dst.mkdir(parents=True, exist_ok=True)
    build_tokenizer(dst, template=template)
    (dst / "config.json").write_text(json.dumps(cfg))
    if not mtp:
        save_weights(model, dst)
        return dst
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import (
        Qwen3_5MTPDraftModel,
    )

    head = Qwen3_5MTPDraftModel(
        Qwen3_5MTPConfig.from_dict(
            {
                "model_type": "qwen3_5_mtp",
                "text_config": cfg["text_config"],
                "block_size": 3,
                "tie_word_embeddings": False,
            }
        )
    )
    head.set_dtype(dtype or mx.float32)
    if repeater:  # predict "the token after the next is the next token": always right
        hidden = cfg["text_config"]["hidden_size"]
        head.fc.weight = mx.concatenate(
            [mx.eye(hidden), mx.zeros((hidden, hidden))], axis=1
        ).astype(head.fc.weight.dtype)
        for layer in head.layers:
            for name in ("self_attn", "linear_attn"):
                block = getattr(layer, name, None)
                if block is not None:
                    proj = block.o_proj if name == "self_attn" else block.out_proj
                    proj.weight = mx.zeros_like(proj.weight)
            layer.mlp.down_proj.weight = mx.zeros_like(layer.mlp.down_proj.weight)
    mx.eval(model.parameters(), head.parameters())
    tensors = dict(tree_flatten(model.parameters()))
    tensors.update({f"mtp.{k}": v for k, v in tree_flatten(head.parameters())})
    mx.save_safetensors(
        str(dst / "model.safetensors"), tensors, metadata={"format": "mlx"}
    )
    weight_map = dict.fromkeys(tensors, "model.safetensors")
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    return dst


def build_dflash2_dir(dst: Path, target: dict | None = None, seed: int = 5) -> Path:
    """A random DFlash2 drafter directory matching ``qwen35_config()`` (YUNSHU_VLM_DRAFT=<dir>)."""
    import mlx.core as mx
    from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

    target = target or qwen35_config()
    text = target["text_config"]
    cfg = dict(
        model_type="qwen3",
        architectures=["DFlash2DraftModel"],
        hidden_size=text["hidden_size"],
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        rms_norm_eps=1e-6,
        vocab_size=text["vocab_size"],
        max_position_embeddings=8192,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        num_target_layers=text["num_hidden_layers"],
        dflash_config=dict(
            block_size=8,
            mask_token_id=text["vocab_size"] - 1,
            target_layer_ids=[1, 3],
            num_target_layers=text["num_hidden_layers"],
            conv_kernel_size=2,
            conv_group_size=64,
            selector_rank=8,
            selector_top_k=16,
        ),
    )
    mx.random.seed(seed)
    model = DFlash2DraftModel(DFlash2Config.from_dict(json.loads(json.dumps(cfg))))
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps(cfg))
    save_weights(model, dst)
    return dst


if __name__ == "__main__":
    print(fetch_qwen_vocab())
