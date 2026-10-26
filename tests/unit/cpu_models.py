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


if __name__ == "__main__":
    print(fetch_qwen_vocab())
