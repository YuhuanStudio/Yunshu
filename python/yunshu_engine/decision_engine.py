# Upstream (derived): Cloudflare/clef (Apache-2.0) joint_schema_model.py @ 2f3de3dd
"""Decision models: typed questions answered with probabilities (OpenAI Decisions / Jev SystemOne).

A decision checkpoint is a backbone plus a trained head that scores every allowed option of every
question in ONE forward pass: no decoding, no sampling. Loading such a checkpoint as a plain LLM
would silently drop the head, so the model detector routes it here and this module fails closed on
any head it does not recognise.

Supported head: Cloudflare Clef's "joint schema head" (Qwen3.5-family backbone loaded through
mlx-vlm, `joint_head.safetensors` + `joint_head_config.json`). The encoding of a record, the head's
arithmetic and the readout (softmax over each question's option logits; a score is the
probability-weighted level index) follow the reference implementation shipped in the checkpoint
(`joint_schema_model.py`, Apache-2.0), re-implemented in MLX. See vendor.json.

Deliberate differences from the reference, all stated here so nothing is implicit:
  * the head runs in float32 (the reference runs it in bfloat16 on CUDA);
  * the "lexical" option vectors are rows of the 4-bit quantized lm_head, dequantized on the fly
    (the reference uses the bf16 matrix);
  * a request whose encoding exceeds the context window is rejected, never truncated.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
MAX_INPUT_TOKENS = 16384  # the reference's training/serving length
MAX_IMAGES = 128

HEAD_FILE = "joint_head.safetensors"
HEAD_CONFIG_FILE = "joint_head_config.json"
_HEAD_CONFIG_KEYS = {
    "hidden_size",
    "width",
    "routing_layers",
    "layers",
    "heads",
    "feedforward",
}


class DecisionError(ValueError):
    """A request the engine cannot answer (the caller's fault: HTTP 400)."""


# ── internal request type (shared by /v1/decisions and /v1/systemone) ─────────────────────


@dataclass(frozen=True)
class Option:
    id: str  # unique within the question; what the model sees as option_id
    value: Any  # what the caller gets back (str | bool for choices, level label for scores)
    description: str | None = None


@dataclass(frozen=True)
class Question:
    kind: str  # "predicate" | "choice" | "score"
    name: str | None
    instructions: str
    options: tuple[Option, ...]  # request order; for a predicate: true, false


@dataclass
class DecisionRequest:
    model: str
    state: Any  # text, or any JSON value (SystemOne), rendered compactly
    questions: list[Question]
    images: list[bytes] = field(default_factory=list)


@dataclass
class DecisionResult:
    # one entry per question, option probabilities in the question's request order;
    # None = no finite distribution, reported as a refusal rather than a made-up value
    probabilities: list[list[float] | None]
    input_tokens: int


# ── record encoding (reference: joint_schema_model.encode_record) ────────────────────────


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


_PREDICATE_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}


def model_options(question: Question) -> list[tuple[str, str | None]]:
    """(option_id, description) in the order the MODEL sees them.

    The reference sorts choices by id (so the model is order-invariant by construction), keeps
    true/false for a predicate and numbers score levels. Callers map back through the ids."""
    if question.kind == "predicate":
        custom = {o.id: o.description for o in question.options}
        return [(k, custom.get(k) or _PREDICATE_CRITERIA[k]) for k in ("true", "false")]
    if question.kind == "choice":
        pairs: list[tuple[str, str | None]] = [
            (o.id, o.description) for o in question.options
        ]
        return sorted(pairs, key=lambda t: t[0])
    return [(o.id, o.description) for o in question.options]


@dataclass(frozen=True)
class EncodedQuestion:
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    media_offset: int = 0  # position where the image tokens start (len(prefix))


def question_ids(questions: list[Question]) -> list[str]:
    """The id each question carries in the prompt: its name, else q<N>. Unique or DecisionError."""
    ids = [q.name if q.name else f"q{i + 1}" for i, q in enumerate(questions)]
    seen: set[str] = set()
    for qid in ids:
        if qid in seen:
            raise DecisionError(f"questions: duplicate question name {qid!r}")
        seen.add(qid)
    return ids


def encode_record(
    tokenize: Callable[[str], list[int]],
    state: Any,
    questions: list[Question],
    media_ids: list[int] | None = None,
    max_length: int = MAX_INPUT_TOKENS,
) -> EncodedRecord:
    """Token layout of the reference: system+state prefix, [image tokens], state, schema, suffix.

    Each fragment is tokenized on its own and concatenated, exactly as the reference does;
    ``tokenize`` must not add special tokens."""
    if not questions:
        raise DecisionError("questions: at least one question is required")
    qids = question_ids(questions)
    schema_ids = tokenize("\n\nSCHEMA FIELDS:\n")
    encoded: list[EncodedQuestion] = []
    for index, (qid, question) in enumerate(zip(qids, questions, strict=True)):
        kind = "noul" if question.kind == "predicate" else question.kind
        schema_ids.extend(
            tokenize(f"\nFIELD {index + 1}\nID: {qid}\nTYPE: {kind}\nINSTRUCTION: ")
        )
        q_start = len(schema_ids)
        schema_ids.extend(tokenize(_render(question.instructions or qid)))
        q_end = len(schema_ids)
        schema_ids.extend(tokenize("\nALLOWED OPTIONS:\n"))
        spans: list[tuple[int, int]] = []
        ids: list[str] = []
        for opt_index, (opt_id, description) in enumerate(model_options(question)):
            schema_ids.extend(tokenize(f"OPTION {opt_index + 1}: "))
            start = len(schema_ids)
            semantics: dict[str, Any] = {"option_id": opt_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(tokenize(_render(semantics)))
            spans.append((start, len(schema_ids)))
            ids.append(opt_id)
            schema_ids.extend(tokenize("\n"))
        schema_ids.extend(tokenize("END FIELD\n"))
        encoded.append(
            EncodedQuestion(
                QUESTION_TYPES[kind], (q_start, q_end), tuple(spans), tuple(ids)
            )
        )

    prefix = tokenize(
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    )
    suffix = tokenize(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    )
    media_offset = len(prefix)
    prefix = prefix + list(media_ids or [])
    state_ids = tokenize(_render(state))
    total = len(prefix) + len(state_ids) + len(schema_ids) + len(suffix)
    if total > max_length:
        raise DecisionError(
            f"input is {total} tokens (state, images and schema together); this model's limit is "
            f"{max_length}. Shorten the input or split the questions."
        )
    offset = len(prefix) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(
            q.question_type,
            (q.question_span[0] + offset, q.question_span[1] + offset),
            tuple((a + offset, b + offset) for a, b in q.option_spans),
            q.option_ids,
        )
        for q in encoded
    )
    return EncodedRecord(
        tuple(prefix + state_ids + schema_ids + suffix), shifted, media_offset
    )


def softmax(logits: list[float]) -> list[float] | None:
    """Plain softmax; None when any logit is not finite (the caller reports a refusal)."""
    if not logits or not all(math.isfinite(x) for x in logits):
        return None
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


# ── head (reference: JointSchemaHead), functional MLX version ────────────────────────────


def expected_head_shapes(cfg: dict[str, int]) -> dict[str, tuple[int, ...]]:
    h, w, ff = cfg["hidden_size"], cfg["width"], cfg["feedforward"]
    shapes: dict[str, tuple[int, ...]] = {
        "hidden_norm.weight": (h,),
        "hidden_norm.bias": (h,),
        "type_embedding.weight": (3, w),
        "option_summary_norm.weight": (w,),
        "option_summary_norm.bias": (w,),
        "field_norm.weight": (w,),
        "field_norm.bias": (w,),
        "option_norm.weight": (w,),
        "option_norm.bias": (w,),
        "residual_scorer.0.weight": (w, 4 * w),
        "residual_scorer.0.bias": (w,),
        "residual_scorer.3.weight": (1, w),
        "residual_scorer.3.bias": (1,),
        "prior_logit_scale": (),
        "joint_logit_scale": (),
        "residual_gate": (),
    }
    for name in (
        "memory_projection",
        "question_projection",
        "option_question_projection",
        "global_projection",
        "option_context_projection",
        "option_lexical_projection",
    ):
        shapes[f"{name}.weight"] = (w, h)

    def mha(prefix: str) -> None:
        shapes[f"{prefix}.in_proj_weight"] = (3 * w, w)
        shapes[f"{prefix}.in_proj_bias"] = (3 * w,)
        shapes[f"{prefix}.out_proj.weight"] = (w, w)
        shapes[f"{prefix}.out_proj.bias"] = (w,)

    def norm(prefix: str) -> None:
        shapes[f"{prefix}.weight"] = (w,)
        shapes[f"{prefix}.bias"] = (w,)

    for i in range(cfg["routing_layers"]):
        p = f"evidence_layers.{i}"
        mha(f"{p}.attention")
        for n in ("query_norm", "memory_norm", "feedforward_norm"):
            norm(f"{p}.{n}")
        shapes[f"{p}.feedforward.0.weight"] = (ff, w)
        shapes[f"{p}.feedforward.0.bias"] = (ff,)
        shapes[f"{p}.feedforward.3.weight"] = (w, ff)
        shapes[f"{p}.feedforward.3.bias"] = (w,)
    for i in range(cfg["layers"]):
        p = f"layers.{i}"
        mha(f"{p}.self_attn")
        mha(f"{p}.multihead_attn")
        for n in ("norm1", "norm2", "norm3"):
            norm(f"{p}.{n}")
        shapes[f"{p}.linear1.weight"] = (ff, w)
        shapes[f"{p}.linear1.bias"] = (ff,)
        shapes[f"{p}.linear2.weight"] = (w, ff)
        shapes[f"{p}.linear2.bias"] = (w,)
    return shapes


def read_head_config(model_dir: Path) -> dict[str, int]:
    path = model_dir / HEAD_CONFIG_FILE
    if not path.is_file():
        raise ValueError(
            f"{model_dir.name}: decision checkpoint without {HEAD_CONFIG_FILE}"
        )
    cfg = json.loads(path.read_text())
    if not isinstance(cfg, dict) or set(cfg) != _HEAD_CONFIG_KEYS:
        raise ValueError(
            f"{HEAD_CONFIG_FILE}: unknown head layout (keys {sorted(cfg) if isinstance(cfg, dict) else cfg}); "
            f"only the Clef joint schema head {sorted(_HEAD_CONFIG_KEYS)} is supported"
        )
    if not all(isinstance(v, int) and v > 0 for v in cfg.values()):
        raise ValueError(f"{HEAD_CONFIG_FILE}: every value must be a positive integer")
    if cfg["width"] % cfg["heads"]:
        raise ValueError(f"{HEAD_CONFIG_FILE}: width must be divisible by heads")
    return cfg


def load_head_weights(model_dir: Path, cfg: dict[str, int]) -> dict[str, Any]:
    """Load the head as float32 arrays; any missing, extra or misshapen tensor is an error."""
    import mlx.core as mx

    path = model_dir / HEAD_FILE
    if not path.is_file():
        raise ValueError(f"{model_dir.name}: decision checkpoint without {HEAD_FILE}")
    raw = cast(dict[str, Any], mx.load(str(path)))
    expected = expected_head_shapes(cfg)
    missing = sorted(set(expected) - set(raw))
    extra = sorted(set(raw) - set(expected))
    if missing or extra:
        raise ValueError(
            f"{HEAD_FILE} does not match the Clef joint schema head: missing {missing[:4]}, "
            f"unexpected {extra[:4]}"
        )
    for name, shape in expected.items():
        if tuple(raw[name].shape) != shape:
            raise ValueError(
                f"{HEAD_FILE}: {name} has shape {tuple(raw[name].shape)}, want {shape}"
            )
    return {k: v.astype(mx.float32) for k, v in raw.items()}


def _layer_norm(x, w, b, eps: float = 1e-5):
    import mlx.core as mx

    return mx.fast.layer_norm(x, w, b, eps)


def _linear(x, w, b=None):
    y = x @ w.T
    return y if b is None else y + b


def _mha(W, prefix: str, q_in, kv_in, heads: int):
    """torch.nn.MultiheadAttention (batch_first, no mask) with packed in_proj."""
    import mlx.core as mx

    w = W[f"{prefix}.in_proj_weight"]
    b = W[f"{prefix}.in_proj_bias"]
    d = w.shape[1]
    q = _linear(q_in, w[:d], b[:d])
    k = _linear(kv_in, w[d : 2 * d], b[d : 2 * d])
    v = _linear(kv_in, w[2 * d :], b[2 * d :])

    def split(t):
        n, s, _ = t.shape
        return t.reshape(n, s, heads, d // heads).transpose(0, 2, 1, 3)

    out = mx.fast.scaled_dot_product_attention(
        split(q), split(k), split(v), scale=(d // heads) ** -0.5
    )
    out = out.transpose(0, 2, 1, 3).reshape(q_in.shape[0], q_in.shape[1], d)
    return _linear(out, W[f"{prefix}.out_proj.weight"], W[f"{prefix}.out_proj.bias"])


def _gelu(x):
    import mlx.nn as nn

    return nn.gelu(x)


def _normalize(x, eps: float = 1e-12):
    import mlx.core as mx

    n = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    return x / mx.maximum(n, eps)


def _span_matrix(spans: list[tuple[int, int]], n: int):
    """[len(spans), n] matrix whose rows average a token span (a mean as one matmul)."""
    import numpy as np

    m = np.zeros((len(spans), n), dtype=np.float32)
    for i, (a, b) in enumerate(spans):
        if b <= a:
            raise DecisionError("empty question or option text")
        m[i, a:b] = 1.0 / (b - a)
    return m


def head_logits(
    W: dict[str, Any],
    cfg: dict[str, int],
    hidden,
    record: EncodedRecord,
    lexical_rows: Callable[[list[int]], Any],
) -> list[list[float]]:
    """Option logits per question. ``hidden`` is [N, hidden_size] (final-norm backbone states);
    ``lexical_rows(token_ids)`` returns the output-embedding rows [len(ids), hidden_size]."""
    import mlx.core as mx

    n = hidden.shape[0]
    heads = cfg["heads"]
    seq = _layer_norm(
        hidden.astype(mx.float32), W["hidden_norm.weight"], W["hidden_norm.bias"]
    )
    memory = _linear(seq, W["memory_projection.weight"])[None]  # [1, N, W]
    global_vec = seq[-1]

    questions = record.questions
    q_spans = [q.question_span for q in questions]
    o_spans = [s for q in questions for s in q.option_spans]
    counts = [len(q.option_spans) for q in questions]
    q_mat = mx.array(_span_matrix(q_spans, n))
    o_mat = mx.array(_span_matrix(o_spans, n))
    qv = q_mat @ seq  # [Q, H]
    cv = o_mat @ seq  # [O, H]

    # lexical vectors: mean of the output-embedding rows of each option's tokens
    ids = list(record.input_ids)
    tok_ids: list[int] = []
    bounds: list[tuple[int, int]] = []
    for a, b in o_spans:
        bounds.append((len(tok_ids), len(tok_ids) + (b - a)))
        tok_ids.extend(ids[a:b])
    rows = lexical_rows(tok_ids).astype(mx.float32)
    lv = mx.array(_span_matrix(bounds, len(tok_ids))) @ rows  # [O, H]

    q_of_option = []
    for qi, c in enumerate(counts):
        q_of_option.extend([qi] * c)
    q_idx = mx.array(q_of_option)

    x = (
        _linear(cv, W["option_context_projection.weight"])
        + _linear(lv, W["option_lexical_projection.weight"])
        + _linear(qv, W["option_question_projection.weight"])[q_idx]
    )[None]  # [1, O, W]
    for i in range(cfg["routing_layers"]):
        p = f"evidence_layers.{i}"
        mem = _layer_norm(
            memory, W[f"{p}.memory_norm.weight"], W[f"{p}.memory_norm.bias"]
        )
        qn = _layer_norm(x, W[f"{p}.query_norm.weight"], W[f"{p}.query_norm.bias"])
        x = x + _mha(W, f"{p}.attention", qn, mem, heads)
        f = _layer_norm(
            x, W[f"{p}.feedforward_norm.weight"], W[f"{p}.feedforward_norm.bias"]
        )
        f = _linear(
            _gelu(
                _linear(f, W[f"{p}.feedforward.0.weight"], W[f"{p}.feedforward.0.bias"])
            ),
            W[f"{p}.feedforward.3.weight"],
            W[f"{p}.feedforward.3.bias"],
        )
        x = x + f
    routed = x[0]  # [O, W]

    base = _linear(qv, W["question_projection.weight"])  # [Q, W]
    width = base.shape[-1]
    summaries = []
    start = 0
    for qi, c in enumerate(counts):
        opts = routed[start : start + c]
        weights = mx.softmax((opts @ base[qi]) / math.sqrt(width), axis=0)
        summaries.append(mx.sum(weights[:, None] * opts, axis=0))
        start += c
    type_ids = mx.array([q.question_type for q in questions])
    fields = (
        base
        + _layer_norm(
            mx.stack(summaries),
            W["option_summary_norm.weight"],
            W["option_summary_norm.bias"],
        )
        + _linear(global_vec, W["global_projection.weight"])[None]
        + W["type_embedding.weight"][type_ids]
    )[None]  # [1, Q, W]
    for i in range(cfg["layers"]):
        p = f"layers.{i}"
        a = _layer_norm(fields, W[f"{p}.norm1.weight"], W[f"{p}.norm1.bias"])
        fields = fields + _mha(W, f"{p}.self_attn", a, a, heads)
        a = _layer_norm(fields, W[f"{p}.norm2.weight"], W[f"{p}.norm2.bias"])
        fields = fields + _mha(W, f"{p}.multihead_attn", a, memory, heads)
        a = _layer_norm(fields, W[f"{p}.norm3.weight"], W[f"{p}.norm3.bias"])
        fields = fields + _linear(
            _gelu(_linear(a, W[f"{p}.linear1.weight"], W[f"{p}.linear1.bias"])),
            W[f"{p}.linear2.weight"],
            W[f"{p}.linear2.bias"],
        )
    fields = _layer_norm(
        fields[0], W["field_norm.weight"], W["field_norm.bias"]
    )  # [Q, W]

    log_cap = math.log(100.0)
    prior_scale = mx.exp(mx.minimum(W["prior_logit_scale"], log_cap))
    joint_scale = mx.exp(mx.minimum(W["joint_logit_scale"], log_cap))
    gate = mx.sigmoid(W["residual_gate"])
    all_opts = _layer_norm(routed, W["option_norm.weight"], W["option_norm.bias"])
    lex_anchor = _normalize(lv)
    out = []
    start = 0
    for qi, c in enumerate(counts):
        anchor = _normalize(qv[qi] + global_vec)
        prior = prior_scale * (lex_anchor[start : start + c] @ anchor)
        opts = all_opts[start : start + c]
        field_row = mx.broadcast_to(fields[qi][None], opts.shape)
        cosine = mx.sum(_normalize(field_row, 1e-8) * _normalize(opts, 1e-8), axis=-1)
        feats = mx.concatenate(
            [field_row, opts, field_row * opts, mx.abs(field_row - opts)], axis=-1
        )
        hid = _gelu(
            _linear(feats, W["residual_scorer.0.weight"], W["residual_scorer.0.bias"])
        )
        residual = _linear(
            hid, W["residual_scorer.3.weight"], W["residual_scorer.3.bias"]
        )[:, 0]
        out.append(prior + gate * (joint_scale * cosine + residual))
        start += c
    mx.eval(out)
    return [[float(v) for v in o.tolist()] for o in out]


def readout(
    question: Question, logits: list[float], option_ids: tuple[str, ...]
) -> list[float] | None:
    """Probabilities in the question's REQUEST order from logits in the model's option order."""
    probs = softmax(logits)
    if probs is None:
        return None
    by_id = dict(zip(option_ids, probs, strict=True))
    return [by_id[o.id] for o in question.options]


# ── engine ───────────────────────────────────────────────────────────────────────────────


def is_decision_checkpoint(model_dir: str | Path) -> bool:
    """A directory with a decision head. Any of the head's files marks it: such a directory must
    never be loaded as a plain LLM/VLM, which would drop the head."""
    p = Path(model_dir)
    return (p / HEAD_FILE).exists() or (p / HEAD_CONFIG_FILE).exists()


class DecisionEngine:
    """Serves ``decide()`` for a Clef-style checkpoint on the single MLX thread."""

    supports_multimodal = False  # set from the checkpoint's config in start()

    def __init__(self, model_path: str, config: Any = None):
        self._model_path = model_path
        self._dir = Path(model_path)
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._head: Any = None
        self._head_cfg: Any = None
        self._lm_head: Any = None
        self._loaded = False
        self._active = 0
        self.max_input_tokens = MAX_INPUT_TOKENS

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._dir.name

    def has_active_requests(self) -> bool:
        return self._active > 0

    async def _run(self, fn):
        from .mlx_executor import get_mlx_executor

        self._active += 1
        try:
            future = asyncio.get_running_loop().run_in_executor(get_mlx_executor(), fn)
        except BaseException:
            self._active -= 1
            raise

        def completed(done):
            self._active -= 1
            if not done.cancelled():
                done.exception()

        future.add_done_callback(completed)
        return await asyncio.shield(future)

    def _validate_checkpoint(self) -> dict[str, int]:
        cfg = json.loads((self._dir / "config.json").read_text())
        text_cfg = cfg.get("text_config", cfg)
        if cfg.get("model_type") != "qwen3_5":
            raise ValueError(
                f"{self.model_name}: the Clef joint schema head is only defined on a qwen3_5 "
                f"backbone, this checkpoint is {cfg.get('model_type')!r}"
            )
        head_cfg = read_head_config(self._dir)
        if head_cfg["hidden_size"] != text_cfg.get("hidden_size"):
            raise ValueError(
                f"{self.model_name}: head hidden_size {head_cfg['hidden_size']} does not match the "
                f"backbone's {text_cfg.get('hidden_size')}"
            )
        return head_cfg

    async def start(self) -> None:
        head_cfg = self._validate_checkpoint()  # fail closed before any weight is read
        self.supports_multimodal = "vision_config" in json.loads(
            (self._dir / "config.json").read_text()
        )

        def load():
            from mlx_vlm.utils import load as vlm_load

            model, processor = vlm_load(self._model_path)
            self._head = load_head_weights(self._dir, head_cfg)
            self._head_cfg = head_cfg
            self._model = model
            self._processor = processor
            self._tokenizer = getattr(processor, "tokenizer", processor)
            self._lm_head = self._output_embedding(model)

        await self._run(load)
        self._loaded = True

    @staticmethod
    def _output_embedding(model):
        lm = model.language_model
        head = getattr(lm, "lm_head", None)
        if head is None:  # tied embeddings
            head = lm.model.embed_tokens
        return head

    def _lexical_rows(self, token_ids: list[int]):
        """Rows of the output-embedding matrix for ``token_ids`` (dequantized when 4/8-bit)."""
        import mlx.core as mx

        layer = self._lm_head
        idx = mx.array(token_ids)
        if hasattr(layer, "scales"):
            return mx.dequantize(
                layer.weight[idx],
                layer.scales[idx],
                layer.biases[idx],
                group_size=layer.group_size,
                bits=layer.bits,
            )
        return layer.weight[idx]

    async def stop(self) -> None:
        def clear():
            import mlx.core as mx

            self._model = self._processor = self._tokenizer = None
            self._head = self._lm_head = None
            mx.clear_cache()

        self._loaded = False
        await self._run(clear)

    def _tokenize(self, text: str) -> list[int]:
        return list(self._tokenizer(text, add_special_tokens=False).input_ids)

    def _encode_images(self, images: list[bytes]):
        import io

        import numpy as np
        from PIL import Image

        pics = []
        for i, raw in enumerate(images):
            try:
                pics.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception as e:
                raise DecisionError(
                    f"input image {i}: not a decodable image ({e})"
                ) from None
        text = IMAGE_PLACEHOLDER * len(pics) + "\n"
        enc = self._processor(text=[text], images=pics, return_tensors="np")
        return (
            [int(t) for t in np.asarray(enc["input_ids"])[0]],
            np.asarray(enc["pixel_values"]),
            np.asarray(enc["image_grid_thw"]),
        )

    def _decide_sync(self, req: DecisionRequest) -> DecisionResult:
        import mlx.core as mx

        media_ids: list[int] = []
        pixels: Any = None
        grid: Any = None
        if req.images:
            media_ids, pixels, grid = self._encode_images(req.images)
        record = encode_record(
            self._tokenize, req.state, req.questions, media_ids, self.max_input_tokens
        )
        ids = mx.array([list(record.input_ids)])
        kwargs: dict[str, Any] = {}
        if pixels is not None:
            kwargs = {
                "pixel_values": mx.array(pixels),
                "image_grid_thw": mx.array(grid),
            }
        feats = self._model.get_input_embeddings(ids, **kwargs)
        hidden = self._model.language_model.model(
            ids, inputs_embeds=feats.inputs_embeds, position_ids=feats.position_ids
        )[0]
        logits = head_logits(
            self._head, self._head_cfg, hidden, record, self._lexical_rows
        )
        probs = [
            readout(q, lg, enc.option_ids)
            for q, lg, enc in zip(req.questions, logits, record.questions, strict=True)
        ]
        return DecisionResult(probs, len(record.input_ids))

    async def decide(self, req: DecisionRequest) -> DecisionResult:
        if not self._loaded:
            raise RuntimeError("decision model is not loaded")
        if len(req.images) > MAX_IMAGES:
            raise DecisionError(f"input: at most {MAX_IMAGES} images per request")
        result: DecisionResult = await self._run(lambda: self._decide_sync(req))
        return result
