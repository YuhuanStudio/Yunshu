"""Text cross-encoders: Qwen3 yes/no and trained sequence-classification heads.

Loads published float and quantized safetensors, preserving trained heads. Unsupported architectures
fail explicitly instead of silently using an embedding or an untrained head.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
QWEN_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sigmoid(x: float) -> float:
    if not math.isfinite(x):
        raise ValueError("Non-finite classification logit")
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def probabilities(logits: list[float], multi_label: bool = False) -> list[float]:
    if not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("Empty or non-finite classification logits")
    if len(logits) == 1 or multi_label:
        return [sigmoid(x) for x in logits]
    exps = [math.exp(x - max(logits)) for x in logits]
    return [x / sum(exps) for x in exps]


def qwen_input_ids(
    tokenizer, query: str, document: str, instruction=None, max_length=8192
):
    """Model-card recipe: truncate the body, always preserve system and answer suffix."""
    prefix = tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
    budget = max_length - len(prefix) - len(suffix)
    if budget <= 0:
        raise ValueError("Context too short for the reranker template")
    body = f"<Instruct>: {DEFAULT_INSTRUCTION if instruction is None else instruction}\n<Query>: {query}\n<Document>: {document}"
    ids = tokenizer(body, truncation=True, max_length=budget)["input_ids"]
    return prefix + ids + suffix


def scoring_max_length(config: dict, tokenizer_max_length=None) -> int:
    limit = int(config["max_position_embeddings"])
    if config.get("model_type") == "qwen3":
        return min(8192, limit)
    if config.get("model_type") in ("roberta", "xlm-roberta"):
        limit -= int(config.get("pad_token_id", 1)) + 1
    return (
        min(limit, int(tokenizer_max_length))
        if tokenizer_max_length is not None
        else limit
    )


def scoring_kind(config: dict, model_path: str) -> str | None:
    architectures = config.get("architectures", [])
    if any(a.endswith("ForSequenceClassification") for a in architectures):
        return "head"
    if (
        config.get("model_type") == "qwen3"
        and "reranker" in model_path.lower()
        and (not architectures or architectures == ["Qwen3ForCausalLM"])
    ):
        return "qwen3"
    return None


class _ClassifierLogits:
    """Adapt upstream's typed output while exposing its trained head/parameters."""

    def __init__(self, model):
        self.model = model

    def __call__(self, **inputs):
        return self.model(**inputs).logits

    def __getattr__(self, name):
        return getattr(self.model, name)


def load_sequence_classifier(path: str, config: dict):
    """Keep the trained head using mlx-vlm's maintained encoder/quantized loader."""
    family = config.get("model_type")
    if family not in ("bert", "roberta", "xlm-roberta"):
        raise ValueError(
            f"Sequence classification architecture '{family}' is not supported; supported: bert, roberta, xlm-roberta"
        )
    if (
        config.get("hidden_act", "gelu") != "gelu"
        or config.get("position_embedding_type", "absolute") != "absolute"
    ):
        raise ValueError(
            "Only GELU / absolute-position encoder classifiers are supported"
        )
    from mlx_vlm.encoder_loader import load_encoder_model

    labels = int(config.get("num_labels", len(config.get("id2label", {})) or 2))
    model = load_encoder_model(
        Path(path),
        model_remapping={"roberta": "xlm_roberta", "xlm-roberta": "xlm_roberta"},
        model_class_name="SequenceClassificationModel",
        config=config,
        config_overrides={"num_labels": labels},
        strict=True,
    )
    return _ClassifierLogits(model)


class TextScoringEngine:
    def __init__(self, model_path: str, config: Any = None):
        self._model_path = model_path
        self._config = json.loads((Path(model_path) / "config.json").read_text())
        self.kind = scoring_kind(self._config, model_path)
        self.num_labels = int(
            self._config.get("num_labels", len(self._config.get("id2label", {})) or 2)
        )
        self.is_reranker = self.kind == "qwen3" or self.num_labels == 1
        self.is_classifier = self.kind == "head"
        self.supports_multimodal = False
        self._model = self._tokenizer = None
        self._loaded = False
        self._active = 0

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def model_name(self):
        return Path(self._model_path).name

    def has_active_requests(self):
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
                done.exception()  # retrieve exceptions even when the HTTP waiter was cancelled

        future.add_done_callback(completed)
        # Cancellation ends the HTTP wait, but the queued/running Metal work remains active.
        return await asyncio.shield(future)

    async def start(self):
        def load():
            from transformers import AutoTokenizer

            if self.kind == "qwen3":
                from mlx_lm import load as load_lm

                model, _ = load_lm(self._model_path)
            elif self.kind == "head":
                model = load_sequence_classifier(self._model_path, self._config)
            else:
                raise ValueError("Not a supported scoring checkpoint")
            tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            return model, tokenizer

        if not self._loaded:
            self._model, self._tokenizer = await self._run(load)
            self._loaded = True

    async def stop(self):
        def clear():
            self._model = self._tokenizer = None
            self._loaded = False

        await self._run(clear)

    def _encode_head(self, first, second=None):
        limit = scoring_max_length(self._config, self._tokenizer.model_max_length)
        return self._tokenizer(first, second, truncation=True, max_length=limit)

    def token_usage(self, texts=None, pairs=None, instruction=None):
        if self._tokenizer is None:
            return {"prompt_tokens": 0, "total_tokens": 0}
        if pairs is not None:
            if self.kind == "qwen3":
                n = sum(
                    len(
                        qwen_input_ids(
                            self._tokenizer,
                            a,
                            b,
                            instruction,
                            scoring_max_length(self._config),
                        )
                    )
                    for a, b in pairs
                )
            else:
                n = sum(len(self._encode_head(a, b)["input_ids"]) for a, b in pairs)
        else:
            n = sum(len(self._encode_head(t)["input_ids"]) for t in texts)
        return {"prompt_tokens": n, "total_tokens": n}

    def _head_logits(self, first, second=None):
        import mlx.core as mx

        if not self._loaded:
            raise RuntimeError("Engine not started")
        encoded = self._encode_head(first, second)
        inputs = {
            k: mx.array([v])
            for k, v in encoded.items()
            if k in ("input_ids", "attention_mask", "token_type_ids")
        }
        logits = self._model(**inputs).astype(mx.float32)
        mx.eval(logits)
        return logits[0].tolist()

    async def score_pairs(
        self, pairs: list[tuple[str, str]], instruction=None, use_activation=True
    ):
        if not self.is_reranker:
            raise ValueError(
                "/v1/score requires a single-label cross-encoder or an embedding model"
            )
        if not all(isinstance(a, str) and isinstance(b, str) for a, b in pairs):
            raise ValueError("Text rerankers require string query/documents")

        def score():
            import mlx.core as mx

            if not self._loaded:
                raise RuntimeError("Engine not started")
            scores = []
            for a, b in pairs:
                if self.kind == "head":
                    value = self._head_logits(a, b)[0]
                    scores.append(sigmoid(value) if use_activation else value)
                else:
                    ids = qwen_input_ids(
                        self._tokenizer,
                        a,
                        b,
                        instruction,
                        scoring_max_length(self._config),
                    )
                    logits = self._model(mx.array([ids]))[0, -1].astype(mx.float32)
                    yes = self._tokenizer.convert_tokens_to_ids("yes")
                    no = self._tokenizer.convert_tokens_to_ids("no")
                    gap = logits[yes] - logits[no]
                    mx.eval(gap)
                    value = float(gap.item())
                    scores.append(sigmoid(value) if use_activation else value)
            if not all(math.isfinite(s) for s in scores):
                raise ValueError("Non-finite classification logit")
            return scores

        return await self._run(score)

    async def rerank(self, query, documents, instruction=None, use_activation=True):
        return await self.score_pairs(
            [(query, d) for d in documents], instruction, use_activation
        )

    def embed(self, texts, normalize=True, **kwargs):
        raise ValueError(
            "Scoring checkpoints do not provide embeddings; load an embedding model"
        )

    def pool(self, texts, pooling_type="CLS"):
        raise ValueError(
            "Scoring checkpoints do not provide pooled embeddings; use /v1/score or /v1/classify"
        )

    async def classify(self, texts: list[str]):
        if not self.is_classifier:
            raise ValueError("This reranker has no sequence classification head")
        return await self._run(
            lambda: [
                probabilities(
                    self._head_logits(t),
                    self._config.get("problem_type") == "multi_label_classification",
                )
                for t in texts
            ]
        )

    @property
    def labels(self):
        mapping = self._config.get("id2label", {})
        return [mapping.get(str(i), f"LABEL_{i}") for i in range(self.num_labels)]
