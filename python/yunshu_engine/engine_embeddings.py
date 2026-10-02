from __future__ import annotations

"""Engine embeddings extracted from batched_engine.

Patchable helpers resolve through the compatibility facade. Concrete self types
retain the shared BatchedEngine state; misc ignores allow that mixin self type.
"""

from typing import Any


class EngineEmbeddingsMixin:
    def _resolve_embedding_pooling(self: _engine.BatchedEngine) -> str:  # type: ignore[misc]
        """The model's intended sentence-embedding pooling: MEAN | CLS | LAST.

        embed() hardcoded MEAN, but CLS-pooled models (the BGE
        family — BAAI/bge-*) are trained to use the CLS token; mean-pooling them yields
        vectors in the wrong pooling space (degraded retrieval). Detect via the
        sentence-transformers `1_Pooling/config.json` shipped in the model dir. Defaults
        to MEAN whenever the dir/config is absent or ambiguous → ZERO regression for the
        mean-pooled models (E5/Nomic/GTE) already served. Cached per engine.
        """
        cached = getattr(self, "_embed_pooling", None)
        if cached is not None:
            return cached
        pooling = "MEAN"
        try:
            import json as _json
            import os as _os

            cand = self.model_name if isinstance(self.model_name, str) else ""
            # Resolve a HF repo id (e.g. "BAAI/bge-base-en-v1.5") to its local
            # snapshot before looking for 1_Pooling/config.json. The old code joined the RAW
            # name → "BAAI/bge-base-en-v1.5/1_Pooling/config.json", a relative path that never
            # exists on disk → isfile False → every repo-id-served model fell through to MEAN.
            # That silently MEAN-pooled CLS-trained BGE models (the exact bug this fix
            # targets), only working if the operator passed an explicit local dir. Mirror the
            # engine's own load-path resolution (line 1203-1206).
            if cand and not _os.path.isdir(cand):
                try:
                    from mlx_lm.utils import hf_repo_to_path

                    cand = str(hf_repo_to_path(self.model_name))
                except Exception:
                    pass
            pcfg = _os.path.join(cand, "1_Pooling", "config.json")
            if cand and _os.path.isfile(pcfg):
                with open(pcfg) as _f:
                    d = _json.load(_f)
                if d.get("pooling_mode_cls_token"):
                    pooling = "CLS"
                elif d.get("pooling_mode_lasttoken"):
                    pooling = "LAST"
        except Exception:
            pooling = "MEAN"
        self._embed_pooling = pooling
        return pooling

    def embed(  # type: ignore[misc]
        self: _engine.BatchedEngine, texts: list[str], normalize: bool = True
    ) -> list[list[float]]:
        """Generate embeddings for the given texts.

        Uses the model's transformer backbone (before LM head) to extract hidden states,
        applies the model's intended pooling (MEAN/CLS/LAST — see
        _resolve_embedding_pooling), and optionally L2-normalizes the result (default:
        True, matching the OpenAI embeddings API contract).

        Runs synchronously — callers in async contexts should wrap with
        ``await loop.run_in_executor(get_mlx_executor(), engine.embed, texts)``.
        """
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Engine not loaded — call start() first")

        import mlx.core as mx

        # Resolve the transformer backbone (before LM head projection).
        backbone = self._get_backbone()
        _pooling = self._resolve_embedding_pooling()

        embeddings: list[list[float]] = []
        for text in texts:
            tokens = self._tokenizer.encode(text)
            if not tokens:
                hidden_size = self._get_hidden_size()
                embeddings.append([0.0] * hidden_size)
                continue

            input_ids = mx.array([tokens])

            if backbone is not None:
                hidden = backbone(input_ids)
            else:
                output = self._model(input_ids)
                hidden = self._extract_hidden_states(output)

            # Pool over the sequence per the model's intended convention.
            if _pooling == "CLS":
                pooled = hidden[:, 0, :].squeeze(0)
            elif _pooling == "LAST":
                pooled = hidden[:, -1, :].squeeze(0)
            else:
                pooled = mx.mean(hidden, axis=1).squeeze(0)

            if normalize:
                norm = mx.sqrt(mx.sum(pooled * pooled) + 1e-12)
                pooled = pooled / norm

            embeddings.append(pooled.tolist())

        return embeddings

    def _compute_prompt_logprobs_sync(  # type: ignore[misc]
        self: _engine.BatchedEngine, input_ids: list[int], top_k: int = 0
    ) -> list:
        """Compute per-prompt-token logprobs (eval/perplexity).

        Runs ONE forward over the prompt (separate from generation — does not
        touch the decode hot path) and returns a vLLM-shaped list aligned to the
        prompt tokens: element i is the logprob the model assigned to the actual
        token at position i given positions <i. Element 0 is None (no context).
        Each non-null element: {"token_id", "logprob", optional "top_logprobs":
        [{"token_id","logprob"}...]}. top_k=0 → only the realized token.

        Caller runs this on the MLX executor. Length-capped by the caller.
        """
        import mlx.core as mx

        if not input_ids or len(input_ids) < 2:
            return [None] * len(input_ids)
        ids = mx.array([input_ids])
        out = self._model(ids)
        logits = out[0] if not isinstance(out, mx.array) else out
        # logits: [1, seq, vocab] → drop batch dim
        logits = logits[0] if logits.ndim == 3 else logits
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        result: list = [None]  # position 0 has no preceding context
        n = len(input_ids)
        k = int(top_k) if top_k and top_k > 0 else 0
        # position i predicts token at i+1: use row i for the realized token i+1.
        for i in range(n - 1):
            tgt = int(input_ids[i + 1])
            row = logp[i]
            entry: dict = {"token_id": tgt, "logprob": float(row[tgt].item())}
            if k > 0:
                topk_idx = mx.argpartition(-row, kth=min(k, row.shape[-1] - 1))[:k]
                topk = sorted(
                    ((int(t), float(row[int(t)].item())) for t in topk_idx.tolist()),
                    key=lambda x: -x[1],
                )
                entry["top_logprobs"] = [
                    {"token_id": t, "logprob": lp} for t, lp in topk
                ]
            result.append(entry)
        return result

    async def _compute_prompt_logprobs_for(  # type: ignore[misc]
        self: _engine.BatchedEngine,
        prompt,
        enable_thinking,
        top_k: int,
    ) -> list | None:
        """Async wrapper: derive input_ids from the prompt (templating chat
        messages) and compute prompt logprobs on the MLX executor. Length-gated
        to bound the [seq, vocab] logits memory."""
        cap = 8192
        if isinstance(prompt, list):
            text = self._apply_chat_template(prompt, enable_thinking=enable_thinking)
        else:
            text = prompt
        input_ids = self._encode_prompt(self._tokenizer, text)
        if not input_ids:
            return None
        if len(input_ids) > cap:
            _engine.logger.warning(
                "prompt_logprobs skipped: prompt %d tokens exceeds cap %d "
                "(logits memory bound)",
                len(input_ids),
                cap,
            )
            return None
        import asyncio as _asyncio

        from .mlx_executor import get_mlx_executor

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_mlx_executor(),
            lambda: self._compute_prompt_logprobs_sync(input_ids, top_k),
        )

    def pool(  # type: ignore[misc]
        self: _engine.BatchedEngine, texts: list[str], pooling_type: str = "MEAN"
    ) -> list[list[float]]:
        """Extract pooled hidden states for the given texts.

        Args:
            texts: List of input strings.
            pooling_type: One of "MEAN", "CLS", "LAST".

        Returns:
            List of pooled hidden-state vectors (NOT normalized).
        """
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Engine not loaded — call start() first")

        import mlx.core as mx

        # Pool over the transformer's HIDDEN states (before the LM head), matching
        # embed() and the OpenAI/vLLM pooling contract. Using self._model() here
        # would pool the vocab-space LOGITS instead — a different (huge) dim that
        # can't be compared with embed()'s output. See _get_backbone().
        backbone = self._get_backbone()

        results: list[list[float]] = []
        for text in texts:
            tokens = self._tokenizer.encode(text)
            if not tokens:
                hidden_size = self._get_hidden_size()
                results.append([0.0] * hidden_size)
                continue

            input_ids = mx.array([tokens])
            if backbone is not None:
                hidden = backbone(input_ids)
            else:
                output = self._model(input_ids)
                hidden = self._extract_hidden_states(output)

            if pooling_type.upper() == "CLS":
                pooled = hidden[0, 0, :]  # batch=0, first token
            elif pooling_type.upper() == "LAST":
                pooled = hidden[0, -1, :]  # batch=0, last token
            else:  # MEAN
                pooled = mx.mean(hidden, axis=1).squeeze(
                    0
                )  # [1, seq, d] -> [1, d] -> [d]

            results.append(pooled.tolist())

        return results

    def _extract_hidden_states(self: _engine.BatchedEngine, output) -> Any:  # type: ignore[misc]
        """Extract hidden states from various model output formats.

        MLX models return either:
        - A plain mx.array (the logits or hidden states)
        - A tuple/list where the first element is hidden states
        - An object with .last_hidden_state attribute
        """
        import mlx.core as mx

        if isinstance(output, mx.array):
            hidden = output  # [batch, seq_len, hidden_size] — OR vocab-space logits
        elif isinstance(output, (tuple, list)):
            hidden = output[0]
        elif hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        else:
            # Fallback: try subscript access, otherwise return as-is
            try:
                hidden = output[0]
            except (TypeError, IndexError):
                return output
        # This is only reached when _get_backbone() returned None, so we ran
        # the FULL model and `hidden` may be vocab-space LOGITS, not hidden states.
        # Pooling those gives a wrong-dimensioned, meaningless embedding. Warn loudly
        # (once) when the last dim doesn't match the model's hidden_size — the router
        # fallback already has this awareness; the engine path was silent.
        try:
            _hs = self._get_hidden_size()
            if (
                getattr(hidden, "ndim", 0) >= 1
                and _hs
                and hidden.shape[-1] != _hs
                and not getattr(self, "_warned_logits_pool", False)
            ):
                _engine.logger.warning(
                    "embed/pool: no transformer backbone resolved for %s — pooling raw "
                    "model output of dim %d (expected hidden_size %d). This is likely "
                    "vocab-space LOGITS → embeddings will be wrong-dimensioned and "
                    "semantically meaningless. Use a model with a recognizable backbone.",
                    self.model_name,
                    hidden.shape[-1],
                    _hs,
                )
                self._warned_logits_pool = True
        except Exception:
            pass
        return hidden

    def _get_hidden_size(self: _engine.BatchedEngine) -> int:  # type: ignore[misc]
        """Get the model's hidden dimension size."""
        if hasattr(self._model, "config"):
            cfg = self._model.config
            for attr in ("hidden_size", "d_model", "n_embd", "embed_dim"):
                val = getattr(cfg, attr, None)
                if val is not None:
                    return val
        # Check args (MLX models store config in .args)
        args = getattr(self._model, "args", None)
        if args is not None:
            # Text config may be nested
            text_config = getattr(args, "text_config", None)
            for obj in (text_config, args):
                if obj is not None:
                    for attr in ("hidden_size", "d_model", "n_embd", "embed_dim"):
                        val = getattr(obj, attr, None)
                        if val is not None:
                            return val
        # Heuristic: check model layers
        if hasattr(self._model, "layers") and len(self._model.layers) > 0:
            layer = self._model.layers[0]
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "fc1"):
                return layer.mlp.fc1.weight.shape[1]
        # Default fallback for common models
        return 768

    def _get_backbone(self):
        """Resolve the transformer backbone (before LM head projection).

        MLX models have varying structures:
        - Qwen3.5: model.language_model.model (inner Qwen3_5TextModel)
        - Standard HF: model.model
        - Some use model.transformer

        The backbone returns hidden states of shape [batch, seq, hidden_size]
        without projecting through the vocabulary head.
        """
        m = self._model
        # Qwen3.5 and similar: model.language_model.model
        lm = getattr(m, "language_model", None)
        if lm is not None:
            inner = getattr(lm, "model", None)
            if inner is not None and callable(inner):
                return inner
        # Standard: model.model
        inner = getattr(m, "model", None)
        if inner is not None and callable(inner):
            return inner
        # Some architectures: model.transformer
        inner = getattr(m, "transformer", None)
        if inner is not None and callable(inner):
            return inner
        return None


# Resolve the facade after definitions to also support direct module imports.
from . import batched_engine as _engine
