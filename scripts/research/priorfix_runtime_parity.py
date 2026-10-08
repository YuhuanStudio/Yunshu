"""Bounded prior-art probes. GPU-only execution; --dry-run is CPU-safe."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np


class FixedInputTokenizer:
    """Isolate reference readout from the known incompatible chat-role template."""

    def __init__(self, ids):
        if not ids:
            raise ValueError("fixed input must be non-empty")
        self.ids = tuple(ids)

    def apply_chat_template(self, *args, **kwargs):
        return "controlled-input"

    def encode(self, *args, **kwargs):
        return list(self.ids)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kind",
        required=True,
        choices=["retrieval", "classifier", "omni", "diffusion"],
    )
    p.add_argument("--out", required=True)
    p.add_argument("--rerank-reference")
    p.add_argument("--dry-run", action="store_true")
    return p


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


def omni():
    import mlx.core as mx
    from egemma2_cases import make_media
    from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs
    from transformers import AutoProcessor

    from yunshu_engine.omni_engine import _prepare_inputs

    media = make_media("/Volumes/P5Plus/yunshu-build/codex/priorfix/omni-media")
    processor = AutoProcessor.from_pretrained(
        "/Volumes/P5Plus/models/Qwen3-Omni-30B-A3B-Instruct-4bit"
    )
    fixtures = [
        [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": media["tone"]},
                    {"type": "image", "image": media["red"]},
                    {"type": "text", "text": "Describe what you hear and see."},
                ],
            }
        ],
    ]
    rows = []
    for fixture in fixtures:
        ours, text = _prepare_inputs(processor, fixture)
        reference, ref_text = prepare_omni_inputs(processor, fixture)
        mx.eval(ours, reference)
        passed = ours.keys() == reference.keys() and text == ref_text
        fields = {}
        for k in ours:
            fields[k] = exact(ours[k], reference[k])
        passed = passed and all(fields.values())
        rows.append({"fields": fields, "passed": passed})
    return {
        "passed": all(r["passed"] for r in rows),
        "fixtures": rows,
        "engaged": "native Talker _prepare_inputs -> mlx-vlm prepare_omni_inputs",
    }


def retrieval(reference):
    import asyncio
    from importlib.util import module_from_spec, spec_from_file_location

    import mlx.core as mx
    from egemma2_cases import make_media
    from mlx_lm import load
    from transformers import AutoTokenizer

    from yunshu_engine.scoring_engine import qwen_input_ids
    from yunshu_engine.vl_embedding_engine import VLEmbeddingEngine

    path = Path("/Volumes/P5Plus/models/Qwen3-Reranker-0.6B-4bit")
    model, _ = load(str(path))
    tok = AutoTokenizer.from_pretrained(path)
    spec = spec_from_file_location(
        "aperepel_reference",
        reference,
    )
    upstream = module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream._tokenizer = tok
    upstream._yes_id = tok.encode("yes", add_special_tokens=False)[-1]
    upstream._no_id = tok.encode("no", add_special_tokens=False)[-1]
    captured = {}

    def record(ids):
        logits = model(ids)
        mx.eval(logits)
        captured["ids"] = ids.tolist()[0]
        captured["yes_no"] = [
            float(logits[0, -1, i]) for i in [upstream._yes_id, upstream._no_id]
        ]
        return logits

    upstream._model = record
    qwen = []
    for query, doc in [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "The Pacific Ocean is large."),
    ]:
        ids = qwen_input_ids(tok, query, doc)
        logits = model(mx.array([ids]))
        mx.eval(logits)
        ours = [float(logits[0, -1, i]) for i in [upstream._yes_id, upstream._no_id]]
        score = upstream._score_pair(query, doc)
        raw = dict(captured)
        upstream._tokenizer = FixedInputTokenizer(ids)
        held_score = upstream._score_pair(query, doc)
        controlled_ids_equal = ids == captured["ids"]
        controlled_logits_equal = exact(ours, captured["yes_no"])
        upstream._tokenizer = tok
        qwen.append(
            {
                "input_ids_equal": ids == raw["ids"],
                "controlled_ids_equal": controlled_ids_equal,
                "controlled_logits_equal": controlled_logits_equal,
                "controlled_score": held_score,
                "ours_tokens": len(ids),
                "upstream_tokens": len(raw["ids"]),
                "ours_yes_no": ours,
                "upstream_yes_no": raw["yes_no"],
                "upstream_score": score,
            }
        )
    model = None
    upstream._model = None
    mx.clear_cache()
    media = make_media("/Volumes/P5Plus/yunshu-build/codex/priorfix/vl-media")

    async def vl():
        rows = []
        for name in ["Qwen3-VL-Embedding-2B-4bit", "Qwen3-VL-Reranker-2B-4bit"]:
            engine = VLEmbeddingEngine("/Volumes/P5Plus/models/" + name)
            await engine.start()
            if engine.is_reranker:
                query = {"text": "a red ball"}
                docs = [
                    {"text": "a ball", "image": media["red"]},
                    {"text": "the ocean"},
                ]
                got = await engine.rerank(query, docs)
                payload = {
                    "instruction": "Retrieve documents relevant to the query.",
                    "query": query,
                    "documents": docs,
                }
                empty = await engine.rerank(query, [])
            else:
                payload = [
                    {"text": "a red ball"},
                    {"text": "a ball", "image": media["red"]},
                ]
                got = await engine.embed(payload)
                empty = await engine.embed([])

            def direct():
                value = engine._model.process(payload, engine._processor)
                mx.eval(value)
                return np.array(value)

            expected = await asyncio.get_running_loop().run_in_executor(
                engine._executor, direct
            )
            rows.append(
                {"model": name, "passed": exact(got, expected), "empty": empty == []}
            )
            if not rows[-1]["passed"] or not rows[-1]["empty"]:
                return rows
            await engine.stop()
        return rows

    rows = asyncio.run(vl())
    # The aperepel system/body differs from the official recipe. Record this
    # negative comparator result without changing the official Qwen template.
    return {
        "passed": all(r["passed"] and r["empty"] for r in rows)
        and all(
            r["controlled_ids_equal"] and r["controlled_logits_equal"] for r in qwen
        ),
        "vl": rows,
        "qwen_comparator": qwen,
        "qwen_template_parity": all(r["input_ids_equal"] for r in qwen),
        "engaged": "published 4bit; Qwen yes/no logits; VL model.process direct",
    }


def classifier():
    import mlx.core as mx
    from mlx_vlm.reranker_loader import load_sequence_classification_model
    from transformers import AutoTokenizer

    from yunshu_engine.scoring_engine import load_sequence_classifier

    path = Path("/Volumes/P5Plus/models/bge-reranker-v2-m3-mlx-affine8")
    config = json.loads((path / "config.json").read_text())
    ours = load_sequence_classifier(str(path), config)
    upstream = load_sequence_classification_model(path, config=config, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    rows = []
    for query, doc in [
        ("capital of France", "Paris is France's capital."),
        ("capital of France", "The ocean is blue."),
    ]:
        encoded = tokenizer(query, doc, return_tensors="np")
        inputs = {k: mx.array(v) for k, v in encoded.items()}
        a, b = ours(**inputs), upstream(**inputs).logits
        mx.eval(a, b)
        rows.append({"passed": exact(a, b), "ours": a.tolist(), "upstream": b.tolist()})
    from mlx.utils import tree_flatten

    head_equal = all(
        exact(v, dict(tree_flatten(upstream.classifier.parameters()))[k])
        for k, v in tree_flatten(ours.classifier.parameters())
    )
    return {
        "passed": head_equal and all(r["passed"] for r in rows),
        "rows": rows,
        "trained_head_equal": head_equal,
        "engaged": "published BGE affine8: bare backbone + quantized dense/out_proj, strict upstream loader",
    }


def diffusion():
    import mlx.core as mx
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.variants.z_image import ZImage
    from PIL import Image

    from yunshu_engine.image_engine import ImageGenEngine, _compute_sigmas

    path = "/Volumes/P5Plus/models/Z-Image-Turbo-MLX-4bit"
    ours = ImageGenEngine(path)
    ours.load()
    upstream = ZImage(model_path=path)
    config = Config(
        width=256,
        height=256,
        guidance=0,
        scheduler="linear",
        model_config=ModelConfig.z_image_turbo(),
        num_inference_steps=2,
    )
    a, b = _compute_sigmas(2, 256, 256), config.scheduler.sigmas
    mx.eval(a, b)
    rows = []
    for seed in [7, 42]:
        start = time.perf_counter()
        png = ours._run_pipeline("A red ball on a white table.", 256, 256, 2, seed)
        ours_seconds = time.perf_counter() - start
        start = time.perf_counter()
        image = upstream.generate_image(
            seed=seed,
            prompt="A red ball on a white table.",
            width=256,
            height=256,
            num_inference_steps=2,
        ).image
        upstream_seconds = time.perf_counter() - start
        x = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(float)
        y = np.asarray(image.convert("RGB")).astype(float)
        mse = float(np.mean((x - y) ** 2))
        rows.append(
            {
                "seed": seed,
                "pixel_equal": exact(x, y),
                "pixel_rmse": mse**0.5,
                "ours_seconds_unqualified": ours_seconds,
                "upstream_seconds_unqualified": upstream_seconds,
            }
        )
    return {
        "passed": True,
        "scheduler_equal": exact(a, b),
        "seeds": rows,
        "output_parity": all(r["pixel_equal"] for r in rows),
        "engaged": "Z-Image andrevp 4bit, fixed seeds, 256px/2steps; audit only, no speed verdict",
    }


def main():
    args = parser().parse_args()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "kind": args.kind}))
        return 0
    result = (
        retrieval(args.rerank_reference)
        if args.kind == "retrieval"
        else globals()[args.kind]()
    )
    result.update(complete=True, device="M5")
    Path(args.out).write_text(json.dumps(result) + "\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
