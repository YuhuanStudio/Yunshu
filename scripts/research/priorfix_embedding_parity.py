"""Published EmbeddingGemma2 formats: same-device upstream and fp32 oracle parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(a, b, floor):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (
        a.shape != b.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("invalid parity vectors")
    cosine = np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    )
    return {
        "min_cosine": float(np.min(cosine)),
        "passed": bool(np.all(cosine >= floor)),
        "max_abs": float(np.max(np.abs(a - b))),
    }


def processor_payload(item):
    """CPU-checkable fixture normalization, before any model is allocated."""
    from yunshu_engine.embedding_gemma2 import (
        _load_audio,
        _load_image,
        _load_video,
        build_text,
    )

    text, media = build_text({"text": item} if isinstance(item, str) else item)
    kw = {"text": [text], "return_tensors": "np"}
    if "image" in media:
        kw["images"] = [[_load_image(x) for x in media["image"]]]
    if "audio" in media:
        kw["audio"] = [_load_audio(x) for x in media["audio"]]
    if "video" in media:
        kw["videos"] = [[_load_video(x) for x in media["video"]]]
    return kw


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def run(args):
    import mlx.core as mx
    from egemma2_parity import TASK_TEXTS, _cases, compare_sets

    from yunshu_engine.embedding_gemma2 import (
        EmbeddingGemma2,
        resolve_prompt,
    )
    from yunshu_engine.embedding_gemma2_loader import load_published_model

    root = Path(args.reference).parent
    cs = _cases(str(root))
    payloads = {name: processor_payload(item) for name, item in cs.items()}
    model = EmbeddingGemma2(args.model)
    # Fresh upstream loader instance, direct processor/model invocation; no Yunshu
    # batching, feature scatter, mean-pooling or task adapter in the oracle arm.
    upstream = load_published_model(Path(args.model))
    rows, single = [], {}
    for name, item in cs.items():
        kw = payloads[name]
        prepared = model.processor(**kw)
        expected = upstream(**{k: mx.array(v) for k, v in prepared.items()}).text_embeds
        mx.eval(expected)
        got = model.embed_items([item])[0]
        result = compare(got, np.array(expected), 0.999999)
        rows.append({"case": name, **result})
        if not result["passed"]:
            raise ValueError(f"upstream parity failed: {rows[-1]}")
        single[name] = got[0].tolist()
    batch = model.embed_items(list(cs.values()))[0]
    got_sets = {"plain": single, "batch": dict(zip(cs, batch.tolist(), strict=True))}
    for task in ("SearchQuery", "Document"):
        vectors = model.embed_items(TASK_TEXTS, task=task)[0]
        got_sets[task] = dict(zip(TASK_TEXTS, vectors.tolist(), strict=True))
        prefix = resolve_prompt(model.prompts, task, None)
        inputs = model.processor(
            text=[prefix + t for t in TASK_TEXTS], padding=True, return_tensors="np"
        )
        expected = upstream(**{k: mx.array(v) for k, v in inputs.items()}).text_embeds
        mx.eval(expected)
        rows.append({"case": task, **compare(vectors, np.array(expected), 0.999999)})
    names = list(cs)[:3]
    got_sets["dims256"] = dict(
        zip(
            names,
            model.embed_items([cs[n] for n in names], dims=256)[0].tolist(),
            strict=True,
        )
    )
    # Quantization changes accuracy: fp32 floor applies only to bf16 format.
    fp32_rows = compare_sets(json.loads(Path(args.reference).read_text()), got_sets)
    quantized = bool(model.config.get("quantization"))
    floor = 0.99985 if not quantized else None
    raw_multishard = None
    shards = sorted(Path(args.model).glob("*.safetensors"))
    if len(shards) > 1:
        raw = EmbeddingGemma2(args.model, dtype="float32")
        name = next(
            n
            for n, item in cs.items()
            if isinstance(item, str) or list(item) == ["text"]
        )
        vector = raw.embed_items([cs[name]])[0]
        reference = json.loads(Path(args.reference).read_text())["plain"][name]
        raw_multishard = compare(vector, [reference], 0.99985)
        if not raw_multishard["passed"]:
            raise ValueError(f"raw multi-shard regression failed: {raw_multishard}")
    passed = all(r["passed"] for r in rows) and (
        floor is None or all(r[2] >= floor for r in fp32_rows)
    )
    return {
        "complete": True,
        "shard_count": len(shards),
        "raw_multishard": raw_multishard,
        "passed": passed,
        "device": "M5",
        "model": args.model,
        "upstream": rows,
        "fp32_cases": len(fp32_rows),
        "fp32_floor": floor,
        "fp32_min_cosine": min(r[2] for r in fp32_rows),
        "engaged": "mlx-vlm embedding loader (native or pinned MIT fallback)",
    }


def main():
    args = parser().parse_args()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "model": args.model}))
        return 0
    result = run(args)
    Path(args.out).write_text(json.dumps(result) + "\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
