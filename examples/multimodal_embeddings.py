"""Yunshu multimodal embeddings + reranking — copy-paste-runnable.

Qwen3-VL-Embedding embeds TEXT, IMAGES, and CROSS-MODAL (text+image) into one
shared vector space; Qwen3-VL-Reranker is a true cross-encoder that scores
query↔document relevance (including image documents).

Prereq: a Yunshu server in multi-model mode with both models available.

    uv pip install "yunshu[embeddings]"
    # put the two models under one dir (folder name = model id), e.g.:
    #   $MODELS/Qwen3-VL-Embedding-2B-8bit
    #   $MODELS/Qwen3-VL-Reranker-2B-8bit
    YUNSHU_MULTI_MODEL=1 YUNSHU_MODELS_DIR=$MODELS yunshu serve --port 8000

Then:
    pip install requests
    python examples/multimodal_embeddings.py path/to/an_image.png
"""

from __future__ import annotations

import base64
import math
import sys

import requests

BASE = "http://localhost:8000"
EMB_MODEL = "Qwen3-VL-Embedding-2B-8bit"
RERANK_MODEL = "Qwen3-VL-Reranker-2B-8bit"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def data_uri(path: str) -> str:
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:image/png;base64,{b64}"


def embed(inputs: list) -> list[list[float]]:
    """inputs: list of str or {"text"?, "image"?, "instruction"?} objects."""
    r = requests.post(
        f"{BASE}/v1/embeddings",
        json={"model": EMB_MODEL, "input": inputs},
        timeout=120,
    )
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


def rerank(query, documents: list, instruction: str | None = None) -> list[dict]:
    """query/documents: str or {"text"?, "image"?} objects."""
    payload = {"model": RERANK_MODEL, "query": query, "documents": documents}
    if instruction:
        payload["instruction"] = instruction
    r = requests.post(f"{BASE}/v1/rerank", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["results"]


def main(image_path: str | None) -> None:
    # 1) Plain text embeddings ------------------------------------------------
    print("TEXT EMBEDDINGS")
    vecs = embed(
        ["the cat sat on the mat", "a feline rested on a rug", "quarterly tax law"]
    )
    print(f"  dim={len(vecs[0])}")
    print(
        f"  cos(cat-sentence, paraphrase) = {cosine(vecs[0], vecs[1]):.3f}  (should be high)"
    )
    print(
        f"  cos(cat-sentence, tax-law)    = {cosine(vecs[0], vecs[2]):.3f}  (should be low)"
    )

    if not image_path:
        print("\n(pass an image path to run the cross-modal + image sections)")
        return

    # 2) Cross-modal: text query vs image, in the SAME space ------------------
    print("\nCROSS-MODAL EMBEDDINGS (text ↔ image)")
    q, img = embed(
        [
            {
                "text": "describe the picture",
                "instruction": "Retrieve relevant images.",
            },
            {"image": data_uri(image_path)},
        ]
    )
    print(
        f"  cos(text-query, image) = {cosine(q, img):.3f}  (related text↔image score)"
    )

    # 3) Cross-encoder rerank with an image document --------------------------
    print("\nMULTIMODAL RERANK (text query → image + text docs)")
    results = rerank(
        "what is shown in the photo",
        [
            {"image": data_uri(image_path)},
            "a recipe for chocolate cake",
            "an unrelated paragraph about stock markets",
        ],
        instruction="Retrieve the document that best matches the query.",
    )
    for item in results:
        doc = item["document"]
        label = "<image>" if "image" in doc else doc["text"][:40]
        print(f"  idx={item['index']} score={item['relevance_score']:.4f}  {label}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
