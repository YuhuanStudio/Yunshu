"""Wave 772 — verify the ANE (CoreML) embeddings path actually runs on-device and
matches the HF mask-weighted-mean reference. Run from repo root:

    PYTHONPATH=. uv run python scripts/realmodel/verify_ane_embeddings_w772.py

Requires coremltools + Apple Silicon (is_ane_available()). Uses the locally-cached
sentence-transformers/all-MiniLM-L6-v2 (a mean-pooled model).
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

from python.yunshu_engine.ane_embedding import ANEEmbeddingConfig, ANEEmbeddingProcessor

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    cfg = ANEEmbeddingConfig(model_name=MODEL, max_seq_length=128,
                             normalize_embeddings=False, compile_on_init=True)
    print("building CoreML model (torch->CoreML)...", flush=True)
    proc = ANEEmbeddingProcessor(cfg)
    print("is_compiled:", proc.is_compiled(), flush=True)

    texts = [
        "The quick brown fox.",
        "A longer sentence about cats and dogs, written to exercise padding well past a few tokens.",
    ]
    ane = np.array(proc.embed(texts))
    print("ANE output shape:", ane.shape, flush=True)

    # Ground truth: HF mask-weighted mean at the SAME seq length.
    import torch
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained(MODEL).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    with torch.no_grad():
        last = m(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    ref = ((last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).numpy()

    ok = True
    for i in range(len(texts)):
        cos = float(np.dot(ane[i], ref[i])
                    / (np.linalg.norm(ane[i]) * np.linalg.norm(ref[i]) + 1e-9))
        print(f"text {i}: cosine(ANE, HF mask-mean) = {cos:.4f}", flush=True)
        ok = ok and cos > 0.99
    passed = proc.is_compiled() and ok
    print("ANE W772 VERIFY:", "PASS (ANE ran on-device, matches reference)" if passed else "FAIL",
          flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
