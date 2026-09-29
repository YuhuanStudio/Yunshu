"""Shared pieces of the accuracy-alignment harness (kld.py, greedy_div.py).

* fixed, versioned corpus (token ids only, stored under docs/research/accuracy)
* model loading and the two configurations under test:
    - ``stock``: mlx-vlm exactly as installed, no Yunshu kernel imported
    - ``serve``: what ``VLMEngine._build_batch_runner`` installs for a Qwen3.5
      family checkpoint (oMLX verify kernels, batch-invariant + NAX-packed
      projections, ragged KV), with the same activation switches the runner
      flips around each step.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
MAIN = Path(os.environ.get("YUNSHU_MAIN", str(ROOT)))
DATA = (
    MAIN / "docs/research/accuracy"
)  # private, gitignored; the main checkout keeps it
CACHE = Path(
    os.environ.get("YUNSHU_ACCURACY_CACHE", "~/.cache/yunshu/accuracy")
).expanduser()
EVAL_DATA = MAIN / "reference/omlx/omlx/eval/data"
CN_JSONL = Path(os.environ.get("YUNSHU_CN_JSONL", "train_CN.jsonl")).expanduser()
if not CN_JSONL.exists():
    print(
        f"note: CN corpus {CN_JSONL} not found; set YUNSHU_CN_JSONL to the jsonl path",
        file=sys.stderr,
    )
CODE_PIN = "7a6d3e2a"  # git commit the code corpus is read from

# A corpus is defined by this builder plus the pinned inputs above; bump the
# date when any of it changes so stored references are never mixed.
CORPUS_VERSION = "2026-09-29"
PREFILL_STEP = 2048  # the runner's PREFILL_STEP
TOPK = 64
BUCKETS = [(0, 512), (512, 2048), (2048, 8192), (8192, 1 << 30)]
BUCKET_NAMES = ["0-512", "512-2K", "2K-8K", "8K+"]


# ── corpus ───────────────────────────────────────────────────────────────
def _en_books() -> list[str]:
    """Three public-domain books (Project Gutenberg 1342, 2701, 1661), body only."""
    out = []
    for name in ("pg1342", "pg2701", "pg1661"):
        t = (CACHE / f"{name}.txt").read_text(encoding="utf-8", errors="replace")
        a = t.find("*** START")
        a = t.find("\n", a) + 1
        b = t.find("*** END")
        out.append(t[a:b].replace("\r\n", "\n").strip())
    return out


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]


def _code_files() -> list[str]:
    names = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", CODE_PIN, "python/yunshu_engine"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.split()
    out = []
    for n in sorted(names):
        if n.endswith(".py") and "/kernels/" not in n:
            src = subprocess.run(
                ["git", "show", f"{CODE_PIN}:{n}"],
                capture_output=True,
                text=True,
                check=True,
                cwd=ROOT,
            ).stdout
            if len(src) > 2000:
                out.append(f"# file: {n}\n{src}")
    return out


def _jsonl(path: Path, n: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def _zh_docs(n: int = 4000) -> list[str]:
    docs = []
    for r in _jsonl(CN_JSONL, n):
        conv = r["conversations"]
        docs.append("\n".join(c["value"] for c in conv))
    return [d for d in docs if len(d) > 300]


def _math_docs() -> list[str]:
    docs = []
    for r in _jsonl(EVAL_DATA / "gsm8k_test.jsonl", 1400):
        docs.append(f"Question: {r['question']}\nAnswer: {r['answer']}")
    return docs


def _pack(tok, docs: list[str], n_tokens: int, start: int = 0) -> list[int]:
    """Concatenate whole documents (blank-line separated) into one id list of
    exactly ``n_tokens``, starting at document ``start``."""
    ids: list[int] = []
    i = start
    while len(ids) < n_tokens:
        ids += tok.encode(docs[i % len(docs)] + "\n\n", add_special_tokens=False)
        i += 1
    return ids[:n_tokens]


# category -> (number of sequences, tokens each)
SHORT = 2048
PLAN = {
    "prose": (14, SHORT),
    "code": (10, SHORT),
    "zh": (10, SHORT),
    "math": (8, SHORT),
}
LONG = [  # (name, category, tokens)
    ("long_prose_32k", "prose", 32768),
    ("long_code_32k", "code", 32768),
    ("long_prose_8k", "prose", 8192),
    ("long_zh_8k", "zh", 8192),
    ("long_math_8k", "math", 8192),
]
# decode windows: (sequence name, first fed position, steps). Windows come in
# equal-length pairs so a stock batch of 2 can carry both.
DECODE_SHORT = [
    "prose_00",
    "prose_01",
    "code_00",
    "code_01",
    "zh_00",
    "zh_01",
    "math_00",
    "math_01",
]
DECODE_SHORT_START, DECODE_SHORT_STEPS = 256, 384
DECODE_LONG = ["long_prose_32k", "long_code_32k"]
DECODE_LONG_START, DECODE_LONG_STEPS = 31488, 128


def decode_windows() -> list[tuple[str, int, int]]:
    w = [(n, DECODE_SHORT_START, DECODE_SHORT_STEPS) for n in DECODE_SHORT]
    w += [(n, DECODE_LONG_START, DECODE_LONG_STEPS) for n in DECODE_LONG]
    return w


def build_corpus(tokenizer) -> dict:
    rng = np.random.default_rng(20260929)
    books = [_paragraphs(b) for b in _en_books()]
    prose = [p for b in books for p in b]
    order = rng.permutation(len(prose))
    prose = [prose[i] for i in order]
    code = _code_files()
    zh = _zh_docs()
    math = _math_docs()
    pools = {"prose": prose, "code": code, "zh": zh, "math": math}
    seqs: list[dict] = []
    cursor = dict.fromkeys(pools, 0)
    for cat, (n, length) in PLAN.items():
        for i in range(n):
            ids = _pack(tokenizer, pools[cat], length, cursor[cat])
            cursor[cat] += 6 if cat != "code" else 1
            seqs.append({"name": f"{cat}_{i:02d}", "cat": cat, "ids": ids})
    # long documents: contiguous text (a book / a code concatenation), not
    # shuffled paragraphs
    books_raw = _en_books()
    long_text = {
        "prose": books_raw[0],
        "code": "\n\n".join(code[12:]),
        "zh": "\n\n".join(zh[200:2200]),
        "math": "\n\n".join(math[400:]),
    }
    for name, cat, length in LONG:
        text = long_text[cat]
        if name == "long_prose_8k":
            text = books_raw[1][40000:]  # past the contents pages
        ids = tokenizer.encode(text, add_special_tokens=False)[:length]
        assert len(ids) == length, (name, len(ids))
        seqs.append({"name": name, "cat": cat, "ids": ids})
    return {"version": CORPUS_VERSION, "seqs": seqs}


def corpus_path() -> Path:
    return DATA / f"corpus-{CORPUS_VERSION}.npz"


def save_corpus(c: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        corpus_path(),
        names=np.array([s["name"] for s in c["seqs"]]),
        cats=np.array([s["cat"] for s in c["seqs"]]),
        lens=np.array([len(s["ids"]) for s in c["seqs"]]),
        ids=np.concatenate([np.array(s["ids"], dtype=np.int32) for s in c["seqs"]]),
        version=np.array(c["version"]),
    )


def load_corpus() -> dict:
    z = np.load(corpus_path())
    lens = z["lens"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    seqs = [
        {
            "name": str(z["names"][i]),
            "cat": str(z["cats"][i]),
            "ids": z["ids"][offs[i] : offs[i + 1]].astype(np.int64),
        }
        for i in range(len(lens))
    ]
    return {"version": str(z["version"]), "seqs": seqs}


def corpus_digest(c: dict) -> str:
    h = hashlib.sha256()
    for s in c["seqs"]:
        h.update(s["name"].encode())
        h.update(np.asarray(s["ids"], dtype=np.int32).tobytes())
    return h.hexdigest()[:16]


# ── models ───────────────────────────────────────────────────────────────
def load_model(path: str):
    """mlx-vlm load, nothing else. Returns (model, tokenizer)."""
    from mlx_vlm import load

    model, processor = load(path)
    return model, getattr(processor, "tokenizer", processor)


def setup_serve(model, *, driver: bool = False) -> dict:
    """Install exactly what ``VLMEngine._build_batch_runner`` installs for a
    Qwen3.5-family checkpoint with an MTP head, minus the drafter itself."""
    from yunshu_engine.kernels import ragged_kv
    from yunshu_engine.kernels.batch_invariant import install as install_invariant
    from yunshu_engine.kernels.batch_invariant import set_active
    from yunshu_engine.kernels.omlx import apply as apply_verify_kernels
    from yunshu_engine.kernels.omlx import is_nax_available

    info: dict = {}
    lm = model.language_model
    if driver:
        from yunshu_engine.kernels import lane_linear

        info["lanes"] = lane_linear.convert(lm)
        if lm.args.tie_word_embeddings:
            lm._yunshu_lane_head = lane_linear.lane_head(lm.model.embed_tokens)
    info["verify"] = apply_verify_kernels(row_exact=False)
    nax = is_nax_available()
    info["nax"] = nax
    info["invariant"] = install_invariant(lm, model=model, packed=nax)
    set_active(False)
    if ragged_kv.supports(lm):
        ragged_kv.install()
        ragged_kv.enable(None)
        info["ragged"] = True
    return info
