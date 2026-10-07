"""EmbeddingGemma 2 parity: our MLX model vs the official sentence-transformers vectors.

  egemma2_parity.py ref     MODEL OUT_DIR          # CPU, torch fp32 (no MLX): run outside gpuq
  egemma2_parity.py mlx     MODEL OUT_DIR          # direct MLX model (run through gpuq)
  egemma2_parity.py compare OUT_DIR                # CPU: cosines per case, exits 1 under the floor

Cases come from egemma2_cases.py (mixed-length text incl. >1.5K tokens, images, audio, interleaved,
video frames), plus task prompts (SearchQuery / Document) and dimensions=256."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egemma2_cases as cases_mod  # noqa: E402
import egemma2_reference as ref  # noqa: E402

FLOOR = 0.999
TASK_TEXTS = ["What causes the northern lights?", "Sun particles hit the atmosphere."]


def _cases(out_dir):
    m = cases_mod.make_media(os.path.join(out_dir, "media"))
    return cases_mod.cases(m)


def cmd_ref(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cs = _cases(out_dir)
    res = {}
    names = list(cs)
    vecs = ref.reference_embed(model, [cs[n] for n in names])
    res["plain"] = dict(zip(names, vecs, strict=True))
    for task in ("SearchQuery", "Document"):
        res[task] = dict(
            zip(
                TASK_TEXTS,
                ref.reference_embed(model, TASK_TEXTS, task=task),
                strict=True,
            )
        )
    res["dims256"] = dict(
        zip(
            names[:3],
            ref.reference_embed(model, [cs[n] for n in names[:3]], dims=256),
            strict=True,
        )
    )
    json.dump(res, open(os.path.join(out_dir, "ref.json"), "w"))
    print("reference:", {k: len(v) for k, v in res.items()})


def cmd_mlx(model, out_dir):
    from yunshu_engine.embedding_gemma2 import EmbeddingGemma2

    cs = _cases(out_dir)
    m = EmbeddingGemma2(model)
    names = list(cs)
    res: dict = {}
    single = {}
    for n in names:  # one item per call
        single[n] = m.embed_items([cs[n]])[0][0].tolist()
    res["plain"] = single
    batch, _ = m.embed_items(
        [cs[n] for n in names]
    )  # all together (padding, mixed modalities)
    res["batch"] = dict(zip(names, (b.tolist() for b in batch), strict=True))
    for task in ("SearchQuery", "Document"):
        v, _ = m.embed_items(TASK_TEXTS, task=task)
        res[task] = dict(zip(TASK_TEXTS, (x.tolist() for x in v), strict=True))
    v, _ = m.embed_items([cs[n] for n in names[:3]], dims=256)
    res["dims256"] = dict(zip(names[:3], (x.tolist() for x in v), strict=True))
    json.dump(res, open(os.path.join(out_dir, "mlx.json"), "w"))
    print("mlx:", {k: len(v) for k, v in res.items()})


def compare_sets(refd, got):
    """[(set, case, cosine)] for every case present in both."""
    rows = []
    for s, cs in got.items():
        rs = refd["plain" if s == "batch" else s]
        for n, v in cs.items():
            rows.append((s, n, ref.cosine(v, rs[n])))
    return rows


def cmd_compare(out_dir):
    refd = json.load(open(os.path.join(out_dir, "ref.json")))
    got = json.load(open(os.path.join(out_dir, "mlx.json")))
    rows = compare_sets(refd, got)
    bad = [r for r in rows if r[2] < FLOOR]
    for s, n, c in rows:
        print(f"{s:12s} {n:16s} cos={c:.5f}{'  <-- FAIL' if c < FLOOR else ''}")
    print(
        f"min cos {min(r[2] for r in rows):.5f} over {len(rows)} vectors; {len(bad)} below {FLOOR}"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ref":
        cmd_ref(sys.argv[2], sys.argv[3])
    elif cmd == "mlx":
        cmd_mlx(sys.argv[2], sys.argv[3])
    elif cmd == "compare":
        sys.exit(cmd_compare(sys.argv[2]))
    else:
        sys.exit(f"unknown command {cmd}")
