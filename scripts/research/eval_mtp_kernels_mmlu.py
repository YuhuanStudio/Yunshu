"""MMLU-Pro drift/accuracy check: AR vs MTP (+ oMLX verify kernels) on Qwen3.8.

Same prompt format and answer extraction as oMLX's built-in MMLU-Pro eval
(thinking on, max 2048 tokens, greedy), questions taken from the user's own
oMLX run file so accuracy is comparable. Each question runs three configs in
turn on one loaded model:

  ar          plain autoregressive decode (reference)
  mtp_exact   MTP draft + bit-exact oMLX kernels (GDN prework/replay, sdpa split)
  mtp_fast    same + oMLX verify_qmm (not bit-exact; tail-ULP differences)

With ``--dflash PATH`` the draft is a DFlash2 block-diffusion drafter instead
of the MTP head and the configs are ``ar``, ``dflash_exact``, ``dflash_fast``.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/research/eval_mtp_kernels_mmlu.py \
        MODEL_DIR QUESTIONS_JSON N OUT.jsonl [--block 4]
"""

import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "reference" / "omlx"))

import mlx_vlm.speculative.mtp as _mtp  # noqa: E402
from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate.ar import BatchGenerator  # noqa: E402
from omlx.patches import (  # noqa: E402
    qwen35_gdn_prework,
    qwen35_gdn_verify_fused,
    qwen35_verify_qmm,
    qwen35_verify_sdpa_split,
)
from omlx.patches.mlx_vlm_mtp import qwen35_verify_linear  # noqa: E402

from yunshu_engine.mlxvlm_mtp import _load_drafter_in_memory  # noqa: E402
from yunshu_engine.mrope import clear_rope_state  # noqa: E402

model_dir, qfile, n, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
block = int(sys.argv[sys.argv.index("--block") + 1]) if "--block" in sys.argv else 4
dflash_path = (
    sys.argv[sys.argv.index("--dflash") + 1] if "--dflash" in sys.argv else None
)
configs = (
    ("ar", "dflash_exact", "dflash_fast")
    if dflash_path
    else ("ar", "mtp_exact", "mtp_fast")
)

qwen35_verify_linear.apply()
qwen35_gdn_prework.apply_qwen35_gdn_prework_patch()
qwen35_verify_sdpa_split.apply_qwen35_verify_sdpa_split_patch()
qwen35_verify_qmm.apply_verify_qmm_patch()
qwen35_gdn_verify_fused.apply_arrays_cache_replay_patch()
ARM = {"on": False}
import mlx_vlm.speculative.dflash as _dfl  # noqa: E402


def _arm(module, name):
    orig = getattr(module, name)

    def wrapped(*a, **k):
        qwen35_verify_qmm.set_verify_qmm_armed(ARM["on"])
        try:
            return orig(*a, **k)
        finally:
            qwen35_verify_qmm.set_verify_qmm_armed(False)

    setattr(module, name, wrapped)


_arm(_mtp, "_mtp_verify_target")
_arm(_dfl, "_dflash_verify_greedy")
_arm(_dfl, "_dflash_verify")

wanted = [q["id"] for q in json.loads(Path(qfile).read_text())["questions"]][:n]
data = {}
for line in (
    (ROOT / "reference/omlx/omlx/eval/data/mmlu_pro_test.jsonl")
    .read_text()
    .splitlines()
):
    item = json.loads(line)
    data[item["id"]] = item


def prompt_of(item):
    parts = [
        "Answer the following question. Answer with just the letter.\n",
        f"Question: {item['question']}\n",
    ]
    parts += [
        f"{lab}. {ch}" for lab, ch in zip(item["labels"], item["choices"], strict=False)
    ]
    parts.append("\nAnswer:")
    return "\n".join(parts)


def extract(resp, letters):
    up = resp.strip().upper()
    pl = "".join(letters)
    m = re.findall(r"(?:answer\s*(?:is|:)\s*)([" + pl + r"])\b", up, re.IGNORECASE)
    if m:
        return m[-1]
    m = re.findall(r"\b([" + pl + r"])\b", up)
    if m:
        return m[-1]
    return up[:1] if up[:1] in letters else ""


model, processor = load(model_dir)
tok = processor.tokenizer
if dflash_path:
    from mlx_vlm.speculative.drafters import load_drafter  # noqa: E402

    drafter, draft_kind = load_drafter(dflash_path, kind="dflash")
else:
    drafter, draft_kind = _load_drafter_in_memory(model_dir), "mtp"
lm = model.language_model


def run(ids, cfg, max_tokens=2048):
    use_draft = cfg != "ar"
    ARM["on"] = cfg.endswith("_fast")
    gen = BatchGenerator(
        lm,
        processor,
        max_tokens=max_tokens,
        draft_model=drafter if use_draft else None,
        draft_kind=draft_kind if use_draft else None,
        draft_block_size=block if use_draft else None,
        greedy_sampling=True,
        compute_logprobs=False,
    )
    clear_rope_state(model)
    kw = model.get_input_embeddings(mx.array(ids)[None], None, mask=None).to_dict()
    t0 = time.perf_counter()
    uid = gen.insert([ids], max_tokens=max_tokens, prompt_kwargs=[kw])[0]
    out, finish = [], None
    try:
        while finish is None:
            _, resps = gen.next()
            for r in resps:
                if r.uid == uid:
                    out.append(int(r.token))
                    if r.finish_reason is not None:
                        finish = r.finish_reason
    finally:
        gen.close()
    return out, finish, time.perf_counter() - t0


with open(out_path, "a") as f:
    for qi, qid in enumerate(wanted):
        item = data[qid]
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt_of(item)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        ids = tok.encode(text, add_special_tokens=False)
        ref = None
        for cfg in configs:
            toks, finish, wall = run(ids, cfg)
            full = tok.decode(toks)
            content = full.split("</think>")[-1] if "</think>" in full else ""
            pred = extract(content, item["labels"]) if content else ""
            if cfg == "ar":
                ref = toks
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(ref, toks, strict=False)) if a != b),
                None if len(ref) == len(toks) else min(len(ref), len(toks)),
            )
            row = {
                "q": qi,
                "id": qid,
                "subject": item["subject"],
                "cfg": cfg,
                "answer": item["answer"],
                "pred": pred,
                "correct": pred == item["answer"],
                "n": len(toks),
                "finish": finish,
                "truncated": "</think>" not in full,
                "wall_s": round(wall, 2),
                "tps": round(len(toks) / wall, 1),
                "identical_to_ar": toks == ref,
                "first_diff_token": first_diff,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)
        mx.clear_cache()
