"""Correct MTP speedup bench: baseline generate_step vs the n_confirmed MTPDecoder
(PR ml-explore/mlx-lm#990's mechanism — MTP head draft + 2-token batched verify +
GatedDeltaNet rollback). Measures REAL speculative speedup (lossless greedy).

NB: the old scripts/bench_mtp_horizontal.py MTP path is a NON-speculative
diagnostic (drafts then does a full backbone step anyway) — its "0.39x" is
meaningless. This script uses the real MTPDecoder.

Speedup SCALES WITH MODEL SIZE (bandwidth- vs compute-bound): Qwen3.5-2B 0.48x,
9B-4bit 0.88x; mlx-lm#990 reports 27B = 1.4x on M3+/M4. On a 27B-class model this
crosses >1x; our largest local model is 9B so we sit just below break-even.

Run: MTP_MODEL=./models/Qwen3.5-9B-MLX-4bit PYTHONPATH=. uv run python scripts/bench_mtp_speedup.py
"""
import logging
import os
import time

logging.basicConfig(level=logging.ERROR)
import mlx.core as mx

MODEL=os.environ.get("MTP_MODEL","./models/Qwen3.5-9B-MLX-4bit"); MT=128
from yunshu_engine.mtp_patch import apply_mtp_patch, load_model_with_mtp
from yunshu_engine.n_confirmed_patch import apply_n_confirmed_patch

apply_mtp_patch(); ncok=apply_n_confirmed_patch()
from mlx_lm.utils import load as _load

_, tok = _load(MODEL)
model = load_model_with_mtp(MODEL)
prompt="Write a detailed 300-word essay explaining how photosynthesis works in plants, step by step."
ids = tok.apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True)
# baseline
from mlx_lm.generate import generate_step


def baseline():
    out=[]; t0=time.perf_counter()
    for t,_ in generate_step(mx.array(ids), model, max_tokens=MT): out.append(int(t))
    return out, time.perf_counter()-t0
b1,_=baseline()  # warmup
b,bt=baseline()
# MTP
from yunshu_engine.mtp_decoder import MTPConfig, MTPDecoder

dec=MTPDecoder(model, tok, MTPConfig(max_tokens=MT))
t0=time.perf_counter(); m=dec.generate(ids, max_tokens=MT); mt=time.perf_counter()-t0
st=dec.stats
print(f"n_confirmed_patch={ncok}")
print(f"baseline: {len(b)/bt:.1f} tok/s")
print(f"MTP:      {len(m)/mt:.1f} tok/s  ({(len(b)/bt and (len(m)/mt)/(len(b)/bt)):.2f}x)  accepts={st.accepts}/{getattr(st,'cycles',getattr(st,'tokens_generated','?'))}")
print(f"lossless(greedy prefix): {b[:30]==m[:30]}  base={tok.decode(b[:40])!r}")
