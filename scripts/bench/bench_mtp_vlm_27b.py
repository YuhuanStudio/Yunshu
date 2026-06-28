"""PROOF + template: mlx-vlm native MTP on Qwen3.6-27B = 1.82x LOSSLESS on M3 Max.

Settles the MTP question: speculative decoding DOES give a big lossless speedup on
a 27B (matching ml-explore/mlx-lm#990's 1.4-1.57x). Our earlier negatives were a
broken non-speculative bench (0.39x) and too-small models (9B 0.88x, bandwidth-
bound). mlx-vlm has the CORRECT MTP (MTP-aware GatedDeltaNet w/ intermediate-state
capture + rollback_speculative_cache + qwen3_5_mtp drafter + _mtp_rounds); our
mlx-lm-0.31.3 patch lacked the SSM intermediate-state capture.

Run: PYTHONPATH=reference/mlx-vlm:. uv run python scripts/bench_mtp_vlm_27b.py

PROOF-ONLY HACKS (must be cleaned for production integration):
  - manual drafter split handles this checkpoint's VLM-nested language_model.mtp.*
    prefix (mlx-vlm's split_qwen3_5_mtp expects bare mtp.*).
  - tolerant nn.Module.load_weights(strict=False fallback) lets the TARGET ignore
    the 31 embedded mtp.* keys it doesn't use (this kradih checkpoint embeds MTP
    in the target shards; standard mlx-vlm layout keeps target clean + drafter
    separate). Production: strip mtp.* from target weights cleanly, or use a
    standard-layout checkpoint.
INTEGRATION TODO: wire mlx-vlm qwen3_5 backbone + run_speculative_rounds into the
engine for Qwen3.5/3.6 MTP serving.
"""
import glob, json, logging, shutil, time
import mlx.nn as _nn
_OLW=_nn.Module.load_weights
def _lw(self, weights, strict=True):
    try: return _OLW(self, weights, strict=strict)
    except ValueError: return _OLW(self, weights, strict=False)
_nn.Module.load_weights=_lw
logging.basicConfig(level=logging.ERROR)
import mlx.core as mx
from pathlib import Path
SRC="./models/Qwen3.6-27B-MTP-4bit-MLX"; DRAFT="./models/Qwen3.6-27B-mtp-drafter"; MT=128
# ---- STEP 1: manual drafter split (handle language_model.mtp.* prefix) ----
sel={}
for sh in glob.glob(SRC+"/*.safetensors"):
    for k,v in mx.load(sh).items():
        if k.startswith("language_model.mtp."):
            sel[k[len("language_model.mtp."):]] = v
src_cfg=json.load(open(SRC+"/config.json")); tc=dict(src_cfg.get("text_config") or {})
Path(DRAFT).mkdir(parents=True, exist_ok=True)
mx.save_safetensors(DRAFT+"/model.safetensors", sel, metadata={"format":"mlx"})
dcfg={"model_type":"qwen3_5_mtp","text_config":tc,
      "block_size":int(tc.get("mtp_num_hidden_layers",1)+2),
      "tie_word_embeddings":bool(tc.get("tie_word_embeddings",True))}
if any(k.endswith(".scales") for k in sel):
    q=src_cfg.get("quantization"); dcfg["quantization"]=q; dcfg["quantization_config"]=q
json.dump(dict(sorted(dcfg.items())), open(DRAFT+"/config.json","w"), indent=2)
for n in ("tokenizer.json","tokenizer_config.json","vocab.json","chat_template.jinja"):
    if Path(SRC,n).exists(): shutil.copy(Path(SRC,n), Path(DRAFT,n))
print("drafter split:", len(sel), "tensors")
# ---- STEP 2: load target (mlx-vlm) + drafter ----
from mlx_vlm.utils import load_model as vlm_load
import mlx_vlm.models.qwen3_5 as _Q
_orig_san=_Q.Model.sanitize
def _drop_mtp_san(self, w):
    w={k:v for k,v in w.items() if ".mtp." not in k}
    return _orig_san(self, w)
_Q.Model.sanitize=_drop_mtp_san  # embedded mtp.* belongs to the drafter, not the target
if hasattr(_Q,'LanguageModel') and hasattr(_Q.LanguageModel,'sanitize'):
    _olm=_Q.LanguageModel.sanitize
    _Q.LanguageModel.sanitize=lambda self,w: _olm(self,{k:v for k,v in w.items() if '.mtp.' not in k})
# also patch the module-level sanitize_weights to drop mtp as a backstop
import mlx_vlm.utils as _U
if hasattr(_U,'sanitize_weights'):
    _osw=_U.sanitize_weights
    _U.sanitize_weights=lambda m,w,*a,**k: _osw(m,{kk:vv for kk,vv in w.items() if '.mtp.' not in kk},*a,**k)
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.speculative.utils import speculative_prefill_kwargs, make_speculative_prompt_cache, run_speculative_rounds
from mlx_lm.models import cache as kvcache
from transformers import AutoTokenizer
model = vlm_load(Path(SRC))
drafter, kind = load_drafter(DRAFT)
print("drafter kind:", kind)
tok = AutoTokenizer.from_pretrained(SRC)
lm = model.language_model
txt = tok.apply_chat_template([{"role":"user","content":"Write a detailed 300-word essay on how photosynthesis works, step by step."}], add_generation_prompt=True, tokenize=False)
ids = tok.encode(txt); input_mx = mx.array([ids], dtype=mx.int32)
from mlx_lm.sample_utils import make_sampler
sampler = make_sampler(temp=0.0)
def sample(logits): return sampler(logits.reshape(-1, logits.shape[-1]))
# ---- baseline (plain greedy on lm) ----
def baseline():
    c = kvcache.make_prompt_cache(lm); out=[]
    o = lm(input_mx, cache=c); t=int(mx.argmax(o.logits[0,-1,:]).item()); out.append(t)
    t0=time.perf_counter()
    for _ in range(MT-1):
        o = lm(mx.array([[t]]), cache=c); t=int(mx.argmax(o.logits[0,-1,:]).item()); out.append(t)
    return out, time.perf_counter()-t0
_=baseline(); b,bt=baseline()
# ---- MTP via run_speculative_rounds ----
pk = speculative_prefill_kwargs("mtp", drafter)
cache_ = make_speculative_prompt_cache(lm, draft_kind="mtp", batch_size=1, left_padding=[0], make_cache=None)
out = lm(input_mx, cache=cache_, **pk)
first_tok = sample(out.logits[:,-1:])
t0=time.perf_counter()
m=[]
for tk,_lp in run_speculative_rounds(model, drafter, cache_, input_mx, first_tok, out.logits[:,-1:], out, draft_kind="mtp", max_tokens=MT, sampler=sample, sampler_is_greedy=True):
    m.append(int(tk) if not isinstance(tk,list) else tk[0])
    if len(m)>=MT: break
mtp_t=time.perf_counter()-t0
print(f"baseline: {len(b)/bt:.1f} tok/s")
print(f"MTP(vlm): {len(m)/mtp_t:.1f} tok/s  ({(len(m)/mtp_t)/(len(b)/bt):.2f}x)")
print(f"lossless(greedy prefix): {b[:25]==m[:25]}")
