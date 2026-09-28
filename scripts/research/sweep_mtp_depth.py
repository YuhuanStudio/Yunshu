"""Cold decode sweep: upstream BatchGenerator MTP draft_block_size vs AR on Qwen3.8.

Same in-memory drafter as probe_apc_mtp_batchgen.py; no APC so every request is
cold. Reports decode tok/s, first-token latency and token parity against AR.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/research/sweep_mtp_depth.py MODEL_DIR [sizes...]
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from mlx_vlm import load  # noqa: E402
from mlx_vlm.generate.ar import BatchGenerator  # noqa: E402

from yunshu_engine.mlxvlm_mtp import _load_drafter_in_memory  # noqa: E402
from yunshu_engine.mrope import clear_rope_state  # noqa: E402

model_dir = sys.argv[1]
_kflag = next((a for a in sys.argv if a.startswith("--omlx-kernels")), None)
use_omlx = _kflag is not None
omlx_set = (
    set(_kflag.split("=", 1)[1].split(","))
    if _kflag and "=" in _kflag
    else {"verify_linear", "gdn_prework", "sdpa_split", "verify_qmm", "gdn_replay"}
)
sizes = [int(x) for x in sys.argv[2:] if not x.startswith("--")] or [3, 4, 5, 6]

if use_omlx:
    # Experiment only: oMLX (Apache-2.0) verify kernels from reference/omlx,
    # armed around upstream mlx-vlm's MTP target verify forward.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference" / "omlx"))
    import mlx_vlm.speculative.mtp as _mtp  # noqa: E402
    from omlx.patches import (  # noqa: E402
        qwen35_gdn_prework,
        qwen35_gdn_verify_fused,
        qwen35_verify_qmm,
        qwen35_verify_sdpa_split,
    )
    from omlx.patches.mlx_vlm_mtp import qwen35_verify_linear  # noqa: E402

    patchers = {
        "verify_linear": qwen35_verify_linear.apply,
        "gdn_prework": qwen35_gdn_prework.apply_qwen35_gdn_prework_patch,
        "sdpa_split": qwen35_verify_sdpa_split.apply_qwen35_verify_sdpa_split_patch,
        "verify_qmm": qwen35_verify_qmm.apply_verify_qmm_patch,
        "gdn_replay": qwen35_gdn_verify_fused.apply_arrays_cache_replay_patch,
    }
    applied = {k: fn() for k, fn in patchers.items() if k in omlx_set}
    print(json.dumps({"omlx_patches": applied}), flush=True)
    _orig_verify = _mtp._mtp_verify_target

    def _armed_verify(*a, **k):
        qwen35_verify_qmm.set_verify_qmm_armed(True)
        try:
            return _orig_verify(*a, **k)
        finally:
            qwen35_verify_qmm.set_verify_qmm_armed(False)

    _mtp._mtp_verify_target = _armed_verify
_yk = next(
    (a.split("=", 1)[1] for a in sys.argv if a.startswith("--yunshu-kernels=")), None
)
if _yk:
    # Vendored copy under python/yunshu_engine/kernels/omlx.
    from yunshu_engine.kernels import omlx as yk  # noqa: E402

    # exact | row_exact (oMLX d403e460: verify rows == stock serial decode)
    print(
        json.dumps(
            {
                "yunshu_kernels": yk.apply(row_exact=_yk == "row_exact"),
                "mode": _yk,
            }
        ),
        flush=True,
    )
    use_omlx, omlx_set = True, {f"yunshu:{_yk}"}


if "--streamed5" in sys.argv:
    from yunshu_engine.kernels.verify_select import (
        install as install_streamed5,  # noqa: E402
    )

    install_streamed5()
    omlx_set = set(omlx_set if use_omlx else set()) | {"streamed5"}
    use_omlx = True

model, processor = load(model_dir)
tok = processor.tokenizer
if "--invariant" in sys.argv:
    from yunshu_engine.kernels.batch_invariant import (
        install as install_invariant,  # noqa: E402
    )

    print(
        json.dumps(
            {
                "batch_invariant": install_invariant(
                    model.language_model,
                    model=model,
                    packed="--invariant-packed" in sys.argv,
                )
            }
        ),
        flush=True,
    )
    omlx_set = set(omlx_set if use_omlx else set()) | {"invariant"}
    use_omlx = True
if "--pack" in sys.argv:
    from yunshu_engine.kernels.omlx import pack_projections  # noqa: E402

    t0 = time.perf_counter()
    print(
        json.dumps(
            {
                "packed_layers": pack_projections(model),
                "pack_s": round(time.perf_counter() - t0, 2),
            }
        ),
        flush=True,
    )
    omlx_set = set(omlx_set) | {"packed"}
    use_omlx = True
_dflash = next(
    (a.split("=", 1)[1] for a in sys.argv if a.startswith("--dflash=")), None
)
if _dflash:
    from mlx_vlm.speculative.drafters import load_drafter  # noqa: E402

    drafter, draft_kind = load_drafter(_dflash, kind="dflash")
else:
    drafter, draft_kind = _load_drafter_in_memory(model_dir), "mtp"
print(
    json.dumps({"draft_kind": draft_kind, "draft": _dflash or "mtp-head"}), flush=True
)
lm = model.language_model

tasks = [
    (
        "code",
        "Write a Python LRU cache class with get, put, delete and resize, with docstrings and type hints. Output code only.",
        384,
    ),
    (
        "prose",
        "Explain in detail how a refrigerator works, covering the refrigerant cycle, compressor, condenser and evaporator.",
        384,
    ),
    (
        "json_like",
        "List ten European capitals with their countries and approximate populations as a markdown table.",
        256,
    ),
]


# --context=N prepends ~N tokens of filler so verify windows cross MLX's SDPA
# plan switches (1024 keys; 16384 on M5), where row exactness can break.
_context = int(
    next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--context=")), "0")
)
_FILLER = "".join(
    f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n" for i in range(8000)
)


def run(name, prompt, max_tokens, block):
    if _context:
        filler_ids = tok.encode(_FILLER, add_special_tokens=False)[:_context]
        prompt = tok.decode(filler_ids) + "\n\n" + prompt
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tok.encode(text, add_special_tokens=False)
    gen = BatchGenerator(
        lm,
        processor,
        max_tokens=max_tokens,
        draft_model=drafter if block else None,
        draft_kind=draft_kind if block else None,
        draft_block_size=block or None,
        greedy_sampling=True,
        compute_logprobs=False,
    )
    clear_rope_state(model)
    kw = model.get_input_embeddings(mx.array(ids)[None], None, mask=None).to_dict()
    t0 = time.perf_counter()
    uid = gen.insert([ids], max_tokens=max_tokens, prompt_kwargs=[kw])[0]
    out, first = [], None
    try:
        while True:
            _, resps = gen.next()
            done = False
            for r in resps:
                if r.uid == uid:
                    first = first or (time.perf_counter() - t0)
                    out.append(int(r.token))
                    done = done or r.finish_reason is not None
            if done:
                break
    finally:
        gen.close()
    wall = time.perf_counter() - t0
    return {
        "task": name,
        "block": block,
        "n": len(out),
        "first_s": round(first, 3),
        "decode_tps": round((len(out) - 1) / (wall - first), 2),
        "tokens": out,
    }


ref = {}
for name, prompt, mt in tasks:
    run(name, prompt, 16, 0)  # warm kernels for this shape
    for block in [0] + sizes:
        r = run(name, prompt, mt, block)
        r["omlx_kernels"] = sorted(omlx_set) if use_omlx else []

        if block == 0:
            ref[name] = r["tokens"]
        r["parity"] = r["tokens"] == ref[name]
        r["tokens_head"] = r["tokens"][:64]
        r.pop("tokens")
        print(json.dumps(r), flush=True)
    mx.clear_cache()
