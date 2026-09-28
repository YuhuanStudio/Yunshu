"""Probe: does mlx-vlm's Qwen3-Omni generate_stream stay correct on turn 2+?

The Yunshu mlx-vlm fork carries 69acca1 (mx.eval mRoPE position_ids) because
turn 2+ in one process used to emit token 0 ("!!!!"). Upstream thinker.py has
no such eval. Run this in any candidate env to decide whether upgrading the
locked mlx-vlm to upstream breaks the Omni path. Prints one JSON line per turn.

    HF_HUB_OFFLINE=1 <python> scripts/research/probe_omni_second_turn.py MODEL_DIR [turns] [audio.wav ...]

With audio paths, each turn sends that clip (cycled) as the spoken question.
"""

import json
import sys
import time

import mlx.core as mx
import mlx_vlm
from mlx_vlm import load
from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs

model_dir = sys.argv[1]
turns = int(sys.argv[2]) if len(sys.argv) > 2 else 3
audios = sys.argv[3:]
prompts = [
    "Reply with exactly one word: ALPHA",
    "What is 2 + 3? Answer with just the number.",
    "Name the capital of France in one word.",
]

t0 = time.perf_counter()
model, processor = load(model_dir)
print(
    json.dumps(
        {
            "model": model_dir,
            "mlx_vlm": mlx_vlm.__version__,
            "mlx": mx.__version__,
            "load_s": round(time.perf_counter() - t0, 2),
        }
    ),
    flush=True,
)

for i in range(turns):
    if audios:
        audio = audios[i % len(audios)]
        prompt = f"audio:{audio.rsplit('/', 1)[-1]}"
        content = [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": "Answer the spoken question in a few words."},
        ]
    else:
        prompt = prompts[i % len(prompts)]
        content = [{"type": "text", "text": prompt}]
    conv = [{"role": "user", "content": content}]
    mi, _ = prepare_omni_inputs(processor, conv)
    t = time.perf_counter()
    text_ids, audio_chunks, first_audio = [], 0, None
    for kind, payload in model.generate_stream(
        mi["input_ids"],
        speaker="Ethan",
        thinker_max_new_tokens=24,
        talker_max_new_tokens=64,
        chunk_size=25,
        **{
            k: mi[k]
            for k in (
                "input_features",
                "feature_attention_mask",
                "audio_feature_lengths",
            )
            if mi.get(k) is not None
        },
    ):
        if kind == "text":
            text_ids = payload.tolist() if hasattr(payload, "tolist") else list(payload)
        elif kind == "audio":
            audio_chunks += 1
            if first_audio is None:
                first_audio = time.perf_counter() - t
    flat = [x for row in text_ids for x in (row if isinstance(row, list) else [row])]
    text = processor.decode(flat, skip_special_tokens=True) if flat else ""
    print(
        json.dumps(
            {
                "turn": i + 1,
                "prompt": prompt,
                "text": text,
                "bang_collapse": text.strip().startswith("!!!"),
                "audio_chunks": audio_chunks,
                "first_audio_s": round(first_audio, 3) if first_audio else None,
                "total_s": round(time.perf_counter() - t, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
