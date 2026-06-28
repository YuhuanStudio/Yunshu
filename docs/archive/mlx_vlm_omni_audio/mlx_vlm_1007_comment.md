Hi — I hit this too. The error in the original post is the **first** of a chain; sharing the full root cause in case it helps.

### 1. The path error (your traceback)
`could not convert string to float: '../AUD-...wav'` happens because the audio **path** is handed straight to the HF feature extractor. Loading the file to an array first gets past it:

```python
from mlx_vlm.utils import load_audio
sr = int(processor.feature_extractor.sampling_rate)
audio = load_audio("AUD-....wav", sr)
generate(model, processor, prompt, audio=[audio], max_tokens=500)
```

### 2. Then a real bug: the audio feature-mask is dropped (param-name mismatch)
After step 1, it fails in the audio tower:

```
File ".../qwen3_omni_moe/audio.py", line 234, in __call__
    padded_feature[i, :, :chunk_len] = input_features[:, start_idx:end_idx]
ValueError: [broadcast_shapes] Shapes (1,1,100,288) and (1,128,100) cannot be broadcast.
```

Root cause — `prepare_inputs` and the model disagree on the mask key (current `main`):

- `mlx_vlm/utils.py:1536` **writes** `model_inputs["feature_attention_mask"]`
- `mlx_vlm/models/qwen3_omni_moe/qwen3_omni_moe.py:65` `get_input_embeddings(...)` only accepts **`input_features_mask`** and forwards *that* (`feature_attention_mask=input_features_mask`, line 78). `input_features_mask` is a key only gemma4/gemma3n set.

So `feature_attention_mask` lands in `**kwargs` (ignored), `input_features_mask` stays `None`, and `thinker.get_audio_features` skips its 2-D reshape (it only reshapes when `feature_attention_mask is not None`) — leaving `input_features` as a raw 4-D `(1,1,100,288)` the audio tower can't use. The HF processor itself is fine: `processor(text="<audio>", audio=[arr], return_tensors="np")` returns `input_features (1,128,288)` + `feature_attention_mask (1,288)`.

**One-line fix** in `qwen3_omni_moe.py` `get_input_embeddings`:

```python
if input_features_mask is None:
    input_features_mask = kwargs.get("feature_attention_mask")
```

(or have `prepare_inputs` also set `input_features_mask`.)

### 3. Heads-up: not fully fixed by the above
After patching #2, a **second** error surfaces at `thinker.py` `masked_scatter`: `Shapes (75776) and (0)` — `audio_features` comes back empty, so the audio tokens have nothing to scatter. So #2's one-liner clears the broadcast error but doesn't make Omni audio fully work yet — this one needs more investigation (audio-encoder output / token alignment).

Tested on `mlx-vlm 0.5.0` (and `main` — same code). Text + vision Omni paths work; audio is the only broken path.
