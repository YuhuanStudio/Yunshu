# mlx_vlm Qwen3-Omni audio — upstream bug + fix (PR-ready)

Status: confirmed in **mlx_vlm 0.5.0 and git HEAD** (the affected code is identical).
Reproduced live on `Qwen3-Omni-30B-A3B-Instruct-4bit`. NOT a Yunshu bug — our
`VLMEngine._audio_arg` correctly pre-loads the audio path to an ndarray; the failure is
entirely inside `mlx_vlm/models/qwen3_omni_moe/`. Text + vision Omni paths work.

## Symptom

Any Qwen3-Omni call with audio input crashes in the audio tower:

```
File ".../qwen3_omni_moe/audio.py", line 234, in __call__
    padded_feature[i, :, :chunk_len] = input_features[:, start_idx:end_idx]
ValueError: [broadcast_shapes] Shapes (1,1,100,288) and (1,128,100) cannot be broadcast.
```

## Bug #1 — feature-mask param-name mismatch (ROOT CAUSE, one-line fix)

`utils.prepare_inputs` emits the audio mask under the key **`feature_attention_mask`**
(`utils.py`, the `if audio_inputs is not None:` block):

```python
model_inputs["feature_attention_mask"] = mx.array(audio_inputs["attention_mask"]).astype(mx.int32)
```

But `qwen3_omni_moe.Model.get_input_embeddings` only accepts **`input_features_mask`**
(a key that ONLY the gemma4 / gemma3n processors set), and forwards *that* to the thinker:

```python
def get_input_embeddings(self, ..., input_features_mask: Optional[mx.array] = None, ..., **kwargs):
    return self.thinker.get_input_embeddings(
        ..., feature_attention_mask=input_features_mask, ...)   # <- always None for Omni
```

So `feature_attention_mask` lands in `**kwargs` (ignored), `input_features_mask` stays `None`,
and `thinker.get_audio_features` skips its 2-D reshape (it only reshapes
`if feature_attention_mask is not None`) — leaving `input_features` as a raw 4-D
`(1,1,100,288)` that the audio tower (expecting 2-D `(mel=128, time)`) cannot broadcast.

Verified: the HF processor itself produces correct `input_features (1,128,288)` +
`feature_attention_mask (1,288)`; only the mlx_vlm name plumbing drops it.

### Fix (mlx_vlm/models/qwen3_omni_moe/qwen3_omni_moe.py)

```python
def get_input_embeddings(self, ..., input_features_mask=None, ..., **kwargs):
    # The shared prepare_inputs path emits `feature_attention_mask`; only
    # gemma4/gemma3n use `input_features_mask`. Accept both.
    if input_features_mask is None:
        input_features_mask = kwargs.get("feature_attention_mask")
    return self.thinker.get_input_embeddings(
        ..., feature_attention_mask=input_features_mask, ...)
```

(Equivalently: change `prepare_inputs` to also set `input_features_mask` for the Omni path.)

## Bug #2 — empty audio_features in masked_scatter (FOLLOW-UP)

After patching #1, a second error surfaces at `thinker.py` `get_input_embeddings`:

```
File ".../qwen3_omni_moe/thinker.py", line 176, in masked_scatter
ValueError: [broadcast_shapes] Shapes (75776) and (0) cannot be broadcast.
```

The audio tower now runs (correct 2-D reshape) but the scattered `audio_features` come out
empty, so the audio-token count (≈75776 flattened positions) has nothing to scatter. The
count-fix block (`thinker.py` ~156-175) only pads/truncates when `audio_features.ndim == 2`.
Needs upstream investigation (audio encoder output shape / token alignment).

## Recommendation

File #1 as a PR (clean one-liner, clear repro). Note #2 as a known follow-up so Omni audio
isn't advertised as working until both land. Yunshu keeps Omni **audio** as
known-blocked-upstream; text + vision are validated and in use.

## Minimal repro

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_audio
m, p = load("<Qwen3-Omni-30B-A3B-Instruct-4bit>")
sr = int(p.feature_extractor.sampling_rate)
audio = load_audio("speech.wav", sr)
msgs = [{"role":"user","content":[{"type":"audio"},{"type":"text","text":"What is said?"}]}]
prompt = apply_chat_template(p, m.config, msgs, num_audios=1)
generate(m, p, prompt, audio=[audio], max_tokens=30)   # -> audio.py:234 broadcast error
```

## Prior-art / duplicate check (GitHub search, Blaizzy/mlx-vlm)

- `feature_attention_mask` → **0 issues/PRs**. `input_features_mask` → **0**. So the exact
  param-name root cause is unreported.
- "audio broadcast_shapes" hits are all **gemma4 / gemma3n / batch** (#923 open is Gemma4, not
  Omni), not the Qwen3-Omni audio tower.
- **Open issue #1007** (2026-04-10, unfixed) "Using command line testing Qwen3-Omni raising
  error" reports a *different, shallower* failure: `ValueError: could not convert string to
  float: '../AUD-...wav'` — the CLI passes the audio PATH straight to the HF feature extractor
  (no `load_audio`). That reporter never reached bug #1/#2 because audio never loaded. This
  independently confirms Omni audio is broken upstream for others (NOT a Yunshu-only issue).
- Closed Omni fixes (#820 integration, #794 visual_embeds UnboundLocalError, #723 destructuring)
  addressed OTHER omni bugs — none the audio feature-mask path.

## Independent-verification checklist (for a reviewer)

1. `pip install "mlx-vlm==0.5.0"` (or git HEAD — identical code).
2. Run the minimal repro above with a real .wav and a Qwen3-Omni checkpoint → expect the
   `audio.py:234` broadcast error.
3. Confirm the processor itself is correct:
   `AutoProcessor(...)(text="<audio>", audio=[arr], return_tensors="np")` yields
   `input_features (1,128,288)` + `feature_attention_mask (1,288)`.
4. Confirm the model only reads `input_features_mask` (grep `get_input_embeddings` in
   `qwen3_omni_moe.py`) while `prepare_inputs` only writes `feature_attention_mask` (grep
   `utils.py`). → the mask is dropped. Apply the one-line fix → error #1 clears, #2 surfaces.

## Suggested action

Comment on / link **#1007** (same area, still open) with: (a) the `load_audio` precursor fix,
(b) bug #1 root cause + the one-line patch, (c) bug #2 as the remaining blocker. Lower-risk and
more helpful than a cold PR — it gives the maintainer the full chain on an already-open thread.

---

## UPDATE — BOTH bugs fixed and VALIDATED end-to-end (Qwen3-Omni audio now works)

Cloned mlx-vlm to `reference/mlx-vlm` (HEAD), applied two minimal fixes, tested on the real
`Qwen3-Omni-30B-A3B-Instruct-4bit`:

**Fix #1** — `models/qwen3_omni_moe/qwen3_omni_moe.py` `get_input_embeddings`: accept the mask
under either key (`if input_features_mask is None: input_features_mask =
kwargs.get("feature_attention_mask")`).

**Fix #2 (the real "#2")** — `prompt_utils.py` `_format_list_with_image`: this formatter (used
by `qwen3_omni_moe`, `LIST_WITH_IMAGE_FIRST`) inserts image placeholders but NOT audio ones, so
`{"type":"audio"}` was dropped → the prompt had ZERO audio tokens → the encoder's 37 audio
features had no slots (`masked_scatter` `(75776) vs (0)`). Added the audio insertion that the
sibling `_format_list_with_image_type` already has:
`if role == "user" and not skip_audio_token and num_audios > 0: content += [audio_message()] * num_audios`.

**Result (live, end-to-end):** prompt now contains `<|audio_start|><|audio_pad|><|audio_end|>`;
generate transcribed the test clip as **"The quick brown fox jumps over the lazy dog."** —
exactly the TTS sentence in the audio.

**Safety:** top-level `apply_chat_template` defaults `num_audios=0`, so the `num_audios > 0`
guard makes fix #2 a no-op for text-only and image-only calls (verified: text-only → no audio
token; audio → token added). Other `LIST_WITH_IMAGE_FIRST` models (qwen2_5_vl, glm4v, …) are
unaffected.

**Diff: 2 files, +12 lines** (in `reference/mlx-vlm`, local only — NOT submitted). Ready for
review; submission is the user's call.

## Robustness + cross-model validation (4 distinct clips)

Generated 4 distinct TTS clips (known ground truth), transcribed with the FIXED Omni, then
cross-checked with the dedicated **Qwen3-ASR** model:

| clip | ground truth | fixed Omni | Qwen3-ASR |
|---|---|---|---|
| fox   | the quick brown fox jumps over the lazy dog | exact | exact |
| tokyo | the weather in Tokyo is sunny today | exact | exact |
| fruit | please add three apples and two oranges | exact | exact |
| ai    | artificial intelligence is changing the world | "Artificial intelligence." | "Artificial intelligence." |

The AI clip: Omni AND ASR independently produced the SAME truncated text → the TTS audio itself
is short, not an Omni error. Two models agreeing = Omni reads the real audio, no hallucination.

Also passing: audio REASONING (topic of AI clip → "AI"), IMAGE input regression (cat → "Cat"),
TEXT-only regression (2+2 → "4"). The fix is robust across content, modalities, and a
cross-model check — not a single lucky case.
