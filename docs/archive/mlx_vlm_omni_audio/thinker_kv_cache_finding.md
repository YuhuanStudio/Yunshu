# Finding: Yunshu's KV prefix cache cannot accelerate the Qwen3-Omni Talker path

**Date:** 2026-06-28 · **Model:** Qwen3-Omni-30B-A3B-Instruct-4bit · **HW:** M3 Max / 36GB

## Question

Can we connect Yunshu's KV prefix cache to the omni model's Thinker so multi-turn
voice conversations reuse the shared prefix (system prompt + history) instead of
re-prefilling each turn? (The intuition: "thinker 可以接入我們的 cache 系統".)

## What we tried

`mlx_vlm`'s `qwen3_omni_moe.generate_stream` calls `generate_step` for the thinker
and forwards any `thinker_<x>` kwarg as `<x>`. `generate_step` **does** accept
`prompt_cache`. So a persistent cache can be injected with no fork:

```python
cache = make_prompt_cache(model.thinker)
model.generate_stream(input_ids, thinker_prompt_cache=cache, ...)  # threads cleanly
```

It runs without error and output stays **coherent** across turns.

## Measured (thinker=8, talker=200, first-audio seconds)

| turn | no cache | shared persistent cache |
|------|----------|-------------------------|
| 1 (warm) | **1.59s** | 3.35s |
| 2 (reuse) | — | **2.69s** |

**Reusing the cache is slower than not using it** (2.69s vs 1.59s).

## Root cause (why a prefix cache cannot help here)

After the thinker decodes, the Talker is conditioned on hidden states produced by
`extract_thinker_hidden_states`, which runs a **full dense forward over the entire
`thinker_result_sequences` (prompt + generated) every turn, with NO cache**:

```python
outputs = self.thinker.language_model(input_ids, inputs_embeds=..., output_hidden_states=True)
```

A prompt cache can remove the thinker's autoregressive *decode* prefill, but it
cannot remove this mandatory full-sequence extraction pass — and that pass is the
cost. Passing the full prompt with a persistent cache only makes the decode's
attention context longer (it grows every turn), so it adds work without removing
any. The only way to get prefill savings (pass just the delta tokens) would starve
the Talker's hidden-state extraction of the prefix context, degrading speech.

## Decision

**Not shipped.** Connecting the KV prefix cache to the Talker path gives no benefit
without forking mlx-vlm's omni glue to make `extract_thinker_hidden_states` itself
cache-aware — which contradicts Yunshu's "wrap MLX, don't fork" principle. The
steady-state thinker latency (~1.6–2.7s) is already fine for voice without it.

A real win would require an **upstream mlx-vlm change** making the hidden-state
extraction reuse a cache. If pursued, that's a PR to `Blaizzy/mlx-vlm`, not a hack
in this repo.
