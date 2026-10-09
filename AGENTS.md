# AGENTS.md

Guidance for coding agents working in this repository. See `CLAUDE.md` for the full version — this is a short
mirror.

你可以報告 但別停下來 持續工作 加油！

## What Yunshu is

A **fast, local, single-node LLM / VLM inference engine for Apple Silicon**: one OpenAI/Anthropic-compatible
process, on-device via MLX (`mlx-lm`, `mlx-vlm`, `mlx-audio`). Primary axes: LLM/VLM decode speed, TTFT (cold and
cached), prefix reuse, capability completeness. Qwen3.8-27B is the first fully tuned model. Speech-to-speech
(Qwen3-Omni), ASR/TTS, Realtime voice, image/video generation and embeddings are supported capabilities, not the
identity — fix them when shared with the LLM/VLM path or explicitly prioritized. [Yunmo](../Yunmo) is one consumer.

**Non-goals (do not reintroduce):** no distributed/multi-node mesh, no throughput/batching race, no
multi-tenant control plane. Custom Metal kernels only when a same-checkpoint A/B proves an end-to-end win with
matching output. Performance = TTFT, decode speed, prefix reuse (cold and warm).
Single-node console exception (2026-10-07): multiple API keys with per-key usage/quotas, writing engine settings and
editing CORS are in scope; multi-node and a wider multi-tenant control plane are not.

This repo is mid-**refocus** away from its old "production platform / distributed Infra" framing. The
whitepaper and the wave-narrative VALIDATION_REPORT are retired — do not cite or resurrect their claims.
Describe only what the code verifiably does.

## Layout & commands

Flat layout: `python/{yunshu_gateway,yunshu_engine,yunshu_kv,yunshu_cli}/`, `tests/`, `scripts/`, `docs/`.
(`yunshu_control` stays only for `audit_log` and `token_counter`; the mesh/API/multi-tenant packages are gone.)

Settings: every `YUNSHU_*` setting goes through `python/yunshu_engine/settings.py` — never read one with
`os.environ`/`getenv` (a unit test enforces it). Add a new flag there, then run
`uv run python scripts/gen_config_docs.py` to regenerate `docs/CONFIGURATION.md`. Experimental flags are
temporary: each needs `decide=` (the measurement that settles it) and `added=`, at most 8 exist, and once
measured the winner becomes the default and the flag plus the losing path are deleted.

```bash
just setup        # uv install
just dev          # dev server on :8000 (YUNSHU_MODEL=/path/to/mlx-model)
just test         # full suite   ·   just lint && just format
PYTHONPATH=. uv run python scripts/realmodel/test_real_model.py
```

Serving: every mlx-vlm model → VLM batch runner (`vlm_batch_runner.py`: shared continuous batching with
per-row sampling, APC prefix cache, image/audio/video via `prepare_media`; Qwen3.5 family also MTP/DFlash spec
decode with batch-invariant kernels so spec on == off). There is no other VLM generation path (the deleted
legacy loop is recorded in `docs/archive/legacy_vlm_loop/`). Text-only mlx-lm models → single-request fast
path (`_generate_fast` → mlx-lm `generate_step`). Both on one MLX thread (`max_workers=1`). Per-request sampler +
SequenceStateMachine (Aho-Corasick), per-request detokenizer (never pool), `uv` only. Constrained JSON-schema
decoding is wired into both paths — keep it working.

## Internal docs

Private notes, decisions and evidence live in `docs/research/` (gitignored; never `git add -f` it). It has three
authoritative files and one rule:

- `docs/research/INDEX.md` — the single entry point (current records, per-line status, archive pointer).
- `docs/research/DECISIONS.md` — every user decision with date and quote; later ones supersede earlier ones explicitly.
- `docs/research/GROUND_TRUTH.md` — facts derived from source (ports, defaults, API counts, gpuq policy, CI commands);
  a doc that contradicts it is wrong, and if reality contradicts it, re-verify from source and update it first.
- **Rule: add or update research by updating INDEX.md first**, and keep it live: every line updates its INDEX entry /
  `docs/research/<line>/HANDOFF.md` on each commit. `scripts/dev/research_index.py` regenerates the auto blocks (per-line
  branch, ahead/behind, last commit, report head, READY TO MERGE, live worker, gpuq jobs); the lead runs it on every
  watchdog event and hourly, and the watchdog raises `index-stale` after 60 min. Superseded material moves to
  `docs/research/archive/` (with an `archive/INDEX.md` row); raw evidence is never deleted.
