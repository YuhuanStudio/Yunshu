# AGENTS.md

Guidance for coding agents working in this repository. See `CLAUDE.md` for the full version — this is a short
mirror.

你可以報告 但別停下來 持續工作 加油！

## What Yunshu is

A **local, single-node, multimodal (omni) inference engine for Apple Silicon**: one OpenAI/Anthropic-compatible
process serving text · vision · OCR · ASR · TTS · Realtime voice · image generation, on-device via MLX
(`mlx-lm`, `mlx-vlm`, `mlx-audio`). It is the local sensory body for [Yunmo](../Yunmo) (a digital-being
framework), and usable standalone.

**Non-goals (do not reintroduce):** no distributed/multi-node mesh, no throughput/batching race, no
multi-tenant control plane. Custom Metal kernels only when a same-checkpoint A/B proves an end-to-end win with
matching output. Performance = TTFT, decode speed, prefix reuse, voice round-trip (cold and warm).

This repo is mid-**refocus** away from its old "production platform / distributed Infra" framing. The
whitepaper and the wave-narrative VALIDATION_REPORT are retired — do not cite or resurrect their claims.
Describe only what the code verifiably does.

## Layout & commands

Flat layout: `python/{yunshu_gateway,yunshu_engine,yunshu_kv,yunshu_cli}/`, `tests/`, `scripts/`, `docs/`.
(`yunshu_control` stays only for `audit_log` and `token_counter`; the mesh/API/multi-tenant packages are gone.)

```bash
just setup        # uv install
just dev          # dev server on :8000 (YUNSHU_MODEL=/path/to/mlx-model)
just test         # full suite   ·   just lint && just format
PYTHONPATH=. uv run python scripts/realmodel/test_real_model.py
```

Serving = single-request fast path (`_generate_fast` → mlx-lm `generate_step`, `max_workers=1`). Per-request
sampler + SequenceStateMachine (Aho-Corasick), per-request detokenizer (never pool), single Metal thread,
`uv` only. Constrained JSON-schema decoding is wired into the fast path — keep it working.
