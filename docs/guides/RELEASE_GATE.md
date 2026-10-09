# Release gate

> The console's Docs section carries this guide in English, 繁體中文 and 简体中文 ([`frontend/docs/developers/release-gate.mdx`](../../frontend/docs/developers/release-gate.mdx)); this file is its GitHub-facing counterpart.

A version ships only when `scripts/release/gate.sh` passes on the commit being released **and** the
`long` stage (32K-128K against the previous release, run by `scripts/dev/yv gate`) passes. The gate
tests Yunshu the way a user gets it: built into a wheel, installed with `uv tool install` into a
clean environment, and driven over HTTP with the official OpenAI and Anthropic SDKs. Every check
prints `PASS`, `FAIL` or `SKIP`. The run ends with a table and exits 1 if anything failed.

Run the public entry point through `yv`, which queues every model stage:

```sh
scripts/dev/yv gate --priority -1
scripts/dev/yv gate --priority -1 --stages install,serve-27b
GATE_SERVICE=1 scripts/dev/yv gate --priority -1 --stages service
```

Do not invoke `gate.sh` directly; `yv` preserves successful stage evidence. For a
complete release checklist, use `scripts/dev/release_check <commit-sha>` after
preparing the final version/changelog commit. CPU-only review uses `--dry-run`.

`scripts/dev/yv gate` runs the same stages as separate gpuq jobs and remembers which passed on the commit, so a rerun after a late failure skips them ([VERIFY.md](VERIFY.md)).

Model paths come from the environment or `scripts/research/local.env`, which is gitignored. The
variable names are in [`scripts/release/local.env.example`](../../scripts/release/local.env.example).
A family whose variable is unset is reported as `SKIP` with the reason.

The gate keeps everything it installs and writes under `$GATE_ROOT`: HOME, the uv tool
directories, uv's cache and Python installs. It never touches your `~/.yunshu` or `~/.local`.
Results and server logs go to `$OUT`, which defaults to `docs/research/runs/<date>-release-gate/`
(local, not in git).

| Stage | What must hold | Time |
|---|---|---|
| `install` | `uv build` succeeds. `yunshu[vision]` and `yunshu[all]` install from the wheel. `--version` matches `pyproject.toml`. `doctor` reports no failed check. `model list` finds the models directory. `pull` of a model already on disk refuses to download (the check runs offline). `config set models_dir` round-trips. `service install --dry-run` prints the agent and writes nothing. | ~1–10 min |
| `serve-27b` | Qwen3.8-27B runs on the installed quickstart binary. The capability matrix is 34/34. The OpenAI SDK passes chat, streaming, tools (auto, forced, streamed deltas), JSON schema, stop, logprobs, reasoning split and image input. The Anthropic SDK passes messages, streaming, tool use and thinking. Dropping a stream mid-generation still leaves the next request answered correctly within 60 s. A 32K-token prompt recalls its needle. 8 concurrent requests answer 8/8. The server decodes as fast as the engine does in-process: `scripts/release/check_server_path.py` runs the same greedy prompts (code and prose, 1K and 8K context) through the batch runner in the gate process and through the server, requires identical text, and fails when the geometric mean of server / in-process decode tok/s over the cases is below 0.965 or any single case is below 0.90 (5 interleaved repeats per case, median; thresholds from 24 gate runs, see `check_server_path.py`). The server log has no Traceback. | ~15 min |
| `families` | Each of these boots, answers its smoke checks, and exits without a Traceback: Qwen3.5 0.8B and 9B (VLM runner), a text-only `mlx-lm` model (fast path: tools, schema), Gemma-4 (tools, image), GLM-OCR (`/v1/ocr`), ASR (`/v1/audio/transcriptions` on a `say`-generated sentence), TTS (`/v1/audio/speech` returns a WAV), Qwen3-Omni (a chat turn), and Z-Image (`/v1/images/generations` at 512×512). A documented modality that fails is a FAIL. | ~15–25 min |
| `soak` | Qwen3.8-27B completes MMLU-Pro 300 at 8 in flight (16384 max tokens, `reasoning_effort=medium`) with 0 errors and accuracy no more than 3 below the 249/300 baseline (a higher score passes). Then a `SOAK_MINUTES` realistic mixed soak (30 by default) with 0 errors and ≥95% correct. After each part the process footprint must come back within 4 GiB of where it started. | ~65 min |
| `agent-sessions` | What long agent traffic does, each check on its own cold server. `tool_session`: a Claude Code-style `/v1/messages` loop (6 Read calls of 8K-token files, the model's own replies echoed back, ~50K tokens) must produce a valid tool call every turn, recall all six needles at the end and reuse >= 85% of the previous prompt from the cache each turn. `concurrent_long`: a 16K-token document, a 4K one, a short question, a tool call and a JSON-schema answer run solo, 2 and 4 at a time (prefix cache off); every answer correct and every concurrent output equal to its solo output. `restart_idle`: after 45 s idle (memory release) and after a SIGTERM restart on the same SSD tier the same request returns the same text with >= 95% / 70% of the prompt cached. `stock_text_lm`, `stock_qwen35_9b`, `stock_gemma_image`: the server's greedy text equals stock mlx-lm / mlx-vlm on the same checkpoint (agreeing >= 150 characters, needle recalled). | ~30 min |
| `long` | `yv ab --suite long` (32K / 64K / 128K, see [VERIFY.md](VERIFY.md)), run by `yv gate` only (not a `gate.sh` stage). Candidate = the commit under test; base = the newest `v*` tag before it (the release reference), else `origin/main`; `GATE_LONG_BASE` overrides. Fail closed: the verdict must exist with overall PASS and every planned stage (identity incl. spec on == off over 2048-token replies, apc, speed, memory, `longqa` needle retrieval vs stock, 2x32K `conc`) PASS. A missing cell, NOT_RUN stage, infra error, identity mismatch, retrieval loss or regression beyond yv's thresholds fails the gate. | ~3-5 h |
| `service` | Off by default, because it loads a real launchd agent. With `GATE_SERVICE=1`, `service install` must serve the model and `service uninstall` must free the port. | ~2 min |

The checks live in `scripts/release/gate_checks.py`. It reuses the research harnesses
`bench_engine_matrix.py`, `probe_concurrency.py`, `soak_mmlu_pro.py` and `soak_realistic.py`.
When a baseline legitimately moves (for example a new default kernel changes MMLU-Pro), update
`MMLU_BASELINE` in the script together with the measurement that justifies it.
