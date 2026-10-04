# Release gate

A version ships only when `scripts/release/gate.sh` passes on the commit being released. The gate
tests Yunshu the way a user gets it: built into a wheel, installed with `uv tool install` into a
clean environment, and driven over HTTP with the official OpenAI and Anthropic SDKs. Every check
prints `PASS`, `FAIL` or `SKIP`. The run ends with a table and exits 1 if anything failed.

```sh
zsh scripts/release/gate.sh                          # install, serve-27b, families, soak
STAGE=install,serve-27b zsh scripts/release/gate.sh  # any subset, comma-separated
STAGE=service GATE_SERVICE=1 zsh scripts/release/gate.sh   # also load a real launchd agent
```

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
| `service` | Off by default, because it loads a real launchd agent. With `GATE_SERVICE=1`, `service install` must serve the model and `service uninstall` must free the port. | ~2 min |

The checks live in `scripts/release/gate_checks.py`. It reuses the research harnesses
`bench_engine_matrix.py`, `probe_concurrency.py`, `soak_mmlu_pro.py` and `soak_realistic.py`.
When a baseline legitimately moves (for example a new default kernel changes MMLU-Pro), update
`MMLU_BASELINE` in the script together with the measurement that justifies it.
