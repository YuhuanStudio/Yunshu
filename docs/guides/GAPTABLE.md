# Cross-engine gap table

Use `scripts/research/gaptable.py` with an installation manifest containing exact release
versions, commit IDs, artifact SHA256 hashes, binary paths and extra-engine launch recipes.
Keep the old installations. Every server runs inside gpuq, on ports 18990–18996, with its
own disposable home. The driver removes homes, caches and logs after each cell; failure
records retain a log tail in the result JSONL.

A public comparison needs current main, latest competitor releases, at least three
interleaved repetitions, and the checkpoint/quantization attached to every row. Different
quantizations are different product samples. Single-repetition smoke results prove only
that the engine and parser run; they do not decide performance priorities.

The full plan includes code and prose at 1K, 32K and 128K, cold decode/TTFT, 32K warm and
follow-up TTFT, eight concurrent independent 1K prompts (aggregate end-to-end tokens/s,
per-request decode tokens/s and TTFT), and a separate Splash `--disable-ane` arm.
Repetition order rotates between engines. Decode uses 512 output tokens; concurrency
uses 256 per request. Early stop remains an error. Aggregate throughput includes prefill
and queueing; per-request decode throughput starts at its own first token.

TensorFold must receive `--parallel 1` for single-request cells. Its automatic eight
streams at a large context can reserve all prefix-cache memory and produce genuine
`cached_tokens=0`; this is not a missing session identifier. Each cell sizes its context
window to prompt length plus 4096, and llama.cpp reserves that amount **per slot**.
Cache counts come from `usage.prompt_tokens_details.cached_tokens`. Missing counts stay
unknown, and invalid values fail. The table driver rejects warm/follow-up 32K cache
misses and incomplete cells, rather than presenting cold prefill as warm TTFT.

Prepare and inspect a plan without starting a server:

```bash
python scripts/research/gaptable.py plan --manifest /path/manifest.json \
  --src /path/to/final-main --out /Volumes/P5Plus/results/final
```

The lead starts the full run only after the final merges:

```bash
ROOT=/path/to/final-main OUT=/Volumes/P5Plus/results/final \
  MANIFEST=/path/manifest.json scripts/research/gaptable_full.sh
```

The shell driver submits one timing cell at a time (`--quiet`, priority 1), polls its rc
at 60-second intervals, and stops on failure. The plan records the main SHA; the runner
refuses a changed HEAD during the run. Do not edit the source tree during measurement.
