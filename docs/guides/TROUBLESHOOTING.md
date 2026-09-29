# Troubleshooting

Start with `yunshu doctor` (add `-m <model>` to check a model too). It checks the platform, MLX /
Metal, memory, settings, the models directory, the port and the service, and prints a fix for each
problem. It exits 1 when something blocks serving.

## Install

**`cannot import mlx` / Python runs under Rosetta.** MLX ships arm64 wheels only. An x86_64 Python,
for example an old Homebrew under `/usr/local`, cannot load it. Install a native interpreter with
`uv python install 3.13`, then reinstall Yunshu with it.

**`requires-python >=3.13`.** Yunshu needs Python 3.13. `uv tool install` picks up a uv-managed
3.13 automatically.

**A Qwen3.5 / 3.6 / 3.8 or vision model fails to load, and doctor shows `mlx-vlm: not installed`.**
Install the `vision` extra: `uv tool install "yunshu[vision]"`, or `uv sync --extra vision` in a
source checkout.

## Starting the server

**`Error: /path: no such directory`.** `yunshu serve` checks the model before loading it:

- `yunshu model list` shows what is on disk (the models directory and the Hugging Face cache).
- `yunshu pull <org/name>` downloads a model.
- A directory that holds several models goes to `--models-dir`, not `--model`.

**`interrupted download` / `N weight shard(s) missing`.** Run the same `yunshu pull` again. It
resumes and fetches only the missing files.

**`weights ... > ... memory`.** The model cannot fit in unified memory. Use a smaller or more
quantized build, for example a 4-bit MLX conversion.

**A warning that weights use most of the GPU working set.** The model loads, but long prompts may
run out of memory. macOS gives the GPU a working set somewhat below total RAM; `yunshu doctor`
prints both numbers.

**`port ... is used by another program`.** Pick another port with `--port`. If the program on the
port is Yunshu (for example the background service), `yunshu status` reports it.

**The server starts but `/health/ready` returns 503.** When a model fails to load (wrong path,
truncated or damaged weights, not enough memory) the server stays up instead of exiting, so a
supervisor does not restart-loop. `curl http://127.0.0.1:8000/health/ready` returns 503 with a
`reason` field, for example `weights are damaged or incomplete; run yunshu pull again`; chat
requests get a 503 `No model loaded` until you fix it and restart. The full traceback is in the
server log (the terminal, or `yunshu service logs`); look for `FATAL: model ... load failed`. The
probe also returns 503 while shutting down or when GPU memory use is above 95%.

**Which speculative path is in use.** `yunshu doctor -m <model>` prints a `speculative` row, and the
startup log has a `Speculative decoding:` line. A DFlash2 drafter that matches the model (for
Qwen3.8-27B: `yunshu pull incoai/Qwen3.8-27B-DFlash2`) is picked up automatically from the models
directory or the Hugging Face cache. `YUNSHU_VLM_DRAFT=mtp` forces the checkpoint's MTP head,
`YUNSHU_VLM_DRAFT=off` turns drafting off, and a directory path picks a specific drafter. If an
automatic drafter cannot be loaded, the server logs a warning and falls back to MTP.

**A request returns 400 about the prompt or `max_tokens`.** The prompt is longer than the model's
context window, or `max_tokens` is above the server's limit (131072). Shorten the prompt or lower
`max_tokens`; the message states both numbers.

**Stopping the server.** Ctrl-C or `kill` (SIGINT / SIGTERM) stops accepting new requests and lets
running ones finish for `YUNSHU_DRAIN_TIMEOUT` seconds (default 30); after that the remaining
connections are cut, their generation stops and the server exits. A client that disconnects
mid-request, streaming or not, cancels its generation at the next chunk: during prefill the next
request waited 0.8 s on a 0.8B model and 1.9 s on Qwen3.8-27B (one 2048-token chunk).

**429 `Rate limit exceeded`.** Rate limiting is off by default. It is on only if
`YUNSHU_RATE_LIMIT_RPM` is set above 0.

**A bad setting stops startup.** `yunshu config` shows every effective value and where it came
from. A misspelled `YUNSHU_*` name only produces a warning with the closest match.

## Clients

**401 `Missing or invalid Authorization header`.** The server was started with `--auth-token`.
Send `Authorization: Bearer <token>`, or `x-api-key: <token>` from Anthropic clients.

**A client in Docker cannot connect.** By default the server listens on `127.0.0.1` only. Start it
with `--host 0.0.0.0 --auth-token <secret>` and connect to `http://host.docker.internal:8000`.

**`model not found` in multi-model mode.** Use an id from `GET /v1/models` (`yunshu model list
--url http://127.0.0.1:8000`). In single-model mode, any name works.

**A reasoning model's answer is empty.** Its tokens went to thinking. Raise `max_tokens`, lower
`reasoning_effort`, or pass `chat_template_kwargs: {"enable_thinking": false}`.

## Reporting a bug

Include:

- the output of `yunshu --json doctor -m <model>`
- the model folder name
- the command or request that failed
- the last ~50 lines of the server log

Use the issue template at https://github.com/YuhuanStudio/Yunshu/issues.
