# Your first local conversation

Yunshu runs on Apple Silicon with macOS 14 or later. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then install the vision runtime for Qwen3.5 / Qwen3.8 models (it also serves their text requests):

```bash
uv tool install "yunshu[vision]" --python 3.13
yunshu --version
yunshu doctor
```

The base `yunshu` install serves text-only `mlx-lm` models. The `vision` extra supplies the shared runner used by vision-capable checkpoints, including text requests to those checkpoints. Audio, image generation and embeddings use separate extras; see the [package extras](../../pyproject.toml).

## Choose a model before downloading

Check what is already on disk:

```bash
yunshu model list
```

A small first-run checkpoint is `mlx-community/Qwen3.5-0.8B-MLX-bf16`. It is a smoke-test model; choose a larger checkpoint for coding quality. Download once, then serve:

```bash
yunshu pull mlx-community/Qwen3.5-0.8B-MLX-bf16
yunshu doctor -m mlx-community/Qwen3.5-0.8B-MLX-bf16
yunshu serve -m mlx-community/Qwen3.5-0.8B-MLX-bf16
```

`pull` resumes an interrupted download and reuses complete models in Yunshu's model directory or the Hugging Face cache. You can also pass a local directory to `serve -m`, or use `yunshu serve /path/to/model`. Options such as `--port` work before or after the positional model; `--model` remains supported. For external model storage, set the directory **before** downloading:

```bash
yunshu config set models_dir /Volumes/MyModels/models
```

This changes where future downloads go; it does not move existing weights. An existing Hugging Face cache remains discoverable. [Configuration](../CONFIGURATION.md) shows effective values and their sources.

For Qwen3.8-27B, use the [main quickstart](../../README.md#qwen38-27b) and check `doctor -m` for memory fit and drafter availability. Do not download an MTP sidecar as if it were a complete model.

## Check readiness, then chat

Keep the server terminal open. In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health/ready
curl --fail http://127.0.0.1:8000/v1/models
yunshu chat
```

Readiness returns HTTP 503 while loading or when the checkpoint cannot load; wait for HTTP 200 before sending your first request. `chat` connects to an existing server. It does not download a model or start another server. Use `/help` for commands, `/clear` to clear conversation turns, and `/quit` to leave the client. Stop the server with Ctrl-C in its terminal.

If port 8000 is occupied, choose another explicit port:

```bash
yunshu serve -m mlx-community/Qwen3.5-0.8B-MLX-bf16 --port 8001
yunshu chat --url http://127.0.0.1:8001
```

Use the chosen port in readiness checks and client URLs too. `serve` reports an unavailable TCP address before checking/loading the model. Another process can still acquire the address between preflight and server startup; the final bind error is authoritative.

## Send an API request

OpenAI clients use `/v1`; Anthropic clients use the host URL without `/v1`.

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64,"temperature":0}'
```

For an agent, review the generated launch settings first:

```bash
yunshu launch claude --dry-run
yunshu launch codex --dry-run
yunshu launch opencode --dry-run
```

Then run the same command without `--dry-run`. The launcher reads the served model's real context window and capabilities. See [coding-agent compatibility](AGENT_COMPAT.md) for exact route coverage and configuration behavior, or [clients](CLIENTS.md) for SDK examples.

## Upgrade or run at login

```bash
uv tool upgrade yunshu
yunshu service install -m mlx-community/Qwen3.5-0.8B-MLX-bf16 --dry-run
```

Remove `--dry-run` to install the login service. See [service management](SERVICE.md) for logs, start/stop and uninstall. For a failure, run `yunshu doctor -m <model>` and follow [troubleshooting](TROUBLESHOOTING.md); check `/health/ready` rather than treating an open port as proof that a model is ready.
