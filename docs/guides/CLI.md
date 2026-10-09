# Yunshu CLI

> The console's Docs section carries this guide in English, 繁體中文 and 简体中文 ([`frontend/docs/guides/cli.mdx`](../../frontend/docs/guides/cli.mdx)); this file is its GitHub-facing counterpart.

Install on an Apple Silicon Mac (Python 3.13+, macOS 14+):

```sh
uv tool install 'yunshu[vision]'
# Or: brew install yuhuanstudio/tap/yunshu
```

The commands below describe the next release's CLI. Installed 0.1.4 users can use
`yunshu model list`, `yunshu model info`, and `yunshu pull`; `setup`, `models`,
`completion`, and `top` require the updated CLI.

## First run

```sh
yunshu doctor
yunshu setup                  # select a complete local model or enter an MLX HF repo
yunshu serve                  # same selection when no model/directory is configured
```

Selection does not silently download a recommended checkpoint. Enter a repository
whose MLX weights fit your memory. `setup` prints a serve command; it does not start
inference or install a service. A missing model in a noninteractive `serve` returns
an error and next steps. For scripts, supply the model explicitly:

```sh
yunshu models pull Jundot/Qwen3.8-27B-oQ4e-mtp
yunshu serve -m Jundot/Qwen3.8-27B-oQ4e-mtp
```

Models default to `~/.yunshu/models`. Change the directory with
`yunshu config set models_dir /Volumes/Models`. Complete HF cache snapshots are
reused in place. Explicit `--revision <tag-or-sha>` resolves that revision even if
another version is already downloaded. An interrupted pull resumes on retry.

## Models

| Command | Result |
|---|---|
| `yunshu models list` | Local models and HF cache; `--no-hf-cache` excludes cache |
| `yunshu models list --url http://localhost:8000` | Running server inventory |
| `yunshu models pull org/name` | Download or resume; `--dir`, `--revision`, `--force` |
| `yunshu models show org/name` | Config, type, bytes and weight-file count |
| `yunshu models rm org/name --yes` | Remove exactly this name from the configured models directory |
| `yunshu model load <model>` / `unload <id>` | Load/unload on the running server |

`model` remains available. Removal prompts without `--yes`; JSON mode requires
`--yes`. External paths, symlinked models and HF cache entries are not removed.
Stop/unload a model before removing its files.

## Server and service

```sh
yunshu serve -m org/name --port 8000
yunshu status
yunshu top                    # refresh the merged health/system/engine API every 2 s
yunshu top --once              # one snapshot

yunshu service install -m org/name --dry-run  # review launchd plist
yunshu service install -m org/name           # start at login, restart after crashes
yunshu service status
yunshu service logs -n 100
yunshu service logs -f
yunshu service restart
yunshu service stop
yunshu service uninstall
```

`top` currently uses the same information as `status`; the unmerged telemetry
branch's power, clock and thermal view is pending integration. Unavailable
monitoring fields are omitted. Set the configured auth token for protected endpoints.
All install methods use `yunshu service`, rather than a second Homebrew service.
`service install --no-start` only writes the agent; `--force` replaces it.
Models and logs survive uninstall. Logs can be rotated with `service rotate-logs`.

## JSON and errors

Put global options **before** the command:

```sh
yunshu --json models list
yunshu --json service status
yunshu --json service logs -n 50
yunshu --json top
yunshu --json doctor --no-metal
yunshu --url http://localhost:8000 --json complete 'Hello'
```

Successful finite commands produce a JSON result on stdout. Human progress goes
to stderr. Failures produce `{"error": "..."}` and a nonzero exit code (typically
2 for input/connectivity errors, 1 for operation failures). Interactive chat rejects
JSON with a pointer to `complete`. `top --json` is one snapshot; JSON logs reject
`--follow`. `serve --json` emits a `starting` event, then remains running; readiness
comes from `/health`, not the startup event. `--json` is not a server-log format.
`--version`, `--help` and shell completion control options retain Typer's normal text behavior.

`doctor` checks platform, dependencies, model fit/completeness, storage, settings,
auth, server tools, launchd and APC tiers. Its 0.1.5 checks cover writable Evals and
stored-completion directories, and explain Realtime secret/Decisions prerequisites. Realtime secrets follow inference
auth: a configured server token is required when auth is enabled; the default
local open-inference mode also permits minting them.
`--no-metal` skips the MLX import and GPU availability probe; it cannot prove that
Metal or a checkpoint works.

## Shell completion

```sh
mkdir -p ~/.zfunc
yunshu completion zsh > ~/.zfunc/_yunshu
# Add to ~/.zshrc before compinit: fpath=(~/.zfunc $fpath)
# Then: autoload -Uz compinit; compinit

yunshu completion bash > ~/.yunshu-completion.bash
# Add to ~/.bashrc: source ~/.yunshu-completion.bash

mkdir -p ~/.config/fish/completions
yunshu completion fish > ~/.config/fish/completions/yunshu.fish
```

Scripts derive options/subcommands from the installed command tree. `completion`
prints a script without editing shell configuration; `--json completion zsh`
returns `{shell, script}`.

## Other commands and prior art

Inference: `complete`, `embed`, `tokenize`, `rerank`, `transcribe`, `speak`, `ocr`,
`image`. Use each command's `--help` for model, payload and output arguments.
`config`, `cache`, `launch`, `statusline`, `eval`, `bench`, and `diagnose` retain
their existing interfaces; GPU diagnostics/benchmarks execute inference work.

The model lifecycle follows familiar [Ollama CLI](https://docs.ollama.com/cli) and
[LM Studio lms](https://lmstudio.ai/docs/cli) operations. Explicit repository/path
startup is also used by [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
and [mlx_lm.server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md).
The serve/inference/benchmark grouping was compared with the
[vLLM CLI](https://docs.vllm.ai/en/latest/cli/). These are UX references; Yunshu
accepts MLX checkpoints and does not imply checkpoint or runtime compatibility
with their other weight formats.
