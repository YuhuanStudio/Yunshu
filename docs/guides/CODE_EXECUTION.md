# Code execution (Anthropic `code_execution`)

The Anthropic `code_execution_20250825`, `_20260120` and `_20260521` tools run on this machine, inside
Anthropic's [sandbox-runtime](https://github.com/anthropics/sandbox-runtime) (`srt`). Off by default.

## Turn it on

```bash
npm install -g @anthropic-ai/sandbox-runtime     # Node >= 22.12; srt is an optional dependency
yunshu serve -m <model> --set code_execution=1
yunshu doctor                                    # "code execution" row: srt version and a self-test
```

| Setting | Default | Meaning |
|---|---|---|
| `YUNSHU_CODE_EXECUTION` | off | Serve the tool. Off: a request that declares it gets a `400` naming this setting. On without srt: `503` (`api_error`) with the install command. A command never runs outside srt. |
| `YUNSHU_CODE_EXECUTION_ROOT` | `~/.yunshu/sandbox` | One sub-directory per container, mode 0700. |
| `YUNSHU_CODE_EXECUTION_TIMEOUT` | 60 s | Wall clock per call (also the CPU-time limit); then `execution_time_exceeded`. |
| `YUNSHU_CODE_EXECUTION_MEMORY_MB` | 2048 | Resident memory of the call's whole process tree, sampled every 100 ms; above it the call is killed. |
| `YUNSHU_CODE_EXECUTION_NETWORK_ALLOW` | none | Comma-separated hosts the sandbox may reach. Empty: no network, loopback included. |
| `YUNSHU_CODE_EXECUTION_TTL_MIN` | 20 | Minutes a container may sit unused before it is deleted. |

## What the API does

Declare `{"type": "code_execution_20250825", "name": "code_execution"}` (no beta header). The model gets the
two sub-tools `bash_code_execution` and `text_editor_code_execution` (`view`, `create`, `str_replace`) and
the response carries `server_tool_use` + `bash_code_execution_tool_result` /
`text_editor_code_execution_tool_result` blocks exactly as in the official docs, streamed the same way (the
sub-tool input as `input_json_delta`, each result block whole in one `content_block_start`), plus
`container: {id, expires_at}` (in `message_delta.delta` when streaming). The anthropic 2.x SDK's pydantic
types validate every shape.

- **Containers.** Pass `container: "<id>"` (or `{id}`) back to keep the working directory; an unknown or
  expired id is a `400`. A container is deleted after the idle TTL; the total is capped at 2 GiB (oldest go
  first).
- **Files.** `container_upload` content blocks copy a Files API file into the working directory. Every bash
  call gets a fresh `$OUTPUT_DIR`; its top-level files become Files API entries returned as
  `bash_code_execution_output {file_id}` (download with `GET /v1/files/{id}/content`).
- **Limits.** stdout and stderr are cut at 64 KiB each (marked), a call that writes more than 1 MiB is killed
  (`output_file_too_large`), background processes die with the call, and the environment the command sees
  holds only `PATH`, `HOME` (the container), `TMPDIR`, `OUTPUT_DIR` and locale variables, never the server's.

## What the sandbox allows

Writes: the container directory only. Reads: `/Users`, `/Volumes`, `/home` and the container root are
denied; only this container and the Python interpreter's prefix are allowed back, so your home, keys,
`~/.yunshu` and other containers are unreadable. Network: none unless listed; loopback is never reachable and
allowed names cannot resolve into private ranges. Unix sockets, Apple Events and `open` are off.
Violations are logged as metadata (container id, return code, byte counts); commands, output and file
contents are never logged. An allowed domain list does not inspect traffic: allowing a broad host such as
`github.com` can still leak data, so keep the list short.

## Differences from Anthropic's container

macOS, not Linux x86_64; the Python is the interpreter Yunshu runs on (standard library only, no pandas or
numpy); no REPL state between calls and no programmatic tool calling (the `20260120` and `20260521` types are
accepted and behave like `20250825`); no Content Credentials. OpenAI `code_interpreter` and `shell` are not
served yet.

The sandbox runs under macOS `sandbox-exec`, which Apple marks deprecated, and srt is a 0.0.x release
(tested: 0.0.81).
