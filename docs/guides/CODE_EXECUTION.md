# Code execution

Yunshu runs model-written code on this machine, inside Anthropic's
[sandbox-runtime](https://github.com/anthropics/sandbox-runtime) (`srt`), for three APIs. Off by default.

| API | What it serves |
|---|---|
| Anthropic Messages | `code_execution_20250522`, `_20250825`, `_20260120`, `_20260521`: bash, text editor and a Python tool, REPL state, programmatic tool calling |
| OpenAI Responses | `code_interpreter` (container `auto` or an id) and the hosted `shell` tool (`environment` `container_auto` / `container_reference`) |
| OpenAI containers | `/v1/containers`: create, list, retrieve, delete, and container files |

All three share one container manager: a container made through one API can be named in the other.

## Turn it on

```bash
npm install -g @anthropic-ai/sandbox-runtime     # Node >= 22.12; srt is an optional dependency
yunshu serve -m <model> --set code_execution=1
yunshu doctor                                    # "code execution" row: srt version and a self-test
```

| Setting | Default | Meaning |
|---|---|---|
| `YUNSHU_CODE_EXECUTION` | off | Serve the tools. Off: a request that declares one gets a `400` naming this setting. On without srt: `503` with the install command. A command never runs outside srt. |
| `YUNSHU_CODE_EXECUTION_ROOT` | `~/.yunshu/sandbox` | One sub-directory per container, mode 0700. |
| `YUNSHU_CODE_EXECUTION_TIMEOUT` | 60 s | Wall clock per call (also the CPU-time limit); then `execution_time_exceeded`. A Python cell that waits for the client's tool results does not count. |
| `YUNSHU_CODE_EXECUTION_MEMORY_MB` | 2048 | Resident memory of the call's whole process tree, sampled every 100 ms; above it the call is killed. |
| `YUNSHU_CODE_EXECUTION_NETWORK_ALLOW` | none | Comma-separated hosts the sandbox may reach. Empty: no network, loopback included. |
| `YUNSHU_CODE_EXECUTION_TTL_MIN` | 20 | Minutes a container may sit unused before it is deleted. |

## Anthropic

Declare `{"type": "code_execution_20260120", "name": "code_execution"}` (no beta header needed). The response
carries `server_tool_use` + `*_tool_result` blocks exactly as in the official docs, streamed the same way (the
sub-tool input as `input_json_delta`, each result block whole in one `content_block_start`), plus
`container: {id, expires_at}` (in `message_delta.delta` when streaming). The anthropic 2.x SDK's pydantic types
validate every shape.

What each version offers (the SDK types show no other schema difference, so these are the behaviors):

| Version | Sub-tools the model sees | Python interpreter |
|---|---|---|
| `20250522` | `code_execution` (Python, input `{code}`) | fresh process every call |
| `20250825` | `bash_code_execution`, `text_editor_code_execution`; `code_execution` too when a client tool lists a code-execution caller | fresh process every call |
| `20260120`, `20260521` | all three | **one process per container: variables, imports and open files persist between calls** |

- **REPL state.** The Python process lives with the container (pass `container: "<id>"` back, in the same
  conversation or a later one). It is killed and replaced cleanly on a timeout, a memory overrun, more than 1 MiB
  of output, an idle period of 10 minutes, a container delete or when more than 4 containers hold one (least
  recently used first); the next call then runs in a clean process and the result says that variables are gone.
  A cell prints like a notebook cell (the last expression's value) and `await` works at top level.
- **Containers.** An unknown or expired id is a `400`. A container is deleted after the idle TTL; the total
  is capped at 2 GiB (oldest go first).
- **Files.** `container_upload` content blocks copy a Files API file into the working directory. A bash call gets
  a fresh `$OUTPUT_DIR` whose top-level files become Files API entries (`bash_code_execution_output {file_id}`); a
  Python cell's new or changed files become `code_execution_output {file_id}` (download with
  `GET /v1/files/{id}/content`).
- **Programmatic tool calling.** A client tool with `"allowed_callers": ["code_execution_20260120"]` (or the
  version you declared) is offered to the model as an `async` Python function inside the interpreter, not as a
  direct tool (add `"direct"` to offer both). When the code awaits it, the turn ends with `stop_reason:
  "tool_use"`: the content holds the `server_tool_use` block and a `tool_use` block per call with
  `caller: {type: "code_execution_...", tool_id: "srvtoolu_..."}`. Send the assistant content back plus a user
  message with one `tool_result` per call (and the `container`); the response then begins with the
  `code_execution_tool_result` for the original `server_tool_use` (or with more `tool_use` blocks if the code
  calls again). Calls issued together (`asyncio.gather`) arrive together and all must be answered. The model never
  sees these inner calls: its history holds one code call and its final result. A `tool_result` with
  `is_error` raises `RuntimeError` in the code. The cell waits at most 270 s for the answer; after that it is
  killed and the continuation request is a `400` saying so. One cell may make 256 tool calls, each argument set up to
  1 MiB. A request that names a code-execution caller without declaring the tool is a `400`. For `20260521` the
  block's `caller.type` is `code_execution_20260120`, the newest the SDK models.

## OpenAI Responses

```python
r = client.responses.create(
    model="m", input="Plot nothing; print 6 * 7 with Python.",
    tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
    include=["code_interpreter_call.outputs"],
)
# output: code_interpreter_call {container_id, code, outputs: [{type: "logs", logs}], status}
```

- `container` is an id (`cntr_...`) or `{"type": "auto", "file_ids": [...], "memory_limit", "network_policy"}`; `auto`
  makes one container per response and, with `previous_response_id`, reuses the container the earlier response
  used. Python state persists inside it (same interpreter as above). `outputs` is `null` unless
  `include` has `code_interpreter_call.outputs`; PNG/JPEG/GIF/WebP files the code makes are `image` outputs whose
  `url` is the container file content route.
- Streaming events: `response.code_interpreter_call.in_progress`, `response.code_interpreter_call_code.delta` /
  `.done`, `.interpreting`, `.completed`, each with `output_index` and `item_id`.
- **Files.** Files the code writes (in its working directory, also reachable as `/mnt/data`) are container
  files. When the answer mentions one (`[name](sandbox:/mnt/data/name)` or its path), the text carries a
  `container_file_citation` annotation with `container_id`, `file_id`, `filename` and the character span.
- **Shell.** `{"type": "shell", "environment": {"type": "container_auto"}}` (or `container_reference` with a
  `container_id`) is run by the server: a `shell_call` item (`action.commands`, `timeout_ms`, `max_output_length`,
  `environment: {type: "container_reference", container_id}`) followed by a `shell_call_output` item with one
  `{stdout, stderr, outcome: {type: "exit", exit_code} | {type: "timeout"}}` per command, and the
  `response.shell_call_command.*` / `response.shell_call_output_content.*` events. Commands run in order in the
  same container as `code_interpreter`. A `shell` tool with `environment` `local` (or none) is the client's: the
  model's call comes back as a `shell_call` item with `environment: {type: "local"}` for the client to run and
  answer with `shell_call_output`.
- One request uses one container: naming two different containers (or an id together with `auto`) is a `400`.
  `programmatic_tool_calling` (OpenAI's variant) is not implemented.

## OpenAI containers

`POST /v1/containers` (`name` required), `GET /v1/containers` (`limit`, `order`, `after`, `name`),
`GET/DELETE /v1/containers/{id}` (delete answers `204`), and for files `POST/GET /v1/containers/{id}/files`,
`GET/DELETE /v1/containers/{id}/files/{file_id}`, `GET .../content`. Upload a multipart `file` or a Files API
`file_id`; the file is `/mnt/data/<name>` (`source: "user"`), files the code makes are `source: "assistant"`. Ids
are `cntr_...` and `cfile_...` (stable per container and path). An expired container is gone: `404`.

Mapping onto the hosted API's knobs (the response always shows what is in force):

| Parameter | Here |
|---|---|
| `expires_after` `{anchor: "last_active_at", minutes}` | the idle lifetime; capped by `YUNSHU_CODE_EXECUTION_TTL_MIN`, which is also the default. Each call that uses the container refreshes it |
| `memory_limit` `1g`..`64g` | a ceiling under `YUNSHU_CODE_EXECUTION_MEMORY_MB`; the response shows the largest label that fits |
| `network_policy` | `disabled`, or `allowlist` of hosts that the server list already allows (anything else is a `400`); never wider than `YUNSHU_CODE_EXECUTION_NETWORK_ALLOW`, also after the setting changes |
| `skills`, `domain_secrets` | not supported (`400`) |

Containers belong to the API key that made them (admins see all).

## What the sandbox allows

Writes: the container directory only. Reads: `/Users`, `/Volumes`, `/home` and the container root are
denied; only this container and the Python interpreter's prefix are allowed back, so your home, keys,
`~/.yunshu` and other containers are unreadable. Network: none unless listed; loopback is never reachable and
allowed names cannot resolve into private ranges. Unix sockets, Apple Events and `open` are off. The gateway
itself never follows a link the sandbox planted: container files are opened component by component with
`O_NOFOLLOW`, names are validated, nothing is built into a shell command line, and every process runs in its own
group that is killed on a timeout, a memory overrun, an output overflow or a container delete. The Python process
talks to the gateway over a private pipe, only for tools the request declared. Violations are logged as metadata
(container id, return code, byte counts); commands, output and file contents are never logged. An allowed domain
list does not inspect traffic: allowing a broad host such as `github.com` can still leak data, so keep the list
short.

## Differences from the hosted tools

macOS, not Linux; the Python is the interpreter Yunshu runs on (standard library only, no pandas, numpy or
matplotlib); no Content Credentials; no skills; no `programmatic_tool_calling` for OpenAI. The sandbox runs under
macOS `sandbox-exec`, which Apple marks deprecated, and srt is a 0.0.x release (tested: 0.0.81).
