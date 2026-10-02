# I8: auxiliary requests yield to interactive agent turns

Evidence is the saved traffic, not a client-name / small-model heuristic. Run
`scripts/research/agentic/characterize_auxiliary.py MAIN/docs/research/runs --out inventory.json`
to inventory physical captured bodies. The inventory snapshot is in
`../research/runs/2026-10-02-i8/fingerprints.json`; later concurrently captured files may change counts.

| Client | Captured request classification | Client model | Output limit | Tools |
| --- | --- | --- | --- | --- |
| opencode | 11 title captures; 68 agent captures | `Qwen3.8-27B-oQ4e-mtp` | `max_tokens=32000` | titles 0; agent 9 |
| Claude Code | 134 agent captures; no captured auxiliary request | `Qwen3.8-27B-oQ4e-mtp` | `max_tokens=32768` | 18 |
| Codex | 43 Responses agent captures; no captured auxiliary request | `Qwen3.8-27B-oQ4e-mtp` | both limits absent | 8 |

These are physical saved bodies (including repeated/copied captures), not independent
sessions. Error-only bodies are excluded. Fingerprints include Anthropic system blocks,
Chat Completions system/developer messages, and Responses instructions/developer
inputs; each source component has its own hash (structured values use sorted-key JSON). No summary / topic-detection / quota-check /
auto-title fingerprint was captured for Claude Code or Codex. Such requests remain
interactive until captured evidence supports a separate exact classifier.

The complete embedded request-log snapshot (`request_logs.json` in the run directory)
has 67 opencode no-tools requests (621–805 prompt tokens, all with 3 messages,
model `Qwen3.8-27B-oQ4e-mtp`, max_tokens 32000), 465 opencode tool requests,
265 Claude requests with 18 tools, and 64 Codex requests with 8 tools. One historical
Claude log lacks model / sampling metadata; the other 264 use the model / limit above.
Unstored no-tools bodies are not assumed to match the title fingerprint.

Coverage is the benchmark traffic: `agents.py` forces Claude's small/haiku model
names to the target and sets `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` (also disables
terminal titles). Codex is captured in CLI exec mode. These captures cannot establish
which side requests the regular Claude UI / Codex app sends. No unobserved fingerprint
is invented or deprioritized.

The opencode title system message is 2,096 characters, starts with
`You are a title generator. You output ONLY a thread title. Nothing else.`, and has
SHA-256 `e7a6848eba328f28c7e870874cf0591e4edbaf90d7602ad8fdfe90601c6e656f`.
There are exactly three plain-text messages, roles `system,user,user`; no tools or
legacy functions. The fixture preserves a real captured body. The known client model
and output limit also have to match. No short-token-budget assumption is made:
the captured title limit is as large as the agent limit. I4/I5's recorded ~650-token
prompt / ~3 s replay TTFT is a prompt measurement, not a requested output limit.

## Scheduling and correctness

`admission.py` classifies the validated gateway Chat Completions body. It attaches
priority -1 to the request record; `RequestTracker` transfers that to the cancellation
event carried across executor threads. Unknown requests retain priority 0. This
metadata never enters the prompt or sampling parameters.

A recognized auxiliary request waits 500 ms at the gateway to let the companion agent
turn arrive (the existing captured replay sends it after 300 ms), then waits for
interactive HTTP requests to drain before entering the engine. This also prevents
auxiliary work entering the serialized text fast path while a primary is present.
The grace costs auxiliary latency and does not add a wait to ordinary requests.

The VLM runner holds pending auxiliary rows while primary work exists (including calls still preparing before their row arrives), keeps the two
priorities in separate batches, and does not step an active auxiliary batch while a
primary is present. Speculative generators retain their own cache / sampling state;
the shared drafter's prompt readout and mRoPE context are restored on switching.
Per-request speculative counters exclude the other lane's work; gateway queue positions
follow the same priority ordering.
The experimental round driver has a separate auxiliary driver with the same model,
chunking and sampling path. Neither implementation cancels, drops, restarts or truncates
the auxiliary output. Cancelled paused rows are still cleaned up at the next slice.

Auxiliary requests have APC disabled (lookup and store); they cannot supersede / evict
primary RAM or disk checkpoints. The text fast path similarly bypasses prefix and
prompt caching for auxiliary requests. Auxiliary prompts do not update the APC
prefill cost model either. Normal requests keep the existing APC policy.

GPU kernels already submitted are non-preemptible. A primary arriving *after* auxiliary
prefill started can wait for that in-flight operation / slice, never for the whole
auxiliary generation. The text-only mlx-lm fast path still runs an already-started
request to completion; its gateway deferral and cache protection do not turn it into a
preemptible runner. The measured claim here is the 27B VLM runner. Under an endless
stream of primary work auxiliary requests can wait indefinitely, subject to existing
client cancellation / deadlines; strict priority cannot also promise bounded auxiliary
latency. Existing queue limits, errors and deadlines remain in force.

`YUNSHU_AUXILIARY_SCHEDULING` is a stable deployment option. The default is decided
only after the paired replay; 0 restores ordinary admission / APC behavior.

## Validation

Regression tests failed on an isolated original HEAD: 16 failures, 1 passing cancellation check, rc=1
(`../research/runs/2026-10-02-i8/unit-before.log`). The final regressions additionally check
pending cancellation, gateway grace, APC cost-model isolation and Responses inventory sources. Tests cover fingerprint near misses, opt-out,
worker propagation, pending and active priority inversion, state-preserving resume,
separate speculative / round-driver lanes, and APC exclusion.

GPU jobs `1002-184051-00-i8-before` and `1002-184051-00-i8-after` were submitted at
priority -1, timeout 20 minutes, stall 5 minutes. Each mode runs three fresh servers
on the same 27B checkpoint, replaying all six captured opencode main turns 0002–0007,
with captured title 0001 sent once, 300 ms before the first main turn. The main
loop never waits for the independent title future between turns. Bodies are unchanged except
streaming transport and seed 42. Original tools, prompts, max_tokens and client model
are preserved; each server uses a new empty APC disk root on P5Plus, keeping the
SSD tier enabled while preventing cross-run persistence; there is no artificial completion cap. All servers use ports
18990–18999, this worktree's Python sources, and cleanup only their own process group.

`scripts/research/agentic/i8_summarize.py` rejects incomplete runs, reports main-turn
TTFT p50/p90 pooled and per run, and compares complete content / reasoning / tool-call /
finish-reason SHA-256 digests for both primary and auxiliary responses. The result
must pass all 18 primary and all 3 title comparisons before a lossless default is enabled.
