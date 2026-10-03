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
interactive HTTP requests to drain, or its 10-second aging threshold, before entering the engine. This also prevents
auxiliary work entering the serialized text fast path while a primary is present.
The grace costs auxiliary latency and does not add a wait to ordinary requests.

The VLM runner holds pending auxiliary rows while primary work exists (including calls still preparing before their row arrives), keeps the two
priorities in separate batches, and pauses an active auxiliary batch while a
primary is present until that row has waited 10 seconds since its last service. Speculative generators retain their own cache / sampling state;
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
preemptible runner. The measured claim here is the 27B VLM runner. On the default runner, aging admits and services an auxiliary row after 10 seconds of waiting,
but this is a service opportunity rather than a completion deadline: in-flight atoms,
other older work, client cancellation and existing deadlines still apply. The
experimental round driver retains strict auxiliary priority and does not use this
work-ordering policy; endless primary driver traffic can still defer auxiliary
service. Existing queue limits, errors and deadlines remain in force.

`YUNSHU_AUXILIARY_SCHEDULING` is a stable deployment option. The default remains off; 0 restores ordinary admission / APC behavior.

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

## Historical uncached-work ordering: CPU simulation (YUNSHU_UNCACHED_SCHEDULING)

`scripts/research/agentic/i8_admission_sim.py` replays three captured opencode
sessions (prompt sizes from the request bodies, 4.15 chars/token, 236 ms +
1.21 ms/token prefill in 512-token atoms, 25 ms batched decode step, 150 decode
tokens and 2 s tool time per turn, sessions starting 0/1.5/3 s apart) through
three policies. Main-turn TTFT, seconds (n = 30 main turns):

| policy | p50 | p90 | max | title TTFT p50 |
|---|---|---|---|---|
| upstream FIFO | 1.86 | 10.93 | 29.0 | 10.5 |
| auxiliary yields | 1.86 | 10.72 | 28.5 | 21.7 |
| auxiliary + uncached HRRN + 250 ms decode quanta | 1.47 | 16.91 | 32.5 | 21.9 |

Shortest-uncached-first lowers the median but lengthens the tail (long cold
first turns wait behind many short suffix turns), and decode quanta lengthen
cold prefills; a 25 ms quantum gives p50 2.12 / p90 13.76. In this trace the
auxiliary deferral alone moves main p90 only about 2%. The ordering is therefore
kept disabled (default off) and is not recommended on this evidence; the model
is crude (no real contention between decode and prefill kernels), so a GPU
A/B would be required before any default change.

## 27B replay, both options on (round 1 of the A/B, job i8c-ab-r1-0103)

Captured opencode fix-cart-discount session (six agent turns plus the title),
seed 42, arm 0 = stock, arm 1 = `YUNSHU_AUXILIARY_SCHEDULING=1` and
`YUNSHU_UNCACHED_SCHEDULING=1`. Main-turn TTFT (s), turns 1-6:

| arm | turn 1 | 2 | 3 | 4 | 5 | 6 | title |
|---|---|---|---|---|---|---|---|
| 0 | 8.66 | 0.50 | 0.86 | 1.74 | 0.62 | 0.30 | 3.04 |
| 1 | 7.78 | 0.27 | 0.87 | 2.60 | 20.43 | 20.41 | 26.0 |

Main outputs are token-identical in both arms. Arm 1 regresses turns 5-6 to 20 s
(prefill_ms 20305): the main prefill is starved while the title row decodes, and
the title output hash differs from arm 0. Both options stay off. The initial hypothesis was upstream speculative decode returning before
prefill. Single-feature arms were still queued when this was written. The
completed bisect and resumed investigation below identify paused-lane decode
debt, rather than a prefill sharing the same speculative generator, as the
actual cause.


## Resumed fixes (2026-10-03)

The single-feature bisect (`1003-032144-00-i8c-bisect-r1-0103`, rc 0)
completed both arms: auxiliary-only main turns 5/6 were 0.702/0.358 s,
uncached-only 0.611/0.298 s. The ~20 s stalls require their combination.
The original upstream-speculation explanation above was too broad: speculative
lanes contain one request each. A paused auxiliary speculative lane was counted
as available decode work, so it blocked primary prefill via decode debt without
being allowed to repay that debt. Restrict debt gating to eligible decode lanes;
primary canonical atoms now progress while auxiliary speculation is paused.

A suspended atom must also remain in the prefill waiting set. Otherwise a
decode-only slice marks its prefill complete and corrupts the remaining-work
estimate. Regression tests cover both failures.

Auxiliary output differences came from disabling APC numerical checkpoints,
including the system-turn boundary, along with cache storage. The auxiliary
coordinator now snapshots only planning metadata and runs the same canonical
checkpoint arithmetic; lookup, storage, reservation, cost-model updates and
primary cache metadata remain excluded. The first repaired replay
(`1003-101854-00-i8-fix-ab-r1-p0-1020`, rc 0) matches all six primary outputs
and the complete title output. Its runner source predates the final resumed
merge integration, so it is preliminary evidence, excluded from the final
three-run summary.

Interactive prefills gain persistent FIFO protection after one overtaking
atom; subsequent short arrivals cannot repeatedly skip a long cold request.
Decode debt is reduced from 250 to 100 ms. This prevents starvation without
changing canonical token spans. It does not promise that every TTFT improves.

The CPU simulator now defaults to the runner's actual 2048-token atoms; use
`--atom 512` to reproduce the earlier approximation. With the same 30-turn
trace and offsets 0/1.5/3 s:

| policy / atom size | main p50 (s) | main p90 (s) | max (s) |
|---|---:|---:|---:|
| FIFO / 2048 | 1.868 | 10.675 | 28.146 |
| old full policy / 2048 | 1.769 | 14.107 | 30.221 |
| repaired full policy / 2048 | 1.802 | 12.624 | 29.046 |
| FIFO / historical 512 | 1.855 | 10.932 | 29.021 |
| repaired full policy / historical 512 | 1.832 | 10.147 | 32.171 |

These are CPU estimates, not measured GPU latencies. In particular, the actual
2048-token approximation still regresses p90 against FIFO. Both settings remain
default off; a single-session GPU replay does not settle mixed long-prefill
policy. The source/engaged-mode receipts and `i8_summarize.py --first-run 5`
reject incomplete sessions, different code versions and output differences.


## Final three-round 27B replay (runs 5/6/7)

`i8_three_runs.sh` consolidates three interleaved fresh-server pairs in one quiet
job (`1003-112620-00-i8-fix-three-ab-1127`), arm order 1/0/1. Same 27B checkpoint,
seed 42, unchanged seven captured bodies, six empty APC roots and six source /
MTP receipts. Results are in `../research/runs/2026-10-03-i8e`; summarize with
`i8_summarize.py ROOT --first-run 5`. All 18 primary and all 3 title response
content/reasoning/tool-call/finish digests match. Auxiliary cached tokens are
0 in all three enabled sessions.

| API main-turn latency | stock ordering | both options on |
|---|---:|---:|
| pooled main TTFT p50 | 0.745 s | 0.797 s |
| pooled main TTFT p90 | 8.666 s | 7.760 s |
| mean main TTFT | 2.122 s | 2.071 s |
| warm main TTFT p50 | 0.628 s | 0.696 s |
| warm main TTFT p90 | 1.755 s | 2.423 s |
| mean sum of six main completion times per session | 40.829 s | 32.366 s |
| mean title TTFT | 3.040 s | 28.778 s |
| mean title completion time | 17.807 s | 37.268 s |

Cold first-turn TTFT per run is 8.671/8.664/8.670 s versus 7.780/7.787/7.752 s.
The repaired turns 5/6 are 0.696/0.384, 0.692/0.389, 0.709/0.390 s; no ~20 s
stall remains. Mean main TTFT improves 2.36%, in the same direction in all
three pairs (2.1208→2.0735, 2.1218→2.0718, 2.1222→2.0690 s), a small measured
win retained in the opt-in path. Pooled p90 improves 10.45%, but median worsens
6.98% and warm p90 worsens 38.1%. Main completion time improves 20.73%, helped
by retaining MTP for the primary while the title waits. These are one captured
session's scheduling tradeoffs, not a general model-speed improvement. Both
options remain default off given the warm-tail and mixed-prefill evidence.

Independent non-quiet correctness job `1003-110707-00-i8-token-parity-1107`
(rc 0, final complete) observes actual runner-emitted token IDs via
`i8_token_receipt.py`, without changing serving code or sampling. All seven
captured prompt token digests match on/off. In both arms, the repeated first
main prompt has 7336 prompt tokens, cold cached=0 versus hit cached=7335,
and the same 63 emitted tokens (SHA-256
`526abd8e8275f19c001e5753d7030987996ce3be962a929d665ebaeb5fe071c3`).
The probe changes request order by adding that repeat, so its timings are not
used in the table above.

The predecessor 0.8B smoke (`1003-102420-00-i8-tiny-smoke-1027`) stalled with
rc -2 while waiting for arm 1's title; only arm 0 completed. It had the captured
32000-token title output limit and no final complete record. It is not a parity
pass and establishes no timing claim for 0.8B. Its servers were cleaned up.


## Next-iteration final-window dispatch (awaiting quiet measurements)

The opt-in work policy carries an interactive request through its remaining
2048-token window and first-token delivery before repaying decode debt. This
continuation is latched after an executed atom enters that window, including a
partially cached suffix that needs more than one atom. Every canonical checkpoint
and token span stays with the generator. A long waiter keeps FIFO protection
after one overtaking atom; the already-selected request can finish its bounded
window before that protection resumes. New arrivals cannot repeat the bypass.
Cancellation removes both the saved prefill and deferred checkpoint captures.

The tiny-model probe's original stall was silent continued generation, rather
than a scheduler with no progress: instrumented output reached 19,949 tokens
before the diagnostic's deadline. On the merged version both 0.8B arms complete
their title streams, but title digests differ. It is a transport/liveness probe,
not lossless evidence for the small model. Both deployment options remain off.

Qualified runners now reserve one 100 ms handoff after a primary completion.
Repeated completions do not extend that grace; actual auxiliary service or an
empty auxiliary lane resets it. Eligibility is checked again after a completion
in the same slice. An aged auxiliary may yield once to the highest-ranked primary
if that primary is in its final window, then regains its service opportunity.
Metadata peeks use the admitted APC namespace, and final-window status is
revalidated after actual lookup/progress so an evicted warm estimate cannot turn
a long cold miss into an unbounded continuation. These handoff changes await
their own quiet replay; both defaults stay off.
