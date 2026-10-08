# Verifying a change: `yv`

`scripts/dev/yv` runs the standard verification of a candidate change end to end and returns one
verdict. Use it instead of writing a measurement script: it builds pinned trees for both arms,
submits every GPU job to gpuq itself with sane memory / timeout / `--quiet` settings, stops at
the first failed stage, remembers what finished, and writes `verdict.json` and `verdict.md`.

```sh
scripts/dev/yv ab --base <base-sha> --cand <candidate-sha> --suite decode --label wide8-lane-fusion
scripts/dev/yv ab --base <base-sha> --cand <candidate-sha> --cand-env YUNSHU_VLM_DRAFT=mtp --suite prefill --label prefill7-x --detach
scripts/dev/yv status <run>      # state of a run (directory or name under .../verify/runs)
scripts/dev/yv wait <run>        # block until the verdict; exit code = the verdict's
scripts/dev/yv suites            # list suites
scripts/dev/yv gate              # release gate with per-stage persistence
scripts/dev/yv gate --ref COMMIT_SHA --label-prefix worker --root /Volumes/P5Plus/yunshu-build/gate-worker --stages install,serve-27b,families
scripts/dev/yv ab --base <base-sha> --cand <candidate-sha> --suite preflight --label docs-check  # CPU only
```

Exit code: 0 every stage passed, 1 a stage failed, 2 infrastructure error (tool, git, gpuq,
parameter mismatch). `yv ab` blocks until done; `--detach` returns at once.

## Arms

For a reviewable verdict, commit both arms and pass their full commit SHAs. Check the
first line of `yv.log` to confirm the intended base and candidate differ. Directory
arms are supported by the tool but do not provide a final committed verdict.

`--base` / `--cand` take a git ref or a checkout directory. A ref becomes a detached worktree
under `/Volumes/P5Plus/yunshu-build/verify/trees/<commit12>`, reused by commit hash. A directory
is used as it is; uncommitted changes are hashed into the arm key (`<commit>-d<hash>`), so a
resume never mixes two states of one tree. Both arms run the same interpreter (the main checkout
`.venv`) with `PYTHONPATH=<tree>/python`, the same way `tfbench.py` and `paired_eval.py` already
did. `--env K=V` applies to both arms, `--cand-env` to the candidate only, `--base-env` to the base.
`--model` defaults to the 27B checkpoint in `scripts/research/local.env`.

## Stages (in this order, fail-fast)

| Stage | What it does | Pass |
|---|---|---|
| `preflight` | CPU only. Both trees import. `git diff base..cand` is mapped to related unit tests (changed tests, `test_<module>*`, tests that import or name the module) and run under `nice -n 15`. | imports ok, tests pass |
| `smoke` | Both arms serve and answer (`tfbench.py --smoke`). `--engaged SPEC` proves the candidate path ran: `log:REGEX` (candidate server log), `field:KEY=REGEX` (`x_yunshu` response field), `spec:MODE` (engaged spec mode). | answers, spec mode engaged as requested, every `--engaged` matched |
| `identity` | Greedy decode digests base == cand over code and prose at each suite context (cold, warm, follow-up turn); with `--spec-off` also cand with `YUNSHU_VLM_DRAFT=off` == cand (spec on == off). The digest is over text, reasoning, tool calls and finish reason plus the token count (`tfbench.py` `sha`). | every cell equal |
| `apc` | From the candidate's identity cells: the warm request (served from the prefix cache) must equal the cold request, and report `cached_tokens > 0` (a request that never hit the cache fails: APC not engaged). `--no-apc-hit-required` for paths without APC. | hit == miss |
| `quality` | 200-item paired MMLU-Pro through `paired_eval.py` (resumable rounds of 14 min per arm, base and cand interleaved). | all items scored in both arms, net difference in correct answers within +-1 |
| `speed` | `tfbench.py` decode cells (cold decode tok/s, cold TTFT, follow-up TTFT) in N quiet reps (default 3), interleaved base, cand, base, cand. Medians, per-rep paired deltas and a noise estimate (half range of the paired deltas) are reported. | no median worsening beyond `max(2%, noise)` (`--speed-tol`); a contended rep is rerun once, then the stage fails. **Confirmation:** when a cell regresses, only the regressing `(ctx, kind)` cells run `speed_confirm_reps` more paired reps (suite setting, default 2; arms still alternate, cells `base-r3@c1024prose`) and the metric is re-judged on all reps with the median of the per-rep paired deltas and noise = 1.4826 x the larger of the deltas' MAD and each arm's own MAD (percent of its median), so an intermittent ~100 ms GPU stall in one or two reps cannot decide; a real regression repeats in every rep and still fails. Stage numbers and `verdict.md` carry both verdicts (`initial: regression ...; confirmed: neutral`). Resume reuses finished confirmation cells |
| `memory` | `memory_ab.py`, one job per arm and rep (default 2 reps, alternating order): peak footprint, footprint after idle, footprint held after a short follow-up. | no metric above base by more than 3% + 0.25 GiB |
| `longqa` | `tfbench.py --part needle`: 10 deterministic key-value needles spliced into the 32K / 64K / 128K prose prompts, one greedy question each (the prefix is cached after the first). Long suites only. | every item scored in both arms, correct count within +-1 |
| `conc` | `tfbench.py --part conc32`: two requests at once, each a warm 32K prefix + ~2K new text, 1024-token reply; TTFT and per-request decode. Long suites only. | TTFT / decode within 10% of base |

`--spec-modes default,mtp,dflash` repeats identity (and spec on == off) once per speculative
method (`mtp` / `dflash` set `YUNSHU_VLM_DRAFT`; `dflash` needs the drafter path, `D` in
`local.env` or `YV_DRAFTER`). The base arm's identity cells are deterministic, so they are shared
across runs through `/Volumes/P5Plus/yunshu-build/verify/cellcache/` (key: base commit, env, model,
context, harness hash, device): the next candidate against the same base reruns only its own arm.

## Suites

`decode` (preflight, smoke, identity at 1K / 8K, apc, speed; `--spec-off` adds spec on == off), `prefill`
(identity up to 32K, apc, quality, speed), `scheduler`, `memory`, `full` (everything, 1K / 8K / 32K),
`long` (32K / 64K / 128K split cells of 2048-token replies via `tfbench --decode-tokens`: identity incl. spec on == off, apc, speed, memory at 32K / 128K, longqa, conc; every stage runs even after a failure), `longtrend` (one rep, 32K / 128K prose, for tag-to-tag comparisons), `tiny` (everything on a small model, for dry runs of the tool). `--suite smoke,identity,speed`
builds an ad-hoc ladder (stages keep their canonical order). Overrides: `--ctx 1024,8192`,
`--reps`, `--mmlu-n`, `--mem-sizes`, `--mem-reps`, `--speed-tol`, `--spec-off`.

## Resume

Everything lands in `/Volumes/P5Plus/yunshu-build/verify/runs/<label>-<cand key>/`: one
`<stage>.jsonl` per stage (`cell_submitted`, `cell_done`, final `stage_complete`), `cells/` with
each cell's evidence, `state.json`, `verdict.*`. A cell's evidence file is written under its final
name only after the job exited 0 AND the file ends with `complete: true`; a job that ended early
leaves no evidence and the stage fails (missing evidence = fail). Run the same command again to
resume: finished cells are reused (the verdict marks them `*`), a job still in the queue from the
killed invocation is re-attached rather than resubmitted, and a failure costs only the in-flight
cell. Changing base, candidate, env, suite or model under an existing label is refused (exit 2):
use another label.

## Verdict

`verdict.json` (schema 1: label, base, cand, env, suite, model, `overall`, `exit_code`,
`failed_stage`, per-stage `status` / `reasons` / `numbers` / `jobs`, job ids) and `verdict.md`
(Traditional Chinese summary and an English block ready for `docs/reports/PERF_TREND.md`: commits,
job ids, numbers, PASS / FAIL per stage, reasons). Paste the block only for measured, shipped
changes.

## `yv gate`

Runs `scripts/release/gate.sh` one stage per gpuq job (`install`, `serve-27b`, `families`,
`soak-mmlu`, `soak-realistic`, `agent-sessions`; port 18993) plus the `long` stage (`yv ab --suite long`, cand = HEAD, base = last `v*` tag; verdict judged fail-closed) and records each in `runs/gate-<commit>/`. `--ref COMMIT_SHA` builds a clean pinned tree; `--label-prefix worker` attributes every GPU job to that worker. `--root PATH` isolates installed tools and caches from other workers. Marked `.yv-own-venv` trees use their own interpreter, and installed wheel RECORD fingerprints invalidate old evidence when an environment changes. `YV_MAX_PENDING` limits a runner to 1–6 queued cells (default 6); use 2 when running several verdicts together. The `serve-27b` timing stage requests quiet admission. A stage
passes when its check rows have no FAIL or CONTENDED, at least one PASS, and the job exited 0.
A rerun on the same commit skips passed stages (`--fresh` reruns all); `install` reruns if
`$GATE_ROOT` holds another commit's install.

The `serve-27b.server_path` check compares the server's decode tok/s with the engine in-process.
The old rule (every case >= 0.95, median of 3) failed about one gate in five on noise. Across 24
gate runs (96 case ratios) the mean ratio is 0.997 and the per-case sd 0.031, split evenly above
and below 1.0 (the in-process side is the noisy one), so there is no server gap. The check now
alternates which side goes first, takes the median of 5, and fails on a geometric mean below
0.965 or a single case below 0.90; a systematic 5% serving overhead still fails.

## `m3sweep` (correctness on the M3 lane)

```bash
scripts/dev/m3sweep [REF] [--only wire,agent,units,routes] [--dry-run] [--no-wait]
scripts/dev/m3sweep --collect DIR      # judge a --no-wait run later
```

Submits `m3lane-sweep-*` jobs to the gpuq M3 lane for HEAD (or REF) and writes one
`/Volumes/P5Plus/yunshu-build/m3sweep/<sha>-<time>/verdict.json` (PASS/FAIL, exit code). Jobs:
`wire` runs the SDK wire-contract matrix (OpenAI chat / completions / responses, Anthropic
messages, Ollama; stream and non-stream; tools, tool_choice, parallel off, JSON schema, stop,
truncation, usage invariants, error shapes) against a real server on Qwen2.5-3B (text path),
Qwen3.5-0.8B and Qwen3.5-9B-4bit (VLM runner); `agent` runs the covaudit tool session and
concurrent-vs-solo identity on Qwen3.5-2B and Qwen2.5-3B; `routes` runs every route the gateway registers against real servers (below); `units` runs the unit files that skip
without small checkpoints (tests find them through `tests/unit/model_paths.py`, which honours
`M3_MODELS`). Fail closed: a missing or incomplete output, a failed job, a wrong device or any
mismatch is FAIL. Only the four allowlisted small checkpoints (Qwen3.5 0.8B / 2B / 9B-4bit, Qwen2.5-3B-4bit) ever reach the laptop. No failure is excused: a forced tool_choice is always grammar-constrained and usage excludes server prefill. The M3 is portability / correctness evidence only: no timing, no tok/s.

`omni-gemma` (needs `gemma-4-e2b-it-4bit` on the allowlist) makes a spoken fixture with Qwen3-TTS, then runs the `omni` checks (audio / image / video in) on Gemma 4 E2B and the `cascade` check (Realtime voice with Qwen3-ASR + Qwen3-TTS). The `native` checks (Qwen3-Omni, 20 GB: `/v1/omni/speech/stream`, native Realtime speech) run as an M5 job: `m3sweep_jobs.py omni --mode native` through `gpuq --device m5`; the M3 route gate leaves them out.

### `routes`: every route has a real check

`scripts/research/route_checks.py` holds one `@check(name, "METHOD /path", ...)` per group of routes (97 routes, websockets included).
The job starts, one after the other, a default-configuration server on Qwen3.5-0.8B (VLM runner) and on Qwen2.5-3B-Instruct-4bit (mlx-lm path)
and runs the checks that need no token, then a token-protected `--models-dir` server (0.8B + 3B, a fake SearXNG / MCP / page backend on
loopback) for model load / unload, auth, web search, web fetch and the MCP connector. Ports 18994-18996 only. Checks use the official
`openai` / `anthropic` SDKs and their typed models wherever they cover the route (files, batches, conversations, the Responses lifecycle,
Messages batches, `count_tokens`, `input_tokens`, `client.responses.connect()`), raw HTTP / `websockets` elsewhere, and cover the full
lifecycles (file -> batch -> poll -> results -> cancel; conversation create -> items -> delete; response create -> retrieve ->
input_items -> cancel -> delete; token counts equal to the real call's usage; websocket happy path and a disconnect that must stop the
generation). Modalities the four checkpoints cannot serve are checked for the absent-capability answer (error shape, status, a message that
names what is served). A 500 fails the check and carries the server's error log lines.

Two gates keep it honest. `tests/unit/test_route_coverage.py` (CPU, in CI) fails when the app registers a route no check or `EXEMPT`
entry names, or a check names a route the app no longer has. The `m3sweep` verdict fails when a registered route was not verified by a
passing check in the same run. Adding a route: add the route's check to `route_checks.py` (or a reasoned `EXEMPT`), run
`scripts/dev/m3sweep --only routes`. [API_SURFACE.md](API_SURFACE.md) says per row what was verified where.

## `agentbench`: real coding agents on the 27B

```bash
scripts/dev/agentbench                       # Claude Code, Codex, opencode x 20 tasks on main, 60 jobs at priority -1
scripts/dev/agentbench --agents claude --tasks polyglot-bowling,cli-add-flag --repeat 2
scripts/dev/agentbench --no-wait ; scripts/dev/agentbench --collect DIR
```

One gpuq job per (agent, task, repeat) (queue timeout 28 min: one agent run is capped at 20 minutes), a fresh server with Yunshu's default
configuration on a pinned tree of `--ref` (default `main`), the pinned agent CLIs in the sandbox of `scripts/research/agentic`, ports
18994-18996. The verdict (`/Volumes/P5Plus/yunshu-build/agentbench/runs/<sha>-<time>/verdict.json`) is fail closed: a missing run, any API
error, malformed tool call or tool-call markup leak is FAIL; the pass rate per agent is compared with the 2026-09-30 baseline (Wilson 95%
interval) and a drop below its lower bound is REGRESSION. Reported per agent: pass rate, API errors, malformed tool calls, markup leaks,
cache-hit ratio, largest prompt, median wall time, peak memory. Where it runs: **nightly at priority -1** (idle GPU time only; one full
matrix is about 6 to 8 hours of 27B time), and **during release verification** through `scripts/dev/release_check`: it submits the same committed candidate
at priority -3 without waiting, prints its later collect command, and treats its verdict as informational.
The gate, M3 sweep and agent compatibility remain blocking. Never leave its result uncollected.

## Adding a new kind of measurement

Ad-hoc scripts are for measurements `yv` does not cover. Put them in `scripts/research/` with a CPU
unit test, make them write a final `complete: true` record, and then add them as a stage or a
cell in `scripts/verify/stages.py` so the next worker does not need the script.

### Web tools

`yv ab --base BASE_SHA --cand CAND_SHA --suite preflight,websearch --label websearch-smoke --model /Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16 --priority -1` exercises the existing `route_checks_tools` search contract against a loopback SearXNG/page fixture, then candidate Responses `open_page`/`find_in_page` and resident Qwen3-Embedding-0.6B fusion, plus ten frozen adversarial fixture replays. The fixture pilot cannot approve the 200-pair quality gate. This optional stage is excluded from the core full/decode ladder. Use the main checkout gpuq (`YV_GPUQ`) and `GPUQ_OWNER=websearch`. CPU probe/validator tests run before queue submission; every result ends with `complete: true`. It is correctness smoke, never a performance or answer-quality verdict.

The frozen eval entry point is `scripts/research/websearch_eval.py`; the dated 130-query seed set is `scripts/research/data/websearch_queries.jsonl`. Capture writes pending gold, never invented answers. Curate references before replay; `--dry-run` makes no answer-quality claim. Raw page snapshots remain private.

### Reranker / classifier oracle

`--suite rerank` runs CPU preflight and the dedicated capability stage: candidate
HTTP responses on Qwen3-Reranker-0.6B, BGE-reranker-base, MiniLM-L-6-v2 and a tiny
SST2 classifier against independent Transformers float32 CPU inference on the
same M5 host. Five text pairs include a long document that exceeds encoder windows;
three classifier inputs exercise the head probabilities. All scores must be within
0.003 and preserve ranking. The registered embed-route checks, scalar/list
broadcasting and raw/activated score consistency run on those same real engines.
The stage records the detected model type and selected engine class. This is a
capability oracle, rather than a comparison with the old bi-encoder implementation.

Use full commit SHAs and the main checkout's gpuq explicitly:

```sh
PATH=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin:$PATH \
YV_GPUQ=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq \
GPUQ_OWNER=rerank GPUQ_DIR=/Volumes/P5Plus/yunshu-gpuq \
scripts/dev/yv ab --base BASE_SHA --cand CAND_SHA --suite rerank \
  --label rerank-TOPIC --priority -1 --detach
```

The scoring models live in `/Volumes/P5Plus/models` by default; `--env
RERANK_MODEL_ROOT=PATH` overrides the model root. Encoder head serving needs the
`embeddings` extra; Torch is used only by the CPU reference. `yv` atomically
promotes a copy of successful evidence and keeps the original gpuq-declared output
so digest and watchdog checks can still verify it. A finished job without an
explicit return code of zero fails verification.

### Evals API correctness

`yv ab --base <main-sha> --cand <candidate-sha> --suite evals --model /Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16 --label evals-smoke-<unique> --priority -1 --detach` runs CPU preflight and the 12-route SDK check against a real server. The `evals` stage exercises normal chat sampling plus local score/label graders; it makes no speed or accuracy comparison. Its JSONL evidence must end with `complete: true`, `passed: true`, 12 routes and three model invocations.
