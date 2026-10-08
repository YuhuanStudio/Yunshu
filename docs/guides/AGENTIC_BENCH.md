# Agentic coding benchmark

Idealised single-request numbers do not tell you how an engine behaves under what people run: multi-turn
coding-agent loops with a growing context, many tool calls, prefix reuse between turns and long sessions.
`scripts/research/agentic/` drives real coding agents against a local server and lets hidden tests decide
whether the code they wrote works.

## What it measures

Per (agent, engine, task, repeat) run:

- pass / fail from tests the agent never sees, wall time, agent exit status, timeouts;
- number of model requests; total prompt, completion and cached tokens; cache-hit ratio; largest prompt;
- per request TTFT and decode tok/s (measured at the wire by a recording proxy, so every engine is
  measured the same way);
- API errors (4xx/5xx, bodies saved), malformed tool-call arguments, tool-call markup leaking into text;
- the server's peak physical footprint during the run.

Agents (pinned copies under `/Volumes/P5Plus/yunshu-test-envs/agentic-clis`, installed by
`install_agents.sh`, never the ones on your PATH): **opencode** (Chat Completions), **Claude Code**
(Messages), **Codex CLI** (Responses). Each run gets a fresh HOME, XDG dirs, TMPDIR, CLAUDE_CONFIG_DIR and
CODEX_HOME, a scrubbed environment and a dummy key, and runs under `sandbox-exec` that blocks all network but
loopback, hides the real `~/.claude`, `~/.codex`, `~/.config`, keychain and shell rc files, and only allows
writes inside the run directory. Sampling is whatever each agent sends by default.

Tasks (20): 12 Exercism Python exercises from the Aider polyglot benchmark (fetched to
`/Volumes/P5Plus/datasets/agentic`, commit recorded in the meta row) and 8 custom multi-file tasks (bug fix,
CLI flag plus tests, three-file refactor, stdlib HTTP JSON API, parser from a spec, long-context locate and
change in a ~2,100-line package, shell round trips over messy data, failing-suite triage).

## Run it

```bash
# one server per job; --serve starts it (Yunshu defaults or TensorFold) and kill -9s it at the end
python scripts/research/agentic/run_agentic.py run --serve yunshu --checkpoint $M \
    --engine-label yunshu-default --agent opencode --tasks all --repeat 3 \
    --output docs/research/runs/DATE-agentic/yunshu-default-opencode.jsonl
# against a server you started yourself
python scripts/research/agentic/run_agentic.py run --engine-url http://127.0.0.1:18990 --engine-label X ...

python scripts/research/agentic/run_agentic.py summarize docs/research/runs/DATE-agentic/*.jsonl
python scripts/research/agentic/run_agentic.py check-tasks   # reference solutions pass, starting repos fail
```

GPU runs go through `scripts/dev/gpuq`. Runs are resumable (a task x repeat already in `--output` is skipped),
`--shard-i/--shard-n` splits the run list, and `--budget-min` stops starting new runs so a job ends inside its
queue timeout. Output files begin with a `meta` row (engine, checkpoint, git SHA, agent version, flags).

## Repeatable run: `scripts/dev/agentbench`

The commands above run one agent against one server. `scripts/dev/agentbench` is the whole matrix as one command (Claude Code, Codex and
opencode on all 20 tasks against the default 27B server on `main`): one low-priority gpuq job per cell, a verdict that fails closed on API
errors, malformed tool calls, markup leaks and missing runs, and the comparison with the 2026-09-30 run. See
[VERIFY.md](VERIFY.md#agentbench-real-coding-agents-on-the-27b). Nightly standalone runs default to priority -1. `scripts/dev/release_check` submits the release candidate
at priority -3 and prints a later collect command; agentbench is informational, while the gate, M3
sweep and agent compatibility block release. Collect and record every submitted result.

## Add a task

Create `scripts/research/agentic/tasks/<id>/` with `task.json` (`id`, `title`, `prompt`, `test` argv, optional
`protected` files restored before grading, optional `generate` script that writes extra files into the working
directory), `repo/` (the starting files), `hidden/` (test files copied to `_hidden_tests/` only at grading time;
`AGENTIC_TASK_DIR` and `AGENTIC_TASK_SRC` are set) and `reference/` (an overlay, or `apply.py`, that solves it).
`check-tasks` must report the starting repo failing and the reference passing. Keep tasks deterministic and
offline. Polyglot exercises are listed in `tasks.py` and materialized from the dataset, never committed.

Raw prompts, recordings and JSONL stay private (`docs/research/`). Publish only dated,
aggregate results with snapshot SHAs, denominators, timeouts and uncertainty, as in
[BENCHMARKS](../BENCHMARKS.md) and [PERF_TREND](../reports/PERF_TREND.md).
The 2026-10-02 matrix is partial; Codex and TensorFold cells have no published result.
