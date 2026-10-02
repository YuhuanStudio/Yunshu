# Serve and develop on one machine

One Mac can run a production Yunshu server for real users and still run development GPU jobs (benchmarks, evals).
The GPU queue (`scripts/dev/gpuq`) is aware of the production server: dev jobs run only while it is idle and are
frozen the instant user traffic arrives.

## How it works

| Piece | Behavior |
|---|---|
| Poller | The queue daemon GETs `/v1/yunshu/status` of each configured server every `poll_s` (default 1 s) and reads only `requests.active` / `requests.queued`. No prompts, no request items. |
| Start gate | A job starts only after every server has been idle (no active, no queued) for `idle_start_s` (default 120 s). |
| Preemption | A request appears: `SIGSTOP` to the job's own process group (the queue starts each job in its own session; nothing else is ever signalled, never the server). |
| Resume | `SIGCONT` after the servers have been idle for `idle_resume_s` (default 30 s). |
| Pause record | Each interval is stored in the job JSON (`pauses: [[t0, t1], ...]`, wall-clock) and in the file named by `$GPUQ_PAUSE_FILE`. |
| Timeouts | `--timeout` and `--stall` count active time only; paused time is excluded. |
| Memory admission | A job declares `--mem-gb` (default 24, 0 with `--serving-ok`). It starts only if `free - mem_gb >= reserve_gb` (default 16, free = `psutil` available or `vm_stat` free + inactive + speculative + purgeable). |
| Bypass | `--serving-ok` skips the idle gate and is never paused (CPU-only jobs). |
| Status | `gpuq status` shows `wait-idle`, `wait-mem`, `paused` and a `pauses=N` count. |

An unreachable server (connection refused) counts as idle; a timeout, 5xx or unparsable answer counts as busy
(fail closed). Without a configured URL nothing changes: no gate, no memory admission, no pauses.

## Enable it

Either environment variables for the daemon (set before the daemon starts) or `$GPUQ_DIR/serving.json`, which is
re-read every loop so it can be edited live:

```json
{"urls": ["http://127.0.0.1:8000"], "idle_start_s": 120, "idle_resume_s": 30,
 "poll_s": 1, "reserve_gb": 16, "api_key": "optional bearer if the server needs one"}
```

Environment overrides: `GPUQ_SERVING_URLS` (comma separated), `GPUQ_IDLE_START_S`, `GPUQ_IDLE_RESUME_S`,
`GPUQ_SERVING_POLL_S`, `GPUQ_RESERVE_GB`, `GPUQ_SERVING_KEY`.

```bash
scripts/dev/gpuq submit --label eval --mem-gb 30 -- python bench.py     # gated, preemptible
scripts/dev/gpuq submit --label lint --serving-ok -- python cpu_job.py  # bypasses the gate
```

## Measurement-integrity rule

A benchmark number that includes paused time measures the user's traffic, not the engine. Harnesses must never
report such a sample.

* `scripts/dev/gpuq_pause.py` (stdlib only): `was_paused(t0, t1)` (wall-clock), `timed(fn)` which discards and
  re-runs a sample that overlapped a pause and raises `PausedSampleError` after repeated overlaps, and
  `pause_intervals()`. It fails closed: an unreadable `$GPUQ_PAUSE_FILE` means "paused"; outside gpuq the variable
  is unset and nothing is ever paused.
* Wired in: `scripts/research/agentic/replay_traffic.py` and `scripts/research/bench_engine_matrix.py` (every timed
  request is wrapped in `timed`), and the release gate's `cancel_then_next` wall-time check (paused seconds are
  subtracted). Other gate checks have no timing thresholds.
* Accuracy evals (`paired_eval`) need nothing: a pause only makes them slower.
* New benchmark scripts must wrap each timed sample in `timed(...)` (or check `was_paused`) before recording it.
* Limit: preemption happens within one poll interval of the first request, so a sample can briefly share the GPU
  with a request before the stop lands. Keep `poll_s` small; the idle gate makes this rare.

## Planned, not implemented

* Online lossless A/B on live traffic, recording metrics only (no prompts).
* Bandit auto-tuning of lossless knobs against those metrics.
* A gated autonomous improvement loop that proposes, measures and lands changes only through the existing gates.
