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
| Bypass | `--serving-ok` skips the idle gate and serving pauses (CPU-only jobs). Priority preemption still applies. |
| Priority preemption (Q01) | Pending p>=0 work can `SIGSTOP` a running p<=-1 job when memory admits both resident jobs; no job is killed to reclaim memory. High-priority jobs run serially, then the low-priority job resumes when no uncancelled p>=0 job remains pending and the serving resume gate permits it. |
| Status | `gpuq status` shows `wait-idle`, `wait-mem`, `wait-cpu`, `paused` and a `pauses=N` count. When pending jobs have no active runner, an `idle:` line explains memory admission, serving, paused jobs, or a stopped daemon. |

An unreachable server (connection refused) counts as idle; a timeout, 5xx or unparsable answer counts as busy
(fail closed). Without configured URLs or an explicit reserve, there is no serving gate, serial memory
admission, or serving pause. Priority preemption still checks memory: the
measured available memory (which already excludes the resident low-priority job) minus the new job's `mem_gb`
must leave `reserve_gb` (default 16 GB). Unknown memory prevents priority preemption; the low-priority job
continues. A blocked pending high-priority job is shown as `wait-mem`. When memory does admit both jobs,
the low-priority job stays paused even if the high-priority job is still waiting for the serving start window.

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

## Submit labels and bounded waits

An active label (pending, running, or paused) is exclusive. Concurrent submissions with the same active label
are rejected. A label can be reused after its earlier job finishes; every submission gets a distinct job id.
Submission never deletes earlier logs, pause records, return codes, or output files.

Declare expected results with repeatable queue-level `--out` arguments **before** the command separator `--`.
`--expect-complete` requires every output to be a nonempty file containing a line with the literal `complete`.
Command-level `--out` / `--output` paths are also checked for older jobs. The command remains responsible for
writing its own outputs; declaring an output does not pass an argument to the command.

```bash
scripts/dev/gpuq submit --label eval-unique --out results/a.jsonl --out results/b.jsonl \
  --expect-complete -- python sweep.py
scripts/dev/gpuq wait --max-seconds 45 JOB_ID OTHER_JOB_ID
scripts/dev/gpuq digest --label-prefix eval- --since 6h
```

`wait` uses one deadline for the entire id list and prints one summary line per job: state, rc, output problems
(`missing_outputs`, including empty/incomplete outputs), and log path. Exit 0 means every job is `done` with rc 0
and verified outputs; exit 1 means a job failed, was cancelled, is missing, or has invalid outputs; exit 2 means
some jobs remain pending/running at the deadline. Exit 2 takes precedence over failures in finished peers so
callers know they must keep waiting. With no `--max-seconds`, wait blocks until all known jobs finish.

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

## Implemented offline measurement support (unreleased main)

`YUNSHU_SERVE_LOG=1` writes a local, size-capped numbers-only event log: timings,
token counts, cache tier, speculative acceptance, concurrency, arm and build. It
never records prompts, outputs or token ids. `YUNSHU_SERVE_LOG_DIR`,
`YUNSHU_SERVE_LOG_MAX_MB`, `YUNSHU_SERVE_LOG_KEEP` control location and rotation;
`YUNSHU_ARM` labels a manually chosen configuration without changing behavior.
The server reports cumulative GPU busy seconds in `/v1/yunshu/status` and
`yunshu_gpu_busy_seconds_total` in `/metrics`. These are busy-time counters, not
hardware utilization samples. Offline switchback, CUPED and mSPRT helpers and
run-history variance profiling are implemented; no online arm selection is enabled.

## Planned, not implemented

* Online lossless A/B on live traffic, recording metrics only (no prompts).
* Bandit auto-tuning of lossless knobs against those metrics.
* A gated autonomous improvement loop that proposes, measures and lands changes only through the existing gates.

## CPU contamination (Q02)

Every started job records `cpu_samples` at start, every 30 seconds and end: `load_1m`, `load_5m`,
`foreign_cpu_pct` and `top_cpu` (PID, name, `cpu_pct`, eight largest). The sum includes **all** non-job
processes, excluding the job's descendant tree and process group. One busy core is 100%. After the initial
`ps` estimate, CPU percentages use CPU-time deltas over the sampling interval. The monitor polls every two
seconds, stores `foreign_cpu_max_pct` / `foreign_cpu_mean_pct` across those polls, and latches `contended`
when the threshold is reached or sampling fails. An observation gap over five seconds resets the quiet
window; time waiting behind other GPU jobs does not count toward the CPU admission timeout. It writes a single CPU summary to the log at completion;
monitoring never adds log heartbeats that could mask a stalled command.

Jobs with priority >= 0, and jobs explicitly submitted with `--quiet`, must have foreign CPU below 150% for
20 continuous seconds before starting. After a 300-second maximum CPU wait they may start with
`quiet_timeout` / `contended` set. Serving and memory admission still apply; `--serving-ok` does not bypass
the CPU gate. Configure per submission (values are saved with the job and exported to its harness):

```bash
scripts/dev/gpuq submit --quiet --priority -1 --cpu-threshold 150 --quiet-window 20 \
  --quiet-max-wait 300 --label cpu-quiet-unique -- python bench.py
```

Environment defaults: `GPUQ_CPU_THRESHOLD`, `GPUQ_QUIET_WINDOW_S`, `GPUQ_QUIET_MAX_WAIT_S` and
`GPUQ_CPU_SAMPLE_S` (stored periodic sample interval, default 30). These are queue settings, not engine
experiment flags. `gpuq status` shows `wait-cpu`; finished `done` jobs with contamination display as
`contended`. `wait` returns **3** for otherwise successful contended perf/quiet jobs; unfinished peers still
return 2, real command/output failures return 1. Contended negative-priority audits are shown but remain
successful. Digest flags contended perf/quiet jobs as untrustworthy, with a distinct family state.

`GPUQ_CONTENTION_FILE` names the atomic JSON flag file. Harnesses can use
`scripts/dev/gpuq_contention.py:was_contended()` for the latched job flag, or
`was_contended(t0, t1)` for one wall-clock attempt. Missing/bad files or flags with a last sample older than ten seconds fail closed when the variable is set;
outside gpuq it returns false. `tfbench` includes `contended` on every result row. The release gate includes
it on each check row and retries a contended server-path comparison once after a bounded quiet wait. Both
attempts are retained in `27b-server-path.json`; a quiet failure is FAIL, two contended timing attempts are
CONTENDED (gate exit 3), and text/spec mismatches remain FAIL. A quiet successful retry can validate that
check, while gpuq's job-wide flag still asks callers to remeasure other perf results from the job.

An adopted pre-Q02 job has no historical CPU evidence: it is flagged `unmonitored_before_adoption` and
receives new samples from adoption onwards. Already-running processes retain their original environment;
only newly launched jobs receive `GPUQ_CONTENTION_FILE`. Point sampling cannot observe every short-lived
process between polls; no tok/s or TTFT improvement is implied by these queue changes.

## Reading finished GPU jobs: `gpuq digest`

A job that finished is not a result anyone has read. `scripts/dev/gpuq digest` (`scripts/dev/gpuq_digest.py`, stdlib
only, python 3.9 safe) lists the jobs finished since the last digest, grouped by label family (`paired-gsm8k-ref-r3`
-> `paired-gsm8k-ref`): states, total time, the last job's log and every output path named by `--out` / `--output`
in the command or declared at submission. It flags first, and exits 1 on, any job that ended `failed`, `lost`,
`stalled`, `timeout` or `cancelled`, has an absent/nonzero rc, or has missing/empty outputs. When
`--expect-complete` was submitted, an absent `complete` line also fails verification. These are the same output
checks used by `wait`, so a clean process exit that wrote nothing is not a result.

```bash
scripts/dev/gpuq digest              # since the last digest (first run: last 24 h); moves the marker
scripts/dev/gpuq digest --peek       # same window, marker untouched
scripts/dev/gpuq digest --since 6h   # 90m / 2d / ISO time / epoch; never moves the marker
scripts/dev/gpuq digest --label-prefix eval- --since 6h # restrict to one worker/sweep
```

The marker is `$GPUQ_DIR/.digest_marker`, the time the previous digest started (taken before it read the jobs, so a
job ending mid-digest appears next time). Rule: the coordinator runs `gpuq digest` at every check-in and acts on
every flagged line (fix, requeue or explain) before starting new work. A label-filtered digest never moves the
global marker, even without `--since`, so inspecting one worker cannot hide another worker's results.

## Reloading daemon code between jobs

The daemon keeps the Python code loaded at startup. After the lead merges a queue change, hold new submissions
and restart the daemon. A job boundary is convenient; running/paused jobs are adopted on restart.
Pre-Q02 running jobs have no earlier CPU samples and are flagged for remeasurement. Use the shared queue directory, terminate only
the PID holding `daemon.lock`, and start the replacement with `/usr/bin/python3` (Python 3.9). Keep all job,
log, lock, and output files in place. These commands are for the lead to execute; workers must not restart the
live daemon during measurement.

```bash
cd /Users/yuhuan/Documents/YuhuanStudio/Yunshu
export GPUQ_DIR=/Volumes/P5Plus/yunshu-gpuq
scripts/dev/gpuq status                    # inspect running/paused jobs (adopted after restart)
GPUQ_DAEMON_PID="$(lsof -t "$GPUQ_DIR/daemon.lock")"
ps -p "$GPUQ_DAEMON_PID" -o pid=,command=   # must be gpuq.py _daemon
kill -TERM "$GPUQ_DAEMON_PID"              # PID only, never a process group
while kill -0 "$GPUQ_DAEMON_PID" 2>/dev/null; do sleep 1; done
nohup /usr/bin/python3 scripts/dev/gpuq.py _daemon \
  >> "$GPUQ_DIR/daemon.log" 2>&1 < /dev/null &
scripts/dev/gpuq status
```
