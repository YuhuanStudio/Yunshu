# Full comparison: runbook

> The full comparison is the fair, pinned, cross-engine benchmark behind every public performance table. It runs
> the same cells on every engine, from one frozen harness commit, with every version pinned and every number
> checked by an audit. This page is the procedure: what it measures, what to check before a run, what the driver
> does on its own, and how to audit, re-run and publish. Companion to [VERIFY.md](VERIFY.md) (the per-change
> `yv` verdicts); a full comparison is the cross-engine view, `yv` is the change-versus-main view.

Entry point: `scripts/research/fullcmp_run.py`. Modules: `scripts/research/fullcmp_*.py`. Report and wandb:
`scripts/dev/fullcmp_report.py`, `scripts/dev/fullcmp_wandb.py`. Private data (census, manifests, raw results,
audit output) lives on the external disks and under `docs/research/`; none of it is committed.

## 1. What it measures

**Batches.** A run is one batch. Batches run in this order: `B1`, `B2`, `B4`, `B5`. `SMOKE` is the pre-flight matrix
and audits only its own rows.

| Batch | Scope |
|---|---|
| `B1` | Core cross, Qwen3.8-27B 4-bit, concurrency 1: context 1K / 8K / 32K / 64K / 128K x kind code / prose x 512 output. Cold TTFT, cold decode, warm TTFT, immediate follow-up (turn 2) TTFT and decode, memory. |
| `B2` | Sweeps: concurrency 2 / 4 / 8 at 8K and 32K code; contexts 4K and 16K; output 128 and 2048; tool-chat, CJK, JSON-schema at 8K; thinking on; generation-config cases; idle 5 s; restart with the disk cache; a mixed load (one 128K request among 7 x 1K); image inputs. |
| `B4` | Behaviour checks, once per engine, one rep (7 checks, about 3 min per engine). Each check is `pass` or `unsupported`. |
| `B5` | Agents: Claude Code only, on Yunshu and one placeholder competitor, every agentbench task x 3 reps. |

**Cell families.** A cell is one `tfbench` invocation with its own server start, so every cell is a cold process.
Families: `cold` (TTFT, decode), `warm` (prefix-cache hit), `turn2` (follow-up), `conc` / `conc-same` (concurrent
requests), `mixed` (long plus short), `restart` (disk cache across a restart), `idle`, `family` (tool-chat, CJK,
JSON-schema, thinking, genconfig), `behaviour` (B4) and `agent` (B5). The cell ids follow
`<batch>.<family>.<engine>.<ctx>.c<conc>.<kind>.o<out>...`; `fullcmp_report.py` parses them.

**Reps.** Every cell runs two reps (`r0`, `r1`). The driver then compares them per metric, and queues an automatic
third rep (`r2`, rotated slot) for any cell whose reps disagree beyond the threshold. Thresholds come from the A/A
control runs (`rep_thresholds.json`): tok/s at least 3 %, TTFT at least 5 %, memory 5 %, each set to 3 x the largest
A/A gap when that is higher. If the cell still disagrees after rep 3 beyond 2 x the threshold, it is reported
`unstable` with every value of every rep, not as one number. Decision of record: 2026-11-03, "a cell whose two reps
disagree beyond the noise threshold gets an automatic 3rd rep" (`docs/research/DECISIONS.md`).

**Engines and version pins.** Each engine is a pinned artifact recorded in the census (`census.json`): version,
release date, binary sha256 (or pinned source tree for Yunshu). The run fails closed when the census is missing or lacks an engine's pin. Yunshu runs from a pinned tree (`yunshu_pin.json`: git sha and the tree's own `.venv/bin/python`); the server
starts as `<pinned python> -m yunshu_cli`, and the provenance check verifies the version and process command line
after launch. Other engines start from their recorded binary; the driver checks the sha before use.

Self-updating engines are the main hazard. Kapitsa replaced its own binary during the 2026-11-04 run, so one table
row mixed four binaries. The rules now are:

- Run a read-only pinned copy of the binary, never the installed one.
- Set the engine's no-update switches (`KAPITSA_NO_AUTO_UPDATE=1`, `CI=true`; Tensorfold: `--no-update-check`).
- Check the binary sha at launch and fail closed on a mismatch.
- An engine whose binary changed during the run is not comparable and goes to the appendix (section 4).

**Fail closed.** A number counts only when its evidence is complete. Concretely:

- A missing, truncated or unparsable record is an error, not a zero.
- Engaged speculative mode is taken from the engine's own log and response evidence (`fullcmp_engaged_evidence.json`),
  never from the mode the request asked for. A mode with no evidence is `unverified`, and an unverified cell is not
  published as engaged.
- A job that exits 0 without a complete record is a failure.
- An audit violation fails the batch (exit 1). A refused job (foreign GPU job, failed preflight, unmapped row) exits 2.

## 2. Before a run

### 2.1 Census and manifest

1. **Census refresh.** `census.json` in the run directory (`/Volumes/P5Plus/yunshu-build/competitors/compsync/fullcmp-1103/`)
   holds every engine's version, release date and pin. Refresh it when an engine version changes, and re-check that
   each pin matches the binary on disk. A stale census fails closed: a run for an engine whose version the census
   lacks is refused, and the tests that read the census fail. Apply any engine-specific census overlay (for example a
   `census.<engine>.json` built for a re-pinned binary) before the run, not during it.
2. **Manifest.** Build the expected-cell manifest with `scripts/research/fullcmp_manifest.py` (`--smoke` for the
   pre-flight matrix). Each row has a status: `unverified` (runs), `unsupported` (reason; zero cost, for example
   no JSON-schema route) or `cut` (removed to fit the budget; kept with its estimate). Nothing is dropped silently.
3. **Engaged evidence.** If an engine or mode changed, rebuild the evidence table with `fullcmp_engaged_merge.py`
   and check the new entries have a real example and a negative example.

### 2.2 Preflight (before anything touches the GPU)

1. **Pre-submit tests.** Run the fast harness set on the M3 so the M5 stays quiet:

   ```sh
   scripts/dev/m3run -- python -m pytest -m fullcmp_quick -q tests/unit
   ```

   `fullcmp_quick` is the fast harness set (under two minutes). The `fullcmp_e2e` chains take 11 to 30 minutes each
   and are not part of the pre-run gate; run them only when the driver changes. `m3run` is for CPU correctness only,
   never for timing.
2. **Swap and quiet checks.** `scripts/research/fullcmp_preflight.py` samples swap for a window and fails when swap
   grows. The driver runs it as a CPU job through gpuq before the first GPU job; a failure stops the batch.
3. **Dry run.** Always run the batch with `--dry-run` first. It prints every job and touches no GPU and no queue:

   ```sh
   GPUQ_DIR=/Volumes/P5Plus/yunshu-gpuq .venv/bin/python scripts/research/fullcmp_run.py \
     --batch B1 --manifest $M/manifest.jsonl --raw-root $RAW --work-dir $WORK --dry-run > $M/dryrun_B1.txt
   ```

   Check, line by line, before launch:

   | Check | Where | Must be |
   |---|---|---|
   | Job count per engine and group | the job list | matches the manifest rows that are `unverified` |
   | Over-margin cells | lines marked `[over-margin: above the 70% limit, run alone]` | each one is a single cell that really needs its own job; no job is above the 20 min cap (such a job is refused) |
   | Job timeouts | the timeout column | est x 1.3, at most 20 min, never below the longest same-kind run |
   | Prompt corpus | `job body ... TFB_WORK=` | one corpus path for the whole batch (`snapshot014-b2`) |
   | Observe state | `job body ... TFB_OBSERVE=` | `0` for every timed cell; `1` only on profile jobs |
   | Rotation | the engine order per rep | rep r starts at engine (r mod n); no two engines back to back across a rep boundary when n >= 3 |
   | Unsupported and cut rows | the skip list | every skipped row has a reason |

   Also run `--self-check`, which runs gpuq's preflight on every job's submit argv and body without touching the queue.
   The 2026-11-04 rule: the lead reads the dry run line by line before B1; no launch-then-fix.

## 3. Launch

1. **Quiet machine.** Before the launch, nothing else may run CPU or GPU work on the M5: no other gpuq job, no builds,
   no editor indexers, no test suite. Tests and analysis go to the M3 (`scripts/dev/m3run`) or wait for the run to end. Tests that need the external disks run on the M5 before the
   quiet period starts, never during it.
   The quiet gate (`fullcmp_quiet.py`) is the enforcement: it refuses a job when a foreign process holds a core
   (single holder at or above 20 %, or total foreign CPU at or above 150 %). Spotlight (`mds_stores`) is not exempt.
2. **Harness commit.** Commit the harness first. The driver snapshots HEAD (`harness-<commit>/`) and runs every job of
   the invocation from that snapshot, never from the live worktree. A harness edit made during a batch can break its
   jobs (2026-11-04: a `NameError` killed a job), so do not edit the harness while a batch runs; use `--snapshot-dir`
   to reuse a frozen snapshot for a follow-on profile batch.
3. **Launch detached.** The driver must outlive the agent or terminal that started it. Start it in a new session,
   not as a child of the Claude Code process, with sleep prevention:

   ```sh
   cd <worktree> && GPUQ_DIR=/Volumes/P5Plus/yunshu-gpuq \
     caffeinate -i .venv/bin/python scripts/research/fullcmp_run.py --batch B1 --manifest $M/manifest.jsonl \
       --raw-root $RAW --work-dir $WORK --log-usage-sample > $M/run_B1.log 2>&1 &
   ```

   Start that line through a new session (`setsid`, or Python `subprocess.Popen(..., start_new_session=True)`) so a
   killed agent does not take the driver with it. Record the driver pid.
4. **Completion watcher.** Arm one watcher at launch, in its own detached session: it waits for the driver pid to exit,
   then writes `<date> B1 rc=<code>` to a status file and runs the audit command of section 5.1 so a finished batch
   is never left unread. Keep the watcher quiet: it writes only the status line.
5. **Live view.** Stream the run to Weights & Biases, unattended and at low priority:

   ```sh
   taskpolicy -b nice -n 19 /Volumes/P5Plus/yunshu-build/venv-wandb/bin/python scripts/dev/fullcmp_wandb.py \
     --raw $RAW --manifest $M/manifest.jsonl --project yunshu-fullcmp --group <run> --until-pid <driver pid>
   ```

   It uploads aggregated numbers and labels only (no prompts, outputs or logs). Needs `wandb login` once.

## 4. During the run

The driver does the following on its own. You do not need to intervene for these:

- **Quiet refusals.** Each job runs the quiet gate first, with 3 retries 30 s apart (log under `quiet/<label>.jsonl`).
  Exhausted retries exit 3 and name the holders.
- **Quiet requeue.** A cell whose quiet snapshot flagged a violation is rerun once in its own slot (`-qretry1`). The
  flagged file is renamed, not overwritten. Each retry must have its own HOME: check the `-qretry` jobs' HOME
  before a batch (the shared-HOME fix is tracked in fix queue item 13).
- **End-of-batch reruns and rep 3.** After the first pass, the driver audits the raw root, lists cells that need rep 3
  (`fullcmp_reps.py`), queues them as a second submission (`-rep3`, rotated slot 2) and waits for them.
- **Audit.** The driver runs `fullcmp_audit.py` over the raw root at the end of the pass. A violation makes the batch
  exit 1.

Where things are:

| What | Path |
|---|---|
| Driver log | `$M/run_<batch>.log` (the `run_*.log` files) |
| Raw cells | `<RAW_ROOT>/<engine>/<cell>-r<rep>.jsonl` (one session header, then one row per request) |
| Server logs and homes | `<WORK>/tfout/` (per-cell server homes) |
| Quiet gate logs | `<RUN_DIR>/quiet/<label>.jsonl` |
| Preflight output | `<RUN_DIR>/preflight/` |
| Audit output | `<RUN_DIR>/audit/` |
| Job state | `$GPUQ_DIR/jobs/` (`gpuq status`, `gpuq log <id>`) |

Wait on the jobs with a single background `gpuq wait --max-seconds 3000 <ids>` (the driver does this itself). Do not
poll in a loop. Read every job's log when it finishes, not only its exit code: a job whose log ends early is a failure.

## 5. After the run

### 5.1 Audit

The audit reads files only (`fullcmp_audit.py --root <raw> --manifest <manifest.json>`) and runs on the CPU. Run it
through gpuq as a CPU job so it never shares the machine with a timing job:

```sh
scripts/dev/gpuq submit --label fullcmp-audit-B2 --timeout 20 -- \
  nice -n 15 .venv/bin/python scripts/research/fullcmp_audit.py --root $RAW --manifest $M/audit_manifest.json
scripts/dev/gpuq wait <id>
scripts/dev/gpuq log <id>
```

It runs 22 guards, in this order. Exit 0 means every guard passed; exit 1 means at least one violation, printed as
`<guard>: <cell>: <reason>`.

| Guard | Checks |
|---|---|
| `usage_fields` (0) | every request row has a real completion count |
| `prompt_identity` (1) | prompt token count and sha are identical across the engines of a comparison group |
| `provenance_identity` (2) | a group mixes only identical provenance (output length and generation config included) |
| `pinned_versions` (3) | version, release date, binary sha256 or pinned tree, and this run's id |
| `manifest_completeness` (4) | every planned cell x rep exists; no unplanned file; no duplicates |
| `record_count` (5) | expected record count; every phase present; complete streams |
| `engaged_mode` (6) | engaged speculative mode derived from evidence |
| `observe_state` (7b) | a timed Yunshu rep ran with observe off |
| `weight_identity` (7) | weight sha256, bits and group size equal within a weight group |
| `cold_cache` (8) | cold cells are a fresh process with `cached_tokens == 0` |
| `gpu_exclusive_rotation` (9) | each job ran alone on the GPU; engines rotate per rep |
| `output_sha` (10) | an output sha per request; prompt identity per (model, ctx, kind) |
| `measured_only` (11) | every number comes from a complete raw row; errors and truncations fail |
| `fairness_controls` (12) | greedy sampling, isolated HOME per engine, A/A and injected-delay controls |
| `family_validity` (12) | every reply of a tool-chat, CJK or JSON-schema cell is valid |
| `telemetry` (13) | every cell has a telemetry series; thermal throttling and swap growth flagged |
| `quiet_machine` (14) | quiet_pre and quiet_post snapshots exist and are within limits |
| `round_time_outlier` (15) | no request whose per-round time is more than 1.5 x the median |
| `attribution_fields` (16) | the two attribution layers are complete |
| `behaviour` (18) | each B4 check is `pass` or `unsupported` |
| `mixed_same_prompt` | a mixed record holds exactly one long request and the short requests share a prompt |
| `harness_identity` (22) | every cell of one batch ran from one harness commit, never a dirty tree |
| `launch_concurrency` | a concurrent cell was launched the way N users would launch it |

Known benign classes (listed so they are not mistaken for defects; each must still be named in the audit note):

- A quiet flag on a cell that was then requeued and passed: the flagged file is kept as `*.superseded-*`, not counted.
- A `*.superseded-*` file: a record replaced by a targeted re-run (see 5.2); ignored by the audit and the report.
- A cell marked `unsupported` in the manifest: its route is expected to answer 404 (for example JSON-schema on an
  engine without it). It is not a missing cell.
- Profile rows under `<raw root>/profile/`: outside the per-cell layout, so the audit does not read them.
- An engine excluded from the batch with `--engines`: its cells expect no file in that batch.

Anything else is a violation. Read the reason, decide whether the cause is the engine, the harness or the run, and
fix the cause. Do not widen a guard to pass.

### 5.2 Targeted re-runs with `--cells`

After a harness fix, re-run only the affected cells:

```sh
... fullcmp_run.py --batch B2 --cells '^B2\.ctx\.<engine>\.16K\.' --raw-root $RAW --work-dir $WORK2
```

`--cells` takes a regex on manifest cell ids. Records are superseded, never overwritten: the selected cells' existing
files are moved to `<file>.superseded-fixN` before the new run writes, and a failed copy is never overwritten. Use a
new work dir per fix so homes and logs do not mix. Then re-run the audit over the whole raw root. A re-run never
replaces a published table: rebuild the table and say which cells changed.

### 5.3 After a driver death: `--resume-reps`

If the driver dies during rep 3, run the same command with `--resume-reps`. It skips reps 1 and 2 that are already on
disk, runs the audit, and queues only the rep-3 jobs still missing. Check the raw root first: the rep-3 cells that are still
missing are the ones queued.

### 5.4 Table and report

```sh
python3 scripts/dev/fullcmp_report.py --raw $RAW --manifest $M/manifest.jsonl --out $OUT
```

This writes `fullcmp_rows.json` (one row per engine x cell x phase, medians over reps, every rep kept) and
`fullcmp_report.html`. Superseded files are ignored; the newest complete record of a cell wins.

Publishing rules:

- Publish only after the audit passes on the whole raw root for every batch in the table.
- Show both reps' values where the cell has two reps, and the median where it has three; label `unstable` cells with all values.
- Kapitsa-style exclusion: an engine whose binary changed during the run, or whose version cannot be pinned, is not
  in the main table or charts. It goes to a footnoted appendix that says why ("self-updated during the run, mixed
  binaries, not comparable"). The report does this for `kapitsa*` engines (`is_appendix`).
- Engine labels follow the census. A preview build is labelled as a preview.

## 6. Lessons from real-1104

Each lesson is one line, with the fix that now enforces it.

| Lesson (what went wrong) | Enforced by |
|---|---|
| Kapitsa self-updated 0.2.26 to 0.2.31 mid-run, mixing four binaries | pinned read-only copy, no-update switches, sha check at launch; appendix rule (section 1) |
| A `--cells` re-run was refused because existing records blocked it, and all 28 jobs failed | records superseded to `.superseded-fixN` first; failed copies never overwritten (commit 1102788e1) |
| A harness edit landed mid-batch and a job died | every invocation runs from a frozen HEAD snapshot (`harness-<commit>/`) |
| A census without the 0.2.31 entry made nine harness tests fail | census refresh and overlay before the run; the tests read the census; check the census entry before launch |
| 22 quiet-retry jobs reused the original job's HOME (`fairness_controls`) | each retry needs its own HOME; open item, check the `-qretry` HOME in the dry run (fix queue item 13) |
| B1 rep-3 files were audited as part of B2 | the audit covers the batch and earlier batches in order only (`audited_batches`) |
| Prompt identity failed where only the chat template differed | prompt identity compares the prompt token sha, not the rendered text |
| Some engines ran without their own speculative decoding or prefix cache (oMLX without DFlash, mlx-vlm without APC) | engaged mode from evidence, and the `observe_state` / `cold_cache` / engaged guards |
| A TensorFold 1.2.0 run was misread as failed | the engaged-evidence table covers the 1.2.0 DFlash2 log line (`bench_engines` detect mode) |
| A skipped lmstudio cell was flagged by the audit | the audit is scoped to the planned cells (`audit_scope_rows`); root cause fixed in the driver (item 13) |
| Spotlight load contaminated a host-bound decode run | `mds_stores` is not exempt from the single-holder quiet rule |
| A run tied to an agent session could die with that session | launch detached in a new session; completion watcher armed at launch |
| A dry run was not read before launch | section 2.2 table; the lead reads the dry run before B1 |
| Earlier comparisons were not rigorous enough to trust | every published table is re-run through this procedure and the audit before it is used |

## 7. Tests

The harness unit tests run in the default suite. `scripts/dev/ci-local` and `just test-unit` run
`pytest tests/unit -m "not serial and not fullcmp_e2e"`, which includes the `fullcmp_quick` set (the `conftest.py`
marks every `test_fullcmp_*` and `test_tfbench_fullcmp*` file). The `fullcmp_e2e` chains (11 to 30 minutes each,
real tfbench against the stub server) are the harness's own preflight: run them before every comparison with
`python -m pytest -m fullcmp_e2e -q -n 8 --dist loadfile tests/unit` (on the M3 via `scripts/dev/m3run` when the M5
is measuring). Fast set only: `python -m pytest -m fullcmp_quick -q tests/unit`. Run the harness tests under `nice -n 15`.
