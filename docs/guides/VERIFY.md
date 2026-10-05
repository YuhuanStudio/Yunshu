# Verifying a change: `yv`

`scripts/dev/yv` runs the standard verification of a candidate change end to end and returns one
verdict. Use it instead of writing a measurement script: it builds pinned trees for both arms,
submits every GPU job to gpuq itself with sane memory / timeout / `--quiet` settings, stops at
the first failed stage, remembers what finished, and writes `verdict.json` and `verdict.md`.

```sh
scripts/dev/yv ab --base main --cand my-branch --suite decode --label wide8-lane-fusion
scripts/dev/yv ab --base main --cand . --cand-env YUNSHU_FOO=1 --suite prefill --label prefill7-x --detach
scripts/dev/yv status <run>      # state of a run (directory or name under .../verify/runs)
scripts/dev/yv wait <run>        # block until the verdict; exit code = the verdict's
scripts/dev/yv suites            # list suites
scripts/dev/yv gate              # release gate with per-stage persistence
```

Exit code: 0 every stage passed, 1 a stage failed, 2 infrastructure error (tool, git, gpuq,
parameter mismatch). `yv ab` blocks until done; `--detach` returns at once.

## Arms

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
| `speed` | `tfbench.py` decode cells (cold decode tok/s, cold TTFT, follow-up TTFT) in N quiet reps (default 3), interleaved base, cand, base, cand. Medians, per-rep paired deltas and a noise estimate (half range of the paired deltas) are reported. | no median worsening beyond `max(2%, noise)` (`--speed-tol`); a contended rep is rerun once, then the stage fails |
| `memory` | `memory_ab.py`, one job per arm and rep (default 2 reps, alternating order): peak footprint, footprint after idle, footprint held after a short follow-up. | no metric above base by more than 3% + 0.25 GiB |
| `longqa` | `tfbench.py --part needle`: 10 deterministic key-value needles spliced into the 32K / 64K / 128K prose prompts, one greedy question each (the prefix is cached after the first). Long suites only. | every item scored in both arms, correct count within +-1 |
| `conc` | `tfbench.py --part conc32`: two requests at once, each a warm 32K prefix + ~2K new text, 1024-token reply; TTFT and per-request decode. Long suites only. | TTFT / decode within 10% of base |

`--spec-modes default,mtp,dflash` repeats identity (and spec on == off) once per speculative
method (`mtp` / `dflash` set `YUNSHU_VLM_DRAFT`; `dflash` needs the drafter path, `D` in
`local.env` or `YV_DRAFTER`). The base arm's identity cells are deterministic, so they are shared
across runs through `/Volumes/P5Plus/yunshu-build/verify/cellcache/` (key: base commit, env, model,
context, harness hash, device): the next candidate against the same base reruns only its own arm.

## Suites

`decode` (preflight, smoke, identity at 1K / 8K, apc, speed; `--spec-off` adds spec on == off (fails on main today, see BACKLOG)), `prefill`
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
`soak-mmlu`, `soak-realistic`; port 18993) and records each in `runs/gate-<commit>/`. A stage
passes when its check rows have no FAIL or CONTENDED, at least one PASS, and the job exited 0.
A rerun on the same commit skips passed stages (`--fresh` reruns all); `install` reruns if
`$GATE_ROOT` holds another commit's install.

The `serve-27b.server_path` check compares the server's decode tok/s with the engine in-process.
The old rule (every case >= 0.95, median of 3) failed about one gate in five on noise. Across 24
gate runs (96 case ratios) the mean ratio is 0.997 and the per-case sd 0.031, split evenly above
and below 1.0 (the in-process side is the noisy one), so there is no server gap. The check now
alternates which side goes first, takes the median of 5, and fails on a geometric mean below
0.965 or a single case below 0.90; a systematic 5% serving overhead still fails.

## Adding a new kind of measurement

Ad-hoc scripts are for measurements `yv` does not cover. Put them in `scripts/research/` with a CPU
unit test, make them write a final `complete: true` record, and then add them as a stage or a
cell in `scripts/verify/stages.py` so the next worker does not need the script.
