# Accuracy alignment

How Yunshu shows that its serving path computes what stock MLX / mlx-vlm computes on
the same checkpoint. A 300-question MMLU-Pro run carries about +-13 questions of 95%
noise, so it cannot see small numeric drift. Three tiers, cheapest and most sensitive first.

| Tier | What it compares | Sensitivity | Cost | Status |
|---|---|---|---|---|
| 1 | next-token distributions on fixed text (teacher forcing) | about 1e-4 nats of KLD | about 10 min per condition | implemented, `scripts/research/accuracy/kld.py` |
| 2 | greedy continuations of 50 fixed prompts x 256 tokens | first differing token | about 15 min per condition | implemented, `scripts/research/accuracy/greedy_div.py` |
| 3 | paired downstream evals (GSM8K, IFEval, MMLU-Pro, needle/RULER, tool calls) | accuracy differences of about 1 point | hours | plan only (below) |

The reference is always stock mlx-vlm on the same checkpoint with no Yunshu kernel imported
(no batch_invariant, ragged KV, lane, omlx, int-code). The run asserts this. The candidate is
what `VLMEngine._build_batch_runner` installs: oMLX verify kernels, batch-invariant and
NAX-packed projections (active only in the speculative lane), ragged KV.

## Tier 1: teacher-forced logit alignment

The method behind llama.cpp `perplexity --kl-divergence` and vLLM's logprob-closeness tests.

* Corpus (`common.build_corpus`, version = date): 176K tokens; 42 windows of 2048 tokens
  (prose, code, Chinese, math) plus long documents at 32K (prose, code) and 8K (prose,
  Chinese, math). Only token ids are stored, privately. Inputs are pinned (three public
  domain books, gsm8k, a Chinese chat corpus, Yunshu sources at a fixed commit).
* Reference file: top-64 logprobs for every position plus the full-vocabulary log-softmax
  for a fixed random subset (256 prefill, 128 decode positions), so later comparisons never
  rerun the reference. Positions outside the subset use top-64 plus one lumped tail bucket,
  a lower bound of the true KLD; the full subset reports both (they agree within about 10%).
* Parts: `prefill` (all positions, 2048-token chunks), `dec1` (one token per step, the
  speculative lane), `dec6` (6-token verify blocks), `decb` / `decb8` (two-row shared batch
  on a ragged bf16 / int8 KV cache), `stockb2` (stock two-row batch, a noise floor). Decode
  windows: 8 windows of 384 steps at position 256, plus two of 128 steps at position 31488.
* Metrics: KLD(ref || cand) mean, median, p99, max; top-1 agreement; mean |dp(top1)|;
  perplexity of both, overall and by corpus type and position bucket.

## Tier 2: greedy divergence

50 fixed chat prompts (English, code, math, Chinese, eight 2-3K-token summarization
prompts), greedy, up to 256 tokens. Scored by exact match of the whole continuation and by
the first differing position. `engine` runs each prompt with speculation off and on
(`allow_draft`); `YUNSHU_ROUND_DRIVER=1` runs the round driver.

## Noise floors and thresholds

A candidate must be judged against what stock MLX does to itself. Measured on Qwen3.8-27B
oQ4e (M5 Max):

| Floor (stock vs stock) | KLD mean | median | p99 | top-1 % | ppl change |
|---|---|---|---|---|---|
| same run twice (prefill, decode) | 0 | 0 | 0 | 100 | 0 |
| decode, batch 2 vs batch 1 | 1.4e-3 | 2.9e-4 | 1.6e-2 | 98.9 | +0.04% |
| prefill, chunk 1024 vs 2048 | 2.7e-3 | 2.0e-4 | 1.3e-2 | 98.8 | +0.01% |

Reruns are bit-exact, so any nonzero difference is a real path difference. Different
reductions (batch size, chunking) already move the mean KLD to about 1e-3 to 3e-3 and top-1
to about 98.8%; that is the scale of a numerically legitimate implementation.

Tier 1 pass criteria (decode / prefill), each slice of type and position:

* mean KLD <= 2x the floor mean: 3e-3 decode, 5e-3 prefill;
* median <= 2x floor median, p99 <= 2x floor p99 (3e-2);
* top-1 agreement >= 98.4%;
* |perplexity change| <= 0.5%;
* hard gates: Yunshu prefill and `dec6` vs `dec1` are bit-identical (KLD exactly 0, spec
  on == spec off through the verify kernels).

Scale check: the 4-bit vs 8-bit checkpoint (a real quantization gap) sits at mean KLD 0.089
prefill / 0.052 decode, top-1 90.6% / 92.4%, ppl +2.4%: 20-60x above these limits, so the
gate resolves it easily.

Tier 2 pass criteria:

* spec on == spec off, 50/50 identical (hard gate);
* exact match against the reference and median first divergence must not fall below the
  recorded baseline for the same checkpoint (the stock chunk-1024 floor is 43/50 identical,
  median first divergence 23; a batched stock decode floor still has to be measured, see
  the results note below);
* stock rerun 50/50 identical (harness sanity).

First measurement (Qwen3.8-27B oQ4e): the engine's speculative-off path (stock decode
kernels inside the runner) and speculative-on path (invariant kernels) match each other in
26/50 prompts, and each matches the stock reference in 26-27/50. The spec-on == spec-off
gate therefore fails when "off" means the runner's plain decode; it holds inside the
invariant path (Tier 1 `dec6` == `dec1`) and with the round driver (spec on and off
identical). The round-driver vs upstream-runner comparison is 27/50.

## Tier 3: paired downstream evaluation

Status: running. GSM8K is final; MMLU-Pro, IFEval, BFCL and needle are still being filled
(see the results table). The plan below is the target; the first subsection records what
the harness (`scripts/research/accuracy/paired_eval.py`) actually does and found.

### Results (Qwen3.8-27B oQ4e-mtp, M5 Max, 2026-10-02)

Method as run: reference = stock `mlx_vlm.server` (same checkpoint, no Yunshu import),
candidate = Yunshu default settings, same HTTP chat endpoint, greedy, thinking on with
`reasoning_effort` medium (needle and BFCL: off). MMLU-Pro is a seeded random sample of
2000 (seed 20260930), not stratified. IFEval is strict prompt level (lm_eval checker; a
truncated answer counts as wrong). Items are paired by id; only items completed without
a transport error in both arms enter the pair. The reference arm is effectively serial
(stock server, no batching), so its jobs time out requests that Yunshu finishes; the
failed attempts are counted below, not hidden. Report command:
`paired_eval.py report --bench B --ref ref --cand default`.

| bench | status | n paired | ref | Yunshu | delta (pts) | CI95 (pts) | b / c | McNemar p |
|---|---|---:|---:|---:|---:|---|---|---:|
| GSM8K | final | 1319 / 1319 | 97.12% | 97.65% | +0.53 | [+0.04, +1.02] | 2 / 9 | 0.065 |
| MMLU-Pro | partial (917 of 2000) | 917 | 83.97% | 83.53% | -0.44 | [-1.44, +0.57] | 13 / 9 | 0.523 |
| IFEval | partial (179 of 541) | 179 | 90.50% | 88.27% | -2.23 | [-6.01, +1.54] | 8 / 4 | 0.388 |
| BFCL | pending | - | - | - | - | - | - | - |
| needle | pending | - | - | - | - | - | - | - |

b = reference right and Yunshu wrong, c = the reverse. No bench shows a significant loss
(p(candidate worse): GSM8K 0.994, MMLU-Pro 0.262, IFEval 0.194). The GSM8K gain is at the
edge of significance and is read as noise around equal accuracy, not as an improvement.
The partial rows are not final: their intervals still contain a loss of 1.4 (MMLU-Pro)
and 6 (IFEval) points, and the reference subset is the items it finished first, so do not
quote them as the benchmark scores.

Failed attempts (requests that returned an error, usually a timeout, in the arm's file):

| bench | arm | attempts | error attempts | items attempted | items unresolved |
|---|---|---:|---:|---:|---:|
| GSM8K | ref | 2638 | 1319 | 1319 | 0 |
| GSM8K | Yunshu | 1319 | 0 | 1319 | 0 |
| MMLU-Pro | ref | 1085 | 168 | 923 | 6 |
| MMLU-Pro | Yunshu | 2016 | 16 | 2000 | 0 |
| IFEval | ref | 185 | 6 | 182 | 3 |
| IFEval | Yunshu | 349 | 1 | 348 | 0 |

Truncation (finish = length) among paired items: GSM8K ref 1 / Yunshu 0; MMLU-Pro 0 / 2;
IFEval 1 / 0. Mean completion tokens: GSM8K 351 / 348; MMLU-Pro 770 / 811; IFEval 1334 /
1284.

Still queued (priority -1): 36 reference rounds for MMLU-Pro (about 30 completed items per
round at concurrency 2, so about 1080 missing items), 6 reference and 2 Yunshu IFEval
rounds, 6 BFCL and 8 needle rounds per arm. When they finish, re-run the report command per
bench and replace the partial rows. IFEval answers are stored unscored; `report` scores
them in memory (it re-executes under the lm_eval venv), so no separate `score` step is
needed.

### Plan

Purpose: confirm that numerics within the Tier 1/2 limits do not change task accuracy, and
catch failures that distribution metrics miss (format, tool calls, long-context retrieval).

* Sets: GSM8K (full 1319, 8-shot chain of thought, exact numeric answer), IFEval (541
  prompts, strict and loose prompt-level), MMLU-Pro (at least 2000 questions, stratified by
  subject, zero-shot chain of thought, thinking off), long-context needle / RULER subset at
  32K and 131K (retrieval, multi-key, variable tracking; 50 items per length), and a
  tool-calling subset (200 items: correct function, valid JSON arguments, exact argument
  match).
* Pairing: the same items, same rendered prompts, greedy decoding, run once against the
  reference (stock mlx-vlm, same checkpoint) and once against Yunshu through the HTTP
  endpoint the users hit. Runs alternate by item to remove drift and thermal effects.
* Scoring: per item correct / incorrect for both; McNemar exact test on the discordant
  pairs (b = reference right and Yunshu wrong, c = the reverse). Report b, c, the paired
  difference with a 95% interval, and p. Independent accuracy intervals are not used
  (they are 2-3x wider than paired ones). At n = 2000 and 5% discordance a difference of
  about 1.6 points is detectable, against 6+ points for an unpaired 300-question run.
* Pass: no significant loss (McNemar p >= 0.05 for b > c) on every set, and pooled
  discordance no higher than the reference against its own batched or reordered rerun (the
  same floor idea: run the reference twice under a different schedule and compare).
* Additional reads: length of continuation, finish reasons, and tool-call parse failures
  by arm.

## Release gate use

* Every change to a kernel, cache layout or scheduler runs Tier 1 (about 40 min for all
  parts) and Tier 2 on Qwen3.8-27B; a failed criterion blocks the merge. The reference
  files are stored once per checkpoint and corpus version and reused, so a Tier 1 check
  costs only the candidate side.
* Tier 3 runs before each release and after any lossy option changes. Lossy options (int8
  KV, weight quantization) are judged against their own published gap, not against the
  lossless limits.
* A checkpoint change rebuilds the reference (`kld.py run --config stock` without `--ref`).
* Commands: `kld.py corpus`, `kld.py run --config stock|serve --part ...`, `kld.py report`,
  `greedy_div.py build|run|compare`.
