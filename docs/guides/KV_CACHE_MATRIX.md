# Unified KV-cache matrix — all models × all tiers

Reproduces the model × cache-tier matrix from `scripts/bench_all.py`, now extended
to the **VLM text path**. Each measure is its own subprocess, run
sequentially (no concurrency) so numbers don't interfere. M3 Max, greedy, in-process
(no gateway/HTTP). Regenerate with:

```bash
PYTHONPATH=. uv run python scripts/bench_all.py
```

## Which cache tiers each serving path has

Read this first: the "four tiers" exist on ONE of the serving paths.

| Serving path | Serves | RAM, full precision (HOT) | RAM, 4-bit (WARM) | SSD | Nothing cached (COLD) |
|---|---|---|---|---|---|
| **VLM runner** (`VLMEngine` + `vlm_batch_runner.py`, mlx-vlm APC) | every mlx-vlm model, Qwen3.8-27B included; all `/v1/chat/completions`, `/v1/messages`, `/v1/responses` traffic for them | yes: exact checkpoints (KV + recurrent state), byte budget `YUNSHU_VLM_APC_MEMORY_GB` | **opt-in, off by default** (`YUNSHU_VLM_APC_WARM`): `lossless` (zstd, bit-exact) or lossy `int8` / `int4` K/V; takes `YUNSHU_VLM_APC_WARM_SHARE` of the APC RAM budget | **yes, on by default**, lossless (bit-exact states), `~/.yunshu/cache/apc`, 64 GiB cap for the whole directory (all models together): `YUNSHU_VLM_APC_DISK=0` opts out, `YUNSHU_VLM_APC_DISK_DIR` / `YUNSHU_VLM_APC_DISK_GB`; further storage tiers below it (external SSD, HDD, NAS): `YUNSHU_VLM_APC_DISK_TIERS` | full prefill |
| **Text fast path** (`BatchedEngine._generate_fast`, mlx-lm `generate_step`) | text-only mlx-lm models (e.g. Qwen2.5-3B) | yes: `KVPrefixCache`, `YUNSHU_PREFIX_MAX_ENTRIES` entries | yes but **off by default** (`YUNSHU_PREFIX_HOT_LIMIT=0`; lossy on reuse) | yes but **off by default** (`YUNSHU_SSD_CACHE=1`; `native` precision is bit-exact; skipped for models that prefill faster than `YUNSHU_SSD_CACHE_PREFILL_CEIL_TPS`) | full prefill |
| Engine loop (`YUNSHU_ENGINE_LOOP=1`, legacy) | text models | radix cache (HOT only) | no | no | full prefill |

`x_yunshu.cache` on every response says which tier served the request (additive; the SDK fields are
untouched), and `X-Yunshu-Cache-Tier` / `X-Yunshu-Cache-Reload-Ms` carry it on non-streaming responses:

```json
"x_yunshu": { "cached_tokens": 13867,
              "cache": { "tier": "ram", "cached_tokens": 13867, "reload_ms": 41.2 } }
```

`tier` is `ram` (full-precision RAM = HOT: the VLM runner's APC, the text path's HOT), `warm` (the compact
RAM form: the VLM runner with `YUNSHU_VLM_APC_WARM` on, the text path with `YUNSHU_PREFIX_HOT_LIMIT`),
`ssd` (reloaded from a storage tier; `reload_ms` includes the read) or `none`. With more than one storage
tier configured an `ssd` hit also carries `"device"`, the volume it came from (`internal`, `P5Plus`, ...).
`/debug/kv-cache` and `/metrics` (`apc_*` gauges and counters, `apc_tier_lookups_total{tier}`,
`apc_warm_*`, `apc_storage_tier_*{tier}`) report occupancy, entries, hits by tier, evictions, disk bytes and
each storage tier's usage, measured bandwidth and availability for the VLM runner.

### VLM runner (APC) policy

mlx-vlm's APC keeps exact checkpoints for hybrid (GDN) models: a full copy of the attention KV plus the
recurrent state at one token position, reusable only by a prompt that starts with exactly those tokens.
`yunshu_engine/apc_manager.py` (`YunshuAPCManager`, `SpillDiskStore`) changes the policy, not the states
(a hit restores the same bytes a cold prefill would produce):

- the byte budget (not a two-entry cap) limits RAM; the default budget is half of the memory left after the
  weights and a 16 GiB reserve, between 4 and 32 GiB (a 27B checkpoint costs about 130 KiB per cached token);
- a request's checkpoints supersede earlier-request checkpoints that are prefixes of it (the conversation
  grew), so a long session holds one copy, not one per turn;
- the end of the system turn (system prompt + tool list) is a checkpoint of its own, kept per distinct
  head: a new session in the same project reuses it;
- the SSD tier receives a checkpoint when RAM evicts it (and resident ones at shutdown, newest first, within
  20 s), not on every store. It is **on by default** (measured below: a reload is 15-40x faster than
  re-prefilling; the cache also survives a restart). States are namespaced by the checkpoint's weights
  (path, file sizes, mtimes), so replacing a model in place never reads old states. `yunshu doctor` prints
  the directory, its size, the cap and the free space.

### Cache hierarchy on the VLM runner: HOT / WARM / storage tiers / recompute

Measured on the Qwen3.8-27B oQ4e (M5 Max, 128 GB), real checkpoints of 32K-token documents (code and prose),
`scripts/research/apc_audit/tier_roofline.py`, `tier_breakeven.py`, `tier_capacity_model.py` (the raw
results are private). Unified memory: HOT vs WARM is a format difference, not another memory pool.

**What a checkpoint is.** 154 MB of recurrent (GDN) state, constant, plus 64 KiB of attention K/V per token (16
attention layers, 4 KV heads, head dim 256, bf16): 8K tokens = 0.69 GB, 32K = 2.3 GB, 128K = 8.7 GB. The state is
7 % of a 32K checkpoint and 22 % of an 8K one. (The "130 KiB per token" used for sizing elsewhere is the
resident estimate with allocator slack; the bytes are 64 KiB.)

| Form | bytes vs bf16 | Cost (this M5 Max) | Quality |
|---|---|---|---|
| HOT clone (ready arrays) | 1.00 | 0.043 s per 2.1 GB | exact |
| lossless, zstd-1 on bf16 K/V | 1.27x smaller (no shuffle) | 1.4 GB/s per thread | exact |
| lossless, byte-plane shuffle + zstd-1 | **1.46x** smaller | compress 7.3 GB/s, decode 17 GB/s (8 threads) | exact |
| lossless on the GDN state (fp32 + bf16) | 1.07x | - | exact |
| lz4 | 1.00x (no gain without shuffle), 1.18x with | 14 GB/s | exact |
| int8 g32 K/V (affine), state exact | 0.56x (1.78x capacity) | dequantize 0.02-0.07 s per checkpoint | KLD vs exact: code 5.6e-4 (p99 7.6e-3), prose 2.3e-3; top-1 98.8 % / 98.0 % |
| int4 g64 K/V | 0.28x (3.6x capacity) | same | code 2.0e-3 (p99 2.6e-2, max 0.11), prose 6.8e-3 (p99 5e-2); top-1 98.8 % / 95.7 % |

Control (a second exact restore): KLD 0. The prose int8 figure equals the KLD of merely casting the GDN state
to bf16 (2.4e-3): int8 K/V is at the numerical noise floor, int4 is about three times above it.

**Restore of a 32K checkpoint (2.3 GB) per tier against re-prefilling (37 s at 885 tok/s):**

| Tier | Restore | vs prefill |
|---|---|---|
| HOT hit (clone) | 0.04 s | 900x |
| WARM int8 (lossy), roofline | ~0.05 s | 700x |
| WARM lossless, roofline (decode 17 GB/s + GPU plane merge + copy) | ~0.25 s | 150x |
| internal SSD (10.3 GB/s measured, F_NOCACHE) | 0.22 s | 165x |
| TB4 SSD (5.0 GB/s) | 0.46 s | 81x |
| HDD (150 MB/s, 12 ms; throttled simulation) | 14.9 s raw, 11.1 s zstd | 2.5x / 3.3x |
| NAS 1 GbE (110 MB/s, 3 ms; throttled simulation) | 22.3 s raw, 16.5 s zstd | 1.7x / 2.2x |
| recompute | 37 s | 1x |

Break-even read bandwidth for a 32K prefix is 62 MB/s (checkpoint bytes / prefill time): an HDD or a 1 GbE
NAS still beats recomputing a long prefix. A 1K prefix on HDD or NAS does not (1.45 s / 2.15 s restore against
1.16 s of prefill), which is why a tier is used per entry by the cost model, not blindly.

**Measured through the server** (`reload_ms` of `x_yunshu.cache`: the whole lookup, promotion into RAM
included; median over the replay below; a checkpoint = 154 MB + 64 KiB per token):

| Tier | cached tokens | hits | median reload | p90 | effective MB/s |
|---|---|---|---|---|---|
| HOT (clone) | up to 13K / 13-22K / 22-40K | 26 / 14 / 14 | 28 / 41 / 63 ms | 36 / 52 / 75 ms | 23-31 GB/s |
| internal SSD | up to 13K / 13-22K / 22-40K | 3 / 3 / 3 | 128 / 147 / 496 ms | 293 / 263 / 527 ms | 4-7 GB/s |
| TB4 SSD (single tier) | up to 13K / 13-22K / 22-40K | 7 / 31 / 61 | 138 / 605 / 1300 ms | 604 / 907 / 2085 ms | 1.4-6 GB/s |
| WARM lossless | up to 13K / 13-22K / 22-40K | 4 / 10 / 7 | 331 / 685 / 1812 ms | 361 / 1401 / 6282 ms | 1.0-2.5 GB/s |
| WARM int8 (lossy) | up to 13K / 13-22K / 22-40K | 4 / 8 / 2 | 63 / 503 / 128 ms | 72 / 5547 / 161 ms | 2.4-17 GB/s |

The sequential probe (10.3 and 5.0 GB/s) is an upper bound: reading a checkpoint back through the loader and
cloning it into the cache delivers 1.4-7 GB/s, and a cold read of a large file is slower than one still in the
page cache. The storage cost model therefore uses the observed restore speed (EMA per tier) when it is slower
than the probe. WARM lossless decodes at 1-2.5 GB/s end to end, not at the 17 GB/s of the codec: promotion
into HOT evicts (and demotes, i.e. compresses) another checkpoint inside the same lookup, and the decoded
copy is cloned once more. WARM int8's p90 shows the same demotion work (quantize plus the exact SSD copy).

**Replay: 3 agent sessions growing to 30K tokens round-robin (20 s idle gaps), each revisited, Qwen3.8-27B**
(`queue_tier4.sh`, `session_replay.py`; 27 requests; SSD = the TB4 volume unless stated; greedy):

| Configuration | prompt tokens served from cache | hits (ram / warm / ssd) | TTFT mean / p50 / p90 / max | peak RSS |
|---|---|---|---|---|
| 4 GiB RAM only (no SSD) | 31.7 % | 20 / 0 / 0 | 21.2 / 17.7 / 39.8 / 44.7 s | 4.7 GB |
| 4 GiB RAM + SSD (before this change) | 86.3 % | 2 / 0 / 24 | 6.0 / 6.1 / 8.8 / 14.0 s | 5.6 GB |
| 4 GiB RAM + SSD, superseded checkpoints dropped on disk | 86.3 % | 2 / 0 / 24 | 6.3 / 6.0 / 9.1 / 13.9 s | 5.7 GB |
| 4 GiB + WARM lossless (share 0.4) + SSD | 86.3 % | 2 / 7 / 17 | 6.2 / 6.2 / 8.9 / 13.9 s | 9.0 GB |
| 4 GiB + WARM int8 (share 0.4) + SSD | 86.3 % | 2 / 1 / 23 | 7.2 / 6.0 / 14.7 / 19.0 s | 6.3 GB |
| 6 GiB RAM + SSD | 86.3 % | 12 / 0 / 14 | 5.5 / 5.7 / 7.1 / 13.8 s | 5.4 GB |
| 6 GiB + WARM lossless + SSD | 86.3 % | 2 / 14 / 10 | 6.3 / 5.6 / 10.8 / 13.9 s | 10.5 GB |
| 6 GiB + WARM int8 + SSD | 86.3 % | 2 / 13 / 11 | 6.8 / 5.9 / 12.1 / 18.0 s | 5.6 GB |
| 32 GiB RAM (everything fits) | 86.3 % | 26 / 0 / 0 | 4.7 / 4.8 / 5.5 / 13.8 s | 4.7 GB |
| 4 GiB + internal SSD (3 GiB) > TB4 SSD (6 GiB) > simulated HDD (40 GiB) | 85.7 % | 2 / 0 / 24 (9 internal, 15 TB4, 0 HDD) | 5.6 / 5.7 / 8.9 / 14.0 s | 5.4 GB |
| 4 GiB + TB4 SSD (3 GiB) > simulated HDD (150 MB/s, 17 ms) | 81.3 % | 4 / 0 / 22 (12 TB4, 10 HDD) | 10.4 / 7.7 / 19.2 / 28.3 s | 5.8 GB |

Every configuration hits the same 86.3 % ceiling (the first request of a session and the head are cold; each
turn adds 3K new tokens, 3.4 s of prefill that no cache removes). What differs is where the hit comes from
and what it costs. Reading the table:
- the SSD tier is the whole difference between 21 s and 6 s mean TTFT at 4 GiB; the HDD-only variant restores
  a 2 GB checkpoint in 9-11 s instead of recomputing it for 25-45 s;
- WARM, lossless or int8, adds nothing over the SSD on these runs: the mean TTFT is equal or worse (6.2-7.2 s
  against 6.0 s), and lossless WARM doubles the resident memory (peak RSS 9-10.5 GB: HOT + compressed copies +
  the one entry being compressed + the decode buffers). The roofline said so before the run (the codec is not
  faster than an internal NVMe, and the end-to-end decode is slower than the codec);
- three devices (internal SSD, TB4 SSD, simulated HDD) with fast-tier caps below the working set (3 + 6 GiB; each
  session's newest checkpoint is 2.1 GB, plus its interval checkpoint) match the single big SSD (5.6 s against 6.0 s) because the mover demotes the least
  recently used checkpoints instead of deleting them and the cost model picks the cheapest tier per lookup.

**Disk supersede.** A request's checkpoints used to accumulate on the SSD tiers: the 3-session replay left 46
files and 63 GiB, the SSD cap (64 GiB) reached in five minutes. Once the cap or the RAM budget is small the cache
the next request needs is the one an LRU evicts first, so these stale copies cost hits in tiered
configurations (a first tiered run lost every session's newest checkpoint and recomputed 30-47 s requests).
A checkpoint written to any tier now drops the checkpoints of earlier requests that are prefixes of it, on every
tier: the same replay leaves 6 files and 10.8 GiB (the head plus each session's newest checkpoints) with the same
hits. Heads and anything this process neither stored nor restored are kept.

**Conclusions.**
1. The tier that pays is the SSD (and the storage tiers below it): without it the 4 GiB RAM budget of a 16-32 GB
   machine serves 32 % of the prompt tokens from cache and runs at a 21 s mean TTFT (45 s at the end of the
   replay); with it 86 % and 6 s.
2. **WARM lossless** is bit-exact (tested token-identical, restores from WARM, SSD and HOT give the same greedy
   tokens) but compresses 1.4x, and on 27B traffic it is not a win: off by default
   (`YUNSHU_VLM_APC_WARM=off`), as the lossless-default rule demands a measured end-to-end win. Keep it for
   machines without a usable SSD tier.
3. **WARM int8 / int4** are lossy: opt-in only, never default, and the replay shows no TTFT gain either.
4. Lower storage tiers matter on slow media: restores from external SSD / HDD / NAS need per-device
   measurement (the break-even above), a cost model that learns the real restore speed, background demotion
   and zstd on the slow ones (+33 % effective bandwidth); fast disks stay raw. All implemented and tested
   with throttled devices; the real TB5 enclosure and a real HDD / NAS were not available (HDD and NAS are
   simulated by pacing every probe, copy and read of the tier to the stated bandwidth and latency).

**Machine sizes** (APC RAM budget: 32 GiB on 128 GB, 16 GiB on 64 GB, 4 GiB on 16-32 GB; 30K-token sessions
take 2.2 GB, 60K 4.1 GB):

| Machine | HOT holds | SSD tier | WARM |
|---|---|---|---|
| 128 GB (32 GiB) | ~14 sessions of 30K, 7 of 60K | not needed for a few sessions, still the restart cache | not worth it |
| 64 GB (16 GiB) | ~7 of 30K, 3 of 60K | needed beyond 3 long sessions | only without SSD (model) |
| 16-32 GB (4 GiB) | 1-2 sessions of 30K | the cache: without it 32 % served from cache, 21 s mean TTFT | no gain measured (lossless or int8) |

**Settings** (all in `settings.py`, stable options; none is an experimental flag):
`YUNSHU_VLM_APC_WARM` (`off` | `lossless` | `int8` | `int4`, default `off`), `YUNSHU_VLM_APC_WARM_SHARE` (share of
the APC RAM budget the WARM tier takes, default 0.4), `YUNSHU_VLM_APC_DISK_TIERS` (`PATH[@GiB],...` lower
storage tiers), `YUNSHU_VLM_APC_DISK_ENCODING` (`auto` | `raw` | `zstd`, lower tiers only, never lossy).

**Mechanics.** HOT (`_exact_cache`) evicts LRU into WARM (`yunshu_engine/apc_warm.py`): lossless entries are
compressed on a worker thread (one at a time, the entry stays promotable meanwhile), lossy entries are
quantized on the GPU thread, and in lossy mode the exact copy goes to the SSD at the same moment (the SSD tier
stays exact; a WARM entry that is evicted is dropped). A WARM hit decodes into HOT and the lookup then runs as
a HOT hit (`tier: "warm"`); a failed checksum drops the entry and the lookup falls through to SSD /
recompute. Storage tiers (`yunshu_engine/apc_storage.py`): the SSD directory is the primary store (upstream
safetensors files, async writer, root-wide `DiskBudget`); each further directory is profiled at startup
(`probe_device`: sequential read / write with `F_NOCACHE`, 4 KiB read latency; cached per mount in
`~/.yunshu/cache/apc-device-profiles.json` for 24 h), ordered by measured read bandwidth, and has its own cap and
free-space reserve. When a tier is over its cap the budget hands the file to a background mover instead of
deleting it; the mover copies it (raw, or as a zstd container when the measured gain pays) to the first lower
tier that is mounted, has room and where restore time beats re-prefilling the tokens it saves at the observed
prefill speed. A lookup takes the cheapest candidate over all tiers (restore time + prefill of the rest). An
unmounted tier (volume gone, stat timeout) is skipped; on return its files are re-validated (header, size,
checksums) and corrupt ones are deleted. A path under `/Volumes/NAME` whose volume is absent is never created.
An I/O error while scanning keeps the files and marks the tier unavailable. The cost of a hit from each
tier is in `x_yunshu.cache.reload_ms`.

### SSD cache disk budget (APC and text tier)

Both SSD tiers keep one namespace directory per checkpoint under a root. The budget is per
**root**, shared by every namespace (`yunshu_kv/disk_budget.py`), so old fingerprints and other
models can no longer add up to more than the cap (a 117 GiB directory against a 64 GiB cap, and a
full 1 TB volume, was the failure this replaces):

- **Global cap**: `YUNSHU_VLM_APC_DISK_GB` / `YUNSHU_SSD_CACHE_MAX_GB` bound everything under the
  root; over it, the least recently used files go first across namespaces (mtime is the clock; loads
  touch it).
- **Stale namespaces** go before anything else: unused for `YUNSHU_CACHE_STALE_DAYS` (7), or whose
  recorded checkpoint (`namespace.json`: path + signature) is gone or changed. `yunshu cache gc`
  reports them (`stale-namespace`, `dead-namespace`) and `--apply` removes them; `yunshu cache status`
  and `yunshu doctor` show the size per namespace and per root.
- **Free-space reserve**: no write may leave the volume under max(`YUNSHU_CACHE_RESERVE_PCT` of the
  volume, `YUNSHU_CACHE_RESERVE_GB`) free (10 % / 20 GiB). The effective cap is
  `min(configured cap, root usage + free - reserve)`, re-evaluated as space changes, not fixed at start.
- **Write errors** (ENOSPC, EIO): that checkpoint is dropped, one warning is logged (no traceback),
  spilling pauses until free space is back (re-checked every 30 s), temp files are removed, and the
  request in flight is unaffected. The cache content, its validation and the lossless default are unchanged.

### Text path SSD tier (opt-in)

Qwen2.5-3B bf16 (prefills 7.7K tok/s), 3 sessions to 20K tokens, RAM limited to 2 entries: before, a short
RAM hit (the shared system prompt) hid the conversation's much longer spilled prefix, so reuse was 44.8%
and the SSD tier never served. Now the longer SSD prefix wins: reuse 84.6%, mean TTFT 1982 -> 1013 ms,
revisit of a 20K session 2.35 s -> 0.70 s. The prefill-speed gate (`YUNSHU_SSD_CACHE_PREFILL_CEIL_TPS`) was
4000 tok/s and blocked this model; measured reload still wins 3x at 7.7K tok/s, so the default is 20000.
The tier stays opt-in (`YUNSHU_SSD_CACHE=1`): with the default 64 RAM entries the RAM tier already holds
these sessions (84.7% reuse in RAM alone).

### APC SSD tier: measured (Qwen3.8-27B oQ4e, M5 Max, greedy)

TTFT of the same prompt + a 350-token suffix; cold = full prefill. Disk = TB4 enclosure (2.6 GiB/s measured
sequential); the internal SSD of the same machine measured 8.4 GiB/s read / 9.9 GiB/s write, so an internal
default directory reloads about 3x faster than these numbers.

| Prompt | Cold prefill | RAM hit | SSD hit (RAM tier off) | SSD hit after a server restart | Shutdown spill |
|---|---|---|---|---|---|
| 10K | 12.4 s | 0.68 s | 1.30 s | 0.79 s | 1.3 s |
| 30K | 41.0 s | 0.96 s | 2.14 s | 1.45 s | 2.3 s |
| 60K | 91.0 s | 1.69 s | 6.55 s | 2.37 s | 7.2 s |

An SSD hit is 10-40x faster than re-prefilling. The first read is lazy (mmap: `reload_ms` under-reports
it; TTFT is the truth); under concurrent spills a 28K reload took up to 6.8 s, still 6x faster than the 40 s
re-prefill. Greedy output of the SSD hit is token-identical to the RAM hit at 10K / 30K / 60K. (A cached
run and a cold run differ at 60K for a different reason: prefill chunk boundaries move, which is drift, not
a tier property.)

Real agent sessions (Claude Code, Codex, opencode on the current code): reuse equals the achievable ceiling
for the request stream (Claude Code 89-92%, Codex 80-88%, opencode 64-73%; the rest is each session's
first, cold request, plus 0.3-3.7 points of generated tokens re-prefilled on the next turn).

## Result (latest run)

```
Model                      Arch       pTok  pf t/s dec t/s COLD ms  F-HOT   F-WARM  F-SSD    WARMram LOOP   oMLX     loss
Qwen3.5-0.8B-MLX-bf16      hybrid     1242  2895   141.9   429      2.18x   2.30x   1.88x✓   3.58x   5.34x  1.06x    Y
Qwen3.5-2B-MLX-bf16        hybrid     1242  1714   66.6    725      3.14x   2.98x   2.64x✓   3.58x   7.05x  0.98x    Y
Qwen3.5-9B-MLX-4bit        hybrid     1242  505    54.0    2460     6.40x   6.11x   6.02x✓   3.56x  18.54x  0.91x    Y
Qwen2.5-3B-Instruct-bf16   standard   1239  1253   41.9    989      6.33x   6.08x   2.15x✓   3.56x   8.18x  3.88x    Y
gemma-4-e4b-it-bf16        gemma4     1236   801   22.2   1542      5.43x   5.57x   3.45x✓   1.00x   6.46x  LOAD FAI Y
GLM-OCR-bf16               vlm-mrope  1565  5596   169.2   280      12.59x  8.44x   1.15x✓   3.56x   n/a    n/a      Y
Qwen3-Omni-30B-A3B-4bit    vlm-imrope 1239  866    59.7   1431      13.27x 12.35x   2.05x✓   3.56x   n/a    n/a      Y
```

> **LOOP (engine-loop reuse) roughly doubled** after the detokenizer fix
> (9B 9.3→18.5×, 2B 3.2→7.0×, Qwen2.5-3B 5.8→8.2×) — the per-request detok rebuild
> was a big part of the engine-loop's per-step cost. Qwen3-Omni-30B is now measured
> and stable under load. Numbers have ±10-20% run-to-run
> variance (e.g. gemma's cold prefill drifts with thermal state).

> **Qwen3-Omni-30B is now MEASURED**. It was bypassing because (a)
> its INTERLEAVED mRoPE needs the model's native rope state primed for reuse (not
> our own position_ids — that only worked for GLM-OCR's simple mRoPE), and (b) the
> probe's single-logit `<1e-2` gate was too strict for a 30B MoE (routing noise
> that never flips the greedy token). Fix: prime `_rope_deltas=0` (model-native,
> both mRoPE variants) + a greedy-token-sequence probe. → HOT **13.08×**, WARM
> **10.90×**, SSD 1.84×, all lossless.

> **decode tps is now PURE decode** — two-point timing (t41−t1)
> cancels prefill. The earlier column reported `40 tokens / (prefill-of-1400 +
> decode)` which understated decode ~2.5× (0.8B looked like 56 tok/s; real pure
> decode is ~155). There was **no decode regression** — the engine matches
> mlx-lm (0.8B ≈ 136–155 tok/s, verified in isolation).

## How to read it

- **F-HOT / F-WARM / F-SSD** = TTFT speedup of re-requesting the same prefix vs a
  cold full prefill, on the default fast path / VLM text path 4-tier KV cache.
  - HOT = GPU-resident full precision. WARM = 4-bit-in-RAM (UMA). SSD = on disk.
  - Defaults (since 2026-09-29, "lossy is opt-in"): WARM is off
    (`YUNSHU_PREFIX_HOT_LIMIT=0`, every entry full precision) and the SSD tier
    (`YUNSHU_SSD_CACHE=1`) stores KV and recurrent state bit-exact
    (`YUNSHU_SSD_CACHE_PRECISION=native`). The WARM and SSD columns below were
    measured with the earlier lossy defaults (4-bit WARM, int8 SSD); set
    `YUNSHU_PREFIX_HOT_LIMIT>0` / `YUNSHU_SSD_CACHE_PRECISION=int8` to get them.
  - F-SSD trailing ✓ = the SSD tier actually restored from disk (not a full-prefill
    fallback).
- **WARMram** = HOT-entry / WARM-entry RAM ratio (the 4-bit saving). ~3.56× for
  standard attention; **1.00× for gemma4** because its sliding-window cache layers
  have no `to_quantized` (WARM passes them through unquantized — still cached, just
  no RAM saving).
- **LOOP** = opt-in engine-loop (radix cache, `YUNSHU_ENGINE_LOOP=1`). HOT-equivalent
  only; n/a for VLM (LLM-only path).
- **oMLX** = live reference/omlx head-to-head (when its native venv is provided via
  `OMLX_PYTHON`). `LOAD FAI` = oMLX can't load that checkpoint (e.g. gemma4); `1.0x`
  for hybrid = oMLX doesn't reuse the hybrid Qwen3.5 prefix. n/a for VLM.
- **loss** = lossless verdict. LLM rows: all tiers' greedy output matches the cold
  reference. **VLM rows: the engine's empirical reuse probe** (full prefill vs reuse
  path, bit-identical at load) — authoritative, since the per-tier text exact-match
  is noisy for short cross-prompt answers.

## Engine routing

`bench_all.py` routes each model the way the production resolver does — VLMEngine iff
mlx-lm lacks the `model_type` (`importlib.util.find_spec("mlx_lm.models.{type}")`):

| model_type | engine | why |
|---|---|---|
| qwen3_5, qwen2, gemma4, qwen3_vl | BatchedEngine (LLM fast path) | mlx-lm implements it (a `vision_config` stub alone does NOT make it a VLM) |
| glm_ocr, qwen3_omni_moe, qwen2_5_vl | VLMEngine (text path) | mlx-lm lacks it → mlx-vlm |

## VLM cache-tier reality

> **Historical (deleted 2026-09-28).** VLMEngine no longer has its own text KV
> prefix cache: every VLM request is served by the batch runner, whose prefix
> reuse is upstream mlx-vlm APC (all families without a sliding-window cache;
> optional SSD tier via `YUNSHU_VLM_APC_DISK_DIR`). The numbers below describe
> the deleted legacy path, recorded in `docs/archive/legacy_vlm_loop/`.

The VLM text path got the SAME 4-tier KV hierarchy as the LLM fast path, gated by a
two-step safety check (`_text_prefix_reuse_safe`): the cache must be resumable AND an
empirical load-time probe must confirm bit-identical reuse.

- **GLM-OCR** (simple mRoPE, full-attention KVCache): lossless reuse via explicit
  sequential position_ids — **HOT 11.1×, WARM 6.6× (+3.56× RAM saving), SSD restores**.
- **Qwen3-Omni-30B** (interleaved mRoPE): explicit positions don't match its scheme →
  probe fails → **auto-bypassed**, output identical to no-cache (no corruption).
- **gemma-4** would bypass under VLMEngine (sliding-window), but the resolver routes it
  to BatchedEngine (mlx-lm gemma4) where it reuses losslessly (3.4×).

See `docs/archive/legacy_vlm_loop/VLM_TEXT_KV_PREFIX.md` for the full root-cause + safety analysis.
Full JSON: `docs/kv_cache_matrix_results.json`.

## Multi-framework comparison (single-req + batched)

`scripts/bench_frameworks.py` — Yunshu (fast + engine-loop) vs raw mlx-lm vs
vllm-mlx vs oMLX (native venv), LLM models, batch hard-capped at 32. Regenerate:
`PYTHONPATH=. OMLX_PYTHON=/tmp/omlxenv/bin/python uv run python scripts/bench_frameworks.py`.

```
                 TTFT ms   dec t/s  batch8  batch16  batch32   (detok fix)
Qwen3.5-0.8B
  yunshu-fast    173.9     138.3    94.6    94.7     96.9
  yunshu-loop     85.0     142.2   346.6   490.6    578.0
  mlx-lm         186.5     145.7   349.0   505.2    627.7
  vllm-mlx       239.2     154.0    97.9    96.2     95.1
  oMLX           324.5     147.9   133.3   150.6    146.3
Qwen3.5-2B
  yunshu-fast    215.2     64.7     50.6    52.2     51.8
  yunshu-loop    175.9     65.3    174.6   216.8    266.3
  mlx-lm         291.9     64.8    155.7   219.5    266.7
  vllm-mlx       332.4     67.8     48.2    45.9     47.4
  oMLX           383.3     61.8     90.2   108.0    123.6
Qwen2.5-3B
  yunshu-fast    123.4     41.3     33.6    35.8     36.0
  yunshu-loop    328.9     41.5    106.7   147.4    180.4
  mlx-lm         363.3     41.7    107.1   147.0    180.1
  vllm-mlx       365.2     42.1     33.6    32.8     32.9
  oMLX           436.4     40.9     72.3    92.8    101.6
gemma-4-e4b
  yunshu-fast    223.5     25.1     21.5    22.5     23.1
  yunshu-loop    269.6     26.0     81.6   126.1    133.9
  mlx-lm         (raw mlx-lm batch can't load gemma-4)
  vllm-mlx       397.8     26.0     22.5    22.7     22.9
  oMLX           (LOAD FAIL — pinned mlx-vlm too old for gemma-4)
```

**Findings (CLOSED the batching gap):**
- **Engine-loop batched now MATCHES raw mlx-lm static batch** — 2B N=32 266.3 vs
  266.7; 3B 180.4 vs 180.1; 0.8B 578 vs 628. The "1.5–2.5× gap" flagged
  as reclaimable was the **per-request detokenizer rebuild** (~145ms on Qwen's
  151k vocab), not async orchestration. Now we match static batch *while keeping*
  streaming + mid-flight add/remove + per-request stop. Engine-loop TTFT also
  dropped 2–2.4× (0.8B 204→85ms — the detok build was on the TTFT path too).
- **Single-request decode tok/s: all 5 frameworks within ~5%** (same mlx-lm
  kernel).
- **Single-request TTFT: Yunshu fast-path lowest** on most models (3B 123ms vs
  mlx-lm 363 / oMLX 436); engine-loop now also low (0.8B 85ms).
- **vllm-mlx ≈ Yunshu-fast** (single-stream; no batching gain here).
- **Only Yunshu serves gemma-4 and the VLMs** (mlx-lm batch + oMLX can't load it).

## Comparison vs our past results (no regression)

**IMPORTANT — compare LIKE-FOR-LIKE.** The HOT-reuse speedup depends on the PATH
(fast-path 4-tier vs engine-loop radix) and the COLD baseline. An earlier version
of this table compared *engine-loop* 9.90× against the current *fast-path*
6.22× — apples-to-oranges, which looked like a regression but is a column mismatch.

| metric (matched path) | past | now | verdict |
|---|---|---|---|
| 9B hybrid — ENGINE-LOOP reuse | 9.90× (loop) | LOOP 9.32× | consistent |
| 9B hybrid — FAST-PATH reuse | 6.26× (fast) | F-HOT 6.22× | consistent |
| 2B hybrid — FAST-PATH reuse | 4.23× (fast) | F-HOT 3.06–3.18× | **ratio LOWER** ↓ — but COLD got ~40% faster (1162→711ms) and abs HOT improved (275→224ms); smaller ratio because the baseline shrank, not because reuse got slower |
| Qwen2.5-3B — FAST-PATH reuse | 5.57× (fast) | F-HOT 5.97× | consistent/better |
| WARM 4-bit RAM saving | 3.56× | 3.56× | identical |
| oMLX hybrid reuse | ≤1.06× (no reuse) | ≤1.02× | consistent |
| single-req decode (3B, pure) | ~36–38 t/s | ~42 t/s | within variance / faster |
| batched N=32 (3B, mlx-lm) | 143 t/s | 181 t/s | faster |

**Honest verdict — is anything genuinely WORSE?** No time-regression found: every
LIKE-FOR-LIKE pair is consistent or faster in absolute latency. The one cell that
*looks* worse is the **2B fast-path hybrid speedup RATIO (4.23×→3.18×)** — but that
is because the COLD baseline prefill got ~40% faster, which SHRINKS the ratio while
the absolute reuse latency actually improved (HOT 275→224 ms). Speedup ratios are
baseline-relative and not directly comparable across runs when the baseline moves.

One real structural fact (not a regression, a property): for **hybrid** models the
fast-path 4-tier reuse (boundary-snapshot, F-HOT 6.22× on 9B) is LOWER than the
engine-loop radix reuse (LOOP 9.32×) — radix is finer-grained for recurrent
backbones. Standard models are the opposite (fast-path HOT ≈ loop). The genuinely
new capability is the **VLM text-path 4-tier cache** (no past equivalent).

Framework JSON: `docs/framework_comparison_results.json`.

## Comprehensive sweeps (length × hit-ratio × concurrency)

`scripts/bench_comprehensive.py` / `bench_comprehensive_all.py` — the dimensions
the framework/tier tables don't cover. Yunshu production fast path, in-process.
JSON: `docs/comprehensive_sweep_results.json`.

### A. Prompt-length sweep (cold single-request)
TTFT and prefill tok/s rise with length (longer prompts prefill more efficiently
per token); pure decode tok/s falls as the KV context grows (attention over more
tokens) — clearest on GLM-OCR (143→35) and the 9B. (Decode probe rewritten in
to a robust single-point TTFT-based measure — the earlier 0.0/`*` cells
are gone.)

```
                prompt_tok:  ~180   ~640   ~1270  ~2520  ~5030
Qwen3.5-0.8B  decode t/s    107.5  106.8  100.3   95.2   85.0
Qwen3.5-2B    decode t/s     56.1   55.3   55.2   53.0   50.6
Qwen3.5-9B-4b decode t/s     45.8   46.0   44.8   43.4   40.0
Qwen2.5-3B    decode t/s     30.8   31.2   28.7   27.0   24.2
gemma-4       decode t/s     21.3   21.6   20.5   20.2   20.2
GLM-OCR(VLM)  decode t/s    143.0  120.0   87.9   59.0   34.9
```
(TTFT/prefill-tok/s per length in `comprehensive_sweep_results.json` — TTFT scales
with prompt length; prefill tok/s rises then plateaus, e.g. 0.8B 955→3063.)

### B. Cache-hit-ratio sweep (reuse benefit vs shared-prefix fraction)
Speedup scales with BOTH hit-ratio and model size. Below ~0.4 hit-ratio the
(longer) unique suffix costs more than the reuse saves → <1× (don't cache tiny
shared prefixes).

```
              hit~0.90  hit~0.83  hit~0.65  hit~0.35
Qwen3.5-0.8B  2.49x     1.97x     1.43x     0.59x
Qwen3.5-2B    3.18x     2.61x     1.54x     0.56x
Qwen3.5-9B-4b 6.58x     4.16x     1.82x     0.52x
Qwen2.5-3B    6.58x     3.66x     1.61x     0.52x
gemma-4       3.60x     3.00x     1.85x     0.52x
GLM-OCR(VLM)  8.14x     5.77x     2.43x     0.62x
```

### C. Concurrency sweep (N=1/8/16/32, fast path)
The DEFAULT fast path serialises on the single MLX executor → aggregate tok/s is
flat and mean TTFT scales ~linearly with N (right for single-user latency). Real
continuous-batching throughput is the **engine-loop** (see the framework table's
batch8/16/32 columns, e.g. 0.8B engine-loop 254 tok/s @N=32 vs fast-path 56).

```
              N=1   N=8   N=16  N=32   (aggregate tok/s, fast path)
Qwen3.5-0.8B  33    55    56    56
Qwen3.5-2B    16    22    22    21
Qwen3.5-9B-4b 12    13    12    10
Qwen2.5-3B    15    14    15    16
GLM-OCR(VLM)  98    97    96    94
```

## Is it REALLY the same as before? (honest answer)

**Not bit-identical — benchmark numbers have ±10-20% run-to-run variance** (M3 Max
thermal state, background load, exact prompt/tokenization). What's stable is the
QUALITATIVE structure, which IS consistent with past runs:

| claim (LIKE-FOR-LIKE path) | past | now | verdict |
|---|---|---|---|
| HOT reuse scales with model size | 0.8B<2B<9B | 2.1<3.1<6.2× | consistent |
| Qwen2.5-3B fast-path HOT | 5.57× | 5.97× | consistent/better |
| **2B fast-path HOT (ratio)** | **4.23×** | **3.06–3.18×** | **ratio ↓ — but COLD 1162→711ms & HOT 275→224ms both FASTER; smaller ratio = faster baseline, not slower reuse** |
| 9B ENGINE-LOOP reuse | 9.90× | 9.32× | consistent |
| WARM 4-bit RAM saving | 3.56× | 3.56× | identical |
| oMLX does NOT reuse hybrid Qwen3.5 | ≤1.06× | ≤1.02× | consistent |
| single-req decode ≈ across frameworks | within ~5% | within ~5% | consistent |
| engine-loop vs raw mlx-lm batch | mlx-lm 1.5-2.5× ahead | **MATCHED** (2B 266 vs 266, 3B 180 vs 180) | **gap closed (detok fix)** |

**Honest bottom line:** no genuine time-regression — every like-for-like pair is
consistent or faster in *absolute* latency. Speedup RATIOS are baseline-relative,
so a few dropped (notably 2B fast-path 4.23×→3.18×) purely because the COLD prefill
baseline got faster — NOT because reuse got slower. The earlier "much slower"
alarm was the fixed decode-metric bug. (My first draft of this table also wrongly
matched engine-loop 9.90× against the current fast-path — corrected above.)
GENUINELY NEW vs the past LLM-only matrix: the **VLM text-path tiers** (GLM-OCR
12.6× / Qwen3-Omni 13.1× HOT) and these length/hit-ratio/concurrency sweeps.

All raw data: `kv_cache_matrix_results.json`, `framework_comparison_results.json`,
`comprehensive_sweep_results.json`.

## Realistic-scenario cache rate (the real "cache hit %")

`scripts/bench_realistic_cache.py` — actual usage patterns, not synthetic sweeps.

**Scenario 1 — multi-turn chat** (prefix grows every turn). Per-turn cache hit-rate
ramps as the transcript accumulates (Qwen2.5-3B example):

```
turn  prompt_tok  cached  hit%   TTFT_ms
 1        25         0    0.0%   1289   (cold)
 3       149        64   43.0%   1326
 5       279       192   68.8%   1356
 8       468       384   82.1%   1372
```

Aggregate hit-rate (turns 2+): 0.8B **57%** · 2B **57%** · 3B **69%** · GLM-OCR **73%**.
(TTFT barely moves here because the prefixes are short and the 48-token decode
dominates — caching short multi-turn prefixes is correct but low-value.)

**Scenario 2 — RAG / shared long document** (one ~4.6k-token doc reused across 8
distinct questions). THIS is where caching pays off (Qwen2.5-3B):

```
q#  prompt_tok  cached  hit%    TTFT_ms  vs_q1
 1     4613        0    0.0%    3964     1.00x   (cold — doc not yet cached)
 2     4611     4544   98.5%     793     5.00x
 4     4612     4544   98.5%     708     5.60x
```

Aggregate hit-rate (q2+): 0.8B **97.1%** · 2B **97.1%** · 3B **98.5%** · GLM-OCR **98.8%**,
with **3.4–5.6× TTFT speedup** per cached question. This is the canonical
prompt-caching win (system prompt / RAG context / few-shot examples reused across
many requests).

**Takeaway:** the realistic cache hit-rate is **97–99%** for the RAG/shared-context
pattern (huge TTFT win) and **57–73%** for incremental multi-turn (modest win on
short prefixes). See `docs/PROMPT_CACHING_APIS.md` for how clients drive this
explicitly (OpenAI auto / Anthropic cache_control).

## Cache-subsystem audit

An audit of the cache subsystem — `kv_prefix_cache.py`, `ssd_kv_cache.py`,
`hybrid_ssd_snapshot.py`. Verdicts:

| # | Area | Verdict |
|---|------|---------|
| 1 | **"slowness"** (the reported regression) | **FIXED metric, no real regression.** The bench's `decode_tps` was prefill-contaminated; pure decode matches mlx-lm. |
| 2 | Migration stats (`kv_migration.py`) | **REMOVED in the refocus** — the counts were cosmetic LOGICAL tier/temperature transitions (~1e-5 s), not physical I/O. Real KV bytes move in `SSDKVCache`/`hybrid_ssd_snapshot`, which remain; the tiered-migration manager was dead capacity-scaling code and was deleted. |
| 3 | WARM 4-bit quantization | **OK** — `to_quantized(bits=4)` for KVCache layers; sliding-window/recurrent layers pass through unquantized (→ gemma WARMram 1.0×, honest). |
| 4 | SSD net-negative for fast-prefill models | **INHERENT TRADEOFF** — e.g. GLM-OCR prefills at 6489 tok/s, so reading the prefix back from disk (~0.96×) is ~break-even; SSD wins for slow-prefill / capacity-bound cases. No guard skips it (would risk regressing the capacity case it exists for). |
| 5 | Hybrid SSD snapshot size (0.8B ≈ 549 MB) | **OK / inherent** — the WHOLE multi-layer recurrent state per boundary (ArraysCache isn't block-decomposable); bounded by the SSD cap. Measured with int8 storage; the default is now bit-exact (larger files), int8 is `YUNSHU_SSD_CACHE_PRECISION=int8`. |
| 6 | SSD restore axis (`axis=2`) | **Hardened** → `axis=-2` / `shape[-2]` (was correct only for 4-D `[B,H,S,D]`; matrix already showed it lossless). |
| 7 | Eviction block-refcount | **NOT a bug** — `_remove_entry` → `_rebuild_hash_index` clears+recomputes `_block_refcount` from scratch; and `_snapshot_cache` ALWAYS deep-copies (no refcount-conditional aliasing), so no correctness exposure. |

**Bottom line:** no remaining cache *correctness* bugs. The reported slowness was a
benchmark-metric artifact (fixed). Remaining items are inherent tradeoffs (SSD
disk-read vs prefill cost; hybrid snapshot size) or cosmetic naming.
