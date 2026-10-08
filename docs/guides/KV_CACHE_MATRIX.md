# KV cache: current serving paths and measured APC tiers

Current architecture on main in the **0.1.5 cycle**. The tier measurements below are historical:
2026-10-02, M5 Max / 128 GB, Qwen3.8-27B oQ4e-mtp; measurement record `d00f61d7`,
identity record `63b00e16`, merge `57272ed7`. Results are replay summaries, not confidence
intervals from repeated sessions. The old M3 Max pre-runner VLM tables are historical;
see [legacy VLM notes](../archive/legacy_vlm_loop/README.md), not current routing guidance.
Scripts used: `scripts/research/apc_audit/session_replay.py`, `queue_tier4.sh`,
`tier_report.py`, `tier_roofline.py`, `tier_breakeven.py`, `tier_capacity_model.py`.
[Benchmark methods and selected results](../BENCHMARKS.md).

## Which cache tiers each serving path has

The VLM APC and text fast-path prefix caches have distinct settings and defaults.

| Serving path | Serves | RAM, full precision (HOT) | RAM, compact (WARM) | SSD | Nothing cached (COLD) |
|---|---|---|---|---|---|
| **VLM runner** (`VLMEngine` + `vlm_batch_runner.py`, mlx-vlm APC) | every mlx-vlm model, Qwen3.8-27B included; all `/v1/chat/completions`, `/v1/messages`, `/v1/responses` traffic for them | yes: exact checkpoints (KV + recurrent state), byte budget `YUNSHU_VLM_APC_MEMORY_GB` | **opt-in, off by default** (`YUNSHU_VLM_APC_WARM`): `lossless` (zstd, bit-exact) or lossy `int8` / `int4` K/V; takes `YUNSHU_VLM_APC_WARM_SHARE` of the APC RAM budget | **yes, on by default**, lossless (bit-exact states), `~/.yunshu/cache/apc`, up to 64 GiB default cap for the whole directory (all models together): `YUNSHU_VLM_APC_DISK=0` opts out, `YUNSHU_VLM_APC_DISK_DIR` / `YUNSHU_VLM_APC_DISK_GB`; further storage tiers below it (external SSD, HDD, NAS): `YUNSHU_VLM_APC_DISK_TIERS` | full prefill |
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
(a hit restores the stored checkpoint exactly; cold/hit parity also depends on the prefill plan):

- the byte budget (not a two-entry cap) limits RAM; the default budget is half of the memory left after weights and an OS/activation
  reserve (a quarter of RAM, clamped to 4–16 GiB), capped at a quarter of RAM and
  32 GiB; it disables caching if less than 1 GiB remains (a 27B checkpoint costs about 130 KiB per cached token);
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

Measured 2026-10-02 on the Qwen3.8-27B oQ4e (M5 Max, 128 GB), real checkpoints of 32K-token documents (code and prose),
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

| Configuration | prompt tokens served from cache | hits (ram / warm / ssd) | TTFT mean / p50 / p90 / max | peak RSS (GiB) |
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

The all-RAM and single-SSD configurations hit the same 86.3 % ceiling; capped multi-device configurations reach 85.7 % or 81.3 % (the first request of a session and the head are cold; each
turn adds 3K new tokens, 3.4 s of prefill that no cache removes). What differs is where the hit comes from
and what it costs. Reading the table:
- the SSD tier is the whole difference between 21 s and 6 s mean TTFT at 4 GiB; the HDD-only variant restores
  a 2 GB checkpoint in 9-11 s instead of recomputing it for 25-45 s;
- WARM, lossless or int8, adds nothing over the SSD on these runs: the mean TTFT is equal or worse (6.2-7.2 s
  against 6.0 s), and lossless WARM doubles the resident memory (peak RSS 9-10.5 GiB: HOT + compressed copies +
  the one entry being compressed + the decode buffers). The roofline said so before the run (the codec is not
  faster than an internal NVMe, and the end-to-end decode is slower than the codec);
- three devices (internal SSD, TB4 SSD, simulated HDD) with fast-tier caps below the working set (3 + 6 GiB; each
  session's newest checkpoint is 2.1 GB, plus its interval checkpoint) match the single big SSD (5.6 s against 6.0 s) because the mover demotes the least
  recently used checkpoints instead of deleting them and the cost model picks the cheapest tier per lookup.

**Output identity.** Every replay sends the same 27 greedy requests (8 new tokens each). Against the all-RAM
replay, every request served with the same cached length returned the same text: 24 HOT, 131 SSD hits (internal,
TB4 and the simulated HDD, raw and zstd files), 21 WARM-lossless hits, 7 cold requests; none differed.

**Disk supersede.** A request's checkpoints used to accumulate on the SSD tiers: the 3-session replay left 46
files and 63 GiB, the SSD cap (64 GiB) reached in five minutes. Once the cap or the RAM budget is small the cache
the next request needs is the one an LRU evicts first, so these stale copies cost hits in tiered
configurations (a first tiered run lost every session's newest checkpoint and recomputed 30-47 s requests).
A checkpoint written to any tier now drops the checkpoints of earlier requests that are prefixes of it, on every
tier: the same replay leaves 6 files and 10.8 GiB (the head plus each session's newest checkpoints) with the same
hits. Heads and anything this process neither stored nor restored are kept.

**Conclusions.**
1. The tier that pays is the SSD (and the storage tiers below it): without it the 4 GiB RAM budget on this 128-GB
   machine served 31.7 % of prompt tokens from cache with 21.2 s mean TTFT (44.7 s max);
   with SSD and disk supersede it served 86.3 % with 6.3 s mean TTFT (13.9 s max).
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

**Capacity estimates.** The replay used a 128-GB Mac with constrained APC budgets;
it did not test the 27B checkpoint on 16/32-GB hardware. Estimate capacity from the
actual model's checkpoint size and memory left after weights, OS/activation reserve,
and transient restore buffers; `yunshu doctor` checks the local machine. Do not infer
that the same checkpoint fits a smaller Mac from an artificially small cache budget.

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

Real agent sessions (Claude Code, Codex, opencode on the measured snapshot): reuse equals the achievable ceiling
for the request stream (Claude Code 89-92%, Codex 80-88%, opencode 64-73%; the rest is each session's
first, cold request, plus 0.3-3.7 points of generated tokens re-prefilled on the next turn).
