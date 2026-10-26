# Review branch handoff (2026-10-26)

Branch `claude/confident-franklin-3pcza4`, based on `main` at `47a87ab`. It is the result of a parallel review of the
whole repository (engine, APC, gateway routers, auth/sandbox, Realtime/audio, console, host-side performance, project
hygiene, upstream/peer projects in `vendor.json`) followed by fixes and a cross-path parity pass. It was produced in a
Linux cloud container (MLX CPU backend, no Metal, no real checkpoints), so everything that needs Apple GPU numbers or real
models is listed under [Verify on the Mac](#verify-on-the-mac) instead of being claimed.

`docs/research/` is gitignored and was not available here: add this branch to `docs/research/INDEX.md` (and the
decisions below to `DECISIONS.md` once made) when you merge.

## How to merge

* Every fix is its own commit with a regression test (fails before, passes after) or, for optimisations, a bit-exact
  equivalence test against a verbatim copy of the old code. Work landed in lanes, each a `Merge ...` commit on this
  branch; `git log --merges 47a87ab..claude/confident-franklin-3pcza4` lists them. A lane or a single commit can be
  dropped or cherry-picked on its own.
* The local path-unification work will conflict with the engine lanes. Before resolving, read
  [What a unified path must keep](#what-a-unified-path-must-keep): the fixes were made on **every** serving path so that
  none is lost whichever path survives.
* `python/yunshu_engine/path_matrix.json` was updated with the branch (rows and cells below) and
  `tests/unit/test_path_parity.py` is green. Keep it green through the unification: it is the checklist.
* `docs/CONFIGURATION.md`, the console configuration pages and `settingdesc.ts` (en / zh-TW / zh-CN) were regenerated
  for every setting change; rerun `uv run python scripts/gen_config_docs.py` after resolving conflicts in `settings.py`.
* `uv.lock` changed (mlx-lm 0.32.2, removed unused packages): run `uv sync` after merging.

## What a unified path must keep

The serving paths in `path_matrix.json` are `mtp`, `dflash`, `batch`, `generic` (VLM runner) and `text`, `loop` (text
engine; the text path also has the n-gram and draft-model speculative sub-paths). `driver` was deleted (measured loser).
The per-fix, per-path status with `file:line` evidence is in:

* [`PATH_PARITY_VLM_SIDE.md`](PATH_PARITY_VLM_SIDE.md): text-path fixes checked and ported to `mtp`, `dflash`, `batch`,
  `generic`.
* [`PATH_PARITY_TEXT_SIDE.md`](PATH_PARITY_TEXT_SIDE.md): VLM-path fixes checked and ported to `text`, n-gram,
  speculative, `loop`.

New `path_matrix.json` features added by this branch (every path has a cell; a unified path must keep each engaged):

| feature | what it guarantees |
|---|---|
| `grammar_prewarm` | llguidance token tables built in the background at model load (`YUNSHU_GRAMMAR_PREWARM`) |
| `fast_top_logprobs` | top_logprobs without a full-vocabulary argsort or per-id syncs; each path keeps its tie order |
| `special_token_text` | the chat template's own structural special-token strings (turn/role markers it writes) typed in `user` / tool-result text stay text in every render, including count/preview routes; `system`, `developer` and `assistant` text (and `tool_calls` arguments) is never touched |
| `empty_prompt_refusal` | a zero-token prompt raises `EmptyPromptError` (HTTP 400 `empty_prompt`) before generation |
| `finite_state_publish` | a checkpoint with non-finite recurrent state is never stored in a prefix cache |
| `grammar_mask_fast` | constrained decoding masks from the bitmask / `allowed_mask`, not a ~240K-id list per token |
| `generation_suppress_default` | `generation_config.json` `suppress_tokens` is the default mask when the request sets none |
| `grammar_dead_end_isolation` | a grammar dead end fails only its own request (no hang, no batch-wide failure) |

Shared helpers written for the unified path to reuse: `text_logit_processors.py` (dense penalty / bias / ban vectors),
`text_logprobs.py` (`top_logprobs_rows`, `top_logprobs_device`), `special_tokens.py`, `generation_suppress.py`,
`transient_errors.py`, `context_window.require_non_empty_prompt`, `json_schema.apply_allowed_mask_from` /
`allowed_membership`, `text_prep.py` (render + encode on one dedicated thread).

## Changes by area

| area | main changes |
|---|---|
| Security | per-key actor and admin scope (an infer key no longer acts as admin over other keys' responses/containers/cancels); loopback servers refuse foreign `Host` and cross-site state-changing requests (DNS rebinding); failed-auth backoff (20/min per IP, 429); request-supplied MCP connector cannot reach private/loopback addresses by default; code-execution editor I/O confined with `O_NOFOLLOW` and non-regular files refused; decoded request media in a private 0700 dir denied to srt; special-token injection neutralised (template-structural strings in user/tool text); `YUNSHU_MCP_SERVERS` masked as secret |
| Gateway | pure-ASGI middleware stack (~120 → ~8 µs per streamed chunk, CPU); `ClosingStreamingResponse` on every inference stream; native tool schemas counted in the token budget; Responses replay rebuilds one assistant turn; missing `previous_response_id` ancestor is a 404; malformed Anthropic `tool_choice` is a 400; Anthropic thinking budget kept below `max_tokens`; 4 KB `input_json_delta` frames; whitespace-only content accepted; requests without `max_tokens` skip the full prompt encode; dead `usage_context` billing box removed |
| Engine (VLM) | a grammar dead end fails only its row; jobs popped from pending always get a terminal; terminal stream items never dropped by a full queue; temp media cleaned on early failure; orphan key hashed once; non-stream timeout is a total deadline; a cancel after disconnect stops the parked orphan; bounded text-prompt cache; `generation_config` suppress_tokens; think tail stays reasoning; empty prompt refused; non-finite state never published; usage counts consumed tokens; batch top_logprobs by device argpartition; token masks built once per vocab; out-of-range `logit_bias`/suppress ids ignored (mlx-lm wrapped `-1` onto the last token) |
| Engine (text) | no default 300 s deadline; wired limit sized correctly; dense penalties/bias/ban; argpartition top_logprobs; inflight register takes the list; llguidance `allowed_mask`; render/tokenize off the event loop; byte-once block hashes; auto-quantised KV not stored in token-keyed caches (fast, stream); the VLM-side fixes above ported (see TEXT_SIDE doc); per-row dead end in the loop; n-gram draft filter tests membership through the mask |
| APC / KV | SSD token tuples pruned with their files; failed WARM promotion keeps the entry; whole-prompt divergence memo; best-fit bounded restore buffers; probe before restore; transient SSD load errors keep the file; manager hit stats count accepted restores only; MTP lane clears the buffer cache on every 256-token boundary (multi-token rounds skipped the old check) |
| Realtime / audio | contiguous VAD silence window; idle silence trimmed; partial assistant item kept on cancel/barge-in; store:false failed turn keeps the last good response; ASR off the receive loop (ordered lane); Silero on the MLX thread; `ws_max_size` follows `YUNSHU_MAX_REQUEST_SIZE`; numpy resample / G.711 (byte-identical); single `response.audio.done`; `input_audio_buffer.cleared`; item ids on VAD events; GA transcription session type; `input_audio_transcription` honoured; explicit TTS `max_tokens`; empty audio → empty transcription |
| Console | Logs tail resets after an engine restart, memoised newest-300 rows; one shared recent-requests / memory poller; fetch timeouts and queued kicks; history refresh after restart; range change clears old points; Escape guard; queued manual refresh; Playground deltas once per frame and Stop cancels on the engine; curl snippet from the connection; token remembered only when applied; CSP `img-src 'self' data: blob:` (no remote images from model output); unused LiveScroll removed |
| Cleanup | `YUNSHU_ROUND_DRIVER` + `round_driver/` and `YUNSHU_MTP_ROW_EXACT` deleted (measured losers, per the flag rule); orphan modules deleted (voice_pipeline, mtp_decoder, mtp_patch, n_confirmed_patch, spec_interface, spec_proposer, bfcl_eval, gateway/engine/engine, gpu_ngram); 10 unused dependencies removed (and the duplicate `dev` extra); mlx-lm ≥ 0.32.2; dead KV event bus removed; retired-document citations removed; `YUNSHU_API_KEY` registered; README zh-TW/zh-CN passage restored; CHANGELOG links, justfile app path, layout lists fixed; doctor minimums follow pyproject; behaviour tests for previously untested modules |
| Upstream adoptions | every peer/upstream item from the review handled row by row in [`UPSTREAM_ADOPTION_2026-10-26.md`](UPSTREAM_ADOPTION_2026-10-26.md) (adopted with commit, not applicable with reason, or Mac step): late system notes in place and cache-store opt-out (omlx), seam checkpoint flag (TensorFold / Rapid-MLX), DFlash window capture only for stored checkpoints, prefix-cache snapshot integrity (ollama), forced think close and grammar after reasoning (omlx, mlx2, vllm, ollama), cleanup steps never strand the terminal (mlx2), row_exact K=64/128 and 2 GiB guards (mlx2), chunked head_dim-256 SDPA flag (omlx), GDN eps and `_release()` readiness (mlx-vlm), mlx-audio fixes, Gemma 4 images (mlx-vlm), greedy top_k 1 (splash), budget marker in system text (omlx) |
| Review of the branch | four independent reviews of this branch's own diff and a completeness audit found regressions that are fixed here: Realtime auto-commit lost every utterance after the idle trim (snapshot at commit time), per-row dead-end isolation was not engaged for real guides, per-boundary sync in hybrid cold prefill, the auto-quant gate disabled prefix caching for dense models, `n_prompt` offset (min_tokens never engaged), overly wide special-token neutralisation, failed-auth backoff blocking correct tokens, host-guard false positives, 22 overwritten parser tests restored, two benches restored, round driver archived in `docs/archive/round_driver/` |
| Upstream | `vendor.json` review recorded for 2026-10-26 (KEEP for omlx `qwen35_verify_sdpa_split`, quiet watches advanced, `ollama`, `mlx-swift-lm`, `vllm-metal`, `mlxcel` added to `watch`); GPU sampler A/B script `scripts/research/ab_gpu_sampler.py` |

## Behaviour changes clients and operators will see

* Loopback-bound servers answer 403 to a non-loopback `Host`, and to POST/PUT/PATCH/DELETE with a foreign `Origin` that
  is not in `YUNSHU_CORS_ORIGINS` (requests with `Authorization` / `x-api-key` are exempt). New setting
  `YUNSHU_ALLOWED_HOSTS` adds names (`*` disables). `yunshu serve --host` now records the bind host in `YUNSHU_HOST`.
  Accepted `Host` forms on a loopback bind: `localhost` (also `localhost.` and `*.localhost`), any 127.0.0.0/8 form
  (`127.0.0.1`, `127.1`), `[::1]` and its mapped form, and `0.0.0.0` / `[::]`, any port. `host.docker.internal` is not
  accepted (a container reaches the host's loopback only when the server listens beyond it, which turns the check off).
  A reverse proxy, `tailscale serve` or `ssh -L` that forwards a public name needs it in `YUNSHU_ALLOWED_HOSTS`; the 403
  names the header value and the setting. `Origin: null` stays refused; custom schemes (`vscode-webview://*`, `app://.`)
  are allowed by listing them in `YUNSHU_CORS_ORIGINS` (`scheme://*` matches every origin of a scheme). Clients that send
  `Authorization` / `x-api-key` are unaffected.
* Managed API keys are their own actors: an infer-scope key no longer reads/cancels objects of other keys or of the
  static token. `/metrics` answers 401/403 JSON
  envelopes (it was a 500), and with managed keys but no static token it needs an `admin`-scope key (it is open only
  when no auth is configured at all).
* Failed-auth backoff: 20 WRONG credentials per minute from one non-loopback client → further wrong credentials get 429
  with `Retry-After`. A correct token or key always passes and resets that client's count; a request with no
  credential at all is a plain 401 and is not counted. The client is the peer IP, or the last `X-Forwarded-For` entry
  when the peer is listed in `YUNSHU_TRUSTED_PROXIES`.
* `YUNSHU_MCP_CONNECTOR_ALLOW_PRIVATE` defaults to false: request-supplied MCP servers on private/loopback/link-local
  addresses are refused. Operator-configured servers are unaffected.
* Special-token strings in `user` and tool-result text get a zero-width space after their first character, so the
  model sees text, not a control token. Scope: only the special tokens the chat template itself writes as structure
  (found literally in the template, or the `bos_token`/`eos_token`... it references), so `<pad>`, `<unk>` and the like
  are left alone; only outside-input roles (`user`, `tool`/`function`/`ipython`, unknown roles), never `system` /
  `developer` (operator text) or `assistant` content and `tool_calls` arguments (replayed model output: for models
  whose tool/channel markers are special tokens, e.g. Mistral `[TOOL_CALLS]`, defusing them would corrupt history).
  An unreadable template falls back to every special string. Raw `/v1/completions` string prompts are not changed.
* Text-only requests have no default generation deadline (was 300 s total).
* Zero-token prompts → 400 `empty_prompt` on every path. Whitespace-only user/system content is accepted.
* Responses: a missing ancestor in a `previous_response_id` chain → 404 `previous_response_not_found`, whose message
  names the missing id, the in-memory store limit (1024, cleared on restart) and the fix (resend `input`);
  failed/cancelled parents replay their input only. Coalescing of assistant items also applies to client-supplied
  `input` (plain valid histories are unchanged; tests pin it).
* The sandbox text editor follows symlinks the sandboxed code created when the target stays inside the container
  (relative, `..` below the root, `/mnt/data/...`); links that leave it, and loops, are still refused with a message
  that says why.
* Without a client `max_tokens`, the full encode is skipped only while a UTF-8 byte upper bound still leaves at least
  32768 tokens (or the whole default) for the answer; then `x_yunshu.budget.prompt_tokens` is that bound and
  `prompt_tokens_basis` says `upper_bound`. Otherwise the exact count is used, so small windows are not over-clamped.
* Auto length-gated 8-bit KV starts at the offset where the KV estimate reaches the gate (not token 0, and not decided by
  `max_tokens`), and a request skips its prefix/prompt-cache store only when its cache actually holds quantised layers.
* `YUNSHU_GRAMMAR_PREWARM` (default on) builds one llguidance table at the tokenizer width (about 0.45 GiB native heap);
  the forced-tool grammar's padded-width table is built on its first use. Turn it off on tight hosts.
* `suppress_tokens` (request or checkpoint `generation_config.json`, also for hub-id models) no longer takes a request
  off the n-gram / Gemma-4 speculative routes; they ban the ids themselves. `min_tokens` / `ignore_eos` still do.
* Text fast/stream paths count `min_tokens` and frequency/presence penalties from the length of the ids given to
  `generate_step` (mlx-lm 0.32.2 hands its processors the whole prefilled suffix).
* Structured output and reasoning: when the prompt opens `<think>` (thinking on), the grammar stays dormant until the
  reasoning closes and then binds from the first answer token, on every path (EOS and the implicit `<tool_call>` end are
  masked inside the reasoning); a request whose reasoning never closes returns empty content, finish `length`, and
  reports `enforced=false, reason=reasoning_not_closed`. When the grammar binds from the first token (thinking off),
  the output is never relabelled reasoning. The forced thinking-budget close is the template's own `\n</think>\n\n`.
* Removed settings: `YUNSHU_ROUND_DRIVER`, `YUNSHU_ROUND_PREFILL_CHUNK`, `YUNSHU_MTP_ROW_EXACT`. `YUNSHU_GPU_SAMPLER` is now
  marked experimental (`decide=` points at the A/B script).
* Console: remote images in model output are blocked by CSP; Logs shows the newest 300 rows with "show earlier" and,
  while you are scrolled up, grows that window by what arrives so the row you read does not move.
* Console: an engine restart in Logs no longer wipes the screen: the old process's lines stay in a collapsed "previous
  engine process" block above the list (the crash trace is not in the new ring); changing a filter clears it.
* Console: Escape in the Requests inspector yields to a focused text field (it no longer closes the inspector while
  you are typing).
* A system/developer note sent right after a tool result (Claude Code hook/reminder notes) stays in place as a
  `[System note] … [/System note]` **user** turn instead of being hoisted into the leading system block (omlx
  5a0a15a7), so earlier turns keep a stable prefix; Claude Code's `<total_tokens>` budget marker is dropped from system
  text as well as from lone messages (omlx 38f420d9). See decision 5.
* `X-Yunshu-Cache-Store: off` (request header): the request reuses cached prefixes but stores nothing (omlx c4bb311e).
* `GET /v1/yunshu/config` needs the admin scope once a static token or managed keys exist (open with no auth), and
  masks `user:pass@` in URL values. Settings that run host code or reach arbitrary host paths (`host_access`: path
  settings, `YUNSHU_MCP_SERVERS`, `YUNSHU_TRUST_REMOTE_CODE`, `YUNSHU_CODE_EXECUTION`, `YUNSHU_ALLOW_LOCAL_FILES`) are
  read-only over `PATCH` for managed keys unless `YUNSHU_ADMIN_WRITE_EXEC_SETTINGS` is set on the host; the owner
  (static token or no auth) can still write them. `yunshu serve --auth-token auto` generates and prints a token;
  startup warns on a token shorter than 16 characters and on a non-loopback bind with no auth.
* Structured output `strict`: `true` closes every object (`additionalProperties: false` at any depth) on Chat and
  Responses alike; `false` / omitted leaves the schema as sent (Responses used to default it to true).
* Anthropic `thinking: {type: "enabled"}` on a model with native reasoning (model card) no longer injects a `<think>`
  instruction and regex grammar; models without native reasoning keep them.
* Responses accepts `max_tokens` as an alias of `max_output_tokens` (a conflicting pair is a 400).
* `top_k: 1` or `min_p: 1` requests decode greedily on every path (splash 8d5e05b); XTC keeps a request sampled.
* A stop string that completes in the final detokenizer flush is reported as a stop (Anthropic `stop_sequence`).
* WebRTC Realtime under backpressure drops incoming audio frames (counted) instead of hanging up.
* New experimental flags (default off, each with `decide=` and an A/B script): `YUNSHU_APC_SEAM_CHECKPOINT`
  (checkpoint at the document/question seam, TensorFold #1262 / Rapid-MLX 421bf658) and `YUNSHU_SDPA_CHUNK`
  (query-chunked head_dim-256 prefill attention on M1-M4, omlx sdpa256). Four experimental flags exist (limit 8).
* mlx-audio fixes carried as local patches until the bump (Whisper `no_speech` on older vocabularies, Kokoro voices
  from the loaded snapshot, Qwen3-ASR per-chunk language); Gemma 4 / EmbeddingGemma 2 PIL images keep channels-last.

## Verify on the Mac

Nothing here ran on Metal or with a real checkpoint. Before merging, run on the M5 (and M3 lane where relevant):

1. The full unit suite including the Metal-only tests that cannot run here (`test_constrained_spec`,
   `test_mtp_lane_tool_guide`, `test_coldfuse_*`, `test_walk_gpu`, `test_commit_fused_tail`, `test_omlx_verify_kernels`,
   `test_fast_tree_wide`, `test_invariant_wide_decode`, `test_yv_verify`, gpuq tests) — in particular **spec on == off**
   after the spec-lane changes (penalty context tail, top_logprobs rows, processor hygiene).
2. `scripts/realmodel/test_real_model.py` and the yv identity checks (APC hit == miss, spec on == off) on Qwen3.8-27B.
3. Perf A/B for the changes whose CPU numbers do not transfer: wired limit now applied on the text path (`71c183e`); VLM
   batch `top_logprobs_device` (uint64 keys + argpartition on Metal; it falls back to the upstream sort if the probe
   fails); pure-ASGI middleware; text prep thread; dense penalties; llguidance prewarm memory.
4. `scripts/research/ab_gpu_sampler.py --checkpoint $TEXT_CKPT --src python --out ab_gpu_sampler.json` to settle
   `YUNSHU_GPU_SAMPLER`.
5. Control-token neutralisation with the real Qwen tokenizer (`tok.encode("<|im_end|>")` inside user text must not yield
   the special id after rendering), and with a Mistral-style tokenizer (`[TOOL_CALLS]` in a replayed assistant turn keeps its
   control id).
6. Console: `pnpm test` including the bundle-size check (route budget raised 304 → 305 KB in `3e8a442`, measured once at
   304.2 KB; revert if you prefer trimming), the full Playwright suite and WebKit.
7. mlx 0.32.4 (gates mlx-vlm ≥ 0.7.9 and mlx-audio 0.6): run the row-invariance microtest (`mx.quantized_matmul` over
   M ∈ {1, 2, 7, 16, 64, 65, 512}, 0.32.3 vs 0.32.4) to find the FP32 SSD APC token-identity break; bump with mlx-vlm
   0.7.10. The eps (`kernels/gdn_qk_eps.py` measures the installed `_normalize_qk`), the `ScaledQuantizedLinear` verify
   rows and the post-`_release()` restore-buffer offer are already prepared; after the bump check fused GDN norm ==
   stock bitwise, cold == warm, spec on == off, and that restore-buffer reuse still engages.
8. Kernel guards adopted from peers: `row_exact_qmv` declines K = 64 / 128 (4- and 8-bit, group 32/64/128) and the
   per-row fallback matches; an untiled weight near the 2 GiB slice offset is declined to stock on M5 / macOS 26.6.
9. Experimental A/Bs: `scripts/research/ab_seam_checkpoint.py` (`YUNSHU_APC_SEAM_CHECKPOINT`) on 27B, and on an M1-M4
   Mac a bitwise A/B of `YUNSHU_SDPA_CHUNK` against stock head_dim-256 SDPA plus prefill peak memory.
10. mlx-audio patches with real checkpoints: Qwen3-ASR on > 20 min two-language audio, whisper-small with the patched
   `no_speech`, Kokoro loading a local voice offline.
11. Remaining peer rows that need a Mac are listed with their step in
   [`UPSTREAM_ADOPTION_2026-10-26.md`](UPSTREAM_ADOPTION_2026-10-26.md).

## Decisions needed

1. `YUNSHU_ENGINE_LOOP` (`engine_core.py`, `scheduler.py`, ~14k lines) conflicts with the "no batching race" non-goal:
   keep or delete. This branch fixed its bugs and brought it to parity so either answer is safe.
2. `YUNSHU_TOOL_GRAMMAR`: the recorded result (lossless 12/12, no failures removed, decode −0..10%) reads as "delete",
   but later work let guided requests keep the fast kernels, so the cost measurement may be stale.
3. Penalty semantics: the text path counts all generated tokens (OpenAI semantics); the VLM path uses mlx-lm's window of
   the last 20 prompt+generated tokens. A unified path needs one; aligning the VLM path to OpenAI changes 27B output
   when penalties are set.
4. top_logprobs tie order: text = descending value then ascending id; VLM = descending value then descending id (both
   preserved by this branch). Pick one for the unified path.
5. Late system notes (after a tool result) now render as a `[System note]` user turn to keep the prefix stable
   (adopted from omlx 5a0a15a7). Confirm the role change is acceptable, or pick another placement. The original issue:
   mid-conversation system/developer messages were hoisted into the leading system block
   (`message_adapter._hoist_system`), so a system note after a tool result changes the prefix and re-prefills
   everything after it on every turn (omlx `5a0a15a7` found the same). Options: keep the note in place where the
   template allows it, or append it to the trailing turn.
6. Files API and conversations have no owner concept yet (per-key isolation covers responses, containers and cancels).
7. Chat `max_tokens: 0`: this branch keeps the documented prompt-count probe (it answers usage with no output); OpenAI
   rejects values below 1. A commit aligning it to OpenAI was reverted (`91ed94d`); decide whether to drop the probe.
8. `xxhash` is an optional import in `yunshu_kv/hash.py` with a blake2b fallback that produces different hash values:
   declare it as a dependency (stable keys everywhere) or drop the optional path.
9. Deliberate non-ports, kept as they are: a plain disconnect after Stop keeps the 120 s orphan window (only an explicit
   cancel stops it; the console's Stop now sends one); the VLM path renders/tokenizes on the MLX thread (moving it would
   change admission order); the VLM path sets no wired limit (needs a Mac A/B, and `idle_memory.py` already manages it).
10. Auto length-gated 8-bit KV on the text path now starts at the threshold offset instead of token 0 (less lossy, but
   long requests' outputs change versus before); confirm.

## Not done, and known limits

* Orphan adoption on the text path does not exist and was not built.
* The draft-model speculative path drives its constraint through allowed-token lists (cross-model decoder).
* A local reverse proxy or `ssh -L` that adds no forwarding header is indistinguishable from a local client for the
  observe-mode loopback check.
* The active-request counter still decrements at response start, so streams are not counted during graceful drain
  (unchanged behaviour).
* Realtime ordered lane: a second utterance finished before the previous commit's ASR completes can lose its start; a
  `response.cancel` while `response.create` waits behind an ASR gets "no active response".
* Logs restart detection misses a restart if the new process logged more records than the old cursor before the probe.
* The streaming n-gram fallbacks (non-trimmable cache) do not forward `suppress_tokens`, `lora_adapter` or
  `timeout_seconds`; `_spec_route` never sends streaming requests there, so they are unreachable today.
* Logs window growth while scrolled up does not compensate when the 4000-row client cap trims from the front.
* Script candidates for deletion (unreferenced but runnable): `scripts/bench/{_bench_concurrency,_bench_concurrency_omlx,
  bench_comparison,bench_kv_prefix,bench_memory_raw,bench_memory_v2,bench_mmlu_compare,bench_perf_compare,
  bench_throughput,bench_ttft_probe,bench_unified,fair_bench,profile_engine_loop}.py` and `scripts/realmodel/{smoke_*,
  test_all_models,test_inversion,test_ocr_real,test_vlm_engine,verify_ane_embeddings}.py`.
* `kernels/verify_select.py` is used only by research scripts and one test; earlier script candidates were listed in
  the cleanup lane's commit messages (`scripts/bench`, `scripts/realmodel` unreferenced but runnable).

## Test evidence

### Full-suite comparison

`pytest -n 6 tests/unit` in the same container (Linux, MLX CPU backend, shared venv synced to this branch's lock):

| tree | failed | passed | skipped |
|---|---|---|---|
| `main` `47a87ab` | 135 | 12372 | 346 |
| this branch (after the parity lanes) | 134 | 12611 | 325 |
| this branch (final, after the adoption and review-fix rounds) | 134 | 12994 | 326 |

* No test fails on the branch that passes on `main`. The 134 failures are the environment set on both trees (Metal-only
  kernels, real checkpoints, gpuq, zsh, srt, tfbench, timing tests under load).
* One `main` failure is fixed (`test_doctor_checks::test_version_guard_ok_and_missing_packages_are_not_version_failures`,
  doctor minimums now follow pyproject).
* About 620 more tests pass than on `main`: new regression and equivalence tests for every fix, minus the tests of the
  deleted modules (the 22 tool-family parser tests that one cleanup commit overwrote were restored in `1d5df01`).
  The final run's only extra failures were `test_api_parity` (fixed in the next commit) and a CPU-timing test that
  passes alone (`test_priority_queue::test_sort_baseline_comparison`).
* `tests/unit/test_path_parity.py` is green with the new feature rows; `ruff check` / `ruff format` are clean.
* Not covered here: Metal, real models, the console bundle-size build and the full Playwright/WebKit suite (see
  [Verify on the Mac](#verify-on-the-mac)).

### After the parity lanes

* The EngineCore `loop` got the same dense penalty / logit_bias / suppress / min_tokens processors and real
  `top_logprobs` entries (`aed9970`); before, the raw vocab-wide logprobs array went into every output whether or not
  logprobs were requested. Every row of the table in [What a unified path must keep](#what-a-unified-path-must-keep) is
  now engaged on every path.
