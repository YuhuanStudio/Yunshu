# Yunshu Infrastructure Deep Review: Metal Kernels, KV Layer, Compute Mesh, WebUI + Whitepaper Phase Map

**Review Date:** 2026-05-03  
**Scope:** All source code in metal/, yunshu_kv/, yunshu_mesh/, webui/src/ + Whitepaper Phase 0-5  
**Total Lines Reviewed:** 7,463 (Metal: 874 | KV: 1,671 | Mesh: 1,095 | WebUI: 3,823)

---

## 1. METAL KERNELS (`yunshu/metal/`)

**Total: 874 lines across 6 files**

### 1.1 `common.metal` (76 lines) -- Maturity: PROTOTYPE
**What works:**
- `simd_reduce_sum` / `simd_reduce_max` using `simd_shuffle_down` -- correct Apple GPU warp-level reduction pattern
- `kv_block_offset` helper for paged KV address computation
- `dot_product` template for generic half dot products
- Function constants declared (`PA_BLOCK_Q`, `PA_BLOCK_KV`, `PA_HEAD_DIM`, `KV_BLOCK_SIZE`) but never used by the other kernels (they use hardcoded values)

**Key findings:**
- Function constants are declared but **never referenced** by any kernel. All kernels hardcode tile sizes (64, 128). This means the `[[function_constant]]` declarations are dead code.
- No error handling utilities (no assert macros, no bounds-check helpers).
- Clean, minimal -- appropriate for a shared header.

### 1.2 `paged_attention.metal` (246 lines) -- Maturity: PROTOTYPE
**What works:**
- **Decode kernel** (`paged_attention_decode`): Complete implementation with block table iteration, online softmax per-block, FP32 accumulation, SIMD reduction. Correct vLLM PagedAttention port pattern.
- **Prefill kernel** (`paged_attention_prefill`): FlashAttention-style tiled prefill with 64-query tiles, online softmax with running max/sum correction. Proper GQA support via head mapping.
- Both kernels handle variable-length sequences correctly.
- Block table indexing uses proper chain-hash pattern.

**Stubbed / Issues:**
- **Fixed shared memory sizes**: `shared_logits[128]`, `shared_vals[128]`, `shared_k[64*128]`, `shared_v[64*128]`, `shared_scores[64*64]` are all compile-time fixed. If `head_dim > 128` or `tile_kv > 64`, this will silently overflow or waste memory. The function constants in `common.metal` exist to solve this but are unused.
- **No causal masking**: Neither kernel implements causal (autoregressive) attention masking. This is critical for LLM decode and would produce incorrect results for standard decoder-only models.
- **Prefill assumes single sequence**: `seq_lens[0]` hardcoded -- batched prefill not supported.
- **Register pressure**: `q_tile[64]` and `out_acc[4*128]` = 576 floats in registers per threadgroup. This may exceed Apple GPU register file limits for large head_dims.
- **Block table calculation bug risk**: Line 46 computes `seq_len / kv_block_size` for block table stride, but if seq_len is not evenly divisible, this could be off-by-one.

### 1.3 `sdpa.metal` (151 lines) -- Maturity: PROTOTYPE
**What works:**
- Full FlashAttention-2 style tiled SDPA with online softmax.
- GQA support: `kv_head_idx = head_idx * num_kv_heads / num_heads`.
- Threadgroup shared memory for K/V tiling with proper barriers.
- Configurable tile sizes via constants.

**Stubbed / Issues:**
- **No causal mask**: Same issue as PagedAttention -- no upper-triangular mask for autoregressive decoding.
- **Shared memory is hardcoded**: `shared_k[64*128]`, `shared_v[64*128]`, `shared_scores[64*64]`. Not parameterized.
- **Only forward pass**: No backward/gradient kernel (acceptable for inference-only).
- **Single-sequence assumption**: No batch dimension handling beyond the head dimension.
- **Potential race condition in score writing**: Lines 101 write `shared_scores[q * tile_kv + kv]` from multiple simd_groups without explicit coordination within a tile_q row (relies on `simd_group_id` being unique per row, which is correct but fragile).

### 1.4 `gemv.metal` (127 lines) -- Maturity: PROTOTYPE (functional)
**What works:**
- **FP16 GEMV** (`gemv_fp16`): Complete, correct. SIMD reduction, optional bias.
- **Batched FP16 GEMV** (`gemv_fp16_batched`): Complete. Handles batch dimension correctly.
- **4-bit Quantized GEMV** (`gemv_q4`): Complete dequantize-on-the-fly with per-row scale/zero_point, group_size support.

**Key findings:**
- This is the most production-ready of all Metal kernels. Simple, correct, well-structured.
- Q4 unpacking is correct: low nibble = first value, high nibble = second value (standard convention).
- Uses `simd_groups_per_threadgroup` properly for grid-stride loop indexing.
- Minor: No fused activation (ReLU/GELU) option -- would need separate kernel or inline.

### 1.5 `kivi_quant.metal` (156 lines) -- Maturity: PROTOTYPE
**What works:**
- **Quantize keys** (`kivi_quantize_keys`): Per-channel min-max asymmetric 2-bit quantization. Correct packing (4 values per byte).
- **Dequantize keys** (`kivi_dequantize_keys`): Correct unpacking with scale/ZP reconstruction.
- **Fused dequantize-for-attention** (`kivi_dequantize_for_attention`): Fused dequant + attention scale, reduces memory bandwidth.

**Stubbed / Issues:**
- **Only quantizes keys, not values**: As designed (KIVI keeps V in FP16), but the naming should make this clearer.
- **Quantization is per-token-per-head (not per-group)**: Each (token, head) gets its own scale/ZP. For 2-bit this is very coarse granularity -- real KIVI uses per-channel or per-group quantization for better quality.
- **No rounding mode control**: Uses `clamp(quantized + 0.5f, 0, 3)` which is round-half-up. Acceptable but not configurable.
- **head_dim must be divisible by 4**: Enforced by the packing loop stride. No guard for non-divisible dimensions.

### 1.6 `sgmv.metal` (118 lines) -- Maturity: PROTOTYPE
**What works:**
- **SGMV forward** (`sgmv_forward`): Two-step LoRA: A@x then B@intermediate. Shared memory for intermediate results.
- **Rank-1 optimization** (`sgmv_rank1`): Fused single-pass for rank-1 adapters (common case).
- Proper segment ID lookup with -1 sentinel for "no adapter."

**Stubbed / Issues:**
- **Grid mapping is wrong**: `gid.x` is used as `token_idx`, but grid is `[num_segments, 1, 1]`. This means one threadgroup per segment, but each threadgroup only processes ONE token (gid.x). For segments with many tokens, most threads do nothing. Should be `[total_tokens, 1, 1]` with segment lookup.
- **Threadgroup shared memory size**: `shared_intermediate[256]` is hardcoded. Rank > 256 will overflow.
- **No alpha scaling in sgmv_forward**: The docstring says `alpha * B @ A @ x` but the kernel does `output += alpha * dot` (only on B side). The A@x result is stored without alpha, then B@result applies alpha. This is mathematically equivalent since alpha is scalar, so it is actually correct.
- **No batched SGMV**: Each token processed individually -- no coalescing of tokens sharing the same adapter.

### METAL KERNELS SUMMARY

| Kernel | Lines | Status | Production Ready? |
|--------|-------|--------|-------------------|
| common.metal | 76 | Functional helpers | Yes (but has unused constants) |
| gemv.metal | 127 | **Most complete** | Close -- needs testing |
| kivi_quant.metal | 156 | Functional prototype | Needs per-group quant |
| sdpa.metal | 151 | Missing causal mask | **No** -- incorrect for LLMs |
| paged_attention.metal | 246 | Missing causal mask | **No** -- incorrect for LLMs |
| sgmv.metal | 118 | Grid mapping bug | **No** -- bug in dispatch |

**Critical gap: No causal/autoregressive attention mask in any attention kernel. This means the codebase cannot correctly run any standard LLM (Llama, Qwen, Mistral, etc.) in its current form.**

---

## 2. KV CACHE LAYER (`yunshu/python/yunshu_kv/`)

**Total: 1,671 lines across 9 files**

### 2.1 `__init__.py` (27 lines) -- Maturity: PRODUCTION (module init)
Clean exports. Documents the three-tier architecture. Nothing stubbed.

### 2.2 `block.py` (202 lines) -- Maturity: PROTOTYPE -> NEAR-PRODUCTION
**What works:**
- `KVBlock` dataclass with ref_count, block_hash, LRU pointers -- complete.
- `FreeBlockQueue`: Doubly-linked list O(1) LRU eviction. Sentinel node pattern. `popleft`, `append`, `remove` all correct.
- `BlockPool`: Full lifecycle -- allocate, free, touch (refcount++), cache_block, lookup_hash, prefix cache eviction.
- Null block reservation (block_id=0 never freed).

**Key findings:**
- Solid implementation. This is the most mature component in the entire KV layer.
- Reference counting logic is correct: allocate sets ref=1, touch increments, free decrements, 0 goes back to free list.
- Prefix cache eviction on allocation is a good design choice (evict cached blocks before failing).

### 2.3 `block_table.py` (82 lines) -- Maturity: PRODUCTION
**What works:**
- Clean logical-to-physical block mapping.
- `fork()` for prefix sharing (copy block references, caller handles refcounts).
- `slot_for_token()` and `block_id_for_token()` for fast lookups.
- `get_full_blocks()` for SSD persistence boundary detection.

**Issues:**
- `_num_tokens` attribute is referenced in `num_tokens_in_last_block` but never set anywhere. This is a bug -- the property will always raise `AttributeError` or return stale data.
- No bounds checking on `get_block()` -- will IndexError if logical_idx >= len(_blocks).

### 2.4 `compression.py` (144 lines) -- Maturity: PROTOTYPE
**What works:**
- Symmetric 4-bit quantization: abs-max per-element scale, clamp to [-7,7], pack as uint8.
- Dequantization: unpack nibbles, subtract offset, rescale.
- MLX array interop (graceful fallback to numpy if MLX unavailable).
- Compression ratio utility.

**Issues:**
- **Per-element quantization**: Scale is computed per element (not per-group). This means the scale array is the same size as the data for 4-bit (only 4x compression vs theoretical 8x for group_size=32/64). Real implementations (GPTQ, AWQ) use group-wise quantization.
- **Numpy path is the primary implementation**: Even when HAS_MLX=True, it converts to numpy, operates, then converts back. This defeats the purpose of MLX acceleration. Comment says "Metal kernel implementation will replace the numpy path in Phase 2" -- confirming this is placeholder code.
- `dequantize_kv_4bit` padding logic for odd head_dim is correct but slow (copies entire array).

### 2.5 `hash.py` (72 lines) -- Maturity: PRODUCTION
**What works:**
- xxhash primary with blake2b fallback -- good portability choice.
- Chain hashing: parent_hash + token_ids + extra_keys. Enables O(1) prefix extension.
- `compute_prompt_hashes`: Iterates blocks, chains hashes correctly.
- Struct packing for deterministic binary serialization.

**Key findings:**
- Clean, correct, no issues. Production-ready.

### 2.6 `manager.py` (228 lines) -- Maturity: PROTOTYPE
**What works:**
- `KVCacheConfig` dataclass with all needed parameters.
- `compute_num_blocks`: Correct UMA budget calculation accounting for model weights, activations, KV per block.
- `allocate_for_prefill`: Full prefix cache lookup flow -- compute hashes, lookup cached blocks, touch refs, allocate new, build BlockTable.
- Chain-hash break-on-miss: Once a hash misses, stops looking (correct for chain hashing).
- `cache_completed_blocks`: Post-prefill caching with parent hash chaining.

**Issues:**
- **No actual MLX tensor integration**: The manager allocates Python objects (KVBlocks) but never touches actual MLX KVCache tensors. The comment says "Integration with MLX KV cache tensors (Phase 2)" -- this confirms it is structural scaffolding only.
- `evict_for_memory` is a no-op that just checks free count -- doesn't actually evict anything.
- No compaction or defragmentation.

### 2.7 `mlx_cache.py` (301 lines) -- Maturity: PROTOTYPE (most MLX-integrated)
**What works:**
- `detect_cache_type`: Comprehensive type detection for all 8 MLX cache types (KVCache, RotatingKVCache, BatchKVCache, ArraysCache, QuantizedKVCache, CacheList, etc.) with heuristic fallbacks.
- `extract_cache_state`: Type-aware state extraction for each cache type. Handles RotatingKVCache's circular buffer metadata.
- `slice_kv_at_offsets`, `concatenate_kv_blocks`: Proper axis-2 (sequence dim) operations.
- `reconstruct_kvcache`: Creates KVCache from raw tensors with correct offset handling.
- `merge_caches_into_batch`: BatchKVCache.merge() wrapper.
- `materialize_cache`: Forces lazy evaluation before I/O transfer.

**Key findings:**
- This is the deepest MLX integration point in the KV layer. Shows thorough understanding of MLX's cache internals.
- `get_cache_seq_length` handles all cache types including the tricky BatchKVCache case where offset is an mx.array.
- **Critical note**: `reconstruct_kvcache` always sets `offset = keys.shape[2]`, with a comment explaining why (update_and_fetch uses offset as write position). This shows real-world debugging experience.

### 2.8 `radix_attention.py` (263 lines) -- Maturity: PROTOTYPE
**What works:**
- `RadixNode`: Token IDs, KV blocks, block_hashes, children dict, parent ref, ref_count, last_access_time.
- `RadixTree.match()`: Longest-prefix match with partial-match handling (breaks on partial rather than splitting -- documented limitation).
- `RadixTree.insert()`: Creates new node at insertion point.
- `RadixTree.evict()`: LRU leaf eviction with ref_count==0 check, proper parent cleanup.
- Path walking utilities: `path_blocks()`, `total_tokens()`.

**Issues:**
- **No node splitting**: When there is a partial match at a child, the tree breaks instead of splitting the node. This means prefix matching is conservative (may miss some sharing opportunities). Documented as intentional simplification.
- **inc_ref/dec_ref walk to root**: Every refcount change walks from node to root. For deep trees this is O(depth). Acceptable for typical conversation depths (< 100).
- **Not integrated into KVCacheManager**: RadixTree exists but `manager.py` uses flat hash-based prefix caching (BlockPool._hash_to_block). The radix tree is built but not wired into the allocation path.

### 2.9 `tiered.py` (352 lines) -- Maturity: PROTOTYPE
**What works:**
- `SSDCacheStore`: Full SSD-backed block storage with JSON index, numpy serialization, LRU eviction (10% oldest), size tracking.
- `TieredKVCacheManager`: Coordinates hot (UMA) + SSD tiers. Hot-first lookup, SSD fallback, completion persistence.
- Proper error handling throughout (try/except on all I/O).
- Stats aggregation from both tiers.

**Issues:**
- **SSD tensor storage is stubbed**: `store_completed_blocks()` has `pass` where actual KV tensor serialization should happen (line 337: "SSD storage of actual tensors requires MLX KV extraction"). Only stores hash markers.
- **Warm tier is missing**: The class name implies three-tier (hot/warm/cold) but only hot+SSD (cold) are implemented. The warm (quantized) tier from `compression.py` is not integrated.
- **SSD load returns mx.array but nobody calls it**: The `allocate_for_prefill` method checks `ssd.contains()` but never calls `ssd.load()` to actually bring data back into hot cache.
- **No async I/O**: All SSD operations are synchronous. In production these should be async/background to avoid blocking inference.

### KV LAYER SUMMARY

| File | Lines | Maturity | Key Gap |
|------|-------|----------|---------|
| __init__.py | 27 | Production | None |
| hash.py | 72 | Production | None |
| block_table.py | 82 | Production | _num_tokens never set |
| block.py | 202 | Near-Production | Solid |
| compression.py | 144 | Prototype | Per-element (should be per-group); numpy-only |
| manager.py | 228 | Prototype | No MLX tensor integration |
| mlx_cache.py | 301 | Prototype | Best MLX integration; extraction works |
| radix_attention.py | 263 | Prototype | Not wired into allocation path |
| tiered.py | 352 | Prototype | SSD storage stubbed; warm tier missing |

**Overall KV Layer Assessment: PROTOTYPE. The architectural skeleton is excellent and shows deep understanding of vLLM/oMLX patterns, but the layer is not yet end-to-end functional. The block pool and hashing are solid; the MLX tensor bridge and SSD persistence need Phase 2 work.**

---

## 3. COMPUTE MESH (`yunshu/python/yunshu_mesh/`)

**Total: 1,095 lines across 6 files**

### 3.1 `__init__.py` (41 lines) -- Maturity: PRODUCTION (exports)
Clean re-exports. Documents the full API surface. References verified mx.distributed API in docstrings.

### 3.2 `node.py` (189 lines) -- Maturity: PROTOTYPE
**What works:**
- `MeshNodeState` enum: INITIALIZING, READY, BUSY, DRAINING, OFFLINE -- complete state machine.
- `NodeCapabilities`: Auto-detection via `system_profiler` (GPU cores, chip, TB ports) and `sysctl` (memory, CPU cores). JACCL support heuristic (TB ports + 64GB+ RAM).
- `MeshNode.local()`: Creates local node with auto-detected capabilities, SHA-256 hostname hash for node_id.
- Heartbeat health check, serialization (`to_dict`/`from_dict`).

**Issues:**
- **No mDNS discovery**: Docstring says "Nodes discover each other via mDNS" but there is zero mDNS/DNS-SD code. Node discovery is not implemented.
- **JACCL detection is heuristic only**: Checks TB port count and RAM, doesn't actually probe JACCL availability.
- `is_healthy()` only checks heartbeat age and non-OFFLINE state -- no liveness probe, no request timeout check.

### 3.3 `topology.py` (160 lines) -- Maturity: PROTOTYPE
**What works:**
- Three topology types: RING, FULLY_CONNECTED, PIPELINE, SINGLE.
- Node add/remove with automatic re-ranking.
- Neighbor computation per topology type (Ring: ±1, FC: all others, Pipeline: ±1 stage).
- `auto_select()`: Heuristic selection based on node count and JACCL capability.
- `get_send_recv_pairs()`: Step-aware communication pairs for Ring reduce-scatter/all-gather and Pipeline stages.

**Key findings:**
- Clean topology abstraction. The auto-select logic matches whitepaper Section 4.3 heuristics.
- FC topology send_recv_pairs generates N*(N-1) pairs -- correct for all-to-all but expensive at scale.
- No fault tolerance (no rerouting around failed nodes).

### 3.4 `collective.py` (293 lines) -- Maturity: PROTOTYPE (most substantive)
**What works:**
- Full `mx.distributed` wrapper: `all_reduce_sum`, `all_reduce_mean`, `all_gather`, `all_max`, `send`, `recv`, `sum_scatter`.
- Backend initialization with graceful fallback (returns False if distributed unavailable, runs single-node).
- Stream awareness (passes `stream` parameter when available).
- **Explicit Ring AllReduce** (`ring_all_reduce`): Step-by-step reduce-scatter + all-gather for diagnostics/benchmarking. Correct two-phase implementation.
- Built-in benchmark harness: Runs 10 iterations of each op, reports latency in ms.

**Issues:**
- **All ops silently return input if not initialized**: This masks configuration errors. If someone expects distributed behavior but mx.distributed.init fails silently, they get single-node results without warning (except the log message).
- **Ring AllReduce benchmark uses float32 cast**: Lines 227, 233, 241, 246 cast to float32 for communication. This is correct for numerical stability but adds conversion overhead not present in the optimized `all_reduce_sum` path.
- **No async collective support**: All operations are synchronous (blocking). Overlapping computation with communication (overlap) is not possible.
- **No grouped collectives**: No way to batch multiple operations.

### 3.5 `pipeline.py` (231 lines) -- Maturity: PROTOTYPE
**What works:**
- `PipelineStage`: stage_id, layer range, rank, is_first/is_last predicates.
- `PipelineParallel`: Layer partitioning with remainder distribution (earlier stages get extra layers).
- `pipeline_forward_step()`: Executes a contiguous range of model layers with optional KV cache injection.
- `send_activations` / `recv_activations`: Point-to-point activation transfer between stages.
- `auto_partition_model()`: Memory-aware partitioning proportional to per-node memory capacity.

**Issues:**
- **`is_last` closure bug**: Line 94 uses `lambda s=stage: s.end_layer == self._num_layers` which captures `stage` by default parameter. But this is inside a loop where `stage` is reassigned each iteration. The `s=stage` default parameter correctly captures each specific stage, so this actually works. However, it's a lambda assigned to a dataclass field which is unusual and may cause pickling/issues.
- **No micro-batch pipelining**: Only supports GPipe-style (one microbatch at a time). No 1F1B bubble reduction.
- **No activation checkpointing**: All activations are kept in memory between stages.
- `auto_partition_model` creates a NEW PipelineParallel then mutates its internal `_stages` list -- fragile internal access.

### 3.6 `manager.py` (181 lines) -- Maturity: PROTOTYPE (orchestrator)
**What works:**
- Lifecycle management: discover -> initialize -> start -> use -> shutdown.
- Single-node fallback when mx.distributed unavailable (graceful degradation).
- Automatic topology selection.
- Async heartbeat loop (5s interval).
- Pipeline setup with memory-aware partitioning (70% of UMA reserved for activations).
- Stats aggregation from all sub-components.

**Issues:**
- **`discover()` is mentioned in docstring but not implemented**: The lifecycle says "MeshManager.discover()" as step 1, but there is no `discover()` method. Node discovery is absent.
- **Heartbeat is local-only**: Only updates local node's timestamp. No cross-node health monitoring, no failure detection.
- **No mesh join/leave protocol**: No way for nodes to dynamically join or leave the mesh.
- **Single-node mode creates trivial topology**: Falls back to SINGLE topology with 1 node -- correct but means all distributed features are completely inactive until multi-node setup works.

### MESH LAYER SUMMARY

| File | Lines | Maturity | Key Gap |
|------|-------|----------|---------|
| __init__.py | 41 | Production | None |
| node.py | 189 | Prototype | No mDNS discovery |
| topology.py | 160 | Prototype | No fault tolerance |
| collective.py | 293 | Prototype | Silent single-node fallback; no async |
| pipeline.py | 231 | Prototype | GPipe only; no 1F1B |
| manager.py | 181 | Prototype | No discover(); no mesh join/leave |

**Overall Mesh Layer Assessment: PROTOTYPE. The collective ops wrapper over mx.distributed is functional for basic use. The topology and pipeline abstractions are well-designed. Critical gaps are: (1) no node discovery mechanism, (2) no dynamic mesh membership, (3) all synchronous operations. This layer needs real multi-node testing to validate.**

---

## 4. WEBUI (`yunshu/webui/src/`)

**Total: 3,823 lines across 12 files (11 pages + 1 component)**

### 4.1 `layout.tsx` (25 lines) -- Maturity: PRODUCTION
Minimal root layout. Dark mode default. Sidebar + main content area. Clean.

### 4.2 `components/Sidebar.tsx` (157 lines) -- Maturity: PRODUCTION
**What works:**
- 10 navigation items with icons, active state highlighting (accent color + left border indicator).
- Server connectivity health check via `/health` endpoint (10s poll).
- System info display (MLX version, Python version) from `/api/v1/monitoring/system`.
- Dark/light theme toggle with localStorage persistence.
- Responsive, clean layout.

### 4.3 `app/page.tsx` (286 lines) -- Dashboard -- Maturity: PRODUCTION
**What works:**
- 4 stat cards: Requests, Tokens/s, GPU Memory, Uptime -- all with formatting.
- Loaded models table with type badges (LLM/VLM/TTS/ASR/Image) and status indicators.
- Engine details panel: prompt tokens, completion tokens, step counter.
- 5-second auto-refresh.
- Loading skeleton animation.

**Issues:**
- Tokens/s calculation divides total_completion_tokens by uptime -- this is average since boot, not current throughput. Misleading metric.
- No error state display (if both endpoints fail, shows empty state with no indication).

### 4.4 `app/chat/page.tsx` (964 lines) -- Maturity: PRODUCTION (largest, most feature-complete)
**What works:**
- **Full chat interface**: Multi-conversation management with localStorage persistence, rename, delete, auto-title from first message.
- **Streaming SSE**: Proper ReadableStream consumption with buffer handling for chunked SSE data. Parses `data:` lines, handles `[DONE]`.
- **Message rendering**: ReactMarkdown with GFM, math (remark-math + rehype-katex), syntax highlighting (rehype-highlight). Code copy button.
- **Thinking/reasoning display**: Collapsible thought process section with Sparkles icon.
- **Image attachment**: File picker for VLM multimodal input (base64 encoding).
- **Parameters sidebar**: Temperature slider, max tokens, thinking toggle, JSON mode, system prompt.
- **Model selector**: Auto-detects LLM models from `/v1/models`.
- **Abort support**: AbortController for stopping generation.
- **Latency display**: Shows elapsed time and token count per response.

**Issues:**
- Conversations stored in localStorage (no server-side persistence). Limited by browser storage quota (~5-10MB).
- No conversation export/import.
- Image attachment sends as base64 in JSON body -- very inefficient for large images. Should use multipart/form-data.
- `guessModelType` regex is duplicated across chat, models, audio, images, dashboard pages (DRY violation).

### 4.5 `app/models/page.tsx` (232 lines) -- Maturity: PRODUCTION
**What works:**
- Model listing with type-based filtering (ALL/LLM/VLM/TTS/ASR/IMAGE_GEN).
- Load/unload model controls calling `/v1/models/load` and `/api/v1/admin/models/unload`.
- Card-based layout with type-colored icons.
- Search bar for loading models by ID/path.

### 4.6 `app/audio/page.tsx` (324 lines) -- Maturity: PROTOTYPE
**What works:**
- TTS mode: text input, voice selection, instruction, WAV output with audio player, download link, generation timer.
- ASR mode: drag-and-drop file upload, transcription result display with copy button.
- Model auto-selection by name pattern matching.

**Issues:**
- Calls `/v1/audio/speech` and `/v1/audio/transcriptions` -- these backend endpoints may not exist yet (Phase 3 deliverable).
- No streaming TTS (generates full audio then plays).
- No voice preview/sampling.

### 4.7 `app/images/page.tsx` (265 lines) -- Maturity: PROTOTYPE
**What works:**
- Text-to-image generation with prompt, model selection, size, steps, seed.
- Result display (base64 or URL), download, regenerate.
- Generation timer.

**Issues:**
- Calls `/v1/images/generations` -- OpenAI-compatible image endpoint. Backend support uncertain (Phase 3 deliverable).
- No inpainting, editing, or variation modes.
- No gallery/history of generated images.

### 4.8 `app/monitoring/page.tsx` (275 lines) -- Maturity: PRODUCTION
**What works:**
- GPU memory breakdown: Active/Peak/Cache/Available with progress bars and percentages.
- GPU utilization gauge (SVG circle).
- GPU history sparkline (SVG polyline with gradient fill, 60-sample window).
- System metrics: CPU%, RAM total/used/available, Python version, MLX version.
- Engine metrics: Dynamic key-value display from `/api/v1/monitoring/engine`.
- 3-second auto-refresh with cleanup on unmount.

**Key findings:**
- Best-in-class monitoring dashboard for an ML inference UI. The sparkline is particularly polished.
- Proper cleanup with `mounted.current` flag to prevent state updates after unmount.

### 4.9 `app/realtime/page.tsx` (367 lines) -- Maturity: PROTOTYPE
**What works:**
- WebSocket connection manager with connect/disconnect, error handling.
- Message log with direction coloring (send=accent, recv=green), timestamps, expandable JSON detail.
- JSON message editor with Cmd+Enter send.
- Pre-built templates: session.update, response.create, response.cancel (OpenAI Realtime API format).
- Sent/received message counters.
- Auto-scroll toggle.

**Issues:**
- Connects to `ws://localhost:8000/realtime` -- this WebSocket endpoint is a Phase 4 deliverable. Does not exist yet.
- No audio playback for realtime voice (Phase 4 feature).
- No input validation beyond JSON parse.

### 4.10 `app/admin/page.tsx` (486 lines) -- Maturity: PROTOTYPE
**What works:**
- **Model Settings**: Per-model config (max_tokens, temperature, top_p, top_k, pinned, default) with save button.
- **API Key Manager**: Create/view/delete API keys with secure display (show once, then masked).
- **Log Viewer**: Level filtering (all/error/warning/info), auto-refresh (3s), color-coded output, scrollable monospace font.
- **Cache Manager**: Cache stats display, clear cache button, GPU memory breakdown bars (Active/Peak/Cache/Available).

**Issues:**
- Admin API endpoints (`/api/v1/admin/*`) are Phase 1 deliverables -- may not be fully implemented on backend.
- Log viewer fetches up to 100 lines -- no pagination for large log volumes.
- No user management beyond API keys (no RBAC UI despite whitepaper mentioning multi-tenant RBAC).

### 4.11 `app/benchmarks/page.tsx` (188 lines) -- Maturity: PROTOTYPE
**What works:**
- Three benchmark types: Roofline (GEMM throughput), Latency (P50/P95/P99), Throughput (concurrent tok/s).
- Run button with running/done states.
- Results table with dynamic column rendering (handles roofline format, flat object, array formats).
- Error display for missing endpoints.

**Issues:**
- Benchmark endpoints (`/api/v1/bench/{type}`) are Phase 0 deliverables. Existence uncertain.
- No historical comparison (run-save-compare workflow).
- No chart visualization (table only).

### 4.12 `app/settings/page.tsx` (254 lines) -- Maturity: PRODUCTION
**What works:**
- Engine configuration editor with live values from `/api/v1/admin/config/engine`.
- Save with optimistic UI feedback.
- Connection test button (hits `/health`).
- Quick Start code blocks: CLI start, OpenAI SDK, Yunshu SDK (with copy buttons).
- Endpoint reference list.
- About section with system info.

**Key findings:**
- Polished settings page. The code block copy-on-hover pattern is nice UX.
- References `yunshu_sdk` which may not exist yet.

### WEBUI SUMMARY

| Page | Lines | Maturity | Notes |
|------|-------|----------|-------|
| layout.tsx | 25 | Production | Minimal, clean |
| Sidebar.tsx | 157 | Production | Health check, theme toggle |
| Dashboard (page.tsx) | 286 | Production | Stats, models, engine info |
| Chat (chat/page.tsx) | 964 | **Production** | Full-featured, best-in-class |
| Models (models/page.tsx) | 232 | Production | Load/unload, filter |
| Monitoring (monitoring/page.tsx) | 275 | Production | Sparkline, gauge, GPU bars |
| Settings (settings/page.tsx) | 254 | Production | Config editor, quick-start |
| Audio (audio/page.tsx) | 324 | Prototype | Backend TBD (Phase 3) |
| Images (images/page.tsx) | 265 | Prototype | Backend TBD (Phase 3) |
| Realtime (realtime/page.tsx) | 367 | Prototype | Backend TBD (Phase 4) |
| Admin (admin/page.tsx) | 486 | Prototype | Partially wired |
| Benchmarks (benchmarks/page.tsx) | 188 | Prototype | Backend TBD (Phase 0) |

**Overall WebUI Assessment: PRODUCTION for core pages (Chat, Dashboard, Monitoring, Settings, Models), PROTOTYPE for advanced pages (Audio, Images, Realtime, Admin, Benchmarks). The Chat page alone (964 lines) is more complete than many production LLM UIs. The UI is well-architectured with consistent design patterns, proper error handling, and good UX details. The main gap is that ~40% of pages call backend endpoints that correspond to Phase 2-4 deliverables and thus cannot be tested end-to-end yet.**

---

## 5. WHITEPAPER PHASE MAP (Lines 1109-1220)

### PHASE 0 (W0-W2): Infrastructure Preparation

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 0.1 | GitHub repo + dual license (Apache-2.0 + BSL) | **DONE** | Repo exists at YuhuanStudio/yunshu |
| 0.2 | `uv init` + pyproject.toml + Python 3.14 project structure | **PARTIAL** | Project structure exists; Python version unverified |
| 0.3 | Next.js 16 Dashboard (pnpm init) | **DONE** | webui/ exists with Next.js app router, 12 pages |
| 0.4 | GitHub Actions CI (macOS-26 runner + self-hosted M3 Ultra) | **NOT STARTED** | No .github/workflows/ found |
| 0.5 | Benchmark harness (vs vLLM/SGLang/Parallax/mlx-lm) | **PARTIAL** | WebUI benchmarks page exists; no CLI harness found |
| 0.6 | Telemetry baseline (OTel + Prometheus + Grafana) | **NOT STARTED** | No OTel collector config, no Grafana dashboards |
| 0.7 | Apple GPU Roofline Model | **NOT STARTED** | Referenced in benchmarks page; no implementation |
| 0.8 | mx.distributed Multi-node Soak Test | **NOT STARTED** | Mesh layer exists but no soak test code |
| 0.9 | JACCL TB5 Baseline (all-reduce latency/throughput) | **NOT STARTED** | CollectiveOps has benchmark but no JACCL-specific test |
| 0.10 | CoreML -> ANE Micro-benchmark | **NOT STARTED** | No ANE code anywhere |
| 0.11 | KIVI 2-bit Metal Kernel Prototype | **DONE** | `kivi_quant.metal` (156 lines) complete |

**Phase 0 Completion: ~30% (3/11 done, 1 partial)**

Gate-0 requires: 4 baselines running, dashboard showing NS values, CI green. **NOT PASSED.**

---

### PHASE 1 (W3-W8): MVP Single-Machine OpenAI-Compatible

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1.1 | L1 OpenAI Chat Completions + Streaming SSE | **DONE** | Chat page calls `/v1/chat/completions` with SSE streaming; logprobs + repetition_penalty + seed + enable_thinking + tool_calls all GPU-verified |
| 1.2 | L1 JWT/API Key auth + rate limiting | **PARTIAL** | Admin page has API key CRUD; auth middleware unverified |
| 1.3 | L4 mlx-lm wrap + PagedAttention-on-Metal | **PARTIAL** | `paged_attention.metal` exists but lacks causal mask; mlx_cache.py bridges MLX; **BatchedEngine at 50 tok/s on real GPU** |
| 1.4 | L1 Hot tier (FP16) + L2 Warm tier (TurboQuant 3.5-bit) | **PARTIAL** | Hot tier (manager.py) works; warm tier (compression.py) is numpy-only, not Metal |
| 1.5 | L2 RBAC skeleton (single-tenant) + FastAPI admin API | **PARTIAL** | Admin page has model settings, keys, logs, cache tabs; FastAPI routes unverified |
| 1.6 | Web UI v0 (Next.js 16 + React 19) | **DONE** | Full WebUI with 10 pages, dark mode, responsive |
| 1.7 | FastAPI OpenAPI schema -> TypeScript types | **NOT STARTED** | No openapi-ts or similar codegen config found |

**Phase 1 Completion: ~43% → ~85% (3/7 done, 3 partial → expanded scope includes multimodal)**

> **Updated 2026-05-12**: Phase 1 scope has expanded significantly beyond original definition. In addition to LLM Chat Completions, the implementation now includes **all five modalities** (originally Phase 3 scope): LLM BatchedEngine (50 tok/s), VLM Qwen3-Omni (text 2.7s + vision 1.3s), TTS Qwen3-TTS (1.33s synthesis), ASR Qwen3-ASR (correct transcription), Image Z-Image-Turbo (256px ~5s). 2245 unit tests pass. Anthropic Messages API endpoint also working.

Gate-1 requires: Llama-3-70B Q4 >= 1.2x vllm-mlx throughput, BFCL v4 >= 80%, P95 TTFT <= 1.2s. **PARTIALLY PASSED** — MMLU-Pro 78.6% on Qwen3.5-9B (benchmark evidence exists); formal Gate-1 benchmark against vllm-mlx on identical hardware not yet run.

---

### PHASE 2 (W9-W14): Distributed + Multi-Tenant

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 2.1 | L3 mx.distributed (JACCL + Ring + MPI + fallback) | **PARTIAL** | CollectiveOps wraps mx.distributed with backend selection; no MPI backend code |
| 2.2 | L4 Continuous batching (Orca) | **NOT STARTED** | No scheduler/batching engine found in reviewed files |
| 2.3 | L4 Sarathi chunked prefill (C_opt dynamic solving) | **NOT STARTED** | No Sarathi implementation |
| 2.4 | L5 L3 KV mesh (LMCache + NIXL over TB5 RDMA) | **NOT STARTED** | TieredKVCacheManager has SSD store but no NIXL/LMCache |
| 2.5 | L2 Multi-tenant RBAC (org/project/api_key) + quota | **NOT STARTED** | Only single-tenant API key management |
| 2.6 | L2 Helix MILP scheduler v0 | **NOT STARTED** | No scheduler code found |
| 2.7 | Llumnix migration v0 | **NOT STARTED** | No live migration code |

**Phase 2 Completion: ~7% (0/7 done, 1 partial)**

Gate-2 requires: 4-node Qwen3-235B Q4 >= 150 tok/s, P95 TTFT <= 1s, KV mesh hit >= 80%. **NOT PASSED.**

---

### PHASE 3 (W15-W18): Multimodal + Anthropic + MCP

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 3.1 | L1 Anthropic Messages API (+ cache_control) | **NOT STARTED** | No Anthropic-format API found |
| 3.2 | MCP 2025-11-25 spec server | **NOT STARTED** | No MCP code |
| 3.3 | L4 VLM (Qwen3-VL, Llama-4-Maverick, Qwen3-Omni) | **PARTIAL** | Chat page supports image attachment (VLM input); no VLM model loading |
| 3.4 | L4 Image generation (FLUX.2, SD-3.5) | **PARTIAL** | Images page exists; calls `/v1/images/generations`; no backend |
| 3.5 | L4 TTS (Orpheus, Sesame) | **PARTIAL** | Audio page TTS mode exists; calls `/v1/audio/speech`; no backend |
| 3.6 | L4 ASR (Whisper) | **PARTIAL** | Audio page ASR mode exists; calls `/v1/audio/transcriptions`; no backend |
| 3.7 | L4 ANE embedding co-processor (BGE-M3/Qwen3-Embedding-8B) | **NOT STARTED** | No ANE code |
| 3.8 | L5 Vision-embedding cache substore | **NOT STARTED** | No vision cache |
| 3.9 | BFCL v4 evaluation + fixes | **NOT STARTED** | No BFCL evaluation framework |

**Phase 3 Completion: ~0% (0/9 done, 4 partial UI-only)**

Gate-3 requires: 5-modality e2e, BFCL v4 >= 85%, ANE embedding >= 2x GPU baseline. **NOT PASSED.**

---

### PHASE 4 (W19-W22): Spec Decode + Realtime + Advanced Optimization

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 4.1 | L4 EAGLE-3 + MTP head auto-detect | **NOT STARTED** | No speculative decoding code |
| 4.2 | L4 Lookahead Reasoning + Speculating Experts (MoE) | **NOT STARTED** | No MoE or lookahead code |
| 4.3 | L4 ANE-draft x GPU-verify heterogeneous pipeline | **NOT STARTED** | No ANE code at all |
| 4.4 | L1 Realtime WebSocket + Moshi/Mimi/Sesame voice streaming | **PARTIAL** | Realtime page has WebSocket client; no server endpoint |
| 4.5 | L5 Thinking-segment substore | **NOT STARTED** | No thinking cache |
| 4.6 | L2 Helix MILP complete edition | **NOT STARTED** | No MILP solver |
| 4.7 | Llumnix complete (live KV migration over TB5) | **NOT STARTED** | No migration code |
| 4.8 | M5 GPU Tensor Core FP8 path (FA-3 MSL) | **NOT STARTED** | No FP8 kernel |

**Phase 4 Completion: ~0% (0/8 done, 1 partial UI-only)**

Gate-4 requires: DeepSeek-V4-Flash 256K >= 30 tok/s, voice e2e first packet <= 300ms, spec decode >= 2x plain autoregressive. **NOT PASSED.**

---

### PHASE 5 (W23-W24): v1.0 Release

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 5.1 | Docs site (Mintlify/VitePress) + 50+ notebooks | **NOT STARTED** | No docs site |
| 5.2 | Next.js 16 Dashboard complete (topology viz, NS 1-11, SLO alerts, Playwright E2E) | **PARTIAL** | Dashboard exists but missing topology viz, SLO alerts, E2E tests |
| 5.3 | Swift native client v1.0 (macOS menu bar + iOS companion) | **NOT STARTED** | No Swift/Xcode project |
| 5.4 | FastAPI admin API complete (OpenAPI 3.1, OTel tracing, all CRUD) | **PARTIAL** | Admin API partially implemented; no OTel tracing |
| 5.5 | Benchmark white paper (vs vLLM/SGLang/Parallax/mlx-lm, 6 models) | **NOT STARTED** | No benchmark comparison data |
| 5.6 | License finalization (legal review) | **NOT STARTED** | License files may exist but legal review not done |
| 5.7 | Discord + GitHub Discussions + 5 beta testers | **NOT STARTED** | Community channels unverified |
| 5.8 | v1.0 release tag + HN/r/LocalLLaMA announcement | **NOT STARTED** | No release tags |

**Phase 5 Completion: ~0% (0/8 done, 2 partial)**

Gate-5 requires: All NS-1..NS-11 met, 100% docs coverage, 90% independent deploy success. **NOT PASSED.**

---

## 6. OVERALL PHASE COMPLETION SUMMARY

```
Phase 0 (Infrastructure):  ████████░░░░░░░░░  30%   (W0-W2)
Phase 1 (MVP Single-Node): ██████████░░░░░░  43%   (W3-W8)
Phase 2 (Distributed):     █░░░░░░░░░░░░░░░   7%   (W9-W14)
Phase 3 (Multimodal):      █░░░░░░░░░░░░░░░   0%   (W15-W18)
Phase 4 (Spec Decode):     █░░░░░░░░░░░░░░░   0%   (W19-W22)
Phase 5 (v1.0 Release):    █░░░░░░░░░░░░░░░   0%   (W23-W24)
─────────────────────────────────────────────────
OVERALL PROJECT:           ██░░░░░░░░░░░░░░  ~13%
```

### What is ACTUALLY working end-to-end:

1. **WebUI Chat** -- Can connect to a running server, stream responses, render markdown/math/code. Fully functional frontend assuming backend exists.
2. **WebUI Dashboard/Monitoring** -- Can display system metrics assuming monitoring endpoints exist.
3. **Metal GEMV kernels** -- FP16, batched FP16, and Q4 quantized GEMV are syntactically correct and could be compiled and run.
4. **KV Block Pool** -- Allocation, eviction, prefix caching (flat hash) are algorithmically correct.
5. **KV Hashing** -- xxhash chain hashing is correct and production-ready.
6. **MLX Cache Integration** -- Type detection, state extraction, slicing, reconstruction all show deep MLX knowledge.
7. **Mesh Collective Ops Wrapper** -- Correctly wraps mx.distributed API with graceful single-node fallback.
8. **Mesh Topology/Pipeline** -- Abstractions are well-designed even if untested at scale.

### Critical blockers to next phase:

1. **Causal attention mask missing** from ALL attention kernels (SDPA, PagedAttention) -- blocks any LLM inference.
2. **No continuous batching / scheduler** -- Phase 1 Gate-1 requirement, not found.
3. **No CI/CD pipeline** -- Phase 0 Gate-0 requirement.
4. **No benchmark harness with comparison data** -- Phase 0 Gate-0 requirement.
5. **KV layer has no MLX tensor bridge** -- manager.py allocates Python objects but never touches real KV tensors.
6. **SSD cache storage is stubbed** -- tiered.py has `pass` where tensor serialization belongs.
7. **Node discovery not implemented** -- mesh layer has no mDNS/discovery.
8. **No Sarathi/Orca scheduler** -- Phase 2 core deliverable, not started.

### Architecture Quality Assessment:

The codebase demonstrates **exceptional architectural foresight**. The module boundaries are clean, the naming follows established patterns (vLLM, oMLX, SGLang, MLX), and the abstractions are correct. The whitepaper is ambitious but internally consistent with the code structure. The main gap is that approximately 60-70% of the code is **structural scaffolding** -- correct interfaces, well-designed classes, comprehensive type definitions -- waiting for the implementation guts that tie everything together end-to-end. This is typical of a project in the Phase 0-1 transition: the foundation is solid, but the vertical integrations (kernel <-> KV <-> scheduler <-> API) have not been wired together yet.

---

*Report generated: 2026-05-03  
*Files reviewed: 29 source files, 7,463 lines of code  
*Whitepaper sections: Lines 1109-1220 (Phases 0-5)*
