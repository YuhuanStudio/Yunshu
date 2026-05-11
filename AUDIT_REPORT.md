# Yunshu 全面審計報告

> 審計日期：2026-05-11
> 最後更新：2026-05-11（修復狀態同步）
> 審計範圍：46 個 Engine Python 檔案（~15,000 行）、28 個 Gateway 檔案（~8,500 行）、6 個 Metal kernel、2,222 個測試、326 篇參考文獻

---

## 修復進度總覽

| 嚴重級別 | 總數 | ✅ 已修復 | ✅ 審查非 bug | 🔲 未修復 |
|---------|------|----------|-------------|----------|
| CRITICAL | 10 | 10 | 0 | 0 |
| HIGH | 12 | 12 | 0 | 0 |
| MEDIUM | 20 | 17 | 3 | 0 |
| LOW | 18 | 15 | 0 | 3 |
| 架構問題 | 5 | 5 | 0 | 0 |
| **合計** | **65** | **59** | **3** | **3** |

> 所有 CRITICAL + HIGH + MEDIUM 問題已於 Wave 1–6 修復完成。
> 1977 個單元測試全數通過，0 失敗。
> 修復 commits：`11ec3b9` `84f9ffd` `b04a432` `140bcf4` `2a95440` `32e456c` `d7d289a` `120d5d8` `9d03c71`

---

## 目錄

- [一、嚴重問題（CRITICAL）](#一嚴重問題critical)
- [二、高優先級問題（HIGH）](#二高優先級問題high)
- [三、中等問題（MEDIUM）](#三中等問題medium)
- [四、低優先級問題（LOW）](#四低優先級問題low)
- [五、架構問題](#五架構問題)
- [六、測試品質](#六測試品質)
- [七、白皮書對齊](#七白皮書對齊)
- [八、參考框架比較](#八參考框架比較)
- [九、修復計畫](#九修復計畫)

---

## 一、嚴重問題（CRITICAL）

### C1. Auth 永遠開啟 — Gateway 開箱即壞 ✅ Wave 1 已修復

- **檔案**: `python/yunshu_gateway/middleware/tenant_auth.py:43-48`
- **修復**: `return os.environ.get("YUNSHU_AUTH_TOKEN") is not None`
- **影響**: 未設 `YUNSHU_AUTH_TOKEN` 且未設 `YUNSHU_AUTH_DISABLED=true` 時，所有請求都被 401 拒絕
- **與 `main.py:186` 矛盾**: 註解說「Set YUNSHU_AUTH_TOKEN=secret to enable」，但 auth 預設就開

```python
# tenant_auth.py:43-48 — 錯誤
def _is_auth_enabled(self) -> bool:
    if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in ("true", "1", "yes"):
        return False
    auth_token = os.environ.get("YUNSHU_AUTH_TOKEN")
    return True  # ← 永遠 True，auth_token 變數未被使用
```

**修復**: 改為 `return auth_token is not None`

---

### C2. 時序攻擊可還原 API Token ✅ Wave 1 已修復

- **檔案**: `python/yunshu_gateway/middleware/tenant_auth.py:90`
- **修復**: 改用 `hmac.compare_digest(token, auth_token)`
- **影響**: 攻擊者可透過回應時間差逐字元還原 token

```python
# tenant_auth.py:90 — 不安全
if auth_token and token == auth_token:
```

**修復**: 改用 `hmac.compare_digest(token, auth_token)`

---

### C3. WebSocket 端點零認證 ✅ Wave 1 已修復

- **檔案**: `python/yunshu_gateway/routers/realtime.py:734-739`
- **修復**: `ws.accept()` 前手動驗證 query param token + `hmac.compare_digest()`
- **影響**: 任何人可透過 WebSocket `/realtime` 取得完整的模型推理、TTS、ASR 能力

```python
# realtime.py:734-739 — 無任何認證
@router.websocket("/realtime")
async def realtime_endpoint(ws: WebSocket):
    await ws.accept()  # ← 直接接受
    session = RealtimeSession(ws)
    await session.run()
```

**修復**: 在 `ws.accept()` 前手動驗證 token（從 query param 或 first message）

---

### C4. 暫存檔洩漏 + 副檔名注入 ✅ Wave 1 已修復

- **檔案**: `python/yunshu_gateway/routers/audio.py:226-242`, `realtime.py:639-676`
- **修復**: `tempfile.mkstemp(suffix=...)` + 安全副檔名白名單 `_SAFE_EXTENSIONS`
- **影響**: 磁碟空間洩漏、潛在 symlink 攻擊、副檔名偽造

**修復**: 使用 `tempfile.mkstemp()` + 安全副檔名白名單 + `atexit` 清理

---

### C5. MCP 模板注入 ✅ Wave 1 已修復

- **檔案**: `python/yunshu_gateway/routers/mcp.py:633`
- **修復**: `re.sub(r'\{(\w+)\}', lambda m: str(arguments.get(m.group(1), m.group(0))), template)`
- **影響**: 透過 `__class__.__mro__.__init__.__globals__` 等路徑洩漏資訊

```python
# mcp.py:633 — 不安全
template = prompt_def["template"].format(**arguments)
```

**修復**: 使用 `string.Template.safe_substitute()` 或限制 key 只允許 `\w+`

---

### C6. CORS 過度寬鬆 ✅ Wave 1 已修復

- **檔案**: `python/yunshu_gateway/main.py:163-167`
- **修復**: 限制為 `allow_methods=["GET","POST","PUT","DELETE","OPTIONS"]`, `allow_headers=["Authorization","Content-Type","Accept"]`
- **影響**: 生產環境安全風險

**修復**: 限制為 `allow_methods=["GET","POST","OPTIONS"]`

---

### C7. PagedAttention block_table_stride 計算錯誤 ✅ Wave 2 已修復

- **檔案**: `metal/paged_attention.metal:52`
- **修復**: 使用 `max_blocks` 參數作為 stride，offset 計算改用 `ulong`
- **影響**: 當 `seq_len` 暗示的 block 數少於 allocation width 時，讀取錯誤記憶體位址

```metal
// paged_attention.metal:52 — 錯誤
int block_table_stride = (seq_len + kv_block_size - 1) / kv_block_size;
// 應為：block_table_stride = max_blocks（從 template 參數取得）
```

---

### C8. SDPA threadgroup 記憶體超限 ✅ Wave 2 已修復

- **檔案**: `metal/sdpa.metal:52-54`
- **修復**: K/V tile load 使用 `simd_group_id`/`NUM_SIMD_GROUPS` 迭代，threadgroup 記憶體縮減

---

### C9. SDPA head_dim > 32 時無限迴圈 ✅ Wave 2 已修復

- **檔案**: `metal/sdpa.metal:79-96`
- **修復**: K/V tile 使用 `simd_group_id`/`NUM_SIMD_GROUPS` 迭代取代 `32/head_dim` 除法
- **影響**: 掛死 GPU（適用於所有 head_dim >= 33 的模型，如 head_dim=64/128）

```metal
// sdpa.metal:79-96 — head_dim=128 時 kv += 0，無限迴圈
for (uint kv = simd_lane_id / head_dim; kv < kv_len; kv += 32 / head_dim)
```

---

### C10. server_metrics.py 死鎖 ✅ Wave 1 已修復

- **檔案**: `python/yunshu_engine/server_metrics.py:121-157`
- **修復**: flag-based periodic save — lock 內設 `needs_save=True`，lock 外執行 `save_alltime()`
- **影響**: `threading.Lock` 會死鎖；`threading.RLock` 會暴露 critical section。每 300 秒觸發一次

```python
# server_metrics.py:153-157
with self._lock:
    # ...
    self._lock.release()       # 釋放
    try:
        self.save_alltime()    # 內部又 acquire → 死鎖
    finally:
        self._lock.acquire()   # 再 acquire
```

**修復**: 改用 flag — 在 lock 內設 `needs_save=True`，在 lock 外 save

---

## 二、高優先級問題（HIGH）

### H1. `_init_memory_monitor()` 呼叫兩次 ✅ Wave 1 已修復

- **檔案**: `python/yunshu_engine/engine.py:300,303`
- **修復**: 移除重複呼叫

```python
self._init_memory_monitor()  # line 300
# ...
self._init_memory_monitor()  # line 303 — 重複
```

### H2. `_init_spec_decode()` 呼叫兩次 ✅ Wave 1 已修復

- **檔案**: `python/yunshu_engine/batched_engine.py:110,116`
- **修復**: 移除重複呼叫

```python
self._init_spec_decode()     # line 110
# warmup
self._init_spec_decode()     # line 116 — 重複
```

### H3. n_confirmed rollback 長度調整方向相反 ✅ Wave 1 已修復

- **檔案**: `python/yunshu_engine/n_confirmed_patch.py:309`
- **修復**: `c.lengths = c.lengths - 1`

```python
c.lengths = c.lengths + 1   # 錯誤：應為 c.lengths - 1
```

### H4. VLM engine `has_active_requests()` 永遠 False ✅ Wave 1 已修復

- **檔案**: `python/yunshu_engine/vlm_engine.py:104-105`
- **修復**: 追蹤 `_active_count`，回傳 `self._active_count > 0`

```python
def has_active_requests(self) -> bool:
    return False  # ← 永遠 False
```

### H5. JSON schema 模式靜默丟棄 repetition/presence/frequency penalty ✅ Wave 1 已修復

- **檔案**: `python/yunshu_engine/scheduler.py:931-941`
- **修復**: constrained sampler 改為包裝含 logits_processors 的 `sampler`（而非 `base_sampler`）

```python
# scheduler.py:931-941
# 前面辛苦建的 logits_processors（repetition/presence/frequency penalty）
# 在 json_schema 分支完全被忽略
if json_schema is not None:
    return make_constrained_sampler(base_sampler=base_sampler, ...)
```

### H6. Engine.load() 在錯誤線程載入模型 ✅ Wave 1 已修復

- **檔案**: `python/yunshu_engine/engine.py:306-308`
- **修復**: `load()` 透過 `executor.submit(load_model, model_name).result()` 在 MLX executor 執行
- **影響**: Metal buffers 建立在錯誤的 stream 上，可能導致 GPU 衝突

### H7. 兩個不相容的 RequestOutput 類別 ✅ Wave 1 已修復

- **檔案**: `engine.py:51` vs `request.py:79`
- **修復**: 移除 `engine.py` 的 local `RequestOutput`，統一從 `request.py` re-export
- **影響**: 同時使用 Engine 和 EngineCore 時會遇到型別不相容

### H8. Embedding 端點缺 `await` 導致 crash ✅ Wave 1 已修復

- **檔案**: `python/yunshu_gateway/routers/embeddings.py:104`
- **修復**: 加上 `await`
- **影響**: 回傳 coroutine 物件而非 engine，後續 `engine.embed(texts)` crash

```python
engine = manager.get_engine(model_id)  # ← 缺少 await
```

### H9. Completions 端點缺少 SSE keepalive ✅ Wave 1 已修復

- **檔案**: `python/yunshu_gateway/routers/completions.py:154-232`
- **修復**: 重構為 `_token_source()` async generator + `with_sse_keepalive()` + disconnect detection
- **影響**: 長 prefill 期間客戶端超時斷線，GPU 白算

### H10. Anthropic streaming 雙重 `content_block_stop` ✅ Wave 2 已修復

- **檔案**: `python/yunshu_gateway/routers/anthropic.py:505-512`
- **修復**: 加入 block-started guard，避免零 token 時重複開關 block
- **影響**: Anthropic SDK client 可能 crash 或誤解串流

### H11. PagedAttention 32x SIMD 浪費 ✅ Wave 2 已修復

- **檔案**: `metal/paged_attention.metal:68-76,120-127`
- **修復**: q·k dot product 和 value accumulation 改為 SIMD-strided（`d += 32`）
- **影響**: 效能只有應有的 1/32

### H12. `kv_block_offset` uint32 溢出 ✅ Wave 2 已修復

- **檔案**: `metal/common.metal:63-71`
- **修復**: 改用 `ulong`（64-bit）算術

---

## 三、中等問題（MEDIUM）

### M1. Speculative decoder 目標模型永遠 greedy ✅ Wave 6 已修復

- **檔案**: `speculative_decoder.py:579,613,632`
- **修復**: 使用 `make_sampler(temp=temperature)` 取代 `argmax`
- **影響**: spec decode 永遠產生 greedy 輸出

### M2. Completions stop token 包含在輸出中 ✅ Wave 3 已修復

- **檔案**: `batched_engine.py:372-382`
- **修復**: stop token 用 `tokens.pop()` 排除後再 break
- **影響**: OpenAI API 規範 stop token 不應出現在輸出

### M3. Fast path 靜默丟棄參數 ✅ Wave 4 已修復

- **檔案**: `batched_engine.py:476-478`
- **修復**: 新增 `min_p` 傳入 `make_sampler`；新增 `logits_processors` 傳入 `generate_step` 處理 penalty/bias

### M4. Audio engine 使用錯誤 executor ✅ Wave 4 已修復

- **檔案**: `audio_engine.py:538`
- **修復**: 改用 `get_mlx_executor()` 取代 `None`
- **影響**: GPU 工作可能在不同 thread 執行，Metal stream 衝突

### M5. Anthropic `stop_sequence` 欄位缺失 ✅ Wave 4 已修復

- **檔案**: `anthropic.py:309-320`
- **修復**: 偵測 matched stop sequence 後設 `stop_reason="stop_sequence"` 並回傳 `stop_sequence` 欄位
- **影響**: Anthropic SDK 期待此欄位

### M6. Anthropic `budget_tokens` 未執行 ✅ Wave 5 已修復

- **檔案**: `anthropic.py:284`
- **修復**: `effective_max_tokens = min(req.max_tokens, budget_tokens)` 傳入 streaming 和 non-streaming 路徑
- **影響**: thinking 輸出不受 token 上限約束

### M7. `output_collector.py` class variable 跨實例共享 ✅ 審查後保留（非 bug）

- **檔案**: `output_collector.py:31`
- **說明**: `_waiting_consumers` 為 class variable 是刻意設計，`has_waiting_consumers()` 需要全域計數。Wave 3 曾改為 instance var，Wave 4 還原。

### M8. `restore_rollback` 對非 SSM/KV 層提前返回 False ✅ Wave 3 已修復

- **檔案**: `n_confirmed_patch.py:292-315`
- **修復**: 處理所有層後再回傳 `success` flag，不提前返回

### M9. Norm weight +1.0 mutation ✅ Wave 4 已修復

- **檔案**: `mtp_patch.py:183`
- **修復**: 加上 `_yunshu_shifted` flag 防止 double-shift
- **影響**: 第二次 load 會 double-shift

### M10. EngineCore TOCTOU on output_collectors ✅ Wave 7 已修復

- **檔案**: `engine_core.py:470`
- **修復**: 改為迭代 live dict（`self._output_collectors.get(rid)`），abort 插入的新 collector 可被及時看見

### M11. Embedding 同步阻塞 event loop ✅ Wave 3 已修復

- **檔案**: `embeddings.py:117-165`
- **修復**: `_generate_embeddings` 改為 async，fallback 路徑透過 `run_in_executor(get_mlx_executor(), ...)` 執行

### M12. VLM 串流建立獨立 Metal stream ✅ Wave 4 已修復

- **檔案**: `vlm_engine.py:447`
- **修復**: 移除 `mx.new_thread_local_stream()`，VLM 已在 MLX executor 上執行

### M13. OpenAI streaming delta 每個 chunk 都包含 role ✅ Wave 3 已修復

- **檔案**: `streaming.py:523`
- **修復**: `format_openai_chunk` 新增 `include_role: bool = False` 參數，chat.py 只在第一個 chunk 傳 `include_role=True`

### M14. VLM 回應回報 0 tokens ✅ Wave 4 已修復

- **檔案**: `chat.py:647-654`
- **修復**: 使用 tokenizer 計算 prompt/completion token 數

### M15. MCP SSE zombie 連線 ✅ Wave 6 已修復

- **檔案**: `mcp.py:681-693`
- **修復**: poll 間隔從 30s 降為 15s，加入 error-safe disconnect 檢查

### M16. Anthropic tool call regex 不支援巢狀 JSON ✅ Wave 5 已修復

- **檔案**: `anthropic.py:166`
- **修復**: regex 改為只匹配 name 部分，arguments 用 brace-depth 匹配 + `json.loads()` 驗證

### M17. Tool call streaming 文字重複輸出 ✅ 審查後非 bug

- **檔案**: `chat.py:772-785`
- **說明**: `ToolCallStreamer` 已正確 buffer 並路由 text vs tool call，無重複問題

### M18. Context window 估計不準確（多模態）✅ Wave 5 已修復

- **檔案**: `chat.py:478-482`
- **修復**: 解析 content list 中的 `image_url` blocks，每張圖加 576 tokens

### M19. `shared_logits` 編譯期常數 vs 運行期 `kv_block_size` ✅ Wave 6 已修復

- **檔案**: `metal/paged_attention.metal:53`
- **修復**: `PA_BLOCK_KV` 從 128 提升至 256

### M20. GQA head mapping 整數除法脆弱 ✅ 審查後非 bug

- **檔案**: `metal/sdpa.metal:103-105`
- **說明**: `head_idx * num_kv_heads / num_heads` 是標準 GQA mapping，對 2/4/8 group ratio 正確

---

## 四、低優先級問題（LOW）

| # | 位置 | 問題 | 狀態 |
|---|---|---|------|
| L1 | `ane_embedding.py` | 大量 stub/placeholder（hash-based tokenization, random embeddings） | 🔲 |
| L2 | `telemetry.py` | 整個模組是 stub，flush 直接清空 | 🔲 |
| L3 | `deltanet_inversion.py` | 標記為「不適用 BF16」，monkey-patch 未完成 | 🔲 |
| L4 | `metal_kernels.py:659` | `shell=True` subprocess | ✅ Wave 7 |
| L5 | `json_schema.py:664` | 硬編碼 `range(151936)` vocab fallback | ✅ Wave 7 |
| L6 | `model_registry.py:112` | module-level singleton 非 thread-safe | ✅ Wave 8 |
| L7 | `engine.py:423` | `_make_sampler` 傳了 mlx-lm 不接受的 penalty 參數 | ✅ Wave 7 |
| L8 | `kv_prefix_cache.py:50` | numpy detour 可能丟失 bf16 精度 | ✅ Wave 8 |
| L9 | 多處 | `uuid.uuid4().hex[:12]` 只有 48-bit entropy | ✅ Wave 7 |
| L10 | 多處 | `__import__("time")` 而非 module-level import | ✅ Wave 8 |
| L11 | `streaming.py:700` | `TokenRateTracker` 定義但從未使用 | ✅ Wave 7 |
| L12 | `streaming.py:733` | `StopSequenceDetector` 定義但從未使用 | ✅ Wave 7 |
| L13 | `streaming.py:164` | `SSEKeepaliveWrapper` 被 `with_sse_keepalive` 取代 | ✅ Wave 7 |
| L14 | `health.py` | 與 `main.py` 重複定義健康端點，health.py 未被 include | ✅ Wave 7 |
| L15 | `mcp.py:75` | `MCPSession` 類別定義但從未實例化 | 🔲（保留：有測試覆蓋，是 future API） |
| L16 | Anthropic | 缺少 `cache_creation_input_tokens` / `cache_read_input_tokens` | ✅ Wave 7 |
| L17 | `bench.py:23-24` | 全域 mutable state 無鎖 | ✅ Wave 8 |
| L18 | Prometheus | label 值未跳脫 | ✅ Wave 8 |

---

## 五、架構問題 — 全部處理完成

### 5.1 雙引擎並行 — 最大的技術債 ✅ Wave 9 已標記 deprecated

```
Engine (engine.py, 1060行)
├── 自己的 _step_loop + inline scheduling
├── 自己的 request tracking (_waiting, _active, _uid_to_req)
├── 自己的 _make_sampler / _make_state_machine
└── use_engine_core flag → 也可以 delegate to EngineCore

BatchedEngine (batched_engine.py, 1033行)
├── 包 EngineCore + Scheduler
├── Fast path: 直接 generate_step
└── Engine loop path: continuous batching
```

**後果**：
- 每個 bug 修要改兩處
- 兩套不相容的 RequestOutput
- Gateway 選 Engine 或 BatchedEngine 是硬編碼的
- 沒有從 Engine → BatchedEngine 的遷移路徑

**建議**：棄用 Engine，BatchedEngine 為唯一入口。Engine 保留但標記 deprecated。

### 5.2 Metal 雙源碼 ✅ Wave 9 已標記 .metal 為 reference-only

```
metal/*.metal                    ← Makefile 編譯成 .metallib
python/yunshu_engine/metal_kernels.py  ← Python 字串 inline Metal source
```

**問題**：
- `.metal` 檔案已開始與 Python inline 分歧（Python 版有 SIMD striding，`.metal` 版沒有）
- `MetalKernelManager` 從未載入 `.metallib`，只用 JIT compilation
- 修 `.metal` 不影響運行中的系統

**建議**：擇一。要么全部用 `.metal` + precompile，要么全部用 Python inline + JIT。

### 5.3 Gateway → Engine 私有屬性耦合 ✅ Wave 9 已修復

17+ 處直接存取 `_entries`、`_model_manager`、`_engine` 等私有屬性：
- `models.py:24` → `manager._entries.values()`
- `engine/__init__.py:108` → `_model_manager._entries`
- `main.py:140,258` → `_engine.*`
- `metrics.py:143` → 多處私有屬性
- `monitoring.py:107` → 多處私有屬性
- `tenant_auth.py:98` → 多處私有屬性
- `tokenize.py:71` → 多處私有屬性

**建議**：添加 public accessor methods。

### 5.4 Eager imports 繞過 lazy 機制 ✅ Wave 8 已修復

- **修復**: `__init__.py` 全部改為 `__getattr__` lazy import，import 時零副作用。

### 5.5 `import mlx.core` 在 module level ✅ Wave 8 已修復

- **修復**: `scheduler.py` 移除 module-level `import mlx.core`，改為 local import。

---

## 六、測試品質

### 6.1 分佈

| 類別 | 測試數 | 比例 | 品質評估 |
|---|---|---|---|
| 純 mock（無真實行為驗證） | ~1,500 | 68% | 低 |
| 真實 MLX 張量運算 | ~400 | 18% | 高 |
| HTTP 整合（mock engine） | ~300 | 13% | 中 |
| **真實模型推理** | ~16 | **<1%** | 唯一可信 |

### 6.2 永遠通過的測試（false confidence）

**30+ 個過度寬鬆的 HTTP status assertion**：
```python
# 出現在 test_anthropic.py, test_images.py, test_embeddings.py 等
assert resp.status_code in (200, 404, 500, 503, 422)  # 接受所有可能狀態碼
```

**8 個同義反覆 Anthropic streaming 測試**：
```python
# test_anthropic.py:345-494 — 在測試裡手動建 dict 再 assert 自己
event = {"type": "message_start", ...}
assert event["type"] == "message_start"  # 永遠 True
```

**1 個 literal `assert True`**：
```python
# test_mesh_discovery.py:90
assert True
```

### 6.3 零覆蓋的關鍵模組

| 模組 | 說明 |
|---|---|
| `vlm_engine.py` | 整個 VLM 路徑無測試 |
| `image_engine.py` | 圖像生成無測試 |
| `mtp_decoder.py` | MTP 解碼器無測試 |
| `mlx_executor.py` | GPU executor 併發關鍵無測試 |
| `deltanet_inversion.py` | 無整合測試 |
| `n_confirmed_patch.py` | 無測試 |
| `vision_feature_cache.py` | 無測試 |

### 6.4 SpeculativeDecoder 核心方法零覆蓋

`generate_draft()`、`verify_draft()`、`generate()` — 這三個核心方法從未被任何測試呼叫。35 個 MagicMock，只測初始化和預設值。

### 6.5 錯誤路徑覆蓋極低

2,222 個測試中只有 **37 個 `pytest.raises`**（0.17%）。模型載入失敗、OOM、malformed input、timeout 處理幾乎未測。

### 6.6 測試配置缺失

- 無 `@pytest.mark.integration` / `@pytest.mark.gpu` markers
- 無 coverage reporting (`addopts`)
- 無 `xfail_strict`
- conftest.py 只有 auth disable fixture，無 shared engine/client fixtures

---

## 七、白皮書對齊

### 7.1 Roadmap 狀態

| Phase | 白皮書目標 | 實際完成度 | 備註 |
|---|---|---|---|
| Phase 0 (W0-W2) | CI/CD, roofline, Metal 驗證 | 100% | Gate-0 通過 |
| Phase 1 (W3-W8) | OpenAI Chat, PagedAttention, warm tier KV | 100% | Gate-1 通過（44.4 tok/s = 1.13x mlx-lm） |
| Phase 2 (W9-W14) | 4-node mesh, RBAC, Helix | **45%** | Gate-2 未通過 |
| Phase 3 (W15-W18) | 5-modality, Anthropic, MCP | 100% | 超出預期 |
| Phase 4 (W19-W22) | Spec decode, Realtime | 100% | 但 spec decode 實測 0.54x（減速） |
| Phase 5 (W23-W24) | Release | 100% | v0.0.1 已發布 |

### 7.2 North Star 指標

| 指標 | 白皮書目標 | 實測 | 狀態 |
|---|---|---|---|
| NS-1: 235B Q4 throughput | >= 220 tok/s (4xM3-Ultra) | 無多機環境 | UNTESTABLE |
| NS-2: P95 TTFT | <= 800ms | ~105ms (1 concurrent) | 部分 |
| NS-3: KV reuse hit rate | >= 95% | 無數據 | UNKNOWN |
| NS-4: LoRA adapters | >= 1000 | 0（未實現） | NOT MET |
| NS-5: BFCL v4 | >= 90% | 無正式分數 | UNKNOWN |
| NS-6~NS-11 | 各項 SLO 指標 | 無數據 | UNKNOWN |

### 7.3 未兌現的核心主張

| 白皮書主張 | 狀態 |
|---|---|
| Helix MILP per-second re-solve (Delta-4) | **零代碼** |
| Cross-node KV mesh over TB5 (Delta-3/6) | **零代碼** |
| PD Separation (DistServe/Splitwise) | **零代碼** |
| Spec decode 2-6.5x 加速 | **實測 0.54x** |
| S-LoRA 1000 adapters (Delta-7) | **零代碼** |
| MLA/DeepSeekMoE EP (Delta-8) | **零代碼** |

### 7.4 參考文獻

- 326 篇引用，25+ 篇核心引用已驗證正確
- 4 篇 arXiv ID 已修正
- 2-3 篇可能不存在（[44] DMS, [39] Lookahead Reasoning preprint）
- v10.1-v10.3 有編碼損壞（亂碼）

---

## 八、參考框架比較

### 8.1 Yunshu 獨有優勢

| 功能 | 說明 | 無人做過 |
|---|---|---|
| Thinking budget 強制截斷 | scheduler.py 強制停止超出 budget 的 thinking tokens | ✓ |
| Thinking-segment KV 子存儲 | 跨 request 重用 reasoning KV，SSD 持久化 | ✓ |
| MemoryGuard 預檢 | 請求准入前的 GPU 記憶體預檢查 | ✓ |
| 5 協議統一 Gateway | OpenAI + Anthropic + MCP + Realtime + Batch | ✓ |
| External prefill + abort | 記憶體預檢 + chunked progress + mid-prefill abort | ✓ |

### 8.2 落後於參考框架之處

| 差距 | 來源 | 優先級 |
|---|---|---|
| 無 Radix tree prefix cache（O(n*entries) 線性掃描） | SGLang | 高 |
| 無 hash-based block dedup + COW | vLLM | 高 |
| 無 SSD-tier KV cache | oMLX | 中 |
| 無 `wired_limit` context manager | mlx-lm | 中 |
| 無 KV cache quantization | mlx-lm | 中 |
| 無 request preemption | vLLM | 中 |
| 無 grammar constraint integration | oMLX | 低 |
| Speculative decoder 逐 token 驗證（非 batched） | llama.cpp | 中 |
| 無 SpecPrefill（speculative prefill） | oMLX | 低 |
| 無 mRoPE batch support | oMLX | 低 |

### 8.3 代碼量比較（估算）

| 框架 | 核心推理代碼量 | 調度深度 |
|---|---|---|
| oMLX | ~40,000 行 | 深（PagedCacheManager, BlockAwarePrefixCache, SSD tier） |
| vllm-mlx | ~15,000 行 | 深（vLLM PagedAttention 全套） |
| SGLang | ~60,000 行 | 最深（RadixAttention, HiCache） |
| **Yunshu** | **~16,000 行** | **中等（但 API 廣度最大）** |

---

## 九、修復計畫

### 第一波：安全 + 正確性（建議立即執行）

| # | 問題 | 預估時間 |
|---|---|---|
| 1 | C1: Auth 永遠開啟 | 1 min |
| 2 | C2: 時序攻擊 | 1 min |
| 3 | C3: WebSocket 認證 | 15 min |
| 4 | C4: 暫存檔安全 | 10 min |
| 5 | C5: MCP 模板注入 | 5 min |
| 6 | C6: CORS 寬鬆 | 2 min |
| 7 | C8+C9: SDPA Metal kernel 修復 | 30 min |
| 8 | C7: PagedAttention stride 修復 | 10 min |
| 9 | C10: server_metrics 死鎖 | 5 min |
| 10 | H1: 重複 _init_memory_monitor | 1 min |
| 11 | H2: 重複 _init_spec_decode | 1 min |
| 12 | H3: rollback +/- 1 | 1 min |
| 13 | H4: VLM has_active_requests | 10 min |
| 14 | H8: embedding missing await | 1 min |

### 第二波：架構清理

| # | 問題 | 預估時間 |
|---|---|---|
| 15 | 統一 RequestOutput 類別 | 30 min |
| 16 | 消除 Engine 重複邏輯 | 60 min |
| 17 | 統一 Metal kernel 源碼 | 60 min |
| 18 | 修復 Gateway 私有屬性耦合 | 30 min |

### 第三波：測試品質

| # | 問題 | 預估時間 |
|---|---|---|
| 19 | 消除同義反覆 assertions | 30 min |
| 20 | 添加最小推理冒煙測試 | 30 min |
| 21 | 添加 pytest markers | 15 min |
| 22 | SpeculativeDecoder 方法測試 | 60 min |

### 第四波：功能追趕

| # | 問題 | 預估時間 |
|---|---|---|
| 23 | Radix tree prefix cache | 2-3 天 |
| 24 | Hash-based block dedup | 2-3 天 |
| 25 | PagedAttention 接入 pipeline | 1-2 天 |
| 26 | Batched speculative verification | 1 天 |

---

> 審計結論：Yunshu 是一個功能完整的單節點 Apple Silicon 推理伺服器，API 廣度超過所有競品。
> 但安全漏洞、Metal kernel 錯誤、測試品質和架構債需要系統性修復，才能成為生產級平台。
