# Yunshu 全面審計報告

> 審計日期：2026-05-11
> 審計範圍：46 個 Engine Python 檔案（~15,000 行）、28 個 Gateway 檔案（~8,500 行）、6 個 Metal kernel、2,222 個測試、326 篇參考文獻

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

### C1. Auth 永遠開啟 — Gateway 開箱即壞

- **檔案**: `python/yunshu_gateway/middleware/tenant_auth.py:43-48`
- **問題**: `_is_auth_enabled()` 讀取 `YUNSHU_AUTH_TOKEN` 但從未使用，永遠回傳 `True`
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

### C2. 時序攻擊可還原 API Token

- **檔案**: `python/yunshu_gateway/middleware/tenant_auth.py:90`
- **問題**: 用 `==` 比對 Bearer token，短路由在第一個不同字元
- **影響**: 攻擊者可透過回應時間差逐字元還原 token

```python
# tenant_auth.py:90 — 不安全
if auth_token and token == auth_token:
```

**修復**: 改用 `hmac.compare_digest(token, auth_token)`

---

### C3. WebSocket 端點零認證

- **檔案**: `python/yunshu_gateway/routers/realtime.py:734-739`
- **問題**: `BaseHTTPMiddleware`（TenantAuthMiddleware 的基類）不攔截 WebSocket 連線
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

### C4. 暫存檔洩漏 + 副檔名注入

- **檔案**: `python/yunshu_gateway/routers/audio.py:226-242`, `realtime.py:639-676`
- **問題**: `NamedTemporaryFile(delete=False)` 在 crash 時不清理；`suffix` 來自用戶 `file.filename`
- **影響**: 磁碟空間洩漏、潛在 symlink 攻擊、副檔名偽造

**修復**: 使用 `tempfile.mkstemp()` + 安全副檔名白名單 + `atexit` 清理

---

### C5. MCP 模板注入

- **檔案**: `python/yunshu_gateway/routers/mcp.py:633`
- **問題**: `str.format(**arguments)` 用戶可控 arguments 可存取 Python 內部屬性
- **影響**: 透過 `__class__.__mro__.__init__.__globals__` 等路徑洩漏資訊

```python
# mcp.py:633 — 不安全
template = prompt_def["template"].format(**arguments)
```

**修復**: 使用 `string.Template.safe_substitute()` 或限制 key 只允許 `\w+`

---

### C6. CORS 過度寬鬆

- **檔案**: `python/yunshu_gateway/main.py:163-167`
- **問題**: `allow_methods=["*"]` 開放 DELETE/PUT/PATCH；`allow_headers=["*"]` 接受任何自訂 header
- **影響**: 生產環境安全風險

**修復**: 限制為 `allow_methods=["GET","POST","OPTIONS"]`

---

### C7. PagedAttention block_table_stride 計算錯誤

- **檔案**: `metal/paged_attention.metal:52`
- **問題**: 用 `seq_len` 計算 stride 而非 `max_blocks`（block table 的 allocation width）
- **影響**: 當 `seq_len` 暗示的 block 數少於 allocation width 時，讀取錯誤記憶體位址

```metal
// paged_attention.metal:52 — 錯誤
int block_table_stride = (seq_len + kv_block_size - 1) / kv_block_size;
// 應為：block_table_stride = max_blocks（從 template 參數取得）
```

---

### C8. SDPA threadgroup 記憶體超限

- **檔案**: `metal/sdpa.metal:52-54`
- **問題**: `shared_k(16KB) + shared_v(16KB) + shared_scores(16KB) = 48KB` 超過 Apple GPU 32KB threadgroup 限制
- **影響**: pipeline 創建失敗或靜默記憶體損壞

---

### C9. SDPA head_dim > 32 時無限迴圈

- **檔案**: `metal/sdpa.metal:79-96`
- **問題**: `32 / head_dim` 當 `head_dim > 32` 時整數除法 = 0，`kv += 0` 導致無限迴圈
- **影響**: 掛死 GPU（適用於所有 head_dim >= 33 的模型，如 head_dim=64/128）

```metal
// sdpa.metal:79-96 — head_dim=128 時 kv += 0，無限迴圈
for (uint kv = simd_lane_id / head_dim; kv < kv_len; kv += 32 / head_dim)
```

---

### C10. server_metrics.py 死鎖

- **檔案**: `python/yunshu_engine/server_metrics.py:121-157`
- **問題**: 在 `with self._lock:` 內 `release()` → `save_alltime()` (再 acquire) → `acquire()`
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

### H1. `_init_memory_monitor()` 呼叫兩次

- **檔案**: `python/yunshu_engine/engine.py:300,303`

```python
self._init_memory_monitor()  # line 300
# ...
self._init_memory_monitor()  # line 303 — 重複
```

### H2. `_init_spec_decode()` 呼叫兩次

- **檔案**: `python/yunshu_engine/batched_engine.py:110,116`

```python
self._init_spec_decode()     # line 110
# warmup
self._init_spec_decode()     # line 116 — 重複
```

### H3. n_confirmed rollback 長度調整方向相反

- **檔案**: `python/yunshu_engine/n_confirmed_patch.py:309`
- **問題**: `cache.advance(S)` 讓 lengths +2，要 undo 1 步應 `-1` 但寫了 `+1`

```python
c.lengths = c.lengths + 1   # 錯誤：應為 c.lengths - 1
```

### H4. VLM engine `has_active_requests()` 永遠 False

- **檔案**: `python/yunshu_engine/vlm_engine.py:104-105`
- **問題**: ModelManager 的 LRU 驅逐依賴 `has_active_requests()`，VLM 永遠 False → 活躍中的 VLM 請求會被驅逐

```python
def has_active_requests(self) -> bool:
    return False  # ← 永遠 False
```

### H5. JSON schema 模式靜默丟棄 repetition/presence/frequency penalty

- **檔案**: `python/yunshu_engine/scheduler.py:931-941`
- **問題**: 當 `json_schema` 設定時，直接用 `base_sampler` 建構 constrained sampler，忽略所有 logits_processors

```python
# scheduler.py:931-941
# 前面辛苦建的 logits_processors（repetition/presence/frequency penalty）
# 在 json_schema 分支完全被忽略
if json_schema is not None:
    return make_constrained_sampler(base_sampler=base_sampler, ...)
```

### H6. Engine.load() 在錯誤線程載入模型

- **檔案**: `python/yunshu_engine/engine.py:306-308`
- **問題**: `load()` 在呼叫線程（通常是 asyncio event loop）同步載入模型，而非 MLX executor 線程
- **影響**: Metal buffers 建立在錯誤的 stream 上，可能導致 GPU 衝突

### H7. 兩個不相容的 RequestOutput 類別

- **檔案**: `engine.py:51` vs `request.py:79`
- **問題**: `engine.py` 的 `RequestOutput` 有 9 個欄位；`request.py` 的有 13 個欄位
- **影響**: 同時使用 Engine 和 EngineCore 時會遇到型別不相容

### H8. Embedding 端點缺 `await` 導致 crash

- **檔案**: `python/yunshu_gateway/routers/embeddings.py:104`
- **問題**: `manager.get_engine(model_id)` 是 async 方法但未 `await`
- **影響**: 回傳 coroutine 物件而非 engine，後續 `engine.embed(texts)` crash

```python
engine = manager.get_engine(model_id)  # ← 缺少 await
```

### H9. Completions 端點缺少 SSE keepalive

- **檔案**: `python/yunshu_gateway/routers/completions.py:154-232`
- **問題**: `_stream_completion` 未使用 `with_sse_keepalive()` 包裝
- **影響**: 長 prefill 期間客戶端超時斷線，GPU 白算

### H10. Anthropic streaming 雙重 `content_block_stop`

- **檔案**: `python/yunshu_gateway/routers/anthropic.py:505-512`
- **問題**: 當引擎生成零 token 時，text block 被 open 又 close 兩次
- **影響**: Anthropic SDK client 可能 crash 或誤解串流

### H11. PagedAttention 32x SIMD 浪費

- **檔案**: `metal/paged_attention.metal:68-76,120-127`
- **問題**: 所有 32 條 SIMD lane 做完全相同的 dot product 和 value accumulation
- **影響**: 效能只有應有的 1/32

### H12. `kv_block_offset` uint32 溢出

- **檔案**: `metal/common.metal:63-71`
- **問題**: `physical_block * kv_block_size * num_heads * head_dim` 大 KV cache 時超過 uint32 範圍
- **修復**: 改用 `ulong`（64-bit）運算

---

## 三、中等問題（MEDIUM）

### M1. Speculative decoder 目標模型永遠 greedy

- **檔案**: `speculative_decoder.py:579,613,632`
- **問題**: 用 `argmax` 選 token，忽略 temperature/top_p 設定
- **影響**: spec decode 永遠產生 greedy 輸出

### M2. Completions stop token 包含在輸出中

- **檔案**: `batched_engine.py:372-382`
- **問題**: stop token 先 append 再 break，tokenizer.decode 包含 stop token
- **影響**: OpenAI API 規範 stop token 不應出現在輸出

### M3. Fast path 靜默丟棄參數

- **檔案**: `batched_engine.py:476-478`
- **問題**: `_stream_generate_fast` 忽略 `min_p`、`frequency_penalty`、`presence_penalty`、`logit_bias`、`enable_thinking`、`json_schema`

### M4. Audio engine 使用錯誤 executor

- **檔案**: `audio_engine.py:538`
- **問題**: `run_in_executor(None, ...)` 用 asyncio 預設 thread pool 而非 MLX executor
- **影響**: GPU 工作可能在不同 thread 執行，Metal stream 衝突

### M5. Anthropic `stop_sequence` 欄位缺失

- **檔案**: `anthropic.py:309-320`
- **問題**: 當 `stop_reason="stop_sequence"` 時未回傳 `stop_sequence` 欄位
- **影響**: Anthropic SDK 期待此欄位

### M6. Anthropic `budget_tokens` 未執行

- **檔案**: `anthropic.py:284`
- **問題**: 提取了 `budget_tokens` 但從未傳入 engine
- **影響**: thinking 輸出不受 token 上限約束

### M7. `output_collector.py` class variable 跨實例共享

- **檔案**: `output_collector.py:31`
- **問題**: `_waiting_consumers: int = 0` 是 class variable 非 instance variable

### M8. `restore_rollback` 對非 SSM/KV 層提前返回 False

- **檔案**: `n_confirmed_patch.py:292-315`
- **問題**: 任何非 trimmable 非 SSM 的 cache 層導致函數返回 False，後續層未處理

### M9. Norm weight +1.0 mutation

- **檔案**: `mtp_patch.py:183`
- **問題**: `weights[k] = v + 1.0` 直接修改傳入的 weights dict
- **影響**: 第二次 load 會 double-shift

### M10. EngineCore TOCTOU on output_collectors

- **檔案**: `engine_core.py:470`
- **問題**: `dict(self._output_collectors)` 快照後，scheduler step 可能清理 collector

### M11. Embedding 同步阻塞 event loop

- **檔案**: `embeddings.py:117-165`
- **問題**: `_generate_embeddings` 同步執行 `model(input_ids)`，阻塞所有併發請求

### M12. VLM 串流建立獨立 Metal stream

- **檔案**: `vlm_engine.py:447`
- **問題**: `mx.new_thread_local_stream()` 違反單 GPU thread 模式

### M13. OpenAI streaming delta 每個 chunk 都包含 role

- **檔案**: `streaming.py:523`
- **問題**: OpenAI 規範 `role` 只在第一個 chunk 出現

### M14. VLM 回應回報 0 tokens

- **檔案**: `chat.py:647-654`
- **問題**: VLM 非串流回應 `prompt_tokens=0, completion_tokens=0`

### M15. MCP SSE zombie 連線

- **檔案**: `mcp.py:681-693`
- **問題**: `is_disconnected()` 每 30 秒檢查一次，殭屍連線浪費資源

### M16. Anthropic tool call regex 不支援巢狀 JSON

- **檔案**: `anthropic.py:166`
- **問題**: `\{[^}]*\}` 只匹配單層 JSON，巢狀物件會失敗

### M17. Tool call streaming 文字重複輸出

- **檔案**: `chat.py:772-785`
- **問題**: 部分 text 和 tool call 同時輸出

### M18. Context window 估計不準確（多模態）

- **檔案**: `chat.py:478-482`
- **問題**: 只計算文字部分，漏掉 image tokens

### M19. `shared_logits` 編譯期常數 vs 運行期 `kv_block_size`

- **檔案**: `metal/paged_attention.metal:53`
- **問題**: `threadgroup float shared_logits[PA_BLOCK_KV]` 可能不夠大

### M20. GQA head mapping 整數除法脆弱

- **檔案**: `metal/sdpa.metal:103-105`

---

## 四、低優先級問題（LOW）

| # | 位置 | 問題 |
|---|---|---|
| L1 | `ane_embedding.py` | 大量 stub/placeholder（hash-based tokenization, random embeddings） |
| L2 | `telemetry.py` | 整個模組是 stub，flush 直接清空 |
| L3 | `deltanet_inversion.py` | 標記為「不適用 BF16」，monkey-patch 未完成 |
| L4 | `metal_kernels.py:659` | `shell=True` subprocess |
| L5 | `json_schema.py:664` | 硬編碼 `range(151936)` vocab fallback |
| L6 | `model_registry.py:112` | module-level singleton 非 thread-safe |
| L7 | `engine.py:423` | `_make_sampler` 傳了 mlx-lm 不接受的 penalty 參數 |
| L8 | `kv_prefix_cache.py:50` | numpy detour 可能丟失 bf16 精度 |
| L9 | 多處 | `uuid.uuid4().hex[:12]` 只有 48-bit entropy |
| L10 | 多處 | `__import__("time")` 而非 module-level import |
| L11 | `streaming.py:700` | `TokenRateTracker` 定義但從未使用 |
| L12 | `streaming.py:733` | `StopSequenceDetector` 定義但從未使用 |
| L13 | `streaming.py:164` | `SSEKeepaliveWrapper` 被 `with_sse_keepalive` 取代 |
| L14 | `health.py` | 與 `main.py` 重複定義健康端點，health.py 未被 include |
| L15 | `mcp.py:75` | `MCPSession` 類別定義但從未實例化 |
| L16 | Anthropic | 缺少 `cache_creation_input_tokens` / `cache_read_input_tokens` |
| L17 | `bench.py:23-24` | 全域 mutable state 無鎖 |
| L18 | Prometheus | label 值未跳脫 |

---

## 五、架構問題

### 5.1 雙引擎並行 — 最大的技術債

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

### 5.2 Metal 雙源碼

```
metal/*.metal                    ← Makefile 編譯成 .metallib
python/yunshu_engine/metal_kernels.py  ← Python 字串 inline Metal source
```

**問題**：
- `.metal` 檔案已開始與 Python inline 分歧（Python 版有 SIMD striding，`.metal` 版沒有）
- `MetalKernelManager` 從未載入 `.metallib`，只用 JIT compilation
- 修 `.metal` 不影響運行中的系統

**建議**：擇一。要么全部用 `.metal` + precompile，要么全部用 Python inline + JIT。

### 5.3 Gateway → Engine 私有屬性耦合

17+ 處直接存取 `_entries`、`_model_manager`、`_engine` 等私有屬性：
- `models.py:24` → `manager._entries.values()`
- `engine/__init__.py:108` → `_model_manager._entries`
- `main.py:140,258` → `_engine.*`
- `metrics.py:143` → 多處私有屬性
- `monitoring.py:107` → 多處私有屬性
- `tenant_auth.py:98` → 多處私有屬性
- `tokenize.py:71` → 多處私有屬性

**建議**：添加 public accessor methods。

### 5.4 Eager imports 繞過 lazy 機制

`python/yunshu_engine/__init__.py:3-5`:
```python
from .batched_engine import BatchedEngine, GenerationOutput
from .engine import Engine, EngineConfig, RequestOutput, RequestState
```

這些 eager imports 立即拉入 `engine_core` → `scheduler` → 整個依賴鏈，使 `__getattr__` lazy import 機制無效。

### 5.5 `import mlx.core` 在 module level

`python/yunshu_engine/scheduler.py:30` 在 import 時就初始化 Metal device context，可能在非 GPU 環境 crash。

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
