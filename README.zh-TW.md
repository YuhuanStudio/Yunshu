<div align="center">

# Yunshu

**為 Apple Silicon 打造的快速本地 LLM / VLM 推論引擎。**

單一進程，相容 OpenAI 與 Anthropic API，透過 MLX 在裝置端執行。為單台 Mac 的低延遲而設計：
首字快、無損解碼快，並重用已經算過的前綴。第一個完整調校的模型是 **Qwen3.8-27B**。

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

[English](README.md) · [简体中文](README.zh-CN.md) · **繁體中文**

</div>

---

## 亮點

- **無損推測解碼**：DFlash2 或 MTP 草稿，搭配不隨批次改變的 kernel，開啟推測解碼時的貪婪輸出
  與關閉時逐 token 相同。
- **混合模型的前綴快取**：為 attention + GatedDeltaNet 模型保存精確 checkpoint，放在 RAM 與 SSD，
  重開後仍在；可再加儲存層。
- **快速路徑上的完整 API**：工具呼叫、JSON schema、停止序列、logprobs、推理、取消，涵蓋 OpenAI
  Chat / Responses、Anthropic Messages 與 Ollama。
- **原生支援程式碼 agent**：Claude Code、Codex、opencode 透過各自的 API 運作，含伺服器端網頁
  搜尋／抓取與 MCP。
- **預設無損**：任何可能改變輸出的東西都是要自己開的設定。
- **本地且私密**：不收集遙測；診斷資料不含 prompt。

## 快速開始

需要 Apple Silicon、macOS 14 以上、Python 3.13 以上與 [uv](https://docs.astral.sh/uv/)。

```bash
uv tool install "yunshu[vision]"
yunshu doctor                                   # 檢查這台 Mac，並說明怎麼修
yunshu pull mlx-community/Qwen3.5-9B-MLX-4bit
yunshu serve -m mlx-community/Qwen3.5-9B-MLX-4bit
```

伺服器在 `http://127.0.0.1:8000`。任何 OpenAI 或 Anthropic 客戶端都能直接使用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")  # 任意 key 皆可
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "用一句話解釋 MLX。"}],
)
print(r.choices[0].message.content)
```

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8000", api_key="local")
msg = client.messages.create(
    model="local", max_tokens=512,
    messages=[{"role": "user", "content": "用一句話解釋 MLX。"}],
)
print(msg.content[0].text)
```

模型放在 `~/.yunshu/models`（用 `yunshu config set models_dir PATH` 搬移）；`serve -m org/name`
也會找 Hugging Face 快取，只有需要時才下載。`yunshu service install -m <model>` 讓伺服器在登入時啟動。

### Qwen3.8-27B

```bash
yunshu pull Jundot/Qwen3.8-27B-oQ4e-mtp
yunshu pull incoai/Qwen3.8-27B-DFlash2          # 選用的草稿模型，會自動使用
yunshu serve -m Jundot/Qwen3.8-27B-oQ4e-mtp
```

裝好草稿模型後，啟動日誌會顯示 `Speculative decoding: dflash`；沒裝時由模型自帶的 MTP 頭產生草稿。
`yunshu doctor -m <model>` 會回報選用的路徑，以及這台 Mac 的記憶體是否放得下。

### 從原始碼執行

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git && cd Yunshu
uv sync --extra vision
uv run yunshu serve -m <model>
```

## 運作方式

```
 OpenAI / Anthropic / Ollama 客戶端 ──► FastAPI gateway（單一進程）
                                          │  請求驗證、工具／推理解析、
                                          │  伺服器端工具（網頁搜尋／抓取／MCP）
                                          ▼
                                  引擎（單一 MLX 執行緒）
          ┌────────────────────────────────┴───────────────────────────────┐
   VLM batch runner（所有 mlx-vlm 模型）                純文字快速路徑（mlx-lm 模型）
   共享連續批次、每列各自取樣                           單請求 generate_step
   推測解碼通道：DFlash2 / MTP / prompt-copy
   前綴快取：RAM ─► SSD ─► 選用的儲存層
```

所有 GPU 工作都在同一條 MLX 執行緒上，請求之間不會互搶 GPU。每個回應在自己的生成結束時就回傳。

### 推測解碼

Qwen3.5 家族的單一請求在推測解碼通道中解碼：

- **DFlash2**：獨立的區塊草稿模型，每輪提出多個 token；裝了相符的草稿模型就自動使用。
- **MTP**：checkpoint 自帶的多 token 預測頭；沒有草稿模型時的後備。
- **Prompt-copy 草稿**：當輸出開始重複 prompt 裡的文字（改程式碼、引用工具結果、多輪 agent），
  通道會提出那段文字的後續，並在同一次驗證中檢查。預設開啟（`YUNSHU_SPEC_COPY_ROWS`，設 `0` 關閉）。

所有解碼與驗證的矩陣乘法都走**不隨批次改變的 kernel**：一個 token 不論單獨驗證或和其他列一起驗證，
算術都相同。這讓開關推測解碼時的貪婪輸出完全一致，取樣輸出在依位置決定的取樣下也精確一致。
用 `YUNSHU_VLM_DRAFT` 選擇草稿（`mtp`、`off` 或草稿模型路徑）。

### 前綴快取（APC）

Qwen3.5 家族混合了 attention 層與遞迴的 GatedDeltaNet 層。遞迴狀態不能像 KV 快取那樣切回較早的
token，所以一般的前綴快取無法使用。Yunshu 在前綴邊界保存**精確的 checkpoint**（KV 加遞迴狀態），
以文字 token 及圖片像素／音訊特徵為鍵，所以快取命中的結果和冷預填完全相同。

| 層 | 位置 | 預設 |
|---|---|---|
| HOT | RAM 中可直接使用的陣列 | 開啟，依可用記憶體決定大小（`YUNSHU_VLM_APC_MEMORY_GB`） |
| WARM | RAM 中壓縮存放（無損 zstd，或有損 int8 / int4） | 關閉（`YUNSHU_VLM_APC_WARM`） |
| SSD | `~/.yunshu/cache/apc`，一個全域磁碟預算並保留剩餘空間，重開後仍在 | 開啟（`YUNSHU_VLM_APC_DISK`、`_DIR`、`_GB`） |
| 儲存層 | 外接 SSD、HDD、NAS（`YUNSHU_VLM_APC_DISK_TIERS`） | 關閉；會量測每個磁碟的速度，只有還原比重算快時才使用 |

重複或修改過的長 prompt、多輪對話與 agent 迴圈，會從最近的 checkpoint 還原，不必重新預填。
對話變長時，同一段對話較舊的 checkpoint 會被取代，不會越堆越多。`yunshu cache status` 與
`yunshu cache gc` 可檢視與清理 SSD 快取。

### 結構化輸出

JSON schema、JSON object、regex 與 grammar 約束在解碼時強制執行（預設使用 llguidance），也包括
工具呼叫的參數。不支援的 schema 寫法會明確回錯，而不是默默忽略。

## API 相容性

| API | 路由 |
|---|---|
| OpenAI | `/v1/chat/completions`、`/v1/completions`、`/v1/responses`（HTTP 與 WebSocket）、`/v1/embeddings`、`/v1/models`、`/v1/audio/*`、`/v1/images/*`、`/v1/realtime`、`/v1/files`、`/v1/batches` |
| Anthropic | `/v1/messages`（thinking、tools、`cache_control`、伺服器工具 `web_search` / `web_fetch`、`mcp_servers`）、`/v1/messages/count_tokens`、`/v1/messages/batches`、Files |
| Ollama | `/api/chat`、`/api/generate` 與模型相關路由 |
| Yunshu 擴充 | 請求即時階段（`/v1/requests`）、以請求 id 取消、預熱、截止時間、佇列標頭、串流中的預填進度、`/v1/yunshu/status` |

Chat 支援的參數：`tools` / `tool_choice` / `parallel_tool_calls`、`response_format`（`json_object`、
strict `json_schema`）、`stop`、`logprobs` / `top_logprobs`（串流也有）、`n`、`seed`、penalty、
`logit_bias`、`reasoning_effort`（推理內容另外回傳）、含 usage 的串流，以及 `usage` 中的快取 token 數。
錯誤使用各 API 自己的格式。擴充欄位都有命名空間（`x_yunshu`、`X-Yunshu-*`），官方 SDK 會忽略。
完整矩陣與每一列的驗證方式見 [API surface](docs/guides/API_SURFACE.md)。

## 程式碼 agent

```bash
yunshu launch claude      # 或：codex、opencode
```

`yunshu launch` 會寫好客戶端設定（base URL、模型、上下文長度與輸出上限、reasoning effort）並啟動
agent。對 Claude Code 還會裝上狀態列，即時顯示預填進度、解碼速度與快取命中。

- **Claude Code**：Messages API，含串流、thinking、`/context` 用的 `count_tokens`、模型探索，以及由
  Yunshu 在伺服器端執行的 WebSearch 工具。
- **Codex**：Responses API，含推理項目、function call、本地壓縮與 `web_search`。
- **opencode**：Chat Completions，含工具與 usage。

伺服器端網頁搜尋使用可設定的後端（例如 SearXNG）；請求中指定的 MCP 伺服器由 gateway 連線。
各 agent 實際呼叫了什麼、怎麼驗證的，見 [Agent 相容性](docs/guides/AGENT_COMPAT.md)。

## 效能

Qwen3.8-27B（oQ4e），M5 Max 128 GB，單一請求，貪婪解碼。方法、原始結果與完整比較表在
[docs/BENCHMARKS.md](docs/BENCHMARKS.md)。

| | Yunshu | TensorFold 0.6.1 |
|---|---|---|
| 冷啟動首字延遲，8K prompt | 8.6 秒 | 8.5 秒 |
| 冷啟動首字延遲，32K prompt | 38.3 秒 | 39.4 秒 |
| 重複或修改過的長 prompt | 從前綴快取還原，不必重新預填 | — |
| 解碼，短程式碼 prompt | 約 90–98 tok/s（DFlash2） | 約 140 tok/s（DFlash2） |

目前 TensorFold 的單請求解碼較快，縮小這個差距是正在進行的主要工作。推測解碼永遠不會改變
Yunshu 的貪婪輸出。相對於原版 MLX 路徑的準確度，分三個層次檢查（logit 對齊、貪婪分歧、成對下游評測），
見 [準確度](docs/guides/ACCURACY.md)。

## 模型

| 模型 | 服務路徑 |
|---|---|
| Qwen3.5 / 3.6 / 3.8 家族（優先調校 Qwen3.8-27B） | VLM batch runner，含前綴快取與 MTP / DFlash2 推測解碼 |
| 其他 mlx-vlm 模型（Gemma、GLM、Qwen-VL、Qwen-Omni…） | 同一個 runner；模型支援時可輸入圖片、音訊、影片；快取版面允許時使用前綴快取 |
| 純文字 mlx-lm 模型 | 單請求快速路徑，含約束、工具與 logprobs |

`/v1/models` 會回傳每個模型的資訊卡：上下文長度、輸出上限，以及實際支援哪些輸入與功能。

## 其他能力

| 能力 | 端點 | Extra |
|---|---|---|
| Qwen3-Omni 語音對語音 | `/v1/omni/speech/stream`（[範例](examples/talk.py)） | `omni` |
| Realtime 語音、ASR、TTS | `/v1/realtime`、`/v1/audio/transcriptions`、`/v1/audio/speech` | `audio` |
| OCR | `/v1/ocr`（GLM-OCR） | `vision` |
| 圖片生成與編輯 | `/v1/images/generations`、`/v1/images/edits` | `generation` |
| Embeddings、rerank、相似度 | `/v1/embeddings`、`/v1/rerank`、`/v1/score` | `embeddings` |

## 命令列

| 指令 | 用途 |
|---|---|
| `yunshu doctor` | 檢查這台 Mac、相依套件與模型，並說明怎麼修 |
| `yunshu pull` / `yunshu model` | 下載與管理模型 |
| `yunshu serve` / `yunshu service` | 執行伺服器，或安裝成登入時啟動的服務 |
| `yunshu launch` / `yunshu statusline` | 啟動接好 Yunshu 的程式碼 agent；引擎即時狀態列 |
| `yunshu chat`、`complete`、`embed`、`transcribe`、`speak`、`ocr`、`image` | 在終端機使用執行中的伺服器 |
| `yunshu status`、`cancel` | 伺服器狀態、取消進行中的請求 |
| `yunshu config` | 實際生效的設定與來源 |
| `yunshu cache status` / `gc` | 檢視與清理 SSD 前綴快取 |
| `yunshu bench`、`eval`、`diagnose` | 效能測試、準確度評測、系統診斷 |

每個指令都有 `--help`。

## 設定

所有設定都走同一個註冊表：環境變數、TOML 檔（`yunshu serve --config yunshu.toml`）或
`--set KEY=VALUE`。`yunshu config` 會顯示實際生效的值與來源。常用的：

| 設定 | 用途 |
|---|---|
| `YUNSHU_VLM_DRAFT` | 草稿選擇：`mtp`、`off` 或草稿模型路徑 |
| `YUNSHU_SPEC_COPY_ROWS` | prompt-copy 草稿寬度（`0` = 關閉） |
| `YUNSHU_VLM_APC_MEMORY_GB`、`YUNSHU_VLM_APC_DISK_GB`、`YUNSHU_VLM_APC_DISK_DIR` | 前綴快取的 RAM、SSD 預算與位置 |
| `YUNSHU_VLM_APC_DISK_TIERS` | 額外的儲存層，例如 `/Volumes/Ext/apc@200,/Volumes/NAS/apc` |
| `YUNSHU_VLM_APC_WARM`、`YUNSHU_KV_PRECISION` | 有損的省記憶體選項（預設關閉） |
| `YUNSHU_AUTH_TOKEN`、`YUNSHU_QUEUE_LIMIT` | API key 與請求佇列上限 |

所有設定見 [設定](docs/CONFIGURATION.md)。

## 文件

- [客戶端](docs/guides/CLIENTS.md)：curl、OpenAI / Anthropic SDK、Open WebUI、agent
- [API surface](docs/guides/API_SURFACE.md) 與 [API 參考](docs/API.md)
- [Agent 相容性](docs/guides/AGENT_COMPAT.md)
- [KV 快取分層](docs/guides/KV_CACHE_MATRIX.md) 與 [prompt caching API](docs/guides/PROMPT_CACHING_APIS.md)
- [效能測試](docs/BENCHMARKS.md) 與 [準確度](docs/guides/ACCURACY.md)
- [背景服務](docs/guides/SERVICE.md)、[疑難排解](docs/guides/TROUBLESHOOTING.md)、[變更紀錄](CHANGELOG.md)

## 隱私

不收集遙測、使用統計或當機報告。Yunshu 只在下載模型、連到你設定的網頁搜尋／MCP 服務，以及請求
要求抓取網頁時才對外連線。

## 基礎與授權

[MLX](https://github.com/ml-explore/mlx)、[mlx-lm](https://github.com/ml-explore/mlx-lm)、
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)、[mlx-audio](https://github.com/Blaizzy/mlx-audio)。
來自 [oMLX](https://github.com/jundot/omlx) 與 [TensorFold](https://github.com/ashhart/TensorFold)
的 kernel 保留其授權聲明，見 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。Yunshu 採用
Apache 2.0：[LICENSE](LICENSE)。
