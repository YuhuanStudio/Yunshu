<div align="center">

# Yunshu

**為 Apple Silicon 打造的快速、本地、單節點 LLM / VLM 推論引擎。**

單一進程，相容 OpenAI／Anthropic API，透過 MLX 在裝置端運行。重點是 LLM/VLM 解碼速度、
冷與快取首 token 延遲（TTFT）、前綴重用，以及完整的推論功能。
**Qwen3.8-27B 是第一個完整調校的模型。** 語音、Realtime 語音、圖片生成、影片輸入與嵌入
是支援能力；核心是 LLM/VLM 服務。Yunmo 是其中一個使用者。

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

[English](README.md) · [简体中文](README.zh-CN.md) · **繁體中文**

</div>

最新發布版：**v0.1.2（2026-10-02）**；**v0.1.1（2026-09-29）**。本頁也涵蓋
**尚未發布的 main**；見 [CHANGELOG](CHANGELOG.md)。

## 推論功能

- **推測解碼。** 自動尋找相符的 DFlash2 草稿模型，否則使用 checkpoint 的 MTP 頭。
  不隨批次大小改變的解碼／驗證 kernel 支援該路徑內的貪婪輸出一致性；取樣草稿使用
  依位置決定的抽樣。main 新增最多 32 列的不變驗證，以及 MTP lane 的提示複製草稿
  （預設開啟，`YUNSHU_SPEC_COPY_ROWS=0` 關閉）。樹狀草稿仍是實驗功能，預設關閉。
  一般 runner 與 invariant lane 並非全面逐 token 相同：見
  [準確度證據](docs/guides/ACCURACY.md) 與 [測量限制](docs/BENCHMARKS.md)。
- **前綴重用。** 混合架構 Qwen3.5 家族的 checkpoint 保留 attention KV 與循環狀態；
  文字與媒體鍵避免錯誤重用。RAM APC 加上預設有容量上限的 SSD 快取，支援重複回合與重啟。
  main 新增可選的 WARM RAM 與下層儲存，以實測還原成本選擇來源。無損 WARM 預設關閉；
  int8/int4 快取格式與 KV 量化是使用者自行開啟的有損選項。見
  [快取層級](docs/guides/KV_CACHE_MATRIX.md)。
- **完整請求功能。** 工具、JSON-schema 約束、stop、logprobs、分離的推理／內容串流、
  模板支援的 reasoning effort 與取消。`/v1/models` 公告模型能力；不支援的請求明確報錯。
  main 的 JSON 約束預設使用 llguidance；無法強制執行的語法會被拒絕。
- **程式代理。** Claude Code、Codex 與 opencode 使用原生 Messages／Responses／Chat API，
  支援伺服器端搜尋／擷取／MCP，以及 Files、Batches、Conversations。`yunshu launch`
  提供模型限制；`yunshu statusline` 在 Claude Code 顯示引擎狀態。
  [相容性證據](docs/guides/AGENT_COMPAT.md) 區分已驗證功能與客戶端限制。
- **本地診斷。** `yunshu doctor`、`yunshu cache status`、`yunshu cache gc`、
  `yunshu diagnose`；診斷資料留在本機，不含提示。

## 快速開始

需要 Apple Silicon、macOS 14+、Python 3.13+ 與 [uv](https://docs.astral.sh/uv/)。

```bash
uv tool install "yunshu[vision]"
yunshu doctor
yunshu pull mlx-community/Qwen3.5-9B-MLX-4bit
yunshu serve -m mlx-community/Qwen3.5-9B-MLX-4bit
```

伺服器監聽 `http://127.0.0.1:8000`。模型存於 `~/.yunshu/models`；`serve -m org/name`
也會尋找 Hugging Face 快取，只在需要時下載。
`yunshu config set models_dir /path/to/models` 可更改位置。

### Qwen3.8-27B

```bash
yunshu pull Jundot/Qwen3.8-27B-oQ4e-mtp
yunshu pull incoai/Qwen3.8-27B-DFlash2
yunshu doctor -m Jundot/Qwen3.8-27B-oQ4e-mtp
yunshu serve -m Jundot/Qwen3.8-27B-oQ4e-mtp
```

安裝相符草稿模型後，啟動日誌應顯示 `Speculative decoding: dflash`；`doctor` 顯示選用路徑。
`YUNSHU_VLM_DRAFT=mtp` 強制 MTP，`YUNSHU_VLM_DRAFT=off` 停用草稿，絕對路徑可指定草稿模型。
記憶體需求取決於權重、上下文、草稿與快取預算；請在自己的 Mac 上用 `doctor` 檢查模型，
不要將 benchmark 佔用量當作最低 RAM 保證。

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
    extra_body={"reasoning_effort": "medium"},
)
print(r.choices[0].message.content)
```

未設定 `--auth-token` 時，任何 key 都可以。單模型模式的 `local` 是佔位名稱；
多模型模式請使用 `/v1/models` 中的 id。

使用尚未發布的 main／從原始碼開發：

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
uv sync --extra vision
uv run yunshu serve -m Jundot/Qwen3.8-27B-oQ4e-mtp
```

`uv.lock` 固定依賴；main 需要 MLX 0.32.3+、mlx-lm 0.32.0+、mlx-vlm 0.7.4+。
`yunshu model list` 列出本地模型。`yunshu service install -m <model>` 安裝登入服務
（[服務指南](docs/guides/SERVICE.md)）；每個指令都有 `--help`。

**沒有遙測。** 模型下載、設定的 MCP／搜尋供應商、請求觸發的網頁擷取可能連外。
不傳送使用分析或崩潰報告。main 可選的 `YUNSHU_SERVE_LOG` 只在本地記錄數字，
不含提示、輸出或 token id。

## 效能

下列每筆測量都來自 [BENCHMARKS](docs/BENCHMARKS.md) 或
[有日期的 PERF_TREND 紀錄](docs/reports/PERF_TREND.md)。除特別註明外，為 M5 Max／128 GB、
Qwen3.8-27B Jundot oQ4e-mtp。日期是測量日期，不是發布日期。

**2026-10-02 tfbench**，記錄於 `4e1338c4`：貪婪、輸出 256 token，每格三次獨立
伺服器 session，取中位數。Yunshu 是當日早上的 **未發布 main（未記錄 SHA），實際使用 MTP**；
TensorFold 使用 DFlash2。這早於 main 的 prefill／提示複製／寬驗證合併，並非相同草稿模型的 A/B。

| Context / output | TensorFold 0.6.1 tok/s / cold TTFT s | Yunshu main (MTP) tok/s / cold TTFT s |
|---|---|---|
| 1K code | 140.5 / 1.2 | 69.3 / 1.4 |
| 1K prose | 72.4 / 1.2 | 52.9 / 1.4 |
| 8K code | 80.6 / 8.4 | 60.2 / 11.1 |
| 8K prose | 67.4 / 8.5 | 57.5 / 11.1 |
| 32K code | 91.2 / 39.4 | 58.7 / 47.2 |
| 32K prose | 55.1 / 39.4 | 46.3 / 47.7 |

TensorFold 在以上各格皆領先。**Splash 在 2026-09-27 探索性 TTFT 測試也領先**：
3,323-token 提示，冷／重複為 3.221／0.130 s，Yunshu 直接 VLMEngine 為 3.511／3.512 s。
Splash 1.1.0 使用自己的量化模型與 INT8 KV；單次測量、共享 GPU，Yunshu SHA 未知。
權重與快取條件不同，不能只歸因於引擎。上述合併之後尚無更新的完整比較。

**2026-10-02 稍後的 main 測量，尚未發布：**

| Workload | Before | After | Commit / source |
|---|---|---|---|
| Cold TTFT, 8K | 11.0 s | 8.57 s | `a4d71bc8`, BENCHMARKS |
| Cold TTFT, 32K | 47.5 s | 38.3 s | `a4d71bc8`, BENCHMARKS |
| MTP prompt-copy, code turn 2, 32K | 63 tok/s | 100–138 tok/s | `c666be70` / `bb4895ca`, PERF_TREND |
| MTP prompt-copy, code turn 2, 8K | 62 tok/s | 86 tok/s | same |

prefill 數字是有日期的合併紀錄；公開資料未提供重複次數／區間。
提示複製的貪婪 digest 開關一致；散文差異在 ±3% 單次測量雜訊內。
冷 prefill 數值可能與舊版不同；APC namespace 包含 prefill 設定。
這些結果不代表全面解碼加速，也不是硬體上限宣告。

**程式代理測試，部分完成，測於 2026-10-02**（PERF_TREND／`2815311c`）：

| Snapshot / agent | Pass / runs | Rate (Wilson 95%) | Wall median / p90 s | Decode median tok/s |
|---|---|---|---|---|
| prod `c4e2b244+` / opencode | 34 / 41 | 83% (69–91%) | 337 / 1200 | 22.2 |
| prod4 `d225f16c` / Claude Code | 14 / 15 | 93% (70–99%) | 93 / 309 | 60.8 |
| prod4 `d225f16c` / opencode | 9 / 10 | 90% (60–98%) | 112 / 137 | 56.9 |

失敗保留在分母，包括 prod 的五次逾時。prod4 涵蓋 20 個任務中的 10 個，各重複 1–3 次；
`d225f16c` snapshot 已包含在 v0.1.2。不同 snapshot 與任務組合不能用來宣稱配對加速。
Codex 與 TensorFold 代理結果仍待完成。準確度 Tier 3 與 APC replay 結果，包含退步與
未完成子集，見 [BENCHMARKS](docs/BENCHMARKS.md)。

## 模型與服務路徑

| 模型 | 服務路徑 | 範圍 |
|---|---|---|
| Qwen3.5／3.6／3.8 家族；先調校 Qwen3.8-27B | VLM batch runner | APC、支援家族模型的 MTP/DFlash、逐列取樣 |
| 其他 mlx-vlm 模型（Gemma、GLM、Qwen-VL、Omni 等） | 同一個 VLM batch runner | 依模型支援媒體與工具；快取結構允許時有 APC；無家族專屬推測解碼 |
| 純文字 mlx-lm 模型 | 單請求 `generate_step` 快速路徑 | 依模型支援前綴快取、約束、工具與 logprobs；並行請求依序執行 |

所有 GPU 工作在同一個 MLX 執行緒。預設 VLM runner 共享並行解碼列；單獨的支援請求可用草稿。
多列草稿使用實驗性的 `YUNSHU_ROUND_DRIVER`，預設關閉，有實測延遲取捨。
main 的 M01 將文字引擎整理為多個模組，未改變服務路徑。實際能力請看模型卡，不能只看家族名稱。

## 其他支援能力

| 能力 | API／範例 | Extra |
|---|---|---|
| Qwen3-Omni 原生語音對語音 | `/v1/omni/speech/stream`、[talk.py](examples/talk.py) | `omni` |
| Realtime 語音、ASR、TTS | `/v1/realtime`、`/v1/audio/transcriptions`、`/v1/audio/speech` | `audio`（原生 Omni 需 `omni`） |
| 圖片生成 | `/v1/images/generations` | `generation` |
| 嵌入／重排 | `/v1/embeddings`、`/v1/rerank` | `embeddings` |
| 文字 WebSocket／Responses WebSocket／Unix socket | `/v1/stream`、`/v1/responses`、`yunshu serve --uds PATH` | core |

這些能力各有模型／後端需求，不在 LLM/VLM 效能表的測試範圍。
適用的 VLM 支援影片輸入；影片生成的引擎後端與公開 HTTP 路由已移除。
WebRTC 與 HTTP/2 尚未實作。[API 表面](docs/guides/API_SURFACE.md) 與
[傳輸方式](docs/guides/TRANSPORTS.md) 列出驗證與限制。

## 設定與文件

設定經過同一個 registry：環境變數、TOML（`yunshu serve --config yunshu.toml`），
或 `yunshu serve --set KEY=VALUE`；`yunshu config` 顯示生效值與來源。

- [連接客戶端](docs/guides/CLIENTS.md)
- [疑難排解](docs/guides/TROUBLESHOOTING.md)
- [API 參考](docs/API.md)
- [設定參考](docs/CONFIGURATION.md)
- [文件索引](docs/README.md)

## 基礎與授權

[MLX](https://github.com/ml-explore/mlx)、[mlx-lm](https://github.com/ml-explore/mlx-lm)、
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)、[mlx-audio](https://github.com/Blaizzy/mlx-audio)。
來自 [oMLX](https://github.com/jundot/omlx) 與 [TensorFold](https://github.com/ashhart/TensorFold)
的 kernel 授權通知見 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。
Yunshu 採 Apache 2.0：[LICENSE](LICENSE)。
