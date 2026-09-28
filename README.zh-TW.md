<div align="center">

# Yunshu

**為 Apple Silicon 打造的快速本地 LLM / VLM 推理引擎。**

單一程序,相容 OpenAI 與 Anthropic API,透過 MLX 在裝置端執行。為單機低延遲而設計:首 token 快、
無損解碼快,並以前綴重用跳過已經算過的部分。第一個完整調校的模型是 **Qwen3.8-27B**。

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

[English](./README.md) · [简体中文](./README.zh-CN.md) · **繁體中文**

</div>

---

## 與眾不同之處

- **無損推測解碼。** 使用 checkpoint 自帶的 MTP 頭,或外部 DFlash drafter。會推測的請求,其所有
  解碼與驗證矩陣乘都走同一顆 batch-invariant kernel,所以 greedy 下開推測與不開推測的輸出逐 token
  相同 —— 即 Splash 所說的無損。非精確的快速驗證也有,但需手動開啟。
- **混合架構模型的前綴快取。** Qwen3.5 家族把注意力層和遞迴的 GatedDeltaNet 層混在一起,一般的
  KV 快取切不開。Yunshu 保存精確的 checkpoint,以文字與圖片像素共同作為鍵,預設 8 GiB 記憶體,
  可再加一層 SSD。重複或只改尾巴的長 prompt 不必重新 prefill。
- **以輸出驗證過的驗證 kernel。** GatedDeltaNet、注意力、5-bit 矩陣乘的驗證 kernel,部分取自
  oMLX,每一顆都經過同 checkpoint A/B 才採用。
- **快速路徑上有完整 API。** 工具呼叫、JSON-schema 約束、停止序列、logprobs、串流推理/內容分離、
  `reasoning_effort` 直接傳給支援它的 chat template(Qwen3.8),以及客戶端斷線時取消生成。

## 快速開始

> **需要 [uv](https://docs.astral.sh/uv/)。** 還沒上 PyPI —— 從原始碼以 `uv sync` 安裝,
> 它會裝 `uv.lock` 中固定的版本(MLX 0.32、上游 `mlx-vlm` 0.7.3+)。

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
uv sync --extra vision       # LLM + VLM(Qwen3.5 / 3.6 / 3.8 需要)
# 或:uv sync --all-extras   # 所有模態

uv run yunshu serve -m /path/to/Qwen3.8-27B-mlx --port 8000
```

任何 OpenAI 客戶端都能直接用:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")  # 任意 key 都可以

# 單模型模式:模型名只是佔位,伺服器服務的是你載入的那個模型。
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "用一句話解釋 MLX。"}],
    extra_body={"reasoning_effort": "medium"},
)
print(r.choices[0].message.content)
```

> **開發環境**:`just setup` 後 `YUNSHU_MODEL=<model> just dev`。
> **文件**:[API 參考](docs/API.md) · [設定參考](docs/CONFIGURATION.md)。

## 效能

量測環境:M5 Max(128 GB)、Qwen3.8-27B、2026-09-28。除特別註明外皆為同一個 Jundot `oQ4e-mtp`
checkpoint;原始資料與方法見
[docs/research/runs/2026-09-28-matrix](docs/research/runs/2026-09-28-matrix/README.md)。

| 引擎 | 能力檢查 | 對話 TTFT(熱) | 8K prompt:冷 / 重複 / 改尾 | 解碼 tok/s |
|---|---|---|---|---|
| **Yunshu**(預設:MTP 深度 6、batch-invariant) | 34/34 | 0.194 s | 8.45 / 0.115 / 0.258 s | 80 |
| **Yunshu**(DFlash2 + 快速驗證,需開啟) | 31/31 | 0.185 s | 8.71 / 0.112 / 0.239 s | 86 |
| mlx-vlm 0.7.3 server(APC) | 27/28 | 0.212 s | 8.60 / 0.108 / 0.265 s | 32 |
| oMLX.app 0.7(MTP + 快取) | 31/31 | 0.312 s | 8.60 / 0.361 / 0.376 s | 85 |
| Splash 1.1(自家量化模型 + DFlash2) | 31/31 | 0.206 s | 7.88 / 0.131 / 7.88 s | 119 |

依輸出類型的無損解碼(同 checkpoint、程序內、greedy、384 token;tok/s):

| 解碼方式 | 程式碼 | 文章 | JSON 類 | 開/關推測相同 |
|---|---|---|---|---|
| **預設**:batch-invariant + packed、MTP 深度 6 | 88.6 | 59.9 | 67.3 | 是(已測)¹ |
| 舊預設:精確驗證 kernel、MTP 深度 3 | 57–67 | 50–53 | 58–62 | 是(已測)¹ |
| 非精確快速驗證(需開啟) | 83.8 | 59.9 | 66.7 | 否 |

¹ 短 prompt 以及 1.2K／16.5K token 上下文下(各 384 token),開推測與不開推測的 greedy 輸出在每個任務
都逐 token 相同。矩陣乘已逐列一致;推測驗證的注意力走的 MLX kernel 和單列解碼不同,所以位元層面的
logits 可能有差,其他輸入下仍可能偶有 token 不同。oMLX 的逐列精確注意力可以消除這點,但解碼慢
30–50%(`YUNSHU_MTP_ROW_EXACT=1`)。

MMLU-Pro 300 題、同時 8 題、最多 16384 token、`reasoning_effort=medium`(準確率與長時間穩定性;
每家設定相同):

| 引擎 | 答對 | 耗時 | 總吞吐 tok/s | 記憶體峰值 |
|---|---|---|---|---|
| **Yunshu**(共用批次,commit fbbb1378) | 250 / 300 | 46.5 分 | 88 | 45 GiB(結束回到 17) |
| Splash 1.1 | 252 / 300 | 17.2 分 | 223 | 67 GiB |
| oMLX.app | 229 / 300(27 題被它的 prefill 記憶體守衛拒絕) | 29.2 分 | 120 | 75 GiB |

速度全測(每個 prompt 唯一、不吃快取;生成 128 token;未註明單位者為 tok/s):

| | Yunshu | oMLX | Splash |
|---|---|---|---|
| 8K / 131K / 200K token 的 TTFT | 8.4 / 209 / 394 s | 8.5 / 214 / 401 s | 7.9 / 207 / 390 s |
| 1K / 32K / 200K 之後的解碼 | 58 / 44 / 17 | 71 / 60 / 29 | 101 / 48 / 66 |
| 8 個 1K prompt 同時,總吞吐 | 61 | 53 | 70 |

現況:
- 前綴重用與熱 TTFT 是量到最好的;冷 prefill 已到硬體上限(三家相差約 5% 內)。
- 準確率與 Splash 相當;每次長時間測試都 0 錯誤。
- **落後 Splash** 的地方:長上下文解碼(它的 KV 用 INT8),以及並發長輸出(上游批次快取把每列補到
  最長那列)。量化 KV 與每列獨立長度的 KV 快取正在驗證中,用來補上這兩點。
- 60 分鐘混合 soak(對話、長文件、圖片、工具、JSON schema、思考、中途斷線)跑完 699 個請求,
  伺服器錯誤 0,記憶體沒有成長(footprint 17–26 GiB)。

Yunshu 的矩陣比舊測試多兩項:logprobs 與串流推理分離。長期基準紀錄見
[docs/reports/PERF_TREND.md](docs/reports/PERF_TREND.md)。

## 支援的模型

| 層級 | 模型 | 路徑 | 可得到的功能 |
|---|---|---|---|
| 1 —— 已調校並量測 | Qwen3.5 / 3.6 / 3.8 家族(文字 + 圖片) | VLM batch runner | 前綴快取(記憶體 + SSD)、MTP / DFlash 無損推測解碼、上述所有 API 功能 |
| 2 —— 支援 | 任何 `mlx-lm` 文字模型 | 單請求快速路徑(`generate_step`) | KV 前綴快取、工具、JSON schema、logprobs;可開啟 n-gram 推測、KV 量化 |
| 2 —— 支援 | 其他 `mlx-vlm` 模型(GLM、Qwen-VL、Gemma-4、Qwen3-Omni、Nemotron-Omni 等) | 同一個 VLM batch runner | 連續批次、前綴快取(使用 sliding window 的模型除外)、圖片 / 音訊 / 影片、上述所有 API 功能;無推測解碼 |

2026-09-28 這一輪只重新量測了第 1 層;第 2 層的 VLM 之後才改走 runner,仍需實機冒煙測試。

## 其他模態

以下功能在同一個伺服器裡,透過可選 extras 安裝。**2026-09-28 這一輪都沒有重新驗證**,
這一輪只涵蓋 LLM/VLM。

| 模態 | 端點 | 後端 | Extra |
|---|---|---|---|
| 原生語音對語音(Qwen3-Omni Thinker→Talker,串流) | `POST /v1/omni/speech/stream` | `mlx-vlm` | `omni` |
| 即時語音 | `WS /v1/realtime` | omni,或 ASR → LLM → TTS | `audio` |
| ASR | `/v1/audio/transcriptions` | `mlx-audio` / Whisper | `audio` |
| TTS | `/v1/audio/speech` | `mlx-audio` | `audio` |
| 圖像生成 | `/v1/images/generations` | 擴散模型 | `generation` |
| 影片生成(Wan 2.x / LTX-2) | `/v1/video/generations` | `mlx-video` | `video` |
| Embeddings / rerank(文字 + 多模態) | `/v1/embeddings`、`/v1/rerank` | `mlx-lm` / `mlx-embeddings` | `embeddings` |

語音對語音:服務一個 Qwen3-Omni 模型(`uv sync --extra omni`),試試
[`examples/talk.py`](examples/talk.py)(麥克風)或 [`examples/quickstart.py`](examples/quickstart.py)
(輸出 WAV,不需音訊硬體)。已確認上游 `mlx-vlm` 0.7.3 多輪 omni 輸出正確
([紀錄](docs/research/runs/2026-09-28-omni/README.md));伺服器的 Realtime 路徑尚未確認。

另外:MCP 伺服器/客戶端,以及相容 Anthropic 的 `/v1/messages`。

## 架構

```
  客戶端(任意 OpenAI / Anthropic SDK)
        │   OpenAI / Anthropic / MCP / Realtime-WS / SSE
  ┌─────┴───────────────────────────────────────────────┐
  │  閘道(FastAPI)       路由 + 中介層                   │
  ├─────────────────────────────────────────────────────┤
  │  引擎                                                 │
  │   · VLM batch runner(所有 mlx-vlm 模型)             │
  │       連續批次 · 前綴快取(記憶體 + SSD)             │
  │       Qwen3.5 家族:MTP / DFlash + batch-invariant    │
  │   · LLM 快速路徑(mlx-lm generate_step)              │
  │       KV 前綴快取 · 約束解碼                          │
  │   · 其他模態:omni、ASR/TTS、圖像、影片、embeddings   │
  └─────────────────────────────────────────────────────┘
        單一 MLX 執行緒 · 經由 Apple MLX 在裝置端執行
```

## 服務模型

所有 GPU 工作都在一條 MLX 執行緒上。VLM(mlx-vlm)模型的並行請求共用一個連續批次,每列有自己的
取樣設定;單獨一個請求時會使用推測解碼(Qwen3.5 家族),期間進來的請求則加入共用批次、不做推測。
純文字的 mlx-lm 模型走單請求快速路徑,並行請求會依序執行。每個回應在它自己的生成結束時就立即返回。

## 設定

所有設定都是 [設定參考](docs/CONFIGURATION.md) 列出的 `YUNSHU_*` 名稱(由同一份登錄表產生)。
可用環境變數、TOML 檔(`yunshu serve --config yunshu.toml`)或 `yunshu serve --set KEY=VALUE`
設定;`yunshu config` 顯示每項的生效值與來源。值無法解析會中止啟動,拼錯的名稱會收到警告。

## 建構於

[MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)。
部分驗證 kernel 取自 [oMLX](https://github.com/jundot/omlx)(Apache-2.0),見
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 授權

Apache 2.0 —— 見 [LICENSE](LICENSE)。
