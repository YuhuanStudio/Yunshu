<div align="center">

# Yunshu（雲述）

**一個快速、本地、為 Apple Silicon 打造的多模態推論引擎。**

相容 OpenAI/Anthropic —— 文字、視覺、OCR、音訊、影像，以及一個即時語音 WebSocket —— 多模型、
全部經由 MLX 在裝置端本地執行。它的與眾不同之處：**原生串流語音對語音**（Qwen3-Omni Talker），
首個音訊約 1.4 秒到達 —— 無雲端、無 ASR + LLM + TTS 串接，模型用它自己的聲音說話。

[![PyPI](https://img.shields.io/pypi/v/yunshu.svg?label=PyPI)](https://pypi.org/project/yunshu/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

[English](./README.md) · [简体中文](./README.zh-CN.md) · **繁體中文**

</div>

---

## 為什麼有這個專案

Yunshu 是一個通用、隨插即用的本地 AI 伺服器 —— 把任意 OpenAI/Anthropic 客戶端指向它，即可在本地
處理文字、視覺、OCR、音訊或影像。真正讓它有別於其他本地伺服器的，是**原生串流語音對語音**。

大多數本地語音管線都是**串接（cascade）**：ASR 把你的語音轉寫成文字 → LLM 生成文字 → TTS 把文字
唸出來。每一跳都增加延遲、流失韻律，而且無法對語氣或聲音本身進行推理。

Qwen3-Omni 的 **Talker** 架構不一樣：單一模型攝入原始音訊、進行推理，並直接解碼出語音 token ——
中間沒有文字。Yunshu 經由 `mlx-vlm` 在 Apple Silicon 上原生暴露這項能力，提供一個串流 SSE 端點，
在你說完話後約 1 秒內就開始輸出音訊區塊。（其他模型也能跑 —— 任意 `mlx-lm`/`mlx-vlm`/`mlx-audio`
模型；非 omni 模型走標準端點。）

```
你（音訊） ──► Qwen3-Omni Thinker（推理） ──► Talker（串流輸出音訊） ──► 你
                     一個統一模型，沒有管線跳轉
```

---

## 快速開始

> **需要 [uv](https://docs.astral.sh/uv/)。** `[omni]` 這個 extra 會把 `mlx-vlm` 固定到一個 fork，
> 該 fork 帶有一個尚未併入上游的 Qwen3-Omni 多輪修正。`pip` 會忽略 fork 的固定並靜默安裝有問題的
> 上游版本。請用 `uv` —— 它會遵守 `[tool.uv.sources]` 的固定。

```bash
# 1. 安裝（需要 uv）
uv pip install "yunshu[omni]"      # 原生 Qwen3-Omni 語音（語音輸入/輸出）
uv pip install "yunshu[all]"       # 全部：文字 + 視覺 + 音訊 + omni + 影像 + 嵌入

# 2. 啟動伺服器
yunshu serve -m /path/to/Qwen3-Omni-30B-A3B-Instruct-4bit --port 8000
# 來自 mlx-community 的任意 4-bit Qwen3-Omni 變體都可以用

# 3. 串流取得一段語音回覆（Server-Sent Events：文字增量 + base64 PCM16 @ 24kHz）
curl -N -X POST http://localhost:8000/v1/omni/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Say hello in one sentence.", "speaker": "Ethan"}'
# 若需語音輸入，加上 "audio_path": "question.wav"（說出來的那一輪）；
# 此時 "text" 用來承載任意系統指令。
```

該端點串流回傳的是 SSE 事件，不是 WAV 檔。若想要一個現成的客戶端來消費這個串流並寫出
`omni_out.wav`，見 [examples/quickstart.py](examples/quickstart.py) —— 它也示範了
文字/視覺/ASR/TTS 端點。

對於**雙向語音代理**路徑（透過 WebSocket 說話進去、聽模型說回來），見
[examples/realtime_voice.py](examples/realtime_voice.py) —— 它針對 OpenAI-Realtime 的
`WS /v1/realtime` 端點跑一輪原生語音對語音（在伺服器上設定 `YUNSHU_REALTIME_OMNI=1`）。

### 把你現有的 OpenAI 客戶端指過來

文字、視覺、嵌入與重排序都說標準 API —— 客戶端不必改動：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")  # 任意 key 都行

# 聊天。在單模型模式下，模型名稱只是佔位符 —— 伺服器會服務你載入的那個模型
#（就像 Ollama / LM Studio），所以填 "local" 即可。
print(
    client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
    ).choices[0].message.content
)

# 嵌入 —— 文字，或多模態（影像 / 跨模態），搭配 Qwen3-VL-Embedding 模型。
client.embeddings.create(model="local", input=["hello", "world"])
```

可執行腳本在 **[examples/](examples/)**：`quickstart.py`（每個端點）、`realtime_voice.py`
（WebSocket 語音對語音）、`multimodal_embeddings.py`（影像 / 跨模態檢索 + 重排序）。

> **開發簽出**：`just setup`，然後 `YUNSHU_MODEL=<model> just dev`。
> **文件**：[API 參考](docs/API.md) · [設定參考](docs/CONFIGURATION.md)。

---

## 這是什麼

一個相容 OpenAI/Anthropic 的單一行程，在**裝置端本地服務每一種模態** —— LLM（大腦）、
VLM/OCR（眼睛）、ASR（耳朵）、TTS + 原生 Talker（嗓音）、嵌入/重排序（檢索），以及影像
生成（想像力）—— 建構於 Apple 的 MLX 堆疊之上。把任意 OpenAI/Anthropic SDK 指向它即可。

它是一個**通用**的本地推論引擎，可獨立使用。一個值得一提的使用者是 Yunmo，一個本地數位生命
框架，它把 Yunshu 當作自己的感官身體 —— 但那是它能驅動什麼的一個範例，而非它本身的定義。

## 這不是什麼

- **不是分散式系統。** 沒有多 Mac 叢集；那個論點已被放棄。
- **不是吞吐量競賽。** 單請求快速路徑；面向一個使用者（一個生命），而非多租戶機群。
- **沒有自訂 Metal kernel。** 封裝 MLX。單串流解碼與 `mlx-lm` 持平。這裡唯一重要的效能軸是
  **語音往返延遲**。

如果你需要生產級多租戶服務或多節點分片，請看
[oMLX](https://github.com/jundot/omlx)、[vllm-mlx](https://github.com/waybarrios/vllm-mlx) 或
[exo](https://github.com/exo-explore/exo)。

## 能力

| 模態 | 端點 | 後端 | Extra |
|---|---|---|---|
| **原生語音對語音**（Qwen3-Omni，串流；預熱後語音輸入約 1.4 秒首音、文字輸入約 1.2 秒） | `POST /v1/omni/speech/stream` | `mlx-vlm` Thinker+Talker | `omni` |
| 文字（工具呼叫、JSON-schema、串流、logprobs） | `/v1/chat/completions`、`/v1/messages` | `mlx-lm` | _(核心)_ |
| 視覺 / OCR | `/v1/chat/completions`（影像內容） | `mlx-vlm` | `vision` |
| ASR | `/v1/audio/transcriptions` | `mlx-audio` / Whisper | `audio` |
| TTS | `/v1/audio/speech` | `mlx-audio` | `audio` |
| 即時語音 WS | `WS /v1/realtime` | ASR + TTS | `audio` |
| 影像生成 | `/v1/images/generations` | 擴散 | `generation` |
| 嵌入（文字 + **多模態**：經由 Qwen3-VL-Embedding 的影像 / 跨模態） | `/v1/embeddings` | `mlx-embeddings` | `embeddings` |
| 重排序（雙編碼器餘弦，或經由 Qwen3-VL-Reranker 的**真正交叉編碼器**） | `/v1/rerank` | `mlx-embeddings` | `embeddings` |

此外還有：單節點 KV 前綴快取（+ 選用的 SSD 持久化 + 按請求 KV 量化）、MCP 伺服端/客戶端、
相容 Anthropic 的 `/v1/messages` 介面。

## 架構

```
  客戶端（Yunmo 常駐程式 / 任意 OpenAI-Anthropic SDK）
        │   OpenAI / Anthropic / MCP / Realtime-WS / SSE
  ┌─────┴──────────────────────────────────────────────┐
  │  閘道（FastAPI）      路由 + 中介層                   │
  ├────────────────────────────────────────────────────┤
  │  引擎                 模態分派 + 服務                 │
  │   · LLM 快速路徑（mlx-lm generate_step）            │
  │   · VLM / OCR（mlx-vlm）  · ASR / TTS（mlx-audio）│
  │   · OmniEngine（Qwen3-Omni Thinker→Talker）         │
  │   · 影像擴散              · KV 前綴快取               │
  └────────────────────────────────────────────────────┘
              經由 Apple MLX 在裝置端本地執行
```

## 狀態

**重新聚焦** —— 從一個過度膨脹的「推論平台」收斂為一個誠實的單節點 omni 引擎：多節點 mesh /
分散式路徑（sharded-load、分離式 prefill/decode）、多租戶控制面，以及分層 KV offload 都已移除。

在預設服務路徑上，一個請求會走**單請求快速路徑**（mlx-lm `generate_step`），帶 KV 前綴 +
prompt 快取、自動 KV 量化（僅當快取會主導頻寬時）、約束解碼（按需的 JSON-schema / regex /
grammar）、按請求的 stop/reasoning 狀態，以及**在貪婪請求上的 n-gram 推測解碼**（無損 ——
驗證器只接受模型自己的 argmax；用 `YUNSHU_NGRAM_DEFAULT=0` 關閉）—— 全部預設開啟。較重或較
依情境的優化則是**按需開啟**、而非預設魔法：替代的推測提議器（經 `spec_decode=true` + 一個草稿
模型的跨模型；經 `YUNSHU_SPEC_PROPOSER=suffix` 的 Suffix Decoding）、top-nσ 取樣器（在
`/v1/chat/completions` 上按請求 `"top_n_sigma"`，或伺服器全域 `YUNSHU_TOP_N_SIGMA`）、
jump-forward（`YUNSHU_JUMP_FORWARD`）、GPU 取樣器（`YUNSHU_GPU_SAMPLER`）、記憶體內
MXFP4/NVFP4 權重量化（`YUNSHU_QUANT_MODE`），以及稀疏 spec-prefill（`YUNSHU_SPEC_PREFILL` +
一個草稿模型）。我們保持它們與時俱進，但絕不在它們沒執行時聲稱它們在執行。跑 `just test` 執行
測試套件；基準趨勢見 `docs/reports/PERF_TREND.md`。

## 建構於

- [MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)

## 授權

Apache 2.0 —— 見 [LICENSE](LICENSE)。
