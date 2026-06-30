<div align="center">

# Yunshu（雲述）

**一個快速、本地、為 Apple Silicon 打造的多模態推論引擎。**

單一行程,相容 OpenAI/Anthropic,全部經由 MLX 在裝置端本地執行:文字、視覺、OCR、音訊、影像、
嵌入,以及一個即時語音 WebSocket。它的與眾不同之處是**原生串流語音對語音** —— 你說話,模型約
1.4 秒後用它自己的聲音回話,無雲端、也沒有語音轉文字 → LLM → 文字轉語音的串接。

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

[English](./README.md) · [简体中文](./README.zh-CN.md) · **繁體中文**

</div>

---

## 原生語音對語音

大多數本地語音方案都是**串接（cascade）**:語音轉文字把你轉寫出來 → LLM 寫出回覆 → 文字轉語音
唸出來。每一跳都增加延遲、流失韻律 —— 系統從不「聽見」你的語氣,也無法塑造自己的語氣。

Qwen3-Omni 的 **Talker** 架構是單一模型:它攝入原始音訊、進行推理,並直接解碼出語音 token ——
中間沒有文字。Yunshu 經由 `mlx-vlm` 在 Apple Silicon 上原生提供這項能力,在你說完話後約 1 秒內
就開始把音訊串流回來。

```
你（音訊） ──► Qwen3-Omni Thinker（推理） ──► Talker（串流輸出音訊） ──► 你
                     一個統一模型,沒有管線跳轉
```

其他一切也都能跑:任意 `mlx-lm` / `mlx-vlm` / `mlx-audio` 模型都走標準端點。

---

## 快速開始

> **需要 [uv](https://docs.astral.sh/uv/)。** 還沒上 PyPI —— 從原始碼安裝。`uv sync` 會遵守
> `[tool.uv.sources]`,所以會拉取 Yunshu 為 Qwen3-Omni 準備的 `mlx-vlm` fork(最新上游 +
> Thinker 早退優化和 omni 修正)。純 `pip install` 會裝到未固定的上游 `mlx-vlm`,所以請用 `uv sync`。

```bash
# 1. 從原始碼安裝
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
uv sync --extra omni        # 原生 Qwen3-Omni 語音（語音輸入/輸出）
# 或: uv sync --all-extras  # 全部:文字 + 視覺 + 音訊 + omni + 影像 + 嵌入

# 2. 啟動模型。來自 mlx-community 的任意 4-bit Qwen3-Omni 變體都可以 —— 原生語音會自動開啟
#    （同一份已載入的模型同時服務文字和語音,不額外佔記憶體）。
uv run yunshu serve -m /path/to/Qwen3-Omni-30B-A3B-Instruct-4bit --port 8000
```

### 跟它對話

```bash
uv run --with sounddevice --with numpy --with websockets python examples/talk.py
```

[`examples/talk.py`](examples/talk.py) 是一段真正的語音對話:按 Enter、說話、再按一次 Enter ——
模型出聲回答,並記得整段對話。

不想接麥克風?[`examples/quickstart.py`](examples/quickstart.py) 會把一段語音回覆串流寫入 WAV
檔,並示範文字端點 —— 不需要任何音訊硬體。

### 用任意 OpenAI 客戶端

文字、視覺、嵌入與重排序都說標準 API —— 客戶端不必改動:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")  # 任意 key 都行

# 在單模型模式下,模型名稱只是佔位符 —— 伺服器會服務你載入的那個模型
#（就像 Ollama / LM Studio），所以填 "local" 即可。
print(
    client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
    ).choices[0].message.content
)
```

> **開發簽出**:`just setup`,然後 `YUNSHU_MODEL=<model> just dev`。
> **文件**:[API 參考](docs/API.md) · [設定參考](docs/CONFIGURATION.md)。

---

## 能力

| 模態 | 端點 | 後端 | Extra |
|---|---|---|---|
| **原生語音對語音**（Qwen3-Omni，串流；預熱後語音輸入約 1.4 秒首音、文字輸入約 1.2 秒） | `POST /v1/omni/speech/stream` | `mlx-vlm` Thinker+Talker | `omni` |
| 文字（工具呼叫、JSON-schema、串流、logprobs） | `/v1/chat/completions`、`/v1/messages` | `mlx-lm` | _(核心)_ |
| 視覺 / OCR | `/v1/chat/completions`（影像內容） | `mlx-vlm` | `vision` |
| ASR | `/v1/audio/transcriptions` | `mlx-audio` / Whisper | `audio` |
| TTS | `/v1/audio/speech` | `mlx-audio` | `audio` |
| 即時語音 WS | `WS /v1/realtime` | omni 或 ASR + TTS | `audio` |
| 影像生成 | `/v1/images/generations` | 擴散 | `generation` |
| 影片生成（Wan 2.x / LTX-2,文字→影片 + 影像→影片） | `/v1/video/generations` | `mlx-video` | `video` |
| 嵌入（文字 + **多模態**：經由 Qwen3-VL-Embedding 的影像 / 跨模態） | `/v1/embeddings` | `mlx-lm` (text) / `mlx-embeddings` (multimodal) | `embeddings` |
| 重排序（雙編碼器餘弦，或經由 Qwen3-VL-Reranker 的**真正交叉編碼器**） | `/v1/rerank` | `mlx-lm` (text) / `mlx-embeddings` (multimodal) | `embeddings` |

此外還有:單節點 KV 前綴快取（+ 選用的 SSD 持久化 + 按請求 KV 量化）、MCP 伺服端/客戶端,以及
相容 Anthropic 的 `/v1/messages` 介面。

## 架構

```
  客戶端（任意 OpenAI / Anthropic SDK）
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

## 效能

Yunshu 為**低延遲**而非吞吐而優化 —— 它一次服務一個請求、走快速路徑,這正是本地單一使用者
伺服器該有的形狀。一個請求經過 mlx-lm 的 `generate_step`,帶 KV 前綴 + prompt 快取,預設開啟;
單串流解碼與 `mlx-lm` 持平。依情境的旋鈕 —— 無損 n-gram 推測解碼(在重複性/agentic 輸出上加速,
但在普通文字上更慢,所以按需開啟)、替代取樣器、記憶體內權重量化、jump-forward —— 都是按需開啟、
絕不靜默生效;見 [設定參考](docs/CONFIGURATION.md)。誠實的基準趨勢見
[docs/reports/PERF_TREND.md](docs/reports/PERF_TREND.md)。

## 建構於

[MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)

## 授權

Apache 2.0 —— 見 [LICENSE](LICENSE)。
