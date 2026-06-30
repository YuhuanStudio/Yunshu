<div align="center">

# Yunshu（云述）

**一个快速、本地、面向 Apple Silicon 的多模态推理引擎。**

单一进程,兼容 OpenAI/Anthropic,全部经由 MLX 在设备本地运行:文本、视觉、OCR、音频、图像、
嵌入,以及一个实时语音 WebSocket。它的与众不同之处是**原生流式语音到语音** —— 你说话,模型约
1.4 秒后用它自己的声音回话,无云端、也没有语音转文字 → LLM → 文字转语音的级联。

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

[English](./README.md) · **简体中文** · [繁體中文](./README.zh-TW.md)

</div>

---

## 原生语音到语音

大多数本地语音方案都是**级联**:语音转文字把你转写出来 → LLM 写出回复 → 文字转语音读出来。
每一跳都增加延迟、丢掉韵律 —— 系统从不"听见"你的语气,也无法塑造自己的语气。

Qwen3-Omni 的 **Talker** 架构是单个模型:它摄入原始音频、进行推理,并直接解码出语音 token ——
中间没有文字。Yunshu 经由 `mlx-vlm` 在 Apple Silicon 上原生提供这一能力,在你说完话后约 1 秒内
就开始把音频流回。

```
你（音频） ──► Qwen3-Omni Thinker（推理） ──► Talker（流式输出音频） ──► 你
                     一个统一模型,没有管线跳转
```

其他一切也都能跑:任意 `mlx-lm` / `mlx-vlm` / `mlx-audio` 模型都走标准端点。

---

## 快速开始

> **需要 [uv](https://docs.astral.sh/uv/)。** 还没上 PyPI —— 从源码安装。`uv sync` 会遵守
> `[tool.uv.sources]`,所以会拉取 Yunshu 为 Qwen3-Omni 准备的 `mlx-vlm` fork(最新上游 +
> Thinker 早退优化和 omni 修复)。纯 `pip install` 会装到未固定的上游 `mlx-vlm`,所以请用 `uv sync`。

```bash
# 1. 从源码安装
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
uv sync --extra omni        # 原生 Qwen3-Omni 语音（语音输入/输出）
# 或: uv sync --all-extras  # 全部:文本 + 视觉 + 音频 + omni + 图像 + 嵌入

# 2. 启动模型。来自 mlx-community 的任意 4-bit Qwen3-Omni 变体都可以 —— 原生语音会自动开启
#    （同一份已载入的模型同时服务文本和语音,不额外占内存）。
uv run yunshu serve -m /path/to/Qwen3-Omni-30B-A3B-Instruct-4bit --port 8000
```

### 跟它对话

```bash
uv run --with sounddevice --with numpy --with websockets python examples/talk.py
```

[`examples/talk.py`](examples/talk.py) 是一段真正的语音对话:按 Enter、说话、再按一次 Enter ——
模型出声回答,并记得整段对话。

不想接麦克风?[`examples/quickstart.py`](examples/quickstart.py) 会把一段语音回复流式写入 WAV
文件,并演示文本端点 —— 不需要任何音频硬件。

### 用任意 OpenAI 客户端

文本、视觉、嵌入和重排序都说标准 API —— 客户端无需改动:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")  # 任意 key 都行

# 在单模型模式下,模型名只是占位符 —— 服务器会服务你加载的那个模型
#（就像 Ollama / LM Studio），所以填 "local" 即可。
print(
    client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
    ).choices[0].message.content
)
```

> **开发签出**:`just setup`,然后 `YUNSHU_MODEL=<model> just dev`。
> **文档**:[API 参考](docs/API.md) · [配置参考](docs/CONFIGURATION.md)。

---

## 能力

| 模态 | 端点 | 后端 | Extra |
|---|---|---|---|
| **原生语音到语音**（Qwen3-Omni，流式；预热后语音输入约 1.4 秒首音、文本输入约 1.2 秒） | `POST /v1/omni/speech/stream` | `mlx-vlm` Thinker+Talker | `omni` |
| 文本（工具调用、JSON-schema、流式、logprobs） | `/v1/chat/completions`、`/v1/messages` | `mlx-lm` | _(核心)_ |
| 视觉 / OCR | `/v1/chat/completions`（图像内容） | `mlx-vlm` | `vision` |
| ASR | `/v1/audio/transcriptions` | `mlx-audio` / Whisper | `audio` |
| TTS | `/v1/audio/speech` | `mlx-audio` | `audio` |
| 实时语音 WS | `WS /v1/realtime` | omni 或 ASR + TTS | `audio` |
| 图像生成 | `/v1/images/generations` | 扩散 | `generation` |
| 嵌入（文本 + **多模态**：经由 Qwen3-VL-Embedding 的图像 / 跨模态） | `/v1/embeddings` | `mlx-lm` (text) / `mlx-embeddings` (multimodal) | `embeddings` |
| 重排序（双编码器余弦，或经由 Qwen3-VL-Reranker 的**真正交叉编码器**） | `/v1/rerank` | `mlx-lm` (text) / `mlx-embeddings` (multimodal) | `embeddings` |

此外还有:单节点 KV 前缀缓存（+ 可选 SSD 持久化 + 按请求 KV 量化）、MCP 服务端/客户端,以及
兼容 Anthropic 的 `/v1/messages` 接口。

## 架构

```
  客户端（任意 OpenAI / Anthropic SDK）
        │   OpenAI / Anthropic / MCP / Realtime-WS / SSE
  ┌─────┴──────────────────────────────────────────────┐
  │  网关（FastAPI）      路由 + 中间件                   │
  ├────────────────────────────────────────────────────┤
  │  引擎                 模态分派 + 服务                 │
  │   · LLM 快速路径（mlx-lm generate_step）            │
  │   · VLM / OCR（mlx-vlm）  · ASR / TTS（mlx-audio）│
  │   · OmniEngine（Qwen3-Omni Thinker→Talker）         │
  │   · 图像扩散              · KV 前缀缓存               │
  └────────────────────────────────────────────────────┘
              经由 Apple MLX 在设备本地运行
```

## 性能

Yunshu 为**低延迟**而非吞吐而优化 —— 它一次服务一个请求、走快速路径,这正是本地单用户服务器
该有的形状。一个请求经过 mlx-lm 的 `generate_step`,带 KV 前缀 + prompt 缓存,以及在贪心请求上
无损的 n-gram 推测解码,全部默认开启;单流解码与 `mlx-lm` 持平。更重或更依场景的旋钮 —— 替代
采样器、内存内权重量化、jump-forward —— 都是按需开启、绝不静默生效;见
[配置参考](docs/CONFIGURATION.md)。诚实的基准趋势见
[docs/reports/PERF_TREND.md](docs/reports/PERF_TREND.md)。

## 构建于

[MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)

## 许可证

Apache 2.0 —— 见 [LICENSE](LICENSE)。
