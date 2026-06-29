<div align="center">

# Yunshu（云述）

**一个快速、本地、面向 Apple Silicon 的多模态推理引擎。**

兼容 OpenAI/Anthropic —— 文本、视觉、OCR、音频、图像，以及一个实时语音 WebSocket —— 多模型、
全部经由 MLX 在设备本地运行。它的与众不同之处：**原生流式语音到语音**（Qwen3-Omni Talker），
首个音频约 1.4 秒到达 —— 无云端、无 ASR + LLM + TTS 级联，模型用它自己的声音说话。

[![PyPI](https://img.shields.io/pypi/v/yunshu.svg?label=PyPI)](https://pypi.org/project/yunshu/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

[English](./README.md) · **简体中文** · [繁體中文](./README.zh-TW.md)

</div>

---

## 为什么有这个项目

Yunshu 是一个通用的、即插即用的本地 AI 服务器 —— 把任意 OpenAI/Anthropic 客户端指向它，即可在本地处理
文本、视觉、OCR、音频或图像。真正让它区别于其他本地服务器的，是**原生流式语音到语音**。

大多数本地语音管线都是**级联**：ASR 把你的语音转写成文字 → LLM 生成文字 → TTS 把文字读出来。
每一跳都增加延迟、丢失韵律，而且无法对语气或声音本身进行推理。

Qwen3-Omni 的 **Talker** 架构不一样：单个模型摄入原始音频、进行推理，并直接解码出语音 token ——
中间没有文字。Yunshu 经由 `mlx-vlm` 在 Apple Silicon 上原生暴露这一能力，提供一个流式 SSE 端点，
在你说完话后约 1 秒内就开始输出音频块。（其他模型也能跑 —— 任意 `mlx-lm`/`mlx-vlm`/`mlx-audio`
模型；非 omni 模型走标准端点。）

```
你（音频） ──► Qwen3-Omni Thinker（推理） ──► Talker（流式输出音频） ──► 你
                     一个统一模型，没有管线跳转
```

---

## 快速开始

> **需要 [uv](https://docs.astral.sh/uv/)。** `[omni]` 这个 extra 把 `mlx-vlm` 固定到一个 fork，
> 该 fork 带有一个尚未合入上游的 Qwen3-Omni 多轮修复。`pip` 会忽略 fork 固定并静默安装有问题的上游版本。
> 请用 `uv` —— 它会遵守 `[tool.uv.sources]` 的固定。

```bash
# 1. 安装（需要 uv）
uv pip install "yunshu[omni]"      # 原生 Qwen3-Omni 语音（语音输入/输出）
uv pip install "yunshu[all]"       # 全部：文本 + 视觉 + 音频 + omni + 图像 + 嵌入

# 2. 启动服务器
yunshu serve -m /path/to/Qwen3-Omni-30B-A3B-Instruct-4bit --port 8000
# 来自 mlx-community 的任意 4-bit Qwen3-Omni 变体都可以用

# 3. 流式获取一段语音回复（Server-Sent Events：文本增量 + base64 PCM16 @ 24kHz）
curl -N -X POST http://localhost:8000/v1/omni/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Say hello in one sentence.", "speaker": "Ethan"}'
# 如需语音输入，加上 "audio_path": "question.wav"（说出来的那一轮）；
# 此时 "text" 用于承载任意系统指令。
```

该端点流式返回的是 SSE 事件，不是 WAV 文件。若想要一个现成的客户端来消费这个流并写出
`omni_out.wav`，见 [examples/quickstart.py](examples/quickstart.py) —— 它也演示了
文本/视觉/ASR/TTS 端点。

对于**双向语音助手**路径（通过 WebSocket 说话进去、听模型说回来），见
[examples/realtime_voice.py](examples/realtime_voice.py) —— 它针对 OpenAI-Realtime 的
`WS /v1/realtime` 端点跑一轮原生语音到语音（在服务器上设置 `YUNSHU_REALTIME_OMNI=1`）。

### 把你现有的 OpenAI 客户端指过来

文本、视觉、嵌入和重排序都说标准 API —— 客户端无需改动：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")  # 任意 key 都行

# 聊天。在单模型模式下，模型名只是占位符 —— 服务器会服务你加载的那个模型
#（就像 Ollama / LM Studio），所以填 "local" 即可。
print(
    client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
    ).choices[0].message.content
)

# 嵌入 —— 文本，或多模态（图像 / 跨模态），配合 Qwen3-VL-Embedding 模型。
client.embeddings.create(model="local", input=["hello", "world"])
```

可运行脚本在 **[examples/](examples/)**：`quickstart.py`（每个端点）、`realtime_voice.py`
（WebSocket 语音到语音）、`multimodal_embeddings.py`（图像 / 跨模态检索 + 重排序）。

> **开发签出**：`just setup`，然后 `YUNSHU_MODEL=<model> just dev`。
> **文档**：[API 参考](docs/API.md) · [配置参考](docs/CONFIGURATION.md)。

---

## 这是什么

一个兼容 OpenAI/Anthropic 的单进程，在**设备本地服务每一种模态** —— LLM（大脑）、
VLM/OCR（眼睛）、ASR（耳朵）、TTS + 原生 Talker（嗓音）、嵌入/重排序（检索），以及图像
生成（想象力）—— 构建于 Apple 的 MLX 栈之上。把任意 OpenAI/Anthropic SDK 指向它即可。

它是一个**通用**的本地推理引擎，可独立使用。一个值得一提的使用者是 Yunmo，一个本地数字生命
框架，它把 Yunshu 当作自己的感官身体 —— 但那是它能驱动什么的一个例子，而非它本身的定义。

## 这不是什么

- **不是分布式系统。** 没有多 Mac 集群；那个论点已被放弃。
- **不是吞吐竞赛。** 单请求快速路径；面向一个使用者（一个生命），而非多租户机群。
- **没有自定义 Metal kernel。** 封装 MLX。单流解码与 `mlx-lm` 持平。这里唯一重要的性能轴是
  **语音往返延迟**。

如果你需要生产级多租户服务或多节点分片，请看
[oMLX](https://github.com/jundot/omlx)、[vllm-mlx](https://github.com/waybarrios/vllm-mlx) 或
[exo](https://github.com/exo-explore/exo)。

## 能力

| 模态 | 端点 | 后端 | Extra |
|---|---|---|---|
| **原生语音到语音**（Qwen3-Omni，流式；预热后语音输入约 1.4 秒首音、文本输入约 1.2 秒） | `POST /v1/omni/speech/stream` | `mlx-vlm` Thinker+Talker | `omni` |
| 文本（工具调用、JSON-schema、流式、logprobs） | `/v1/chat/completions`、`/v1/messages` | `mlx-lm` | _(核心)_ |
| 视觉 / OCR | `/v1/chat/completions`（图像内容） | `mlx-vlm` | `vision` |
| ASR | `/v1/audio/transcriptions` | `mlx-audio` / Whisper | `audio` |
| TTS | `/v1/audio/speech` | `mlx-audio` | `audio` |
| 实时语音 WS | `WS /v1/realtime` | ASR + TTS | `audio` |
| 图像生成 | `/v1/images/generations` | 扩散 | `generation` |
| 嵌入（文本 + **多模态**：经由 Qwen3-VL-Embedding 的图像 / 跨模态） | `/v1/embeddings` | `mlx-embeddings` | `embeddings` |
| 重排序（双编码器余弦，或经由 Qwen3-VL-Reranker 的**真正交叉编码器**） | `/v1/rerank` | `mlx-embeddings` | `embeddings` |

此外还有：单节点 KV 前缀缓存（+ 可选 SSD 持久化 + 按请求 KV 量化）、MCP 服务端/客户端、
兼容 Anthropic 的 `/v1/messages` 接口。

## 架构

```
  客户端（Yunmo 守护进程 / 任意 OpenAI-Anthropic SDK）
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

## 状态

**重新聚焦** —— 从一个过度膨胀的"推理平台"收敛为一个诚实的单节点 omni 引擎：多节点 mesh /
分布式路径（sharded-load、分离式 prefill/decode）、多租户控制面，以及分层 KV offload 都已移除。

在默认服务路径上，一个请求会走**单请求快速路径**（mlx-lm `generate_step`），带 KV 前缀 +
prompt 缓存、自动 KV 量化（仅当缓存会主导带宽时）、约束解码（按需的 JSON-schema / regex /
grammar）、按请求的 stop/reasoning 状态，以及**在贪心请求上的 n-gram 推测解码**（无损 ——
校验器只接受模型自己的 argmax；用 `YUNSHU_NGRAM_DEFAULT=0` 关闭）—— 全部默认开启。更重或更
依场景的优化是**按需开启**、而非默认魔法：替代的推测提议器（经 `spec_decode=true` + 一个草稿
模型的跨模型；经 `YUNSHU_SPEC_PROPOSER=suffix` 的 Suffix Decoding）、top-nσ 采样器（在
`/v1/chat/completions` 上按请求 `"top_n_sigma"`，或服务器全局 `YUNSHU_TOP_N_SIGMA`）、
jump-forward（`YUNSHU_JUMP_FORWARD`）、GPU 采样器（`YUNSHU_GPU_SAMPLER`）、内存内
MXFP4/NVFP4 权重量化（`YUNSHU_QUANT_MODE`），以及稀疏 spec-prefill（`YUNSHU_SPEC_PREFILL` +
一个草稿模型）。我们保持它们与时俱进，但绝不在它们没运行时声称它们在运行。跑 `just test` 执行
测试套件；基准趋势见 `docs/reports/PERF_TREND.md`。

## 构建于

- [MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)

## 许可证

Apache 2.0 —— 见 [LICENSE](LICENSE)。
