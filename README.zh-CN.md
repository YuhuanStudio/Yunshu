<div align="center">

# Yunshu

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

> **需要 [uv](https://docs.astral.sh/uv/)。** 还没上 PyPI —— 从源码以 `uv sync` 安装,
> 它会安装 `uv.lock` 中固定的版本(MLX 0.32、上游 `mlx-vlm` 0.7.3+)。

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
| 视频生成（Wan 2.x / LTX-2，文本→视频 + 图像→视频） | `/v1/video/generations` | `mlx-video` | `video` |
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
  │   · VLM runner（Qwen3.5/3.6/3.8）：前缀缓存 +       │
  │     MTP / DFlash 推测解码 + 验证 kernel             │
  │   · VLM / OCR（mlx-vlm）  · ASR / TTS（mlx-audio）│
  │   · OmniEngine（Qwen3-Omni Thinker→Talker）         │
  │   · 图像扩散              · KV 前缀缓存               │
  └────────────────────────────────────────────────────┘
              经由 Apple MLX 在设备本地运行
```

## 性能

Yunshu 一次服务一个请求,优化的是延迟:首 token(冷启动与缓存命中都算)、解码速度、前缀复用。
第一个完整调校的模型是 **Qwen3.8-27B**。Qwen3.5 家族 VLM(Qwen3.5 / 3.6 / 3.8)走构建在 `mlx-vlm`
生成器之上的专用 runner:

- **前缀缓存(APC)**:混合架构的精确 checkpoint,以文本与图片像素共同作为键;默认 8 GiB 内存,
  可选 SSD 层。重复或只改结尾的长 prompt 无需重新 prefill。
- **推测解码**:使用 checkpoint 自带的 MTP 头,或外部 DFlash drafter(`YUNSHU_VLM_DRAFT`)。
  默认验证 kernel 是精确的:greedy 下开启与关闭推测的输出逐 token 相同。更快但非精确的验证
  kernel 需手动开启(`YUNSHU_MTP_FAST_VERIFY=1`)。
- 流式推理分离、工具调用、JSON-schema 约束、停止序列、logprobs、取消在这条路径上都可用。

测量环境:M5 Max(128 GB)、Qwen3.8-27B、2026-09-28。除特别注明外均为同一个 Jundot `oQ4e-mtp`
checkpoint;原始数据与方法见
[docs/research/runs/2026-09-28-matrix](docs/research/runs/2026-09-28-matrix/README.md)。

| 引擎 | 能力检查 | 对话 TTFT（热） | 8K prompt：冷 / 重复 / 改尾 | 解码 tok/s |
|---|---|---|---|---|
| **Yunshu**（默认：MTP、精确验证） | 31/31 | 0.196 s | 8.62 / 0.095 / 0.253 s | 57 |
| **Yunshu**（DFlash2 + 快速验证，需开启） | 31/31 | 0.185 s | 8.71 / 0.112 / 0.239 s | 86 |
| mlx-vlm 0.7.3 server（APC） | 27/28 | 0.212 s | 8.60 / 0.108 / 0.265 s | 32 |
| oMLX.app 0.7（MTP + 缓存） | 31/31 | 0.312 s | 8.60 / 0.361 / 0.376 s | 85 |
| Splash 1.1（自家量化模型 + DFlash2） | 31/31 | 0.206 s | 7.88 / 0.131 / 7.88 s | 119 |

现状:前缀复用与 TTFT 是测到最好的;这个 checkpoint 的 prefill 已到硬件上限;**默认解码仍落后
oMLX 与 Splash**,它们用了更快(非精确或自定义量化)的 kernel。在不放弃精确输出的前提下追上,
是当前的主要工作。其他场景旋钮(n-gram 推测、替代采样器、KV 量化、jump-forward)都需手动开启,
见 [配置参考](docs/CONFIGURATION.md)。长期基准记录见
[docs/reports/PERF_TREND.md](docs/reports/PERF_TREND.md)。

## 构建于

[MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)。
部分验证 kernel 取自 [oMLX](https://github.com/jundot/omlx)(Apache-2.0),见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 许可证

Apache 2.0 —— 见 [LICENSE](LICENSE)。
