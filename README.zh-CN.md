<div align="center">

# Yunshu

**为 Apple Silicon 打造的快速本地 LLM / VLM 推论引擎。**

单一进程，兼容 OpenAI 与 Anthropic API，通过 MLX 在设备端运行。为单台 Mac 的低延迟而设计：
首字快、无损解码快，并重用已经算过的前缀。第一个完整调校的模型是 **Qwen3.8-27B**。

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

[English](README.md) · **简体中文** · [繁體中文](README.zh-TW.md)

</div>

---

## 亮点

- **无损推测解码**：DFlash2 或 MTP 草稿，搭配不随批次改变的 kernel，打开推测解码时的贪婪输出
  与关闭时逐 token 相同。
- **混合模型的前缀缓存**：为 attention + GatedDeltaNet 模型保存精确 checkpoint，放在 RAM 与 SSD，
  重开后仍在；可再加存储层。
- **快速路径上的完整 API**：工具调用、JSON schema、停止序列、logprobs、推理、取消，涵盖 OpenAI
  Chat / Responses、Anthropic Messages 与 Ollama。
- **原生支持编程 agent**：Claude Code、Codex、opencode 通过各自的 API 运作，含服务器端网页
  搜索／抓取与 MCP。
- **默认无损**：任何可能改变输出的东西都是要自己开的设置。
- **本地且私密**：不发送使用统计；诊断数据不含 prompt。

## 快速开始

模型选择、外接存储、就绪检查与升级请参阅[首次使用指南](docs/guides/FIRST_RUN.md)。

需要 Apple Silicon、macOS 14 以上、Python 3.13 以上与 [uv](https://docs.astral.sh/uv/)。

```bash
uv tool install --python 3.13 "yunshu[vision]"      # or: brew install yuhuanstudio/tap/yunshu
yunshu doctor                                   # checks this Mac and says how to fix problems
yunshu pull mlx-community/Qwen3.5-9B-MLX-4bit
yunshu serve -m mlx-community/Qwen3.5-9B-MLX-4bit
```

服务器在 `http://127.0.0.1:8000`。任何 OpenAI 或 Anthropic 客户端都能直接使用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")  # any key works
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
)
print(r.choices[0].message.content)
```

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8000", api_key="local")
msg = client.messages.create(
    model="local", max_tokens=512,
    messages=[{"role": "user", "content": "Explain MLX in one sentence."}],
)
print(msg.content[0].text)
```

模型放在 `~/.yunshu/models`（`yunshu config set models_dir PATH` 指定后续下载目录，不移动现有权重）；`serve -m org/name`
也会找 Hugging Face 缓存，只有需要时才下载。`yunshu service install -m <model>` 让服务器在登录时启动。

### 本地 console

源代码包含 YunUI 引擎 console，入口为 `/console/`，提供状态、资源图表、模型操作、请求查看／取消与流式诊断。

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm build
```

[Console](docs/CONSOLE.md)

### Qwen3.8-27B

```bash
yunshu pull Jundot/Qwen3.8-27B-oQ4e-mtp
yunshu pull incoai/Qwen3.8-27B-DFlash2          # optional drafter, picked up automatically
yunshu serve -m Jundot/Qwen3.8-27B-oQ4e-mtp
```

装好草稿模型后，启动日志会显示 `Speculative decoding: dflash`；没装时由模型自带的 MTP 头产生草稿。
`yunshu doctor -m <model>` 会回报选用的路径，以及这台 Mac 的内存是否放得下。

### 从原代码运行

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git && cd Yunshu
uv sync --extra vision
uv run yunshu serve -m <model>
```

## 运作方式

```
 OpenAI / Anthropic / Ollama clients ──► FastAPI gateway (one process)
                                           │  request validation, tool/reasoning parsing,
                                           │  server-side tools (web search / fetch / MCP)
                                           ▼
                                  engine (one MLX thread)
          ┌────────────────────────────────┴───────────────────────────────┐
   VLM batch runner (every mlx-vlm model)                 text fast path (mlx-lm models)
   shared continuous batch, per-row sampling              single-request generate_step
   speculative lane: DFlash2 / MTP / prompt-copy
   prefix cache: RAM ─► SSD ─► optional storage tiers
```

所有 GPU 工作都在同一条 MLX 线程上，请求之间不会互抢 GPU。每个回应在自己的生成结束时就回传。

### 推测解码

Qwen3.5 家族的单一请求在推测解码通道中解码：

- **DFlash2**：独立的区块草稿模型，每轮提出多个 token；装了相符的草稿模型就自动使用。
- **MTP**：checkpoint 自带的多 token 预测头；没有草稿模型时的后备。
- **Prompt-copy 草稿**：当输出开始重复 prompt 里的文本（改代码、引用工具结果、多轮 agent），
  通道会提出那段文本的后续，并在同一次验证中检查。默认打开（`YUNSHU_SPEC_COPY_ROWS`，设 `0` 关闭）。

所有解码与验证的矩阵乘法都走**不随批次改变的 kernel**：一个 token 不论单独验证或和其他列一起验证，
算术都相同。这让开关推测解码时的贪婪输出完全一致，采样输出在依位置决定的采样下也精确一致。
用 `YUNSHU_VLM_DRAFT` 选择草稿（`mtp`、`off` 或草稿模型路径）。

### 前缀缓存（APC）

Qwen3.5 家族混合了 attention 层与递归的 GatedDeltaNet 层。递归状态不能像 KV 缓存那样切回较早的
token，所以一般的前缀缓存无法使用。Yunshu 在前缀边界保存**精确的 checkpoint**（KV 加递归状态），
以文本 token 及图片像素／音频特征为键，所以缓存命中的结果和冷预填完全相同。

| 层 | 位置 | 默认 |
|---|---|---|
| HOT | RAM 中可直接使用的数组 | 打开，依可用内存决定大小（`YUNSHU_VLM_APC_MEMORY_GB`） |
| WARM | RAM 中压缩存放（无损 zstd，或有损 int8 / int4） | 关闭（`YUNSHU_VLM_APC_WARM`） |
| SSD | `~/.yunshu/cache/apc`，一个全域磁盘预算并保留剩余空间，重开后仍在 | 打开（`YUNSHU_VLM_APC_DISK`、`_DIR`、`_GB`） |
| 存储层 | 外置 SSD、HDD、NAS（`YUNSHU_VLM_APC_DISK_TIERS`） | 关闭；会量测每个磁盘的速度，只有还原比重算快时才使用 |

重复或修改过的长 prompt、多轮对话与 agent 循环，会从最近的 checkpoint 还原，不必重新预填。
对话变长时，同一段对话较旧的 checkpoint 会被取代，不会越堆越多。`yunshu cache status` 与
`yunshu cache gc` 可查看与清理 SSD 缓存。

### 结构化输出

JSON schema、JSON object、regex 与 grammar 约束在解码时强制执行（默认使用 llguidance），也包括
工具调用的参数。不支持的 schema 写法会明确回错，而不是默默忽略。

## API 兼容性

| API | 路由 |
|---|---|
| OpenAI | `/v1/chat/completions`、`/v1/completions`、`/v1/responses`（HTTP 与 WebSocket）、`/v1/embeddings`、`/v1/models`、`/v1/audio/*`、`/v1/images/*`、`/v1/realtime`、`/v1/files`、`/v1/batches` |
| Anthropic | `/v1/messages`（thinking、tools、`cache_control`、服务器工具 `web_search` / `web_fetch`、`mcp_servers`）、`/v1/messages/count_tokens`、`/v1/messages/batches`、Files |
| Ollama | `/api/chat`、`/api/generate` 与模型相关路由 |
| Yunshu 扩展 | 请求实时阶段（`/v1/requests`）、以请求 id 取消、预热、截止时间、队列标头、流式输出中的预填进度、`/v1/yunshu/status` |

Chat 支持的参数：`tools` / `tool_choice` / `parallel_tool_calls`、`response_format`（`json_object`、
strict `json_schema`）、`stop`、`logprobs` / `top_logprobs`（流式输出也有）、`n`、`seed`、penalty、
`logit_bias`、`reasoning_effort`（推理内容单独返回）、含 usage 的流式输出，以及 `usage` 中的缓存 token 数。
错误使用各 API 自己的格式。扩充字段都有命名空间（`x_yunshu`、`X-Yunshu-*`），官方 SDK 会忽略。
完整矩阵与每一列的验证方式见 [API surface](docs/guides/API_SURFACE.md)。

main 的 0.1.5 周期已提供本地决策（`/v1/decisions`、`/v1/systemone`）、存储聊天响应、Evals（`/v1/evals`）与 Realtime client secrets。决策需要支持的决策 checkpoint，不使用聊天解码器。

[决策](docs/guides/DECISIONS.md)、[Evals](docs/guides/EVALS.md)、[网页搜索](docs/guides/WEB_SEARCH.md)与 [Tavily API](docs/guides/TAVILY.md)

## 编程 agent

```bash
yunshu launch claude      # or: codex, opencode
```

`yunshu launch` 会写好客户端设置（base URL、模型、上下文长度与输出上限、reasoning effort）并启动
agent。对 Claude Code 还会装上状态栏，即时显示预填进度、解码速度与缓存命中。

- **Claude Code**：Messages API，含流式输出、thinking、`/context` 用的 `count_tokens`、模型探索，以及由
  Yunshu 在服务器端运行的 WebSearch 工具。
- **Codex**：Responses API，含推理项目、function call、本地压缩与 `web_search`。
- **opencode**：Chat Completions，含工具与 usage。

服务器端网页搜索使用可设置的后端（例如 SearXNG）；请求中指定的 MCP 服务器由 gateway 连接。
各 agent 实际调用了什么、怎么验证的，见 [Agent 兼容性](docs/guides/AGENT_COMPAT.md)。

## 性能

Qwen3.8-27B（oQ4e），M5 Max 128 GB，单一请求，贪婪解码。方法、原始结果与完整比较表在
[docs/BENCHMARKS.md](docs/BENCHMARKS.md)。

| | Yunshu | TensorFold 0.6.1 |
|---|---|---|
| 冷启动首字延迟，8K prompt | 8.6 秒 | 8.5 秒 |
| 冷启动首字延迟，32K prompt | 38.3 秒 | 39.4 秒 |
| 重复或修改过的长 prompt | 从前缀缓存还原，不必重新预填 | — |
| 后续回合首字延迟，8K / 32K 代码 | 512 / 721 毫秒 | 505 / 670 毫秒 |
| 解码，短代码 prompt | 约 110 tok/s（DFlash2） | 约 140 tok/s（DFlash2） |
| JSON schema / 工具调用输出，warm | 111 / 78 tok/s（推测解码照常启用） | — |

目前 TensorFold 的单请求解码与 32K 后续回合仍较快，缩小这些差距是正在进行的主要工作。结构化输出也照常使用推测解码（不用时为 23 tok/s）。推测解码永远不会改变
Yunshu 的贪婪输出。相对于原版 MLX 路径的准确度，分三个层次检查（logit 对齐、贪婪分歧、成对下游评测），
见 [准确度](docs/guides/ACCURACY.md)。

## 模型

| 模型 | 服务路径 |
|---|---|
| Qwen3.5 / 3.6 / 3.8 家族（优先调校 Qwen3.8-27B） | VLM batch runner，含前缀缓存与 MTP / DFlash2 推测解码 |
| 其他 mlx-vlm 模型（Gemma、GLM、Qwen-VL、Qwen-Omni…） | 同一个 runner；模型支持时可输入图片、音频、视频；缓存版面允许时使用前缀缓存 |
| 纯文本 mlx-lm 模型 | 单请求快速路径，含约束、工具与 logprobs |

`/v1/models` 会回传每个模型的信息卡：上下文长度、输出上限，以及实际支持哪些输入与功能。

## 其他能力

| 能力 | 端点 | Extra |
|---|---|---|
| Qwen3-Omni 语音对语音 | `/v1/omni/speech/stream`（[范例](examples/talk.py)） | `omni` |
| Realtime 语音、ASR、TTS | `/v1/realtime`、`/v1/audio/transcriptions`、`/v1/audio/speech` | `audio` |
| OCR | `/v1/ocr`（GLM-OCR） | `vision` |
| 图片生成与编辑 | `/v1/images/generations`、`/v1/images/edits` | `generation` |
| Embeddings、rerank、相似度 | `/v1/embeddings`、`/v1/rerank`、`/v1/score` | `embeddings` |

## 命令行

| 指令 | 用途 |
|---|---|
| `yunshu doctor` | 检查这台 Mac、相依套件与模型，并说明怎么修 |
| `yunshu pull` / `yunshu model` | 下载与管理模型 |
| `yunshu serve` / `yunshu service` | 运行服务器，或安装成登录时启动的服务 |
| `yunshu launch` / `yunshu statusline` | 启动接好 Yunshu 的编程 agent；引擎即时状态栏 |
| `yunshu chat`、`complete`、`embed`、`transcribe`、`speak`、`ocr`、`image` | 在终端机使用运行中的服务器 |
| `yunshu status`、`cancel` | 服务器状态、取消进行中的请求 |
| `yunshu top` | （0.1.5）实时查看服务器健康、内存、引擎与模型（`--json` 输出单次快照） |
| `yunshu config` | 实际生效的设置与来源 |
| `yunshu cache status` / `gc` | 查看与清理 SSD 前缀缓存 |
| `yunshu bench`、`eval`、`diagnose` | 性能测试、准确度评测、系统诊断 |

每个指令都有 `--help`。

## 设置

所有设置都走同一个注册表：环境变量、TOML 档（`yunshu serve --config yunshu.toml`）或
`--set KEY=VALUE`。`yunshu config` 会显示实际生效的值与来源。常用的：

| 设置 | 用途 |
|---|---|
| `YUNSHU_VLM_DRAFT` | 草稿选择：`mtp`、`off` 或草稿模型路径 |
| `YUNSHU_SPEC_COPY_ROWS` | prompt-copy 草稿宽度（`0` = 关闭） |
| `YUNSHU_VLM_APC_MEMORY_GB`、`YUNSHU_VLM_APC_DISK_GB`、`YUNSHU_VLM_APC_DISK_DIR` | 前缀缓存的 RAM、SSD 预算与位置 |
| `YUNSHU_VLM_APC_DISK_TIERS` | 额外的存储层，例如 `/Volumes/Ext/apc@200,/Volumes/NAS/apc` |
| `YUNSHU_VLM_APC_WARM`、`YUNSHU_KV_PRECISION` | 有损的省内存选项（默认关闭） |
| `YUNSHU_AUTH_TOKEN`、`YUNSHU_QUEUE_LIMIT` | API key 与请求队列上限 |

所有设置见 [设置](docs/CONFIGURATION.md)。

## 文档

- [CLI](docs/guides/CLI.md)：命令行：首次启动、模型、launchd、JSON 与 shell completion

- [客户端](docs/guides/CLIENTS.md)：curl、OpenAI / Anthropic SDK、Open WebUI、agent
- [API surface](docs/guides/API_SURFACE.md) 与 [API 参考](docs/API.md)
- [Agent 兼容性](docs/guides/AGENT_COMPAT.md)
- [KV 缓存分层](docs/guides/KV_CACHE_MATRIX.md) 与 [prompt caching API](docs/guides/PROMPT_CACHING_APIS.md)
- [性能测试](docs/BENCHMARKS.md) 与 [准确度](docs/guides/ACCURACY.md)
- [背景服务](docs/guides/SERVICE.md)、[疑难排解](docs/guides/TROUBLESHOOTING.md)、[变更纪录](CHANGELOG.md)

## 隐私

不发送使用统计或崩溃报告。主机遥测（功耗、温度、内存）只在本机采样供控制台使用，不会离开这台机器（[说明](docs/guides/TELEMETRY.md)）。Yunshu 只在下载模型、连到你设置的网页搜索／MCP 服务，以及请求
要求抓取网页时才对外连接。

## 基础与授权

[MLX](https://github.com/ml-explore/mlx)、[mlx-lm](https://github.com/ml-explore/mlx-lm)、
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)、[mlx-audio](https://github.com/Blaizzy/mlx-audio)。
来自 [oMLX](https://github.com/jundot/omlx) 与 [TensorFold](https://github.com/ashhart/TensorFold)
的 kernel 保留其授权声明，见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。Yunshu 采用
Apache 2.0：[LICENSE](LICENSE)。
