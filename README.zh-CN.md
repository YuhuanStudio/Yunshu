<div align="center">

# Yunshu

**为 Apple Silicon 打造的快速本地 LLM / VLM 推理引擎。**

单一进程,兼容 OpenAI 与 Anthropic API,通过 MLX 在设备端运行。为单机低延迟而设计:首 token 快、
无损解码快,并以前缀复用跳过已经算过的部分。第一个完整调校的模型是 **Qwen3.8-27B**。

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

[English](./README.md) · **简体中文** · [繁體中文](./README.zh-TW.md)

</div>

---

## 与众不同之处

- **无损推测解码。** 使用 checkpoint 自带的 MTP 头,或外部 DFlash drafter。会推测的请求,其所有
  解码与验证矩阵乘都走同一颗 batch-invariant kernel,所以 greedy 下开启与关闭推测的输出逐 token
  相同 —— 即 Splash 所说的无损。非精确的快速验证也有,但需手动开启。
- **混合架构模型的前缀缓存。** Qwen3.5 家族把注意力层和循环的 GatedDeltaNet 层混在一起,普通的
  KV 缓存切不开。Yunshu 保存精确的 checkpoint,以文本与图片像素共同作为键,默认 8 GiB 内存,
  可再加一层 SSD。重复或只改结尾的长 prompt 无需重新 prefill。
- **以输出验证过的验证 kernel。** GatedDeltaNet、注意力、5-bit 矩阵乘的验证 kernel,部分取自
  oMLX,每一颗都经过同 checkpoint A/B 才采用。
- **快速路径上有完整 API。** 工具调用、JSON-schema 约束、停止序列、logprobs、流式推理/内容分离、
  `reasoning_effort` 直接传给支持它的 chat template(Qwen3.8),以及客户端断开时取消生成。

## 快速开始

需要 Apple Silicon 的 Mac(macOS 14 以上)和 [uv](https://docs.astral.sh/uv/)。

```bash
# 安装。vision extra 覆盖 Qwen3.5 / 3.6 / 3.8 系列和所有 VLM。
uv tool install "yunshu[vision] @ git+https://github.com/YuhuanStudio/Yunshu"

yunshu doctor                                   # 检查这台 Mac,并列出修复方法
yunshu pull mlx-community/Qwen3.5-9B-MLX-4bit   # 下载到 ~/.yunshu/models/
yunshu serve -m ~/.yunshu/models/mlx-community/Qwen3.5-9B-MLX-4bit
```

服务器监听 `http://127.0.0.1:8000`,任何 OpenAI 客户端都能直接用:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")  # 任意 key 都可以

# 单模型模式:模型名只是占位,服务器服务的是你加载的那个模型。
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "用一句话解释 MLX。"}],
    extra_body={"reasoning_effort": "medium"},
)
print(r.choices[0].message.content)
```

要在登录时于后台运行:`yunshu service install -m <model>`
([服务指南](docs/guides/SERVICE.md))。每个命令都有 `--help`。`yunshu model list`
会列出本机模型,包括 Hugging Face 缓存。

**从源码**(开发用):克隆仓库,运行 `uv sync --extra vision`(或 `--all-extras`),
然后 `uv run yunshu serve -m <model>`。`uv.lock` 固定了确切版本(MLX 0.32、`mlx-vlm` 0.7.3+)。

**文档:**
- [连接客户端](docs/guides/CLIENTS.md)(OpenAI / Anthropic SDK、编程代理、Open WebUI)
- [故障排查](docs/guides/TROUBLESHOOTING.md)
- [API 参考](docs/API.md)
- [配置参考](docs/CONFIGURATION.md)

## 性能

测量环境:M5 Max(128 GB)、Qwen3.8-27B、2026-09-28。除特别注明外均为同一个 Jundot `oQ4e-mtp`
checkpoint;原始数据与方法见
[docs/research/runs/2026-09-28-matrix](docs/research/runs/2026-09-28-matrix/README.md)。

| 引擎 | 能力检查 | 对话 TTFT(热) | 8K prompt:冷 / 重复 / 改尾 | 解码 tok/s |
|---|---|---|---|---|
| **Yunshu**(默认:MTP 深度 6、batch-invariant) | 34/34 | 0.194 s | 8.45 / 0.115 / 0.258 s | 80 |
| **Yunshu**(DFlash2 + 快速验证,需开启) | 31/31 | 0.185 s | 8.71 / 0.112 / 0.239 s | 86 |
| mlx-vlm 0.7.3 server(APC) | 27/28 | 0.212 s | 8.60 / 0.108 / 0.265 s | 32 |
| oMLX.app 0.7(MTP + 缓存) | 31/31 | 0.312 s | 8.60 / 0.361 / 0.376 s | 85 |
| Splash 1.1(自家量化模型 + DFlash2) | 31/31 | 0.206 s | 7.88 / 0.131 / 7.88 s | 119 |

按输出类型的无损解码(同 checkpoint、进程内、greedy、384 token;tok/s):

| 解码方式 | 代码 | 文章 | JSON 类 | 开/关推测相同 |
|---|---|---|---|---|
| **默认**:batch-invariant + packed、MTP 深度 6 | 88.6 | 59.9 | 67.3 | 是(已測)¹ |
| 旧默认:精确验证 kernel、MTP 深度 3 | 57–67 | 50–53 | 58–62 | 是(已測)¹ |
| 非精确快速验证(需开启) | 83.8 | 59.9 | 66.7 | 否 |

¹ 短 prompt 以及 1.2K/16.5K token 上下文下(各 384 token),开启与关闭推测的 greedy 输出在每个任务
都逐 token 相同。矩阵乘已逐行一致;推测验证的注意力走的 MLX kernel 和单行解码不同,所以位元层面的
logits 可能有差,其他输入下仍可能偶有 token 不同。oMLX 的逐行精确注意力可以消除这点,但解码慢
30–50%(`YUNSHU_MTP_ROW_EXACT=1`)。

MMLU-Pro 300 题、同时 8 题、最多 16384 token、`reasoning_effort=medium`(准确率与长时间稳定性;
每家设置相同):

| 引擎 | 答对 | 耗时 | 总吞吐 tok/s | 内存峰值 |
|---|---|---|---|---|
| **Yunshu**(共用批次,commit fbbb1378) | 250 / 300 | 46.5 分钟 | 88 | 45 GiB(结束回到 17) |
| Splash 1.1 | 252 / 300 | 17.2 分钟 | 223 | 67 GiB |
| oMLX.app | 229 / 300(27 题被它的 prefill 内存守卫拒绝) | 29.2 分钟 | 120 | 75 GiB |

速度全测(每个 prompt 唯一、不命中缓存;生成 128 token;未注明单位者为 tok/s):

| | Yunshu | oMLX | Splash |
|---|---|---|---|
| 8K / 131K / 200K token 的 TTFT | 8.4 / 209 / 394 s | 8.5 / 214 / 401 s | 7.9 / 207 / 390 s |
| 1K / 32K / 200K 之后的解码 | 58 / 44 / 17 | 71 / 60 / 29 | 101 / 48 / 66 |
| 8 个 1K prompt 同时,总吞吐 | 61 | 53 | 70 |

现状:
- 前缀复用与热 TTFT 是测到最好的;冷 prefill 已到硬件上限(三家相差约 5% 内)。
- 准确率与 Splash 相当;每次长时间测试都 0 错误。
- **落后 Splash** 的地方:长上下文解码(它的 KV 用 INT8),以及并发长输出(上游批次缓存把每行补到
  最长那行)。量化 KV 与每行独立长度的 KV 缓存正在验证中,用来补上这两点。
- 60 分钟混合 soak(对话、长文档、图片、工具、JSON schema、思考、中途断线)跑完 699 个请求,
  服务器错误 0,内存没有增长(footprint 17–26 GiB)。

Yunshu 的矩阵比旧测试多两项:logprobs 与流式推理分离。长期基准记录见
[docs/reports/PERF_TREND.md](docs/reports/PERF_TREND.md)。

## 支持的模型

| 层级 | 模型 | 路径 | 可得到的功能 |
|---|---|---|---|
| 1 —— 已调校并测量 | Qwen3.5 / 3.6 / 3.8 家族(文本 + 图片) | VLM batch runner | 前缀缓存(内存 + SSD)、MTP / DFlash 无损推测解码、上述所有 API 功能 |
| 2 —— 支持 | 任何 `mlx-lm` 文本模型 | 单请求快速路径(`generate_step`) | KV 前缀缓存、工具、JSON schema、logprobs;可开启 n-gram 推测、KV 量化 |
| 2 —— 支持 | 其他 `mlx-vlm` 模型(GLM、Qwen-VL、Gemma-4、Qwen3-Omni、Nemotron-Omni 等) | 同一个 VLM batch runner | 连续批处理、前缀缓存(使用 sliding window 的模型除外)、图片 / 音频 / 视频、上述所有 API 功能;无推测解码 |

2026-09-28 这一轮只重新测量了第 1 层;第 2 层的 VLM 之后才改走 runner,仍需实机冒烟测试。

## 其他模态

以下功能在同一个服务器里,通过可选 extras 安装。**2026-09-28 这一轮都没有重新验证**,
这一轮只覆盖 LLM/VLM。

| 模态 | 端点 | 后端 | Extra |
|---|---|---|---|
| 原生语音到语音(Qwen3-Omni Thinker→Talker,流式) | `POST /v1/omni/speech/stream` | `mlx-vlm` | `omni` |
| 实时语音 | `WS /v1/realtime` | omni,或 ASR → LLM → TTS | `audio` |
| ASR | `/v1/audio/transcriptions` | `mlx-audio` / Whisper | `audio` |
| TTS | `/v1/audio/speech` | `mlx-audio` | `audio` |
| 图像生成 | `/v1/images/generations` | 扩散模型 | `generation` |
| 视频生成(Wan 2.x / LTX-2) | `/v1/video/generations` | `mlx-video` | `video` |
| Embeddings / rerank(文本 + 多模态) | `/v1/embeddings`、`/v1/rerank` | `mlx-lm` / `mlx-embeddings` | `embeddings` |

语音到语音:服务一个 Qwen3-Omni 模型(`uv sync --extra omni`),试试
[`examples/talk.py`](examples/talk.py)(麦克风)或 [`examples/quickstart.py`](examples/quickstart.py)
(输出 WAV,无需音频硬件)。已确认上游 `mlx-vlm` 0.7.3 多轮 omni 输出正确
([记录](docs/research/runs/2026-09-28-omni/README.md));服务器的 Realtime 路径尚未确认。

另外:MCP 服务器/客户端,以及兼容 Anthropic 的 `/v1/messages`。

## 架构

```
  客户端(任意 OpenAI / Anthropic SDK)
        │   OpenAI / Anthropic / MCP / Realtime-WS / SSE
  ┌─────┴───────────────────────────────────────────────┐
  │  网关(FastAPI)       路由 + 中间件                   │
  ├─────────────────────────────────────────────────────┤
  │  引擎                                                 │
  │   · VLM batch runner(所有 mlx-vlm 模型)             │
  │       连续批处理 · 前缀缓存(内存 + SSD)             │
  │       Qwen3.5 家族:MTP / DFlash + batch-invariant    │
  │   · LLM 快速路径(mlx-lm generate_step)              │
  │       KV 前缀缓存 · 约束解码                          │
  │   · 其他模态:omni、ASR/TTS、图像、视频、embeddings   │
  └─────────────────────────────────────────────────────┘
        单一 MLX 线程 · 通过 Apple MLX 在设备端运行
```

## 服务模型

所有 GPU 工作都在一条 MLX 线程上。VLM(mlx-vlm)模型的并发请求共用一个连续批次,每行有自己的
采样设置;单独一个请求时会使用推测解码(Qwen3.5 家族),期间进来的请求则加入共用批次、不做推测。
纯文本的 mlx-lm 模型走单请求快速路径,并发请求会依次执行。每个响应在它自己的生成结束时就立即返回。

## 配置

所有设置都是 [配置参考](docs/CONFIGURATION.md) 列出的 `YUNSHU_*` 名称(由同一份注册表生成)。
可用环境变量、TOML 文件(`yunshu serve --config yunshu.toml`)或 `yunshu serve --set KEY=VALUE`
设置;`yunshu config` 显示每项的生效值与来源。值无法解析会中止启动,拼错的名称会收到警告。

## 构建于

[MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)。
部分验证 kernel 取自 [oMLX](https://github.com/jundot/omlx)(Apache-2.0),见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 许可证

Apache 2.0 —— 见 [LICENSE](LICENSE)。
