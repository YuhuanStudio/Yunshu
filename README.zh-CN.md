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

- **无损推测解码。** 优先使用已安装的 DFlash2 drafter(自动选用),否则用 checkpoint 自带的 MTP 头。会推测的请求,其所有
  解码与验证矩阵乘都走同一颗 batch-invariant kernel,所以 greedy 下开启与关闭推测的输出逐 token
  相同 —— 即 Splash 所说的无损。
- **并发请求各行独立的 KV。** 共享批次里每一行保有自己的 KV 长度,短请求不会读到长请求的补齐
  (Qwen3.5 家族;8 路并发的 MMLU-Pro 88 → 139 tok/s,131K 上下文解码 24 → 47 tok/s)。
- **有损的只在你要求时才开。** 所有默认值都是无损的。会改变输出的省内存选项(int8 KV、KV 量化、
  4-bit 缓存前缀、int8 SSD 缓存)都是需要手动开启的设置。
- **混合架构模型的前缀缓存。** Qwen3.5 家族把注意力层和循环的 GatedDeltaNet 层混在一起,普通的
  KV 缓存切不开。Yunshu 保存精确的 checkpoint,以文本与图片像素共同作为键,默认 8 GiB 内存,
  另有默认开启的 SSD 层(`~/.yunshu/cache/apc`,每个缓存根目录一个全局磁盘预算并保留剩余空间,
  `YUNSHU_VLM_APC_DISK=0` 可关闭)。重复或只改结尾的长 prompt 无需重新 prefill,重启后也一样。
- **以输出验证过的验证 kernel。** GatedDeltaNet、注意力、5-bit 矩阵乘的验证 kernel,部分取自
  oMLX,每一颗都经过同 checkpoint A/B 才采用。
- **快速路径上有完整 API。** 工具调用、JSON-schema 约束、停止序列、logprobs、流式推理/内容分离、
  `reasoning_effort` 直接传给支持它的 chat template(Qwen3.8),以及客户端断开时取消生成。`/v1/models` 会标明各模型支持的功能
  (工具、结构化输出、logprobs、媒体、context),请求用到模型没有的功能会得到明确的 400。
  regex 与 JSON-schema 约束是精确执行的(超出内置子集的 schema 交给 llguidance);无法执行的
  语法(如 `uniqueItems`、`not`、`if / then / else`、`contains`)返回 400,不会被默默忽略。
- **编程代理兼容。** Claude Code、Codex、opencode 都能使用:Messages 与 Responses 的原生工具、
  服务端 `web_search` / `web_fetch` / MCP connector、Files、Batches、Conversations,以及给
  Claude Code 状态栏用的 `yunshu statusline`。采样请求(temperature 大于 0,代理发送的就是这种)
  也使用推测解码。各功能的证据见 [AGENT_COMPAT.md](docs/guides/AGENT_COMPAT.md)。
- **诊断。** `yunshu doctor`(每个问题附修复方法)、`yunshu cache status|gc`、`yunshu diagnose`
  (本机诊断包,不含 prompt,不上传)。

## 快速开始

需要 Apple Silicon 的 Mac(macOS 14 以上)和 [uv](https://docs.astral.sh/uv/)。

```bash
# 安装。vision extra 覆盖 Qwen3.5 / 3.6 / 3.8 系列和所有 VLM。
uv tool install "yunshu[vision]"

yunshu doctor                                   # 检查这台 Mac,并列出修复方法
yunshu pull mlx-community/Qwen3.5-9B-MLX-4bit   # 下载到 ~/.yunshu/models/
yunshu serve -m mlx-community/Qwen3.5-9B-MLX-4bit
```

其他安装方式:`pipx install "yunshu[vision]"`、Homebrew(`brew install yuhuanstudio/tap/yunshu`),
或最新的 `main`(`uv tool install "yunshu[vision] @ git+https://github.com/YuhuanStudio/Yunshu"`)。

`yunshu serve -m org/name` 会直接使用模型目录或 Hugging Face 缓存里已有的模型,两边都没有才下载。
模型默认放在 `~/.yunshu/models`;要放到别处,执行 `yunshu config set models_dir /path/to/models`
(保存在 `~/.yunshu/config.toml`)。

### Qwen3.8-27B(已调优的模型)

实测内存占用约 21 GiB(1K prompt)、29 GiB(32K),包含权重、drafter 与 KV,建议 32 GB 及以上的
Mac;131K 上下文需要更多。

```bash
yunshu pull Jundot/Qwen3.8-27B-oQ4e-mtp            # 模型:4-bit,自带 MTP 头
yunshu pull incoai/Qwen3.8-27B-DFlash2             # drafter:比 MTP 更快
yunshu doctor -m Jundot/Qwen3.8-27B-oQ4e-mtp       # 「speculative」一行显示将使用的推测路径
yunshu serve -m Jundot/Qwen3.8-27B-oQ4e-mtp
```

drafter 放在模型目录或 Hugging Face 缓存时,`yunshu serve` 会自动找到并使用,无需任何开关;启动日志会
打印 `Speculative decoding: dflash`。要自行指定:`YUNSHU_VLM_DRAFT=mtp` 强制使用 checkpoint 的 MTP 头,
`YUNSHU_VLM_DRAFT=off` 关闭推测,`YUNSHU_VLM_DRAFT=/path/to/drafter` 指定某个 drafter。每条路径都是无损的:
greedy 下开启与关闭推测的输出相同。

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

**不收集遥测。** Yunshu 不收集、不发送任何使用数据、分析或崩溃报告。但这不等于“从不联网”:只有在你或你的请求触发时才会
对外连接,包括你要求的模型下载、你配置的 MCP 服务器、你设定的服务端 `web_search` 提供方,以及 `web_fetch`
(默认开启,只抓取请求指定的 URL,除非你允许,否则会拦截私有地址,见 `YUNSHU_WEB_FETCH`)。维护者的上游检查
(`just vendor-check`)由人工手动执行。

**文档:**
- [连接客户端](docs/guides/CLIENTS.md)(OpenAI / Anthropic SDK、编程代理、Open WebUI)
- [故障排查](docs/guides/TROUBLESHOOTING.md)
- [API 参考](docs/API.md)
- [配置参考](docs/CONFIGURATION.md)

## 性能

在 M5 Max(128 GB)上以 Qwen3.8-27B 测量,2026-09-28/29。除非另有注明,都用同一个 Jundot
`oQ4e-mtp` checkpoint。各表的测量方法与脚本见
[docs/BENCHMARKS.md](docs/BENCHMARKS.md);原始数据由维护者保存,不随 repo 发布。

| 引擎 | 能力检查 | 对话 TTFT(热) | 8K prompt:冷 / 重复 / 改尾巴 | 解码 tok/s |
|---|---|---|---|---|
| **Yunshu 0.1.1**(默认:MTP block 6、batch-invariant、ragged KV) | 34/34 | 0.192 s | 8.42 / 0.115 / 0.259 s | 73 |
| mlx-vlm 0.7.3 server(APC) | 27/28 | 0.212 s | 8.60 / 0.108 / 0.265 s | 32 |
| oMLX.app 0.7(MTP + 缓存) | 31/31 | 0.312 s | 8.60 / 0.361 / 0.376 s | 85 |
| Splash 1.1(自家量化模型 + DFlash2) | 31/31 | 0.206 s | 7.88 / 0.131 / 7.88 s | 119 |
| TensorFold 0.3.6.1(MTP,parallel 8) | 23/34 | — | — | 28 |

Yunshu 的检查项比旧的测量多(logprobs、流式推理分离);TensorFold 在图片、工具、JSON schema、
logprobs 几项没有通过。

单个请求的无损解码,按输出类型(同一 checkpoint、进程内、greedy、384 token;tok/s):

| 上下文 | 代码 | 散文 | 类 JSON | 开推测 == 关推测 |
|---|---|---|---|---|
| 1K | 82.1 | 57.8 | 69.0 | 是(已测)¹ |
| 32K | 75.4 | 51.2 | 64.2 | 是(已测)¹ |
| 131K | 59.7 | 43.8 | 46.0 | 是(已测)¹ |

¹ 上面每个上下文、每个任务,开推测与不开推测的 greedy 输出都逐 token 相同。矩阵乘与行数无关;
解码与验证的注意力走同一个逐行 kernel,一个 token 的结果不会因为一起验证的 token 数而改变。

MMLU-Pro,300 题,8 路并行,上限 16384 token,`reasoning_effort=medium`(同时检验准确率与长时间
稳定性;所有引擎设置相同):

| 引擎 | 答对 | 耗时 | 总吞吐 tok/s | 内存峰值 |
|---|---|---|---|---|
| **Yunshu**(ragged KV) | 249 / 300 | 28.3 分 | 139 | 33.7 GiB |
| Yunshu 0.1.0 时期的共享批次(补齐 KV) | 250 / 300 | 46.5 分 | 88 | 45 GiB |
| TensorFold 0.3.6.1(MTP,parallel 8) | 250 / 300 | 24.9 分 | 159 | 35.1 GiB |
| Splash 1.1 | 252 / 300 | 17.2 分 | 223 | 67 GiB |
| oMLX.app | 229 / 300(27 题被它的 prefill 内存保护拒绝) | 29.2 分 | 120 | 75 GiB |

开启 `YUNSHU_KV_PRECISION=int8`(有损,需手动开启)时 Yunshu 答对 251 / 300,峰值 28.1 GiB
(在较早版本的 ragged 缓存上测量,34.6 分)。

**Qwen3.8-27B 的默认推测路径:DFlash2 drafter**(成本感知的链深度、8-bit drafter 权重),server、greedy、
生成 128 token、单个请求、prompt 各不相同,tok/s。两种语料:小说散文(`novel_en`,难以草拟)与 Python
代码(`code_python`,容易草拟):

| 上下文 | novel_en | code_python |
|---|---|---|
| 1K | 57.1 | 82.0 |
| 8K | 48.2 | 89.1 |
| 32K | 46.1 | 70.5 |
| 131K | 32.9 | 79.9 |

131K 的 TTFT 约 207 s(冷 prefill,约 640 tok/s,已在硬件上限);能力矩阵 34/34。同一 server 上对照 checkpoint
自带的 MTP 头,novel_en 在 1K 为 57.1 对 47.5 tok/s。数字每次运行会有几 tok/s 的浮动,因为接受率取决于文本。

同语料对照(novel_en,单个请求,tok/s;各引擎的量化可能不同;两者均于 2026-09-29 测量):

| 上下文 | Yunshu(DFlash2) | TensorFold 0.3.6.1(DFlash2) |
|---|---|---|
| 1K | 57.1 | 76.4 |
| 8K | 48.2 | 67.2 |
| 32K | 46.1 | 63.4 |
| 131K | 32.9 | 38.9 |
| 131K 的 TTFT | 207 s | 280 s |

在这组小说散文上,TensorFold 每个上下文的解码都比 Yunshu 快(131K 约 1.2 倍,1K–32K 约 1.3–1.4 倍);
Yunshu 的 prefill 更快(131K:207 对 280 s)。Splash 与 oMLX 没有在这组语料上跑过;下表中它们的数字来自较早、
不同的 prompt 集合,不能与上面各列直接比较。

较早、使用 MTP 草稿的速度扫描(每个 prompt 都不同,不命中缓存;生成 128 token;未注明单位者为 tok/s):

| | Yunshu | oMLX | Splash | TensorFold(MTP) |
|---|---|---|---|---|
| 8K / 131K token 的 TTFT | 8.6 / 207 s | 8.5 / 214 s | 7.9 / 207 s | 9.8 / 293 s |
| 1K / 32K / 131K 之后的解码 | 59² / 59 / 47 | 71 / 60 / 38 | 101 / 48 / 68 | 26 / 57 / 19 |
| 8 个 1K prompt 并发,总吞吐 | 64 | 53 | 70 | 65 |

² 单个请求的解码速度取决于草稿被接受多少,会随 prompt 变动;Yunshu 的 1K 数字是 8 次的平均
(单次介于 40–70)。其他格都是单次测量。

Yunshu 目前的位置:
- 前缀重用与热 TTFT 是测到最好的;冷 prefill 已达硬件上限(各引擎相差约 10% 以内)。
- 准确率与其他引擎相当;每次长时间运行都是 0 错误。
- **落后 Splash** 的地方:长上下文解码(131K:47 vs 68 tok/s)与并发长输出(MMLU-Pro:139 vs
  223 tok/s)。TensorFold 在后者也领先(159),因为它每一行都起草;Yunshu 只在请求单独运行时起草。
  多行推测解码开发中(`YUNSHU_ROUND_DRIVER`,实验性)。
- 其他请求解码时若有长 prompt 进来,prefill 期间其他请求的解码会停住;测过的每个引擎都是如此。
- 2026-09-28 版本的 60 分钟混合压测(对话、长文档、图片、工具、JSON schema、思考、断线)完成
  699 个请求,服务器 0 错误,内存没有增长(17–26 GiB)。

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
| Embeddings / rerank(文本 + 多模态) | `/v1/embeddings`、`/v1/rerank` | `mlx-lm` / `mlx-embeddings` | `embeddings` |

语音到语音:服务一个 Qwen3-Omni 模型(`uv sync --extra omni`),试试
[`examples/talk.py`](examples/talk.py)(麦克风)或 [`examples/quickstart.py`](examples/quickstart.py)
(输出 WAV,无需音频硬件)。已确认上游 `mlx-vlm` 0.7.3 多轮 omni 输出正确
(维护者于 2026-09-28 确认);服务器的 Realtime 路径尚未确认。

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
