<div align="center">

# Yunshu

**为 Apple Silicon 打造的快速、本地、单节点 LLM / VLM 推理引擎。**

单一进程，兼容 OpenAI／Anthropic API，透过 MLX 在设备端运行。重点是 LLM/VLM 解码速度、
冷与缓存首 token 延迟（TTFT）、前缀复用，以及完整的推理功能。
**Qwen3.8-27B 是第一个完整调优的模型。** 语音、Realtime 语音、图片生成、视频输入与嵌入
是支持能力；核心是 LLM/VLM 服务。Yunmo 是其中一个用户。

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

[English](README.md) · **简体中文** · [繁體中文](README.zh-TW.md)

</div>

最新发布版：**v0.1.2（2026-10-02）**；**v0.1.1（2026-09-29）**。本页也涵盖
**尚未发布的 main**；见 [CHANGELOG](CHANGELOG.md)。

## 推理功能

- **推测解码。** 自动寻找相符的 DFlash2 草稿模型，否则使用 checkpoint 的 MTP 头。
  不随批次大小改变的解码／验证 kernel 支持该路径内的贪婪输出一致性；采样草稿使用
  依位置决定的抽样。main 新增最多 32 列的不变验证，以及 MTP lane 的提示复制草稿
  （默认开启，`YUNSHU_SPEC_COPY_ROWS=0` 关闭）。树状草稿仍是实验功能，默认关闭。
  一般 runner 与 invariant lane 并非全面逐 token 相同：见
  [准确度证据](docs/guides/ACCURACY.md) 与 [测量限制](docs/BENCHMARKS.md)。
- **前缀复用。** 混合架构 Qwen3.5 家族的 checkpoint 保留 attention KV 与循环状态；
  文字与媒体键避免错误复用。RAM APC 加上默认有容量上限的 SSD 缓存，支持重复回合与重启。
  main 新增可选的 WARM RAM 与下层存储，以实测还原成本选择来源。无损 WARM 默认关闭；
  int8/int4 缓存格式与 KV 量化是用户自行开启的有损选项。见
  [缓存层级](docs/guides/KV_CACHE_MATRIX.md)。
- **完整请求功能。** 工具、JSON-schema 约束、stop、logprobs、分离的推理／内容流式输出、
  模板支持的 reasoning effort 与取消。`/v1/models` 公告模型能力；不支持的请求明确报错。
  main 的 JSON 约束默认使用 llguidance；无法强制执行的语法会被拒绝。
- **编程代理。** Claude Code、Codex 与 opencode 使用原生 Messages／Responses／Chat API，
  支持服务器端搜索／抓取／MCP，以及 Files、Batches、Conversations。`yunshu launch`
  提供模型限制；`yunshu statusline` 在 Claude Code 显示引擎状态。
  [兼容性证据](docs/guides/AGENT_COMPAT.md) 区分已验证功能与客户端限制。
- **本地诊断。** `yunshu doctor`、`yunshu cache status`、`yunshu cache gc`、
  `yunshu diagnose`；诊断资料留在本机，不含提示。

## 快速开始

需要 Apple Silicon、macOS 14+、Python 3.13+ 与 [uv](https://docs.astral.sh/uv/)。

```bash
uv tool install "yunshu[vision]"
yunshu doctor
yunshu pull mlx-community/Qwen3.5-9B-MLX-4bit
yunshu serve -m mlx-community/Qwen3.5-9B-MLX-4bit
```

服务器监听 `http://127.0.0.1:8000`。模型存于 `~/.yunshu/models`；`serve -m org/name`
也会寻找 Hugging Face 缓存，只在需要时下载。
`yunshu config set models_dir /path/to/models` 可更改位置。

### Qwen3.8-27B

```bash
yunshu pull Jundot/Qwen3.8-27B-oQ4e-mtp
yunshu pull incoai/Qwen3.8-27B-DFlash2
yunshu doctor -m Jundot/Qwen3.8-27B-oQ4e-mtp
yunshu serve -m Jundot/Qwen3.8-27B-oQ4e-mtp
```

安装相符草稿模型后，启动日志应显示 `Speculative decoding: dflash`；`doctor` 显示选用路径。
`YUNSHU_VLM_DRAFT=mtp` 强制 MTP，`YUNSHU_VLM_DRAFT=off` 停用草稿，绝对路径可指定草稿模型。
内存需求取决于权重、上下文、草稿与缓存预算；请在自己的 Mac 上用 `doctor` 检查模型，
不要将 benchmark 占用量当作最低 RAM 保证。

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

未设定 `--auth-token` 时，任何 key 都可以。单模型模式的 `local` 是占位名称；
多模型模式请使用 `/v1/models` 中的 id。

使用尚未发布的 main／从源代码开发：

```bash
git clone https://github.com/YuhuanStudio/Yunshu.git
cd Yunshu
uv sync --extra vision
uv run yunshu serve -m Jundot/Qwen3.8-27B-oQ4e-mtp
```

`uv.lock` 固定依赖；main 需要 MLX 0.32.3+、mlx-lm 0.32.0+、mlx-vlm 0.7.4+。
`yunshu model list` 列出本地模型。`yunshu service install -m <model>` 安装登入服务
（[服务指南](docs/guides/SERVICE.md)）；每个指令都有 `--help`。

**没有遥测。** 模型下载、设定的 MCP／搜索供应商、请求触发的网页抓取可能连外。
不传送使用分析或崩溃报告。main 可选的 `YUNSHU_SERVE_LOG` 只在本地记录数字，
不含提示、输出或 token id。

## 效能

下列每笔测量都来自 [BENCHMARKS](docs/BENCHMARKS.md) 或
[有日期的 PERF_TREND 纪录](docs/reports/PERF_TREND.md)。除特别注明外，为 M5 Max／128 GB、
Qwen3.8-27B Jundot oQ4e-mtp。日期是测量日期，不是发布日期。

**2026-10-02 tfbench**，记录于 `4e1338c4`：贪婪、输出 256 token，每格三次独立
服务器 session，取中位数。Yunshu 是当日早上的 **未发布 main（未记录 SHA），实际使用 MTP**；
TensorFold 使用 DFlash2。这早于 main 的 prefill／提示复制／宽验证合并，并非相同草稿模型的 A/B。

| Context / output | TensorFold 0.6.1 tok/s / cold TTFT s | Yunshu main (MTP) tok/s / cold TTFT s |
|---|---|---|
| 1K code | 140.5 / 1.2 | 69.3 / 1.4 |
| 1K prose | 72.4 / 1.2 | 52.9 / 1.4 |
| 8K code | 80.6 / 8.4 | 60.2 / 11.1 |
| 8K prose | 67.4 / 8.5 | 57.5 / 11.1 |
| 32K code | 91.2 / 39.4 | 58.7 / 47.2 |
| 32K prose | 55.1 / 39.4 | 46.3 / 47.7 |

TensorFold 在以上各格皆领先。**Splash 在 2026-09-27 探索性 TTFT 测试也领先**：
3,323-token 提示，冷／重复为 3.221／0.130 s，Yunshu 直接 VLMEngine 为 3.511／3.512 s。
Splash 1.1.0 使用自己的量化模型与 INT8 KV；单次测量、共享 GPU，Yunshu SHA 未知。
权重与缓存条件不同，不能只归因于引擎。上述合并之后尚无更新的完整比较。

**2026-10-02 稍后的 main 测量，尚未发布：**

| Workload | Before | After | Commit / source |
|---|---|---|---|
| Cold TTFT, 8K | 11.0 s | 8.57 s | `a4d71bc8`, BENCHMARKS |
| Cold TTFT, 32K | 47.5 s | 38.3 s | `a4d71bc8`, BENCHMARKS |
| MTP prompt-copy, code turn 2, 32K | 63 tok/s | 100–138 tok/s | `c666be70` / `bb4895ca`, PERF_TREND |
| MTP prompt-copy, code turn 2, 8K | 62 tok/s | 86 tok/s | same |

prefill 数字是有日期的合并纪录；公开资料未提供重复次数／区间。
提示复制的贪婪 digest 开关一致；散文差异在 ±3% 单次测量杂讯内。
冷 prefill 数值可能与旧版不同；APC namespace 包含 prefill 设定。
这些结果不代表全面解码加速，也不是硬体上限宣告。

**编程代理测试，部分完成，测于 2026-10-02**（PERF_TREND／`2815311c`）：

| Snapshot / agent | Pass / runs | Rate (Wilson 95%) | Wall median / p90 s | Decode median tok/s |
|---|---|---|---|---|
| prod `c4e2b244+` / opencode | 34 / 41 | 83% (69–91%) | 337 / 1200 | 22.2 |
| prod4 `d225f16c` / Claude Code | 14 / 15 | 93% (70–99%) | 93 / 309 | 60.8 |
| prod4 `d225f16c` / opencode | 9 / 10 | 90% (60–98%) | 112 / 137 | 56.9 |

失败保留在分母，包括 prod 的五次逾时。prod4 涵盖 20 个任务中的 10 个，各重复 1–3 次；
`d225f16c` snapshot 已包含在 v0.1.2。不同 snapshot 与任务组合不能用来宣称配对加速。
Codex 与 TensorFold 代理结果仍待完成。准确度 Tier 3 与 APC replay 结果，包含退步与
未完成子集，见 [BENCHMARKS](docs/BENCHMARKS.md)。

## 模型与服务路径

| 模型 | 服务路径 | 范围 |
|---|---|---|
| Qwen3.5／3.6／3.8 家族；先调优 Qwen3.8-27B | VLM batch runner | APC、支持家族模型的 MTP/DFlash、逐列采样 |
| 其他 mlx-vlm 模型（Gemma、GLM、Qwen-VL、Omni 等） | 同一个 VLM batch runner | 依模型支持媒体与工具；缓存结构允许时有 APC；无家族专属推测解码 |
| 纯文字 mlx-lm 模型 | 单请求 `generate_step` 快速路径 | 依模型支持前缀缓存、约束、工具与 logprobs；并行请求依序执行 |

所有 GPU 工作在同一个 MLX 线程。默认 VLM runner 共享并行解码列；单独的支持请求可用草稿。
多列草稿使用实验性的 `YUNSHU_ROUND_DRIVER`，默认关闭，有实测延迟取舍。
main 的 M01 将文字引擎整理为多个模组，未改变服务路径。实际能力请看模型卡，不能只看家族名称。

## 其他支持能力

| 能力 | API／范例 | Extra |
|---|---|---|
| Qwen3-Omni 原生语音对语音 | `/v1/omni/speech/stream`、[talk.py](examples/talk.py) | `omni` |
| Realtime 语音、ASR、TTS | `/v1/realtime`、`/v1/audio/transcriptions`、`/v1/audio/speech` | `audio`（原生 Omni 需 `omni`） |
| 图片生成 | `/v1/images/generations` | `generation` |
| 嵌入／重排 | `/v1/embeddings`、`/v1/rerank` | `embeddings` |
| 文字 WebSocket／Responses WebSocket／Unix socket | `/v1/stream`、`/v1/responses`、`yunshu serve --uds PATH` | core |

这些能力各有模型／后端需求，不在 LLM/VLM 效能表的测试范围。
适用的 VLM 支持视频输入；视频生成的引擎后端与公开 HTTP 路由已移除。
WebRTC 与 HTTP/2 尚未实现。[API 表面](docs/guides/API_SURFACE.md) 与
[传输方式](docs/guides/TRANSPORTS.md) 列出验证与限制。

## 设定与文件

设定经过同一个 registry：环境变数、TOML（`yunshu serve --config yunshu.toml`），
或 `yunshu serve --set KEY=VALUE`；`yunshu config` 显示生效值与来源。

- [连接客户端](docs/guides/CLIENTS.md)
- [故障排查](docs/guides/TROUBLESHOOTING.md)
- [API 参考](docs/API.md)
- [设定参考](docs/CONFIGURATION.md)
- [文件索引](docs/README.md)

## 基础与授权

[MLX](https://github.com/ml-explore/mlx)、[mlx-lm](https://github.com/ml-explore/mlx-lm)、
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)、[mlx-audio](https://github.com/Blaizzy/mlx-audio)。
来自 [oMLX](https://github.com/jundot/omlx) 与 [TensorFold](https://github.com/ashhart/TensorFold)
的 kernel 授权通知见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。
Yunshu 采 Apache 2.0：[LICENSE](LICENSE)。
