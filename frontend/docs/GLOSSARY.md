# Translator glossary

Source of truth for zh-TW and zh-CN wording in `frontend/docs` and the console strings. Mined from the established pages (API, getting started, the guides, the architecture / contributing / verification / release-gate / upstream-tracking developer pages) and `src/i18n/locales/{zh-TW,zh-CN}`. When a term has a row, use it. Where the old pages disagree, the dominant form (by count) wins and the note says so. This file is not a docs page: the docs scripts only read `.mdx`.

## Style rules

1. **CJK / Latin spacing.** The old pages are not uniform (`27B 模型` and `27B模型` both occur). Use one half-width space between Chinese and a Latin word or number, and none next to full-width punctuation. Be consistent inside a page.
2. **Punctuation.** Use full-width punctuation in Chinese sentences (，。；：？！、（）). Use half-width inside code, URLs, numbers, table separators and English fragments. Alternatives in prose use `／` or `、`, not a half-width `/` between Chinese words.
3. **Keep in English, in backticks:** commands (`yunshu serve`), `YUNSHU_*` settings, flags, endpoints, header names, JSON fields, model ids, file paths, environment variables. Never translate or re-space them.
4. **Product and technical names stay Latin:** Yunshu, MLX, mlx-lm, mlx-vlm, MTP, DFlash, APC, launchd, Metal, OpenAI, Anthropic, Ollama, MCP, SSD, GPU, RAM, KV, TTFT, tok/s. The brand 雲樞 (zh-TW) / 云枢 (zh-CN) is used only where the old pages use it.
5. **No OpenCC artefacts.** Never convert zh-TW to zh-CN (or back) mechanically. Known traps: 裡/里 (`里程`, `後面`), 著/着 (`著作` keeps 著), 發/发 (`髮` vs `發`), 乾/干, 麵/面, 佈/布, 迴/回, 係/繫/系, 製/制, 臺/檯/台, 並/并 (`並行` vs `并行`). Translate the sentence, then hand-check these characters.
6. **zh-CN is not zh-TW in simplified glyphs.** Use mainland vocabulary: 默认 (not `预设`), 内存 (not `记忆体`), 服务器 (not `伺服器`), 进程 (not `行程`), 缓存 (not `快取`), 文件 (not `档案`), 日志 (not `日誌`), 令牌 (not `权杖`), 密钥 (not `金钥`), 项目 (not `專案`), 配置 for the act of configuring and 设置 for settings pages.
7. **zh-TW is Taiwan usage**, not mainland words in traditional glyphs: 預設, 記憶體, 伺服器, 行程, 快取, 檔案, 日誌, 權杖, 金鑰, 設定 (the old pages have 289 `設定` against 4 `配置`).
8. **Keep `token` as `token`** for LLM tokens in both locales (`每個 token`, `首 token 延遲`). 權杖 (zh-TW) and 令牌 (zh-CN) are for auth tokens only.
9. **Measure words and units.** zh-TW 個, zh-CN 个. Units stay Latin with a half-width space (`32K token`, `22 tok/s`, `1.2 秒`).
10. **Headings** are noun phrases without a final period. Link text follows the translated heading. Never change an English heading's anchor id.
11. **Do not translate** error `code`, `type`, status names and `stop_reason` values. Translate only the human sentence around them.
12. **Quotes.** 「」 in zh-TW prose, “” in zh-CN prose, straight quotes in code.

## Terms

| English | zh-TW | zh-CN | Note |
| --- | --- | --- | --- |
| engine | 引擎 | 引擎 | the inference process |
| gateway | 閘道 | 网关 | |
| server | 伺服器 | 服务器 | the HTTP server |
| service | 服務 | 服务 | |
| background service | 背景服務 | 后台服务 | the `yunshu service` install |
| launchd service | launchd 服務 | launchd 服务 | keep `launchd` Latin |
| console | 主控台 | 主控台 | dominant in docs (zh-CN 30 vs 控制台 4) and UI; a few UI strings still say 控制台, new text uses 主控台 |
| dashboard | 儀表板 | 仪表板 | |
| CLI | 命令列 | 命令行 | `CLI` itself usually stays Latin |
| command | 指令 | 命令 | |
| terminal | 終端機 | 终端 | |
| endpoint | 端點 | 端点 | |
| route | 路由 | 路由 | |
| request | 請求 | 请求 | |
| response | 回應 | 响应 | |
| header | 標頭 | 标头 | |
| client | 用戶端 | 客户端 | |
| user | 使用者 | 用户 | |
| model | 模型 | 模型 | |
| weights | 權重 | 权重 | |
| checkpoint (model) | 檢查點 | 检查点 | |
| checkpoint (prefix cache) | 檢查點 | 检查点 | exact restorable state at a 64-aligned boundary |
| prefix | 前綴 | 前缀 | |
| prefix cache | 前綴快取 | 前缀缓存 | dominant (44 hits each); APC stays Latin |
| cache | 快取 | 缓存 | |
| cache hit | 命中 | 命中 | verb form `命中快取` / `命中缓存` |
| cache miss | 未命中 | 未命中 | |
| warm / cold | 暖 / 冷 | 暖 / 冷 | `暖請求`, `暖層`; cold start 冷啟動 / 冷启动 |
| warm tier | 暖層 | 暖层 | RAM tier of evicted checkpoints |
| eviction / evict | 淘汰 | 淘汰 | `逐出` appears only for counter names |
| restore | 還原 | 还原 | |
| persist | 持久化 | 持久化 | |
| prefill | 預填 | 预填 | zh-CN also 预填充 (UI phase labels); keep one form per page |
| decode (phase) | 解碼 | 解码 | |
| decode speed | 解碼速度 | 解码速度 | |
| speculative decoding | 推測解碼 | 推测解码 | |
| draft | 草稿 | 草稿 | |
| drafter | 草稿器 | 草稿器 | `DFlash drafter` stays Latin when naming the checkpoint |
| drafted tokens | 草稿 token | 草稿 token | |
| accepted tokens | 被接受的 token | 被接受的 token | |
| accept rate | 接受率 | 接受率 | |
| draft depth | 草稿深度 | 草稿深度 | |
| next-turn rehearsal | 下一回合預演 | 下一回合预演 | |
| TTFT | TTFT | TTFT | spoken form below |
| first-token latency | 首 token 延遲 | 首 token 延迟 | |
| latency | 延遲 | 延迟 | |
| throughput | 吞吐量 | 吞吐量 | |
| tok/s | tok/s | tok/s | never translate |
| token | token | token | see rule 8 |
| auth token | 權杖 | 令牌 | |
| API key | API 金鑰 | API 密钥 | |
| quota | 配額 | 配额 | |
| rate limit | 速率限制 | 速率限制 | 限流 only in casual UI text |
| usage | 用量 | 用量 | |
| batch (Batch API) | 批次 | 批处理 | |
| batch (decoding) | 批次 | 批处理 | |
| batch-invariant | 批次不變 | 批次不变 | |
| continuous batching | 連續批次處理 | 连续批处理 | |
| concurrent | 並行 | 并行 / 并发 | zh-CN 并发 for limits |
| queue / queued | 排隊 | 排队 | |
| KV cache | KV 快取 | KV 缓存 | |
| quantization | 量化 | 量化 | |
| lossless | 無損 | 无损 | |
| lossy | 有損 | 有损 | |
| identical / identity | 一致 / 完全相同 | 一致 / 完全相同 | "token-identical" is token 完全一致 |
| bit-exact | 位元精確 | 位精确 | |
| kernel | kernel | kernel | prose keeps Latin `kernel`; headings may use 核心 (zh-TW) / 内核 (zh-CN) |
| fused kernel | 融合核心 | 融合内核 | |
| unified memory | 統一記憶體 | 统一内存 | |
| memory | 記憶體 | 内存 | |
| memory ledger | 記憶體帳本 | 内存账本 | |
| memory pressure | 記憶體壓力 | 内存压力 | |
| RAM | RAM | RAM | |
| disk | 磁碟 | 磁盘 | |
| SSD tier | SSD 層 | SSD 层 | |
| idle | 閒置 | 空闲 | zh-CN 空闲 for the state, 闲置 for unused resources |
| keep-alive | 保持載入 | 保持加载 | |
| load / unload | 載入 / 卸載 | 加载 / 卸载 | |
| restart | 重新啟動 | 重启 | 重啟 acceptable in short zh-TW UI text |
| start | 啟動 | 启动 | |
| stream / streaming | 串流 | 流式 | |
| server-sent events | 伺服器傳送事件 | 服务器发送事件 | `SSE` stays Latin |
| tool | 工具 | 工具 | |
| tool call | 工具呼叫 | 工具调用 | |
| function calling | 函式呼叫 | 函数调用 | |
| tool result | 工具結果 | 工具结果 | |
| reasoning | 推理 | 推理 | |
| thinking | 思考 | 思考 | |
| thinking budget | 思考 token 上限 | 思考 token 上限 | |
| inference | 推論 | 推理 | zh-TW 推論 for "inference" in general; reasoning content is 推理 in both |
| structured output | 結構化輸出 | 结构化输出 | |
| JSON schema | JSON schema | JSON schema | |
| constrained decoding | 受限解碼 | 约束解码 | |
| grammar | 文法 | 文法 | |
| stop sequence | 停止序列 | 停止序列 | |
| logprobs | logprobs（對數機率） | logprobs（对数概率） | |
| sampling | 取樣 | 采样 | audio sample rate: 取樣率 / 采样率 |
| temperature | temperature（溫度） | temperature（温度） | field names stay Latin |
| prompt | 提示詞 | 提示词 | UI buttons use 提示 |
| system prompt | 系統提示詞 | 系统提示词 | |
| context | 上下文 | 上下文 | |
| conversation | 對話 | 对话 | |
| message | 訊息 | 消息 | |
| turn | 回合 | 回合 | |
| session | 工作階段 | 会话 | rare; in product text prefer 對話 / 对话 |
| embedding | 嵌入 / embedding | 嵌入 / embedding | multimodal embedding stays Latin |
| rerank | 重排序 | 重排序 | |
| vector | 向量 | 向量 | |
| image | 圖片 | 图片 | 影像 / 图像 for image generation and pixel data |
| vision / VLM | 視覺 / 視覺語言模型 | 视觉 / 视觉语言模型 | |
| audio | 音訊 | 音频 | |
| video | 影片 | 视频 | |
| speech | 語音 | 语音 | |
| transcription | 轉錄 | 转录 | transcript text: 逐字稿 |
| speech synthesis (TTS) | 語音合成 | 语音合成 | |
| realtime | 即時 | 实时 | |
| sandbox | 沙盒 | 沙箱 | zh-TW 沙盒 (2 hits); zh-CN 沙箱 preferred (沙盒 also 2 hits) |
| code execution | 程式碼執行 | 代码执行 | |
| web search | 網路搜尋 | 联网搜索 | |
| server tool | 伺服器工具 | 服务器工具 | |
| MCP client | MCP 用戶端 | MCP 客户端 | |
| authentication | 驗證 | 验证 | |
| authorization | 授權 | 授权 | |
| permission | 權限 | 权限 | |
| credential | 憑證 | 凭证 | |
| setting | 設定 | 设置 | |
| configuration | 設定 | 配置 | see rules 6 and 7 |
| default | 預設 | 默认 | 缺省 appears in zh-CN CLI tables; new text uses 默认 |
| option / flag | 選項 / 旗標 | 选项 / 标志 | CLI flags stay `--flag` |
| effective value | 有效設定 | 有效设置 | |
| environment variable | 環境變數 | 环境变量 | |
| log | 日誌 | 日志 | |
| record (verb) | 記錄 | 记录 | |
| metric | 指標 | 指标 | |
| status | 狀態 | 状态 | |
| error | 錯誤 | 错误 | |
| warning | 警告 | 警告 | |
| timeout | 逾時 | 超时 | |
| retry | 重試 | 重试 | |
| cancel | 取消 | 取消 | `abort` is 中止 |
| fallback | 後備 / 退回 | 回退 / 退回 | |
| supported | 支援 | 支持 | |
| unsupported | 不支援 | 不支持 | |
| compatible | 相容 | 兼容 | |
| upstream | 上游 | 上游 | |
| vendored | 內附（vendored） | 内置（vendored） | keep `vendored` in parentheses |
| track (upstream) | 追蹤 | 追踪 | |
| port (code) | 移植 | 移植 | |
| release | 發布 | 发布 | |
| release gate | 發版關卡 | 发布关卡 | |
| benchmark | 基準測試 | 基准测试 | |
| verification | 驗證 | 验证 | |
| regression | 退步 | 退步 | |
| contribute | 貢獻 | 贡献 | |
| branch | 分支 | 分支 | |
| commit | commit | commit | |
| pull request | pull request | pull request | |
| install | 安裝 | 安装 | |
| upgrade | 升級 | 升级 | |
| worker | 工作程序 | 工作进程 | |
| process | 行程 | 进程 | |
| thread | 執行緒 | 线程 | |
| namespace | 命名空間 | 命名空间 | |
| alias | 別名 | 别名 | |
| file | 檔案 | 文件 | |
| document | 文件 | 文档 | zh-CN 文件 is the dominant form for both file and document in old pages |
| upload | 上傳 | 上传 | |
| download | 下載 | 下载 | |
| path | 路徑 | 路径 | |
| address | 位址 | 地址 | URL: 網址 / 网址 |
| port | 連接埠 | 端口 | |
| bind | 綁定 | 绑定 | |
| local | 本機 | 本机 | |
| multimodal | 多模態 | 多模态 | |
