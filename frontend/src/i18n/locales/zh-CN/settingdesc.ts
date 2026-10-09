const settingdesc = {
  YUNSHU_MODEL:
    "单模型模式下提供服务的模型路径或 Hugging Face id；所有请求的模型名称都会映射到它。",
  YUNSHU_MULTI_MODEL:
    "多模型模式：发现 `YUNSHU_MODELS_DIR` 下的模型并按需加载；设置了 `YUNSHU_MODEL` 时忽略。",
  YUNSHU_MODELS_DIR:
    "多模型模式的模型文件夹目录（文件夹名称即模型 id）；未设置时为 `~/.yunshu/models`，也是 `yunshu pull` 的下载位置。",
  YUNSHU_HF_CACHE_MODELS:
    "多模型模式：同时提供 Hugging Face 缓存中已有的模型（名称冲突时以模型目录为准）；`yunshu serve -m org/name` 无论如何都会使用缓存副本。",
  YUNSHU_MODEL_TTL_SECONDS:
    "多模型模式：模型空闲达此秒数后卸载；未设置则永不卸载。",
  YUNSHU_ALLOW_AUTO_LOAD:
    "多模型模式：允许音频请求加载尚未加载的模型（否则会被拒绝）。",
  YUNSHU_MAX_LORAS: "同时保持加载的 LoRA 适配器数量上限（文本模型）。",
  YUNSHU_TRUST_REMOTE_CODE:
    "允许需要 `trust_remote_code` 的 Hugging Face 分词器和处理器。",
  YUNSHU_WARM_PROMPTS:
    "启动时预填充以预热前缀缓存的提示词：以 `||` 分隔的文本或文件路径。",
  YUNSHU_CONFIG: "包含 `YUNSHU_*` 设置的 TOML 配置文件（优先级低于环境变量）。",
  YUNSHU_MAX_CONCURRENT: "同时受理请求数的上限；未设置则自适应，起始值为 8。",
  YUNSHU_UNCACHED_SCHEDULING:
    "按预估未命中缓存的工作量和等待时间排序 VLM 预填充原子，保护交互式预填充，并在原子之间预留实测的解码时间，不改变任何 token 区间。仅在合格的批不变内核下默认开启，其他后端除非启用否则保持先进先出；停用可恢复上游排序。",
  YUNSHU_AUXILIARY_SCHEDULING:
    "降低已识别的 opencode 标题请求的优先级：交互式 VLM 任务先运行，辅助行在 GPU 时间片之间暂停，且不读写 APC。仅在合格的批不变内核下默认开启，其他后端需显式启用；进行中的 GPU 运算无法中断。",
  YUNSHU_QUEUE_LIMIT:
    "同时进行的生成请求数（运行中与等待中合计）；超出时立即以 429 拒绝，并在 `error.x_yunshu` 中附带 `Retry-After` 和队列深度。0 表示不限制。",
  YUNSHU_MEMORY_PRESSURE_REJECT:
    "Metal 工作集使用比例的阈值；超过时若仍有其他请求在运行，新的生成请求会以 503 和 `Retry-After` 拒绝，以避免内存不足。空闲的服务器不会拒绝。0 表示关闭。",
  YUNSHU_COMPLETION_BATCH_SIZE: "文本引擎：一起解码的序列数上限。",
  YUNSHU_DEFAULT_MAX_TOKENS:
    "聊天或 responses 请求省略 `max_tokens` 时的生成长度；仍受上下文预算限制。",
  YUNSHU_MAX_PREFILL_TOKENS:
    "拒绝超过此 token 数的提示词（0 表示除模型上下文外不设限）。",
  YUNSHU_STARTUP_TIMEOUT: "等待模型加载的秒数，超时则启动失败。",
  YUNSHU_DRAIN_TIMEOUT: "关闭时等待进行中请求完成的秒数。",
  YUNSHU_KEEP_ALIVE_TIMEOUT: "空闲 HTTP 连接保持打开的秒数。",
  YUNSHU_UDS:
    "改在此 Unix domain socket 上提供服务，而非 TCP 端口（同一个应用；可用 `curl --unix-socket`、httpx `uds=`）。",
  YUNSHU_WS_MAX_INFLIGHT:
    "文本 WebSocket（`/v1/stream`、wss `/v1/responses`）：每个连接的并发请求上限。",
  YUNSHU_WS_PING_INTERVAL:
    "文本 WebSocket：服务器心跳 ping 的间隔秒数（0 表示禁用）。",
  YUNSHU_WS_SEND_QUEUE:
    "文本 WebSocket：每个连接在暂停生成（背压）前可缓冲的出站事件数。",
  YUNSHU_MAX_REQUEST_SIZE: "请求体大小上限（字节）。",
  YUNSHU_PROGRESS_INTERVAL_S:
    "流式聊天／补全：首个 token 之前，`: yunshu-progress` SSE 注释（队列和预填充进度、预计时间）的间隔秒数；0 表示关闭。严格的 SSE 客户端会忽略注释行。",
  YUNSHU_SLOW_REQUEST_THRESHOLD: "请求耗时超过此秒数时记录警告。",
  YUNSHU_CORS_ORIGINS:
    "以逗号分隔的允许 CORS 来源（`*` 表示任意）；同时用于检查 Realtime WebSocket 的 Origin 头。",
  YUNSHU_RESPONSE_CACHE: "在内存中缓存相同的非流式响应。",
  YUNSHU_BATCH_MAX_ITEMS: "Batch API：每个批次的请求数上限。",
  YUNSHU_BATCH_TIMEOUT: "Batch API：每个批次的默认超时秒数。",
  YUNSHU_ALLOW_LOCAL_FILES:
    "允许请求引用任意本地文件路径（默认仅限 `YUNSHU_MEDIA_DIR` 之下）。",
  YUNSHU_MEDIA_DIR:
    "本地媒体路径必须位于的目录；未设置时为 `$TMPDIR/yunshu_media`。",
  YUNSHU_FILES_DIR:
    "本地 Files／Batch API 存储目录；未设置时为 `~/.yunshu/files`。",
  YUNSHU_FILES_MAX_BYTES:
    "Files API：单个上传文件的大小上限（字节，默认 512 MB）。",
  YUNSHU_FILES_TTL_DAYS:
    "Files API：上传的文件超过此天数后删除；未设置则永久保留。",
  YUNSHU_FILES_MAX_TOTAL_BYTES:
    "Files API：存储区可容纳的总字节数；超出的上传会以 413 `storage_quota_exceeded` 失败（会先清除过期文件）。0 表示不限。",
  YUNSHU_CONVERSATIONS_DIR:
    "Conversations API 存储目录（JSON，每个对话一个文件）；未设置时为 `~/.yunshu/conversations`。",
  YUNSHU_CHAT_COMPLETIONS_DIR:
    "已存储聊天补全的目录（`store=true`；JSON，每条补全一个文件）；未设置时为 `~/.yunshu/chat_completions`。",
  YUNSHU_CHAT_COMPLETIONS_MAX: "保留的已存储聊天补全数量；超过时淘汰最旧的。",
  YUNSHU_CONVERSATION_MAX_ITEMS:
    "Conversations API：单个对话可容纳的条目数上限。",
  YUNSHU_COMPACT_MAX_TOKENS: "Responses 压缩：模型所写摘要的 token 数上限。",
  YUNSHU_AUTH_TOKEN:
    "Bearer 令牌。设置后，除 health、version 和 docs 外的所有请求都需要它；未设置时推理开放，但运维类端点被拒绝。",
  YUNSHU_AUTH_DISABLED:
    "完全禁用认证（运维类端点也会开放），仅供本地开发使用。",
  YUNSHU_DEBUG_ROUTES:
    "挂载 `/debug/*` 诊断路由（engine、system、kv-cache、spec-decode 等），需要认证令牌或 `YUNSHU_AUTH_DISABLED`；`/metrics` 始终挂载。",
  YUNSHU_DEBUG_STREAM_CAPTURE:
    "调试辅助：每次 VLM 运行器生成都在此文件追加一行 JSON，包含生成的 token id、交给网关的文本片段及其拼接结果，以便比对送出的回答与实际生成内容；未设置则关闭。",
  YUNSHU_ACTOR_IDENTITY: "审计日志中为已认证请求记录的身份。",
  YUNSHU_RATE_LIMIT_RPM:
    "每个客户端的请求速率上限（每分钟请求数）；0（默认）表示关闭。本地单用户的引擎不需要，服务器对其他机器开放时再设置。",
  YUNSHU_TRUSTED_PROXIES:
    "以逗号分隔的代理 IP，其 `X-Forwarded-For` 头会被信任。",
  YUNSHU_FOOTPRINT_SAMPLE_MS:
    "在后台线程中每 N ms 采样本进程的 `phys_footprint`，并将峰值以 `yunshu_process_footprint_bytes` 导出到 `/metrics`；0（默认）表示关闭。",
  YUNSHU_MAX_MEMORY_GB:
    "多模型模式的内存上限（GiB），例如 `48` 或 `48GB`；`disabled` 会关闭强制机制。未设置时为统一内存的 80%。",
  YUNSHU_PREFILL_STEP_SIZE:
    "文本引擎：每次预填充前向传递处理的提示词 token 数；在内存较小的机器上调低可限制预填充激活值的峰值。",
  YUNSHU_MEM_PRESSURE_THRESHOLD:
    "文本引擎：内存使用超过此值（百分比，或不大于 1 的小数）时淘汰前缀缓存条目。",
  YUNSHU_PREFIX_MAX_ENTRIES: "文本引擎：前缀 KV 缓存的条目数。",
  YUNSHU_PREFIX_HOT_LIMIT:
    "文本引擎：仅让这么多条前缀 KV 条目保持完整精度，较旧的在内存中以 4-bit 存储（复用时有损，以质量换内存）。0 表示全部完整精度。",
  YUNSHU_SSD_CACHE: "文本引擎：将前缀 KV 持久化到 SSD。",
  YUNSHU_SSD_CACHE_DIR: "文本引擎：SSD 前缀缓存目录。",
  YUNSHU_SSD_CACHE_PRECISION:
    "文本引擎：SSD 前缀缓存的存储精度：`native`（KV 和循环状态按位精确，无损）或 `int8`（逐张量 int8，磁盘占用约为 bf16 的一半，复用时有损）。",
  YUNSHU_SSD_CACHE_PREFILL_CEIL_TPS:
    "文本引擎：实测预填充速度超过此值（tok/s）时跳过 SSD 前缀恢复，因为此时重新预填充与读回 KV 一样快。",
  YUNSHU_SSD_CACHE_MAX_GB:
    "文本引擎：SSD 前缀缓存的大小上限（GiB），整个目录（所有模型合计）共用一份预算。",
  YUNSHU_CACHE_RESERVE_PCT:
    "SSD 前缀缓存（APC 和文本）：缓存写入不得占用的剩余空间，以卷百分比计。保留量取此值与 `YUNSHU_CACHE_RESERVE_GB` 中较大者，也会限制缓存根目录的实际上限。",
  YUNSHU_CACHE_RESERVE_GB:
    "SSD 前缀缓存（APC 和文本）：卷上至少保留的剩余空间（GiB，见 `YUNSHU_CACHE_RESERVE_PCT`）。会使剩余空间低于此值的写入将被丢弃，溢写暂停直到空间恢复。",
  YUNSHU_CACHE_STALE_DAYS:
    "SSD 前缀缓存（APC 和文本）：检查点命名空间超过此天数未使用，或其检查点已不存在或已变更，会在淘汰其他内容之前先被移除（0 表示不按时间清除）。",
  YUNSHU_KV_QUANT_BITS:
    "文本引擎 KV 缓存量化（有损，以质量换内存）：`off`（无损）、`auto`（KV 缓存将超过约 2 GiB 时改用 8-bit），或固定 2/3/4/8 bit。",
  YUNSHU_VLM_APC_MEMORY_GB:
    "VLM 运行器前缀缓存（APC）的内存预算（GiB）；0 表示禁用。未设置时取扣除权重与系统／激活值保留后剩余内存的一半，上限为机器的四分之一和 32 GiB，剩余不足 1 GiB 时关闭。27B 检查点每个缓存 token 约需 130 KiB。",
  YUNSHU_VLM_APC_DISK:
    "APC 的 SSD 层：被内存淘汰的前缀检查点（以及关闭时仍在内存中的）会写入磁盘，之后读回而非重新预填充（按位精确、无损；27B 检查点加载比预填充快约 20 倍）。设为 0 则前缀缓存只放内存。",
  YUNSHU_VLM_APC_DISK_DIR:
    "APC SSD 层的目录；未设置时为 `~/.yunshu/cache/apc`（内置磁盘）。可放到高速卷上以保持内置磁盘干净。",
  YUNSHU_KV_PRECISION:
    "Qwen3.5 系列运行器共享解码批的 KV 缓存精度：`bf16`（无损）或 `int8`（KV 内存和读取带宽约为 0.53 倍，注意力有小幅误差，以质量换内存）。单个请求和推测解码通道保持 bf16。",
  YUNSHU_VLM_APC_DISK_GB:
    "APC SSD 层的大小上限（GiB），整个目录共用一份预算（跨命名空间先淘汰最久未使用的文件；0 表示不设上限，仍受剩余空间保留限制）。未设置时为卷的四分之一，最多 64 GiB。27B 检查点每个 token 约 130 KiB，因此 64 GiB 约可容纳 500K token。",
  YUNSHU_PREFILL_MATMUL:
    "M5 级 GPU 上 Qwen3.5 系列 VLM 运行器：超过 512 行的预填充块所用的矩阵乘法。`stock` 对未分块的权重运行 MLX 量化矩阵乘法（27B 冷预填充约快 25%，块的比特结果取决于行数）；`lane` 对所有行数都使用行不变的 lane 内核。属于每个 APC 键和 SSD 命名空间的一部分。",
  YUNSHU_PREFILL_BUFFER_CACHE_GB:
    "VLM 运行器：在预填充步骤之间保留的 MLX 已释放缓冲缓存大小（GiB）。未设置时为物理内存的 5%，最多 6 GiB。可让长前缀恢复的缓冲在请求之间保持热状态；0 表示沿用上游每个块之后清除。仅影响分配器，输出不变。",
  YUNSHU_PREFILL_GDN:
    "Qwen3.5 系列 VLM 运行器：64 个 token 以上预填充块的 GatedDeltaNet 内核。`chunked` 使用 MLX 的 `mx.fast.gated_delta_update`（每层快 2.7 倍，与 `step` 相差在 bf16 舍入内）；`step` 保持 mlx-vlm 的逐 token 内核。属于每个 APC 键和 SSD 命名空间的一部分。",
  YUNSHU_VLM_APC_DISK_TIERS:
    "SSD 目录之下的其他 APC 存储层，以逗号分隔的 `PATH[@GiB]`（外接 SSD、HDD、NAS）。启动时会测量各卷并按读取速度排序；被淘汰的检查点会下移而非删除，且仅在恢复比重新预填充更快时才由该层服务命中。未挂载的层会跳过。无损；未设置则只用 SSD。",
  YUNSHU_VLM_APC_DISK_ENCODING:
    "较低层 APC 存储（`YUNSHU_VLM_APC_DISK_TIERS`）保存检查点的方式：`raw`（SSD 文件的副本）、`zstd`（无损的字节平面重排加 zstd），或 `auto`（仅在实测能提升有效读取带宽时使用 zstd，例如慢速磁盘和网络共享）。绝不有损。",
  YUNSHU_VLM_APC_WARM:
    "APC WARM 层：前缀检查点离开内存（HOT）层、进入 SSD 之前的处理方式。`off`：直接写入 SSD；`lossless`：压缩后留在内存（按位精确，消耗 CPU）；`int8`／`int4`：注意力 K/V 以量化码保存（有损；SSD 层仍保留精确状态）。WARM 层占用 APC 内存预算中 `YUNSHU_VLM_APC_WARM_SHARE` 的比例。",
  YUNSHU_VLM_APC_WARM_SHARE:
    "启用 `YUNSHU_VLM_APC_WARM` 时，WARM 层占 APC 内存预算（`YUNSHU_VLM_APC_MEMORY_GB`）的比例，其余由 HOT 层使用；APC 内存总量不会增加。",
  YUNSHU_VLM_MAX_IMAGE_BYTES: "请求可通过 URL 引用的图片大小上限（字节）。",
  YUNSHU_VLM_INSECURE_SSL: "TLS 验证失败时，改用不验证的方式重试图片下载。",
  YUNSHU_MTP:
    "Qwen3.5 系列 VLM：使用检查点的 MTP 头进行草稿（批不变；开启与关闭推测解码结果相同）。",
  YUNSHU_VLM_DRAFT:
    "Qwen3.5 系列 VLM：推测解码草稿覆盖。可填 DFlash 草稿模型目录；`mtp` 强制使用检查点的 MTP 头；`off` 禁用草稿。未设置时，若模型目录或 Hugging Face 缓存中有匹配的 DFlash2 草稿模型则自动使用，否则使用 MTP 头。",
  YUNSHU_MTP_BLOCK_SIZE:
    "草稿块大小（对 DFlash 而言是其按接受率调整的深度上限）。未设置时，MTP 为 6，DFlash 为草稿模型训练时的块大小。",
  YUNSHU_SPEC_COPY_ROWS:
    "Qwen3.5 系列单请求推测解码通道：提示词复制回合可使用的验证行数（复制草稿数 = 行数 - 1）。复制回合会在提示词和已生成文本中找出当前尾部最长的先前出现处，提出其后续内容，并由同一个验证步骤检查，因此输出不变。引用上下文的代理、代码编辑和多轮流量，每回合可提交数倍的 token。默认 16 行，并受后端认证宽度限制；0 表示关闭复制回合。",
  YUNSHU_DRAFT_BITS:
    "Qwen3.5 系列 DFlash 草稿模型的权重位数：8（默认）、4，或 0 表示保留原出厂的 bf16。草稿由目标模型验证，因此任何值的输出 token 都相同；位数越少，草稿模型每回合读取的字节越少，但接受率可能下降。",
  YUNSHU_SPEC_TREE:
    "Qwen3.5 系列单请求推测解码通道：`off` 保持训练好的链式加复制；`tree` 强制使用草稿树验证器；`auto` 对有界的 1K 级贪婪请求使用已认证的 M5 Q4 DFlash2 快速树，其余保持链式加复制。贪婪解码的 token 与普通解码相同。",
  YUNSHU_NGRAM_DEFAULT:
    "文本模型：默认对贪婪请求启用无损的 n-gram 推测解码（也可由单次请求的 `spec_decode` 启用）。在重复性高的输出上有优势。",
  YUNSHU_SPEC_PROPOSER: "文本模型：n-gram 推测解码所用的提案器类型。",
  YUNSHU_GEMMA4_ASSISTANT:
    "文本 Gemma-4 模型：辅助草稿模型目录（共享 KV 的推测解码草稿模型）。",
  YUNSHU_GPU_SAMPLER:
    "文本模型：在 GPU 上进行 Gumbel-max 采样（每个 token 无需 GPU 到 CPU 的同步）。",
  YUNSHU_JUMP_FORWARD:
    "文本模型：JSON schema 输出中由语法强制决定的结构性 token 直接输出，不经过前向传递。",
  YUNSHU_JSON_SCHEMA_ENGINE:
    "负责 JSON schema 和 `json_object` 约束解码掩码的引擎：`llguidance`（248K 词表上每 token 掩码中位数约 0.25 ms；属性按 schema 顺序）或 `inhouse`（Python 状态机，中位数约 3 ms；属性顺序不限）。llguidance 无法编译的 schema，在内置引擎支持时会回退到内置引擎。",
  YUNSHU_GRAMMAR_BITMASK:
    "约束解码改用 xgrammar 风格的位掩码引擎，而非允许列表采样器。",
  YUNSHU_TOOL_GRAMMAR:
    "工具调用约束解码（结构标签）：在工具调用起始标记之前为自由文本，之后调用体会被掩码为本次请求精确的调用语法（工具名称、参数键、带类型的值、结尾）。强制的 `tool_choice` 始终受约束；此开关仅控制 `auto`。关闭时，自动选择的工具调用不受约束地解码，事后再修复。",
  YUNSHU_QUANT_MODE:
    "加载时在内存中量化权重（有损，以质量换内存）：`mxfp4`、`nvfp4`、`mxfp8` 或 `affine`（留空则保持检查点原样）。",
  YUNSHU_QUANT_CONFIG:
    "`affine` 内存内量化的位数和分组大小：JSON（含 `bits` 和 `group_size` 键）或 `bits`／`bits,group` 格式。",
  YUNSHU_ROUND_PREFILL_CHUNK:
    "回合驱动器：每个预填充区间的提示词 token 数。解码中的请求只在预填充前向传递之间推进，因此区间越小，越能在长提示词旁保持运行，代价是预填充速度约降低 20%。区间按提示词固定，输出不受其他运行中任务影响；以不同大小预填充的提示词各自一致，但彼此并非按位相同。",
  YUNSHU_ROUND_DRIVER:
    "稠密 Qwen3.5 系列 VLM：以Yunshu的回合驱动器处理文本请求（对所有解码行做打包前向传递，与批量预填充步骤交替；每行都有 MTP 草稿；按位置键采样；见 `docs/guides/ROUND_DRIVER.md`）。关闭时使用上游 BatchGenerator 加单请求推测解码通道。图片提示词、int8 KV 和 MoE 无论如何都走上游路径。",
  YUNSHU_MTP_ROW_EXACT:
    "Qwen3.5 系列运行器：使用 oMLX 逐行精确验证（验证行与单行解码按位相同），取代批不变内核。",
  YUNSHU_ENGINE_LOOP:
    "文本模型：使用 EngineCore 连续批处理循环，取代单请求快速路径。",
  YUNSHU_OVERLAP:
    "文本引擎循环：让 CPU 与 GPU 工作重叠（`cpu_gpu`），或将批拆成两个相互重叠的一半（`two_batch`）。",
  YUNSHU_SPEC_UNVERIFIED:
    "文本模型：显式启用未经验证的外部 mlx-lm 草稿实验。`eagle` 是沿用的路由名称，普通语言模型草稿即可，不需要训练过的 EAGLE 头。仅适用于使用默认采样和惩罚的贪婪、非流式请求；schema、自定义处理器、停止字符串、思考预算、LoRA 和 token 掩码选项仍走常规快速路径。",
  YUNSHU_DRAFT_MODEL:
    "供 `YUNSHU_SPEC_UNVERIFIED=eagle` 使用的外部 mlx-lm 草稿检查点（普通语言模型草稿，而非训练过的 EAGLE 头）。",
  YUNSHU_REALTIME_OMNI:
    "Realtime 连接上的原生 Qwen3-Omni 语音：`auto`（提供可说话的模型时启用）、`on` 或 `off`（改用 ASR、LLM、TTS 级联流程）。",
  YUNSHU_OMNI_MODEL:
    "语音路径所用的模型（与所服务模型不同时设置；omni 模型作为服务模型时会自动复用）。",
  YUNSHU_OMNI_PRELOAD: "启动时预热 omni 模型，让第一个语音请求不必冷启动。",
  YUNSHU_OMNI_THINKER_MAX:
    "omni Thinker 每个语音回合最多写出的 token 数（口语回复的长度上限）。",
  YUNSHU_OMNI_PERSONA:
    "请求未提供时使用的 Realtime 系统角色设定；空字符串表示禁用。未设置时使用内置的简洁口语风格。",
  YUNSHU_REALTIME_SILENCE_MS: "服务端 VAD：模型回答前的停顿时间（ms）。",
  YUNSHU_REALTIME_BARGE_IN_MS: "在模型回复途中打断它所需的持续说话时间（ms）。",
  YUNSHU_REALTIME_VAD_THRESHOLD: "服务端 VAD 的语音检测阈值。",
  YUNSHU_REALTIME_PREFIX_PADDING_MS:
    "服务端 VAD：检测到语音之前保留的音频长度（ms）。",
  YUNSHU_REALTIME_VAD: "服务端 VAD 实现：`energy` 或 `silero`。",
  YUNSHU_REALTIME_VAD_MODEL:
    "`YUNSHU_REALTIME_VAD=silero` 时使用的 Silero VAD 模型 id。",
  YUNSHU_REALTIME_MAX_INPUT_AUDIO_BYTES:
    "Realtime：可缓冲的输入音频大小上限（字节）。",
  YUNSHU_REALTIME_MAX_CONVERSATION_ITEMS:
    "Realtime：每个会话保留的对话条目数。",
  YUNSHU_DIFFUSION_SCHEDULER: "图像生成采样器覆盖（留空则使用管线自带的）。",
  YUNSHU_ANE_EMBEDDINGS:
    "可用时通过 CoreML 在 Apple Neural Engine 上计算嵌入。",
  YUNSHU_ANE_EMBEDDING_MODEL: "ANE 路径所使用的嵌入模型。",
  YUNSHU_MCP_CONFIG: "列出工具服务器的 MCP 客户端配置文件（JSON／YAML）。",
  YUNSHU_MCP_SERVERS:
    "以 JSON 数组指定的 MCP 工具服务器（`YUNSHU_MCP_CONFIG` 的替代方式）。",
  YUNSHU_WEB_SEARCH_PROVIDER:
    "服务端 `web_search` 工具的搜索后端。`auto` 依次选用第一个已配置的 searxng、brave、tavily、exa；`none` 表示禁用。未配置时，请求会收到 API 的 `unavailable` 错误并附带提示。",
  YUNSHU_SEARXNG_URL:
    "自行部署的 SearXNG 实例的基础 URL（需启用 JSON 输出），例如 `http://127.0.0.1:8080`；是推荐的注重隐私的默认选择。",
  YUNSHU_BRAVE_API_KEY: "Brave Search API 密钥。",
  YUNSHU_TAVILY_API_KEY: "Tavily API 密钥。",
  YUNSHU_EXA_API_KEY: "Exa API 密钥。",
  YUNSHU_WEB_SEARCH_RESULTS: "每次 `web_search` 调用返回的结果数。",
  YUNSHU_WEB_FETCH:
    "提供服务端 `web_fetch` 工具（不需要搜索提供方）。关闭时，`web_fetch` 请求会收到 `unavailable` 错误。",
  YUNSHU_WEB_FETCH_ALLOW_PRIVATE:
    "允许 `web_fetch` 访问私有、回环和链路本地地址。关闭（默认）时会拦截这些地址，包括重定向和 DNS 解析之后（SSRF 防护）。",
  YUNSHU_WEB_FETCH_MAX_BYTES: "`web_fetch` 下载的响应体大小上限。",
  YUNSHU_WEB_FETCH_TIMEOUT: "`web_fetch` 等待网页的秒数。",
  YUNSHU_WEB_FETCH_MAX_TEXT_CHARS:
    "交给模型的提取网页文本会截断至此字符数（请求的 `max_content_tokens` 可进一步调低）。",
  YUNSHU_MCP_CONNECTOR:
    "提供 MCP 连接器：Anthropic 的 `mcp_servers` 和 OpenAI Responses 的 `mcp` 工具由本服务器执行，并通过可流式 HTTP／SSE 连接到指定的 MCP 服务器。",
  YUNSHU_MCP_CONNECTOR_ALLOW_PRIVATE:
    "允许 MCP 连接器访问私有和回环的 MCP 服务器（本地工具服务器是常见情况）。关闭时仅限公网地址。",
  YUNSHU_MCP_CONNECTOR_TIMEOUT:
    "MCP 连接器单次调用（initialize、`tools/list`、`tools/call`）可耗用的秒数，含 DNS 解析。",
  YUNSHU_MCP_CONNECTOR_MAX_BYTES:
    "MCP 连接器服务器单次回复（JSON 体或单个 SSE 事件）解压缩后的大小上限。",
  YUNSHU_SERVER_TOOL_MAX_ITERATIONS:
    "单个请求最多可进行的“生成、运行工具、继续”回合数。",
  YUNSHU_MODEL_ALIASES:
    "多模型模式：将代理请求的模型名称（`claude-sonnet-4-5`、`opus`、`gpt-5`）映射到实际提供的模型，格式为“模式到模型 id”的 JSON 对象；模式可为完整名称、`prefix*` 或 `*`（先匹配者优先；真实模型名称始终优先）。单模型模式本来就会响应所有名称。",
  YUNSHU_LOG_LEVEL: "Yunshu日志记录器的日志级别（第三方记录器保持 WARNING）。",
  YUNSHU_AUDIT_LOG_FILE: "同时将审计日志写入此文件。",
  YUNSHU_LOG_MAX_MB:
    "服务日志（launchd）：日志文件达到此大小（MiB）时轮转；0 表示关闭按大小轮转。",
  YUNSHU_LOG_ROTATE_HOURS:
    "服务日志：距上次轮转超过此小时数时也进行轮转；0 表示关闭按时间轮转。",
  YUNSHU_LOG_KEEP: "服务日志：保留的已轮转文件数（gzip 压缩，机密已脱敏）。",
  YUNSHU_LOG_RETENTION_DAYS:
    "服务日志：删除超过此天数的已轮转文件；0 表示保留到 `YUNSHU_LOG_KEEP` 将其清除为止。",
  YUNSHU_SERVE_LOG:
    "每完成一次生成请求，就将一行仅含数字的 JSON（耗时、token 数、推测解码接受率、缓存层、并发度、arm、build）写入本地有大小上限并会轮转的文件。绝不包含提示词、输出或 token id。默认关闭，数据不会离开本机。",
  YUNSHU_SERVE_LOG_DIR:
    "服务记录（serve log）的目录；未设置时为 `~/.yunshu/logs`。",
  YUNSHU_SERVE_LOG_MAX_MB:
    "服务记录：达到此大小（MiB）时轮转；配合 `YUNSHU_SERVE_LOG_KEEP`，目录大小上限为 max * (keep + 1)。",
  YUNSHU_SERVE_LOG_KEEP: "服务记录：保留的已轮转文件数。",
  YUNSHU_ARM:
    "在服务记录中记录本服务器所运行配置组（arm）的标签，用于离线 A/B 分析；不会改变任何行为。",
  YUNSHU_GATEWAY_URL: "`yunshu` CLI 客户端命令所使用的服务器 URL。",
  YUNSHU_HF_ENDPOINT:
    "`yunshu serve` 使用的 Hugging Face Hub 端点（会导出为 `HF_ENDPOINT`）。",
  YUNSHU_HISTORY_INTERVAL_S:
    "控制台历史：内存中历史环的取样间隔（秒），数据包括吞吐、请求数、内存、TTFT 百分位，供 GET /v1/yunshu/history 使用；0 表示关闭取样。环的大小固定，不会增长。",
  YUNSHU_HISTORY_HOURS:
    "控制台历史：历史环保留的小时数（容量 = 小时数 × 3600 ÷ 取样间隔，启动时一次分配；5 秒间隔下 12 小时约 0.4 MiB）。",
  YUNSHU_EVALS_DIR:
    "Evals API 本地 JSON 存储的文件夹；未设置时为 `~/.yunshu/evals`。",
  YUNSHU_VLM_MAX_VIDEO_BYTES: "请求通过网址引用的视频大小上限（字节）。",
  YUNSHU_WEB_SEARCH_PROVIDER_TIMEOUT:
    "聚合搜索每个来源的期限（秒）；慢的来源不会拖住整个查询。",
  YUNSHU_WEB_SEARCH_HEALTH_FILE:
    "搜索来源健康状态的小型快照文件（不含查询、搜索结果或凭证），供 `yunshu config` 读取。",
  YUNSHU_WEB_MWMBL:
    "在自动聚合搜索中加入 Mwmbl 开放小型网站索引（数据为 CC-BY-NC-SA 4.0，限非商业）；明确指定 provider=mwmbl 也算同意。",
  YUNSHU_WEB_KEYLESS:
    "允许不需密钥的 DuckDuckGo（尽力而为，可能被封锁）与 Wikipedia 搜索；查询文字与 IP 会离开这台机器。",
  YUNSHU_MOJEEK_API_KEY:
    "Mojeek 独立索引的 API 密钥；设置后会加入自动聚合搜索。",
  YUNSHU_MARGINALIA_API_KEY:
    "明确选用 Marginalia 小型网站搜索；公开 API 数据为 CC-BY-NC-SA 4.0（非商业），商业密钥另有条款，没有默认的公用密钥。",
  YUNSHU_SERPER_API_KEY: "Serper（Google 搜索结果）的 API 密钥。",
  YUNSHU_PERPLEXITY_API_KEY:
    "Perplexity Search 的 API 密钥（原始搜索结果，不是 Sonar）。",
  YUNSHU_WEB_RESEARCH:
    "用来源页面、不受信任的摘录与本地排序补强搜索摘要；稳定的可选功能，质量评估前不默认开启；抓取的网址会离开这台机器。",
  YUNSHU_WEB_RENDER:
    "高级 Tavily 抓取／爬取／地图在遇到只有 JavaScript 外壳的页面时，改用本地 Chromium 后备（需 web-render 套件与已安装的 Playwright Chromium；只发同源 GET，不带 cookie，不会自动下载浏览器）。",
  YUNSHU_WEB_RESEARCH_BUDGET: "补强搜索摘要的整体期限（秒，最多 4）。",
  YUNSHU_WEB_RESEARCH_PAGES: "每次补强最多取用的来源页面数（上限 6）。",
  YUNSHU_WEB_RESEARCH_MODEL:
    "已加载的本地嵌入模型 id；不会自动加载模型，没有或不可用时只用 BM25。建议 Qwen3-Embedding-0.6B。",
  YUNSHU_TELEMETRY:
    "免权限的 Apple 功耗、GPU 与温度采样器（`on` 或 `off`）；需重启。",
  YUNSHU_TELEMETRY_INTERVAL_S: "主机遥测采样间隔（秒）；需重启。",
  YUNSHU_SERVE_LOG_RETENTION_DAYS:
    "历史 API 元数据的保留天数；0 表示不按时间过滤。",
  YUNSHU_CONSOLE:
    "`yunshu serve` 同时以并行进程启动主控台进程（网页主控台、文档、历史记录器）。它不加载 MLX，引擎重启或崩溃时仍持续运行。`--no-console` 可关闭。",
  YUNSHU_CONSOLE_PORT:
    "主控台进程的端口。8100 避开引擎的 8000 与常见的开发服务器。",
  YUNSHU_CONSOLE_HOST:
    "主控台进程的绑定主机。未设置时：并行启动沿用引擎的主机，单独运行为 127.0.0.1。",
  YUNSHU_CONSOLE_ENGINE:
    "主控台进程监看并转发请求的引擎地址。未设置时：启动它的引擎，否则为 http://127.0.0.1:8000。",
  YUNSHU_CONSOLE_ENGINE_TOKEN:
    "主控台进程记录历史时读取引擎所用的 Bearer 令牌。未设置时使用 YUNSHU_AUTH_TOKEN。",
  YUNSHU_CONSOLE_POLL_S:
    "主控台进程读取引擎的间隔（秒）；1 秒对应 1 秒的历史分辨率。",
  YUNSHU_CONSOLE_HISTORY:
    "把指标历史与请求记录（仅元数据）写入 ~/.yunshu/console-history.sqlite，引擎离线期间保留为缺口与事件。",
  YUNSHU_CONSOLE_RETENTION_DAYS:
    "主控台保留 1 分钟历史、请求记录与引擎事件的天数。",
  YUNSHU_CONSOLE_DB_MAX_MB:
    "主控台历史文件的大小上限（MiB）；超过时删除最旧的数据。",
};
export default settingdesc;
