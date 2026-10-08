const settingdesc = {
  YUNSHU_MODEL:
    "單一模型模式下提供服務的模型路徑或 Hugging Face id；所有請求的模型名稱都會對應到它。",
  YUNSHU_MULTI_MODEL:
    "多模型模式：探索 `YUNSHU_MODELS_DIR` 下的模型並依需求載入；設定 `YUNSHU_MODEL` 時忽略。",
  YUNSHU_MODELS_DIR:
    "多模型模式的模型資料夾目錄（資料夾名稱即模型 id）；未設定時為 `~/.yunshu/models`，也是 `yunshu pull` 的下載位置。",
  YUNSHU_HF_CACHE_MODELS:
    "多模型模式：同時提供 Hugging Face 快取中已有的模型（名稱衝突時以模型目錄為準）；`yunshu serve -m org/name` 無論如何都會使用快取副本。",
  YUNSHU_MODEL_TTL_SECONDS:
    "多模型模式：模型閒置達此秒數後卸載；未設定則永不卸載。",
  YUNSHU_ALLOW_AUTO_LOAD:
    "多模型模式：允許音訊請求載入尚未載入的模型（否則會被拒絕）。",
  YUNSHU_MAX_LORAS: "同時保持載入的 LoRA 適配器數量上限（文字模型）。",
  YUNSHU_TRUST_REMOTE_CODE:
    "允許需要 `trust_remote_code` 的 Hugging Face 分詞器與處理器。",
  YUNSHU_WARM_PROMPTS:
    "啟動時預填以暖機前綴快取的提示詞：以 `||` 分隔的文字或檔案路徑。",
  YUNSHU_CONFIG: "含 `YUNSHU_*` 設定的 TOML 設定檔（優先順序低於環境變數）。",
  YUNSHU_MAX_CONCURRENT: "同時受理請求數的上限；未設定則自適應，起始值為 8。",
  YUNSHU_UNCACHED_SCHEDULING:
    "依預估未命中快取的工作量與等待時間排序 VLM 預填原子，保護互動式預填，並在原子之間預留實測的解碼時間，不改變任何 token 區段。僅在合格的批次不變核心下預設開啟，其他後端除非啟用否則維持先進先出；停用可還原上游排序。",
  YUNSHU_AUXILIARY_SCHEDULING:
    "降低已識別的 opencode 標題請求的優先順序：互動式 VLM 工作先執行，輔助列在 GPU 時間片之間暫停，且不讀寫 APC。僅在合格的批次不變核心下預設開啟，其他後端需明確選用；進行中的 GPU 運算無法中斷。",
  YUNSHU_QUEUE_LIMIT:
    "同時進行的生成請求數（執行中與等待中合計）；超出時立即以 429 拒絕，並在 `error.x_yunshu` 附上 `Retry-After` 與佇列深度。0 表示不限制。",
  YUNSHU_MEMORY_PRESSURE_REJECT:
    "Metal 工作集使用比例的門檻；超過時若仍有其他請求執行，新的生成請求會以 503 與 `Retry-After` 拒絕，以避免記憶體不足。閒置的伺服器不會拒絕。0 表示關閉。",
  YUNSHU_COMPLETION_BATCH_SIZE: "文字引擎：一起解碼的序列數上限。",
  YUNSHU_DEFAULT_MAX_TOKENS:
    "聊天或 responses 請求省略 `max_tokens` 時的生成長度；仍受上下文預算限制。",
  YUNSHU_MAX_PREFILL_TOKENS:
    "拒絕超過此 token 數的提示詞（0 表示除模型上下文外不設限）。",
  YUNSHU_STARTUP_TIMEOUT: "等待模型載入的秒數，逾時則啟動失敗。",
  YUNSHU_DRAIN_TIMEOUT: "關閉時等待進行中請求完成的秒數。",
  YUNSHU_KEEP_ALIVE_TIMEOUT: "閒置 HTTP 連線保持開啟的秒數。",
  YUNSHU_UDS:
    "改在此 Unix domain socket 上提供服務，而非 TCP 連接埠（同一個應用；可用 `curl --unix-socket`、httpx `uds=`）。",
  YUNSHU_WS_MAX_INFLIGHT:
    "文字 WebSocket（`/v1/stream`、wss `/v1/responses`）：每個連線的並行請求上限。",
  YUNSHU_WS_PING_INTERVAL:
    "文字 WebSocket：伺服器心跳 ping 的間隔秒數（0 表示停用）。",
  YUNSHU_WS_SEND_QUEUE:
    "文字 WebSocket：每個連線在暫停生成（背壓）前可緩衝的外送事件數。",
  YUNSHU_MAX_REQUEST_SIZE: "請求主體大小上限（位元組）。",
  YUNSHU_PROGRESS_INTERVAL_S:
    "串流聊天／補全：首個 token 之前，`: yunshu-progress` SSE 註解（佇列與預填進度、預估時間）的間隔秒數；0 表示關閉。嚴格的 SSE 用戶端會忽略註解行。",
  YUNSHU_SLOW_REQUEST_THRESHOLD: "請求耗時超過此秒數時記錄警告。",
  YUNSHU_CORS_ORIGINS:
    "以逗號分隔的允許 CORS 來源（`*` 表示任意）；同時用於檢查 Realtime WebSocket 的 Origin 標頭。",
  YUNSHU_RESPONSE_CACHE: "在記憶體中快取相同的非串流回應。",
  YUNSHU_BATCH_MAX_ITEMS: "Batch API：每個批次的請求數上限。",
  YUNSHU_BATCH_TIMEOUT: "Batch API：每個批次的預設逾時秒數。",
  YUNSHU_ALLOW_LOCAL_FILES:
    "允許請求引用任意本機檔案路徑（預設僅限 `YUNSHU_MEDIA_DIR` 之下）。",
  YUNSHU_MEDIA_DIR:
    "本機媒體路徑必須位於的目錄；未設定時為 `$TMPDIR/yunshu_media`。",
  YUNSHU_FILES_DIR:
    "本機 Files／Batch API 儲存目錄；未設定時為 `~/.yunshu/files`。",
  YUNSHU_FILES_MAX_BYTES:
    "Files API：單一上傳檔案的大小上限（位元組，預設 512 MB）。",
  YUNSHU_FILES_TTL_DAYS:
    "Files API：上傳的檔案超過此天數後刪除；未設定則永久保留。",
  YUNSHU_FILES_MAX_TOTAL_BYTES:
    "Files API：儲存區可容納的總位元組數；超出的上傳會以 413 `storage_quota_exceeded` 失敗（會先清除過期檔案）。0 表示不限。",
  YUNSHU_CONVERSATIONS_DIR:
    "Conversations API 儲存目錄（JSON，每個對話一個檔案）；未設定時為 `~/.yunshu/conversations`。",
  YUNSHU_CHAT_COMPLETIONS_DIR:
    "已儲存聊天補全的目錄（`store=true`；JSON，每筆補全一個檔案）；未設定時為 `~/.yunshu/chat_completions`。",
  YUNSHU_CHAT_COMPLETIONS_MAX: "保留的已儲存聊天補全數量；超過時淘汰最舊的。",
  YUNSHU_CONVERSATION_MAX_ITEMS:
    "Conversations API：單一對話可容納的項目數上限。",
  YUNSHU_COMPACT_MAX_TOKENS: "Responses 壓縮：模型所寫摘要的 token 數上限。",
  YUNSHU_AUTH_TOKEN:
    "Bearer 權杖。設定後，除 health、version 與 docs 外的所有請求都需要它；未設定時推論開放，但管理類端點被拒絕。",
  YUNSHU_AUTH_DISABLED:
    "完全停用驗證（管理類端點也會開放），僅供本機開發使用。",
  YUNSHU_DEBUG_ROUTES:
    "掛載 `/debug/*` 診斷路由（engine、system、kv-cache、spec-decode 等），需要驗證權杖或 `YUNSHU_AUTH_DISABLED`；`/metrics` 一律掛載。",
  YUNSHU_DEBUG_STREAM_CAPTURE:
    "除錯輔助：每次 VLM 執行器生成都在此檔案追加一行 JSON，內含生成的 token id、交給閘道的文字片段及其串接結果，以便比對送出的回答與實際生成內容；未設定則關閉。",
  YUNSHU_ACTOR_IDENTITY: "稽核日誌中為已驗證請求記錄的身分。",
  YUNSHU_RATE_LIMIT_RPM:
    "每個用戶端的請求速率上限（每分鐘請求數）；0（預設）表示關閉。本機單人使用的引擎不需要，伺服器對其他機器開放時再設定。",
  YUNSHU_TRUSTED_PROXIES:
    "以逗號分隔的代理 IP，其 `X-Forwarded-For` 標頭會被信任。",
  YUNSHU_FOOTPRINT_SAMPLE_MS:
    "在背景執行緒中每 N ms 取樣本行程的 `phys_footprint`，並將峰值以 `yunshu_process_footprint_bytes` 匯出到 `/metrics`；0（預設）表示關閉。",
  YUNSHU_MAX_MEMORY_GB:
    "多模型模式的記憶體上限（GiB），例如 `48` 或 `48GB`；`disabled` 會關閉強制機制。未設定時為統一記憶體的 80%。",
  YUNSHU_PREFILL_STEP_SIZE:
    "文字引擎：每次預填前向傳遞處理的提示詞 token 數；在記憶體較小的機器上調低可限制預填啟動值的峰值。",
  YUNSHU_MEM_PRESSURE_THRESHOLD:
    "文字引擎：記憶體使用超過此值（百分比，或不大於 1 的小數）時淘汰前綴快取項目。",
  YUNSHU_PREFIX_MAX_ENTRIES: "文字引擎：前綴 KV 快取的項目數。",
  YUNSHU_PREFIX_HOT_LIMIT:
    "文字引擎：僅讓這麼多筆前綴 KV 項目保持完整精度，較舊的在記憶體中以 4-bit 儲存（重用時有損，以品質換記憶體）。0 表示全部完整精度。",
  YUNSHU_SSD_CACHE: "文字引擎：將前綴 KV 持久化到 SSD。",
  YUNSHU_SSD_CACHE_DIR: "文字引擎：SSD 前綴快取目錄。",
  YUNSHU_SSD_CACHE_PRECISION:
    "文字引擎：SSD 前綴快取的儲存精度：`native`（KV 與遞迴狀態位元精確，無損）或 `int8`（逐張量 int8，磁碟用量約為 bf16 的一半，重用時有損）。",
  YUNSHU_SSD_CACHE_PREFILL_CEIL_TPS:
    "文字引擎：實測預填速度超過此值（tok/s）時略過 SSD 前綴還原，因為此時重新預填與讀回 KV 一樣快。",
  YUNSHU_SSD_CACHE_MAX_GB:
    "文字引擎：SSD 前綴快取的大小上限（GiB），整個目錄（所有模型合計）共用一份預算。",
  YUNSHU_CACHE_RESERVE_PCT:
    "SSD 前綴快取（APC 與文字）：快取寫入不得占用的剩餘空間，以磁碟區百分比計。保留量取此值與 `YUNSHU_CACHE_RESERVE_GB` 中較大者，也會限制快取根目錄的實際上限。",
  YUNSHU_CACHE_RESERVE_GB:
    "SSD 前綴快取（APC 與文字）：磁碟區上至少保留的剩餘空間（GiB，見 `YUNSHU_CACHE_RESERVE_PCT`）。會使剩餘空間低於此值的寫入將被捨棄，溢寫暫停直到空間恢復。",
  YUNSHU_CACHE_STALE_DAYS:
    "SSD 前綴快取（APC 與文字）：檢查點命名空間超過此天數未使用，或其檢查點已不存在或已變更，會在淘汰其他內容之前先被移除（0 表示不依時間清除）。",
  YUNSHU_KV_QUANT_BITS:
    "文字引擎 KV 快取量化（有損，以品質換記憶體）：`off`（無損）、`auto`（KV 快取將超過約 2 GiB 時改用 8-bit），或固定 2/3/4/8 bit。",
  YUNSHU_VLM_APC_MEMORY_GB:
    "VLM 執行器前綴快取（APC）的記憶體預算（GiB）；0 表示停用。未設定時取扣除權重與系統／啟動值保留後剩餘記憶體的一半，上限為機器的四分之一與 32 GiB，剩餘不足 1 GiB 時關閉。27B 檢查點每個快取 token 約需 130 KiB。",
  YUNSHU_VLM_APC_DISK:
    "APC 的 SSD 層：被記憶體淘汰的前綴檢查點（以及關閉時仍在記憶體中的）會寫入磁碟，之後讀回而非重新預填（位元精確、無損；27B 檢查點載回比預填快約 20 倍）。設為 0 則前綴快取只放記憶體。",
  YUNSHU_VLM_APC_DISK_DIR:
    "APC SSD 層的目錄；未設定時為 `~/.yunshu/cache/apc`（內建磁碟）。可放到高速磁碟區以保持內建磁碟乾淨。",
  YUNSHU_KV_PRECISION:
    "Qwen3.5 系列執行器共用解碼批次的 KV 快取精度：`bf16`（無損）或 `int8`（KV 記憶體與讀取頻寬約為 0.53 倍，注意力有小幅誤差，以品質換記憶體）。單一請求與推測解碼通道維持 bf16。",
  YUNSHU_VLM_APC_DISK_GB:
    "APC SSD 層的大小上限（GiB），整個目錄共用一份預算（跨命名空間先淘汰最久未使用的檔案；0 表示不設上限，仍受剩餘空間保留限制）。未設定時為磁碟區的四分之一，最多 64 GiB。27B 檢查點每個 token 約 130 KiB，因此 64 GiB 約可容納 500K token。",
  YUNSHU_PREFILL_MATMUL:
    "M5 級 GPU 上 Qwen3.5 系列 VLM 執行器：超過 512 列的預填區塊所用的矩陣乘法。`stock` 對未分塊的權重執行 MLX 量化矩陣乘法（27B 冷預填約快 25%，區塊的位元結果取決於列數）；`lane` 對所有列數都使用列不變的 lane 核心。屬於每個 APC 鍵與 SSD 命名空間的一部分。",
  YUNSHU_PREFILL_BUFFER_CACHE_GB:
    "VLM 執行器：在預填步驟之間保留的 MLX 已釋放緩衝快取大小（GiB）。未設定時為實體記憶體的 5%，最多 6 GiB。可讓長前綴還原的緩衝在請求之間保持熱狀態；0 表示沿用上游每個區塊後清除。僅影響配置器，輸出不變。",
  YUNSHU_PREFILL_GDN:
    "Qwen3.5 系列 VLM 執行器：64 個 token 以上預填區塊的 GatedDeltaNet 核心。`chunked` 使用 MLX 的 `mx.fast.gated_delta_update`（每層快 2.7 倍，與 `step` 相差在 bf16 捨入內）；`step` 維持 mlx-vlm 的逐 token 核心。屬於每個 APC 鍵與 SSD 命名空間的一部分。",
  YUNSHU_VLM_APC_DISK_TIERS:
    "SSD 目錄之下的其他 APC 儲存層，以逗號分隔的 `PATH[@GiB]`（外接 SSD、HDD、NAS）。啟動時會量測各磁碟區並依讀取速度排序；被淘汰的檢查點會下移而非刪除，且僅在還原比重新預填更快時才由該層服務命中。未掛載的層會略過。無損；未設定則只用 SSD。",
  YUNSHU_VLM_APC_DISK_ENCODING:
    "較低層 APC 儲存（`YUNSHU_VLM_APC_DISK_TIERS`）保存檢查點的方式：`raw`（SSD 檔案的副本）、`zstd`（無損的位元組平面重排加 zstd），或 `auto`（僅在實測能提升有效讀取頻寬時使用 zstd，例如慢速磁碟與網路共享）。絕不有損。",
  YUNSHU_VLM_APC_WARM:
    "APC WARM 層：前綴檢查點離開記憶體（HOT）層、進入 SSD 之前的處理方式。`off`：直接寫入 SSD；`lossless`：壓縮後留在記憶體（位元精確，耗用 CPU）；`int8`／`int4`：注意力 K/V 以量化碼保存（有損；SSD 層仍保留精確狀態）。WARM 層占用 APC 記憶體預算中 `YUNSHU_VLM_APC_WARM_SHARE` 的比例。",
  YUNSHU_VLM_APC_WARM_SHARE:
    "啟用 `YUNSHU_VLM_APC_WARM` 時，WARM 層占 APC 記憶體預算（`YUNSHU_VLM_APC_MEMORY_GB`）的比例，其餘由 HOT 層使用；APC 記憶體總量不會增加。",
  YUNSHU_VLM_MAX_IMAGE_BYTES: "請求可透過 URL 引用的圖片大小上限（位元組）。",
  YUNSHU_VLM_INSECURE_SSL: "TLS 驗證失敗時，改用不驗證的方式重試圖片下載。",
  YUNSHU_MTP:
    "Qwen3.5 系列 VLM：使用檢查點的 MTP 頭進行草稿（批次不變；開啟與關閉推測解碼結果相同）。",
  YUNSHU_VLM_DRAFT:
    "Qwen3.5 系列 VLM：推測解碼草稿覆寫。可填 DFlash 草稿模型目錄；`mtp` 強制使用檢查點的 MTP 頭；`off` 停用草稿。未設定時，若模型目錄或 Hugging Face 快取中有相符的 DFlash2 草稿模型則自動使用，否則使用 MTP 頭。",
  YUNSHU_MTP_BLOCK_SIZE:
    "草稿區塊大小（對 DFlash 而言是其依接受率調整的深度上限）。未設定時，MTP 為 6，DFlash 為草稿模型訓練時的區塊大小。",
  YUNSHU_SPEC_COPY_ROWS:
    "Qwen3.5 系列單一請求推測解碼通道：提示詞複製回合可使用的驗證列數（複製草稿數 = 列數 - 1）。複製回合會在提示詞與已生成文字中找出目前尾端最長的先前出現處，提出其後續內容，並由同一個驗證步驟檢查，因此輸出不變。引用上下文的代理、程式碼編輯與多輪流量，每回合可提交數倍的 token。預設 16 列，並受後端認證寬度限制；0 表示關閉複製回合。",
  YUNSHU_DRAFT_BITS:
    "Qwen3.5 系列 DFlash 草稿模型的權重位元數：8（預設）、4，或 0 表示保留原出貨的 bf16。草稿由目標模型驗證，因此任何值的輸出 token 都相同；位元數越少，草稿模型每回合讀取的位元組越少，但接受率可能下降。",
  YUNSHU_SPEC_TREE:
    "Qwen3.5 系列單一請求推測解碼通道：`off` 維持訓練好的鏈式加複製；`tree` 強制使用草稿樹驗證器；`auto` 對有界的 1K 級貪婪請求使用已認證的 M5 Q4 DFlash2 快速樹，其餘維持鏈式加複製。貪婪解碼的 token 與一般解碼相同。",
  YUNSHU_NGRAM_DEFAULT:
    "文字模型：預設對貪婪請求啟用無損的 n-gram 推測解碼（也可由單次請求的 `spec_decode` 啟用）。在重複性高的輸出上有優勢。",
  YUNSHU_SPEC_PROPOSER: "文字模型：n-gram 推測解碼所用的提案器類型。",
  YUNSHU_GEMMA4_ASSISTANT:
    "文字 Gemma-4 模型：輔助草稿模型目錄（共用 KV 的推測解碼草稿模型）。",
  YUNSHU_GPU_SAMPLER:
    "文字模型：在 GPU 上進行 Gumbel-max 取樣（每個 token 無需 GPU 到 CPU 的同步）。",
  YUNSHU_JUMP_FORWARD:
    "文字模型：JSON schema 輸出中由文法強制決定的結構性 token 直接輸出，不經過前向傳遞。",
  YUNSHU_JSON_SCHEMA_ENGINE:
    "負責 JSON schema 與 `json_object` 約束解碼遮罩的引擎：`llguidance`（248K 詞表上每 token 遮罩中位數約 0.25 ms；屬性依 schema 順序）或 `inhouse`（Python 狀態機，中位數約 3 ms；屬性順序不拘）。llguidance 無法編譯的 schema，在內建引擎支援時會退回內建引擎。",
  YUNSHU_GRAMMAR_BITMASK:
    "約束解碼改用 xgrammar 風格的位元遮罩引擎，而非允許清單取樣器。",
  YUNSHU_TOOL_GRAMMAR:
    "工具呼叫約束解碼（結構標籤）：在工具呼叫起始標記之前為自由文字，之後呼叫本體會被遮罩為本次請求精確的呼叫文法（工具名稱、參數鍵、具型別的值、結尾）。強制的 `tool_choice` 一律受約束；此旗標僅控制 `auto`。關閉時，自動選擇的工具呼叫不受約束地解碼，事後再修復。",
  YUNSHU_QUANT_MODE:
    "載入時在記憶體中量化權重（有損，以品質換記憶體）：`mxfp4`、`nvfp4`、`mxfp8` 或 `affine`（留空則維持檢查點原樣）。",
  YUNSHU_QUANT_CONFIG:
    "`affine` 記憶體內量化的位元數與群組大小：JSON（含 `bits` 與 `group_size` 鍵）或 `bits`／`bits,group` 格式。",
  YUNSHU_ROUND_PREFILL_CHUNK:
    "回合驅動器：每個預填區段的提示詞 token 數。解碼中的請求只在預填前向傳遞之間推進，因此區段越小，越能在長提示詞旁維持執行，代價是預填速度約降低 20%。區段依提示詞固定，輸出不受其他執行中工作影響；以不同大小預填的提示詞各自一致，但彼此並非位元相同。",
  YUNSHU_ROUND_DRIVER:
    "稠密 Qwen3.5 系列 VLM：以Yunshu的回合驅動器處理文字請求（對所有解碼列做打包前向傳遞，與批次預填步驟交替；每列皆有 MTP 草稿；依位置鍵取樣；見 `docs/guides/ROUND_DRIVER.md`）。關閉時使用上游 BatchGenerator 加單一請求推測解碼通道。圖片提示詞、int8 KV 與 MoE 無論如何都走上游路徑。",
  YUNSHU_MTP_ROW_EXACT:
    "Qwen3.5 系列執行器：使用 oMLX 逐列精確驗證（驗證列與單列解碼位元相同），取代批次不變核心。",
  YUNSHU_ENGINE_LOOP:
    "文字模型：使用 EngineCore 連續批次迴圈，取代單一請求快速路徑。",
  YUNSHU_OVERLAP:
    "文字引擎迴圈：讓 CPU 與 GPU 工作重疊（`cpu_gpu`），或將批次拆成兩個互相重疊的一半（`two_batch`）。",
  YUNSHU_SPEC_UNVERIFIED:
    "文字模型：明確啟用未經驗證的外部 mlx-lm 草稿實驗。`eagle` 是沿用的路由名稱，一般語言模型草稿即可，不需要訓練過的 EAGLE 頭。僅適用於使用預設取樣與懲罰的貪婪、非串流請求；schema、自訂處理器、停止字串、思考預算、LoRA 與 token 遮罩選項仍走一般快速路徑。",
  YUNSHU_DRAFT_MODEL:
    "供 `YUNSHU_SPEC_UNVERIFIED=eagle` 使用的外部 mlx-lm 草稿檢查點（一般語言模型草稿，而非訓練過的 EAGLE 頭）。",
  YUNSHU_REALTIME_OMNI:
    "Realtime 連線上的原生 Qwen3-Omni 語音：`auto`（提供可說話的模型時啟用）、`on` 或 `off`（改用 ASR、LLM、TTS 串接流程）。",
  YUNSHU_OMNI_MODEL:
    "語音路徑所用的模型（與所服務模型不同時設定；omni 模型作為服務模型時會自動沿用）。",
  YUNSHU_OMNI_PRELOAD: "開機時預熱 omni 模型，讓第一個語音請求不必冷啟動。",
  YUNSHU_OMNI_THINKER_MAX:
    "omni Thinker 每個語音回合最多寫出的 token 數（口語回覆的長度上限）。",
  YUNSHU_OMNI_PERSONA:
    "請求未提供時使用的 Realtime 系統角色設定；空字串表示停用。未設定時使用內建的簡潔口語風格。",
  YUNSHU_REALTIME_SILENCE_MS: "伺服器端 VAD：模型回答前的停頓時間（ms）。",
  YUNSHU_REALTIME_BARGE_IN_MS: "在模型回覆途中打斷它所需的持續說話時間（ms）。",
  YUNSHU_REALTIME_VAD_THRESHOLD: "伺服器端 VAD 的語音偵測門檻。",
  YUNSHU_REALTIME_PREFIX_PADDING_MS:
    "伺服器端 VAD：偵測到語音之前保留的音訊長度（ms）。",
  YUNSHU_REALTIME_VAD: "伺服器端 VAD 實作：`energy` 或 `silero`。",
  YUNSHU_REALTIME_VAD_MODEL:
    "`YUNSHU_REALTIME_VAD=silero` 時使用的 Silero VAD 模型 id。",
  YUNSHU_REALTIME_MAX_INPUT_AUDIO_BYTES:
    "Realtime：可緩衝的輸入音訊大小上限（位元組）。",
  YUNSHU_REALTIME_MAX_CONVERSATION_ITEMS:
    "Realtime：每個工作階段保留的對話項目數。",
  YUNSHU_DIFFUSION_SCHEDULER: "圖像生成取樣器覆寫（留空則使用管線自帶的）。",
  YUNSHU_ANE_EMBEDDINGS:
    "可用時透過 CoreML 在 Apple Neural Engine 上計算嵌入。",
  YUNSHU_ANE_EMBEDDING_MODEL: "ANE 路徑所使用的嵌入模型。",
  YUNSHU_MCP_CONFIG: "列出工具伺服器的 MCP 用戶端設定檔（JSON／YAML）。",
  YUNSHU_MCP_SERVERS:
    "以 JSON 陣列指定的 MCP 工具伺服器（`YUNSHU_MCP_CONFIG` 的替代方式）。",
  YUNSHU_WEB_SEARCH_PROVIDER:
    "伺服器端 `web_search` 工具的搜尋後端。`auto` 依序選用第一個已設定的 searxng、brave、tavily、exa；`none` 表示停用。未設定時，請求會收到 API 的 `unavailable` 錯誤並附上提示。",
  YUNSHU_SEARXNG_URL:
    "自行架設的 SearXNG 實例的基礎 URL（需啟用 JSON 輸出），例如 `http://127.0.0.1:8080`；是推薦的注重隱私的預設選擇。",
  YUNSHU_BRAVE_API_KEY: "Brave Search API 金鑰。",
  YUNSHU_TAVILY_API_KEY: "Tavily API 金鑰。",
  YUNSHU_EXA_API_KEY: "Exa API 金鑰。",
  YUNSHU_WEB_SEARCH_RESULTS: "每次 `web_search` 呼叫回傳的結果數。",
  YUNSHU_WEB_FETCH:
    "提供伺服器端 `web_fetch` 工具（不需要搜尋供應商）。關閉時，`web_fetch` 請求會收到 `unavailable` 錯誤。",
  YUNSHU_WEB_FETCH_ALLOW_PRIVATE:
    "允許 `web_fetch` 存取私有、回送與本機鏈路位址。關閉（預設）時會封鎖這些位址，包含重新導向與 DNS 解析之後（SSRF 防護）。",
  YUNSHU_WEB_FETCH_MAX_BYTES: "`web_fetch` 下載的回應主體大小上限。",
  YUNSHU_WEB_FETCH_TIMEOUT: "`web_fetch` 等待網頁的秒數。",
  YUNSHU_WEB_FETCH_MAX_TEXT_CHARS:
    "交給模型的擷取網頁文字會截斷至此字元數（請求的 `max_content_tokens` 可進一步調低）。",
  YUNSHU_MCP_CONNECTOR:
    "提供 MCP 連接器：Anthropic 的 `mcp_servers` 與 OpenAI Responses 的 `mcp` 工具由本伺服器執行，並透過可串流 HTTP／SSE 連線到指定的 MCP 伺服器。",
  YUNSHU_MCP_CONNECTOR_ALLOW_PRIVATE:
    "允許 MCP 連接器存取私有與回送的 MCP 伺服器（本機工具伺服器是常見情況）。關閉時僅限公開位址。",
  YUNSHU_MCP_CONNECTOR_TIMEOUT:
    "MCP 連接器單次呼叫（initialize、`tools/list`、`tools/call`）可耗用的秒數，含 DNS 解析。",
  YUNSHU_MCP_CONNECTOR_MAX_BYTES:
    "MCP 連接器伺服器單次回覆（JSON 主體或單一 SSE 事件）解壓縮後的大小上限。",
  YUNSHU_SERVER_TOOL_MAX_ITERATIONS:
    "單一請求最多可進行的「生成、執行工具、繼續」回合數。",
  YUNSHU_MODEL_ALIASES:
    "多模型模式：將代理要求的模型名稱（`claude-sonnet-4-5`、`opus`、`gpt-5`）對應到實際提供的模型，格式為「模式到模型 id」的 JSON 物件；模式可為完整名稱、`prefix*` 或 `*`（先符合者優先；真實模型名稱永遠優先）。單一模型模式本來就會回應所有名稱。",
  YUNSHU_LOG_LEVEL: "Yunshu日誌記錄器的日誌等級（第三方記錄器維持 WARNING）。",
  YUNSHU_AUDIT_LOG_FILE: "同時將稽核日誌寫入此檔案。",
  YUNSHU_LOG_MAX_MB:
    "服務日誌（launchd）：日誌檔達到此大小（MiB）時輪替；0 表示關閉依大小輪替。",
  YUNSHU_LOG_ROTATE_HOURS:
    "服務日誌：距上次輪替超過此小時數時也進行輪替；0 表示關閉依時間輪替。",
  YUNSHU_LOG_KEEP: "服務日誌：保留的已輪替檔案數（gzip 壓縮，機密已遮蔽）。",
  YUNSHU_LOG_RETENTION_DAYS:
    "服務日誌：刪除超過此天數的已輪替檔案；0 表示保留至 `YUNSHU_LOG_KEEP` 將其清除為止。",
  YUNSHU_SERVE_LOG:
    "每完成一次生成請求，就將一行僅含數字的 JSON（耗時、token 數、推測解碼接受率、快取層、並行度、arm、build）寫入本機有大小上限並會輪替的檔案。絕不包含提示詞、輸出或 token id。預設關閉，資料不會離開本機。",
  YUNSHU_SERVE_LOG_DIR:
    "服務紀錄（serve log）的目錄；未設定時為 `~/.yunshu/logs`。",
  YUNSHU_SERVE_LOG_MAX_MB:
    "服務紀錄：達到此大小（MiB）時輪替；搭配 `YUNSHU_SERVE_LOG_KEEP`，目錄大小上限為 max * (keep + 1)。",
  YUNSHU_SERVE_LOG_KEEP: "服務紀錄：保留的已輪替檔案數。",
  YUNSHU_ARM:
    "在服務紀錄中記錄本伺服器所執行設定組（arm）的標籤，用於離線 A/B 分析；不會改變任何行為。",
  YUNSHU_GATEWAY_URL: "`yunshu` CLI 用戶端指令所使用的伺服器 URL。",
  YUNSHU_HF_ENDPOINT:
    "`yunshu serve` 使用的 Hugging Face Hub 端點（會匯出為 `HF_ENDPOINT`）。",
  YUNSHU_HISTORY_INTERVAL_S:
    "主控台歷史：記憶體內歷史環取樣的間隔秒數（吞吐、請求數、記憶體、TTFT 百分位，供 GET /v1/yunshu/history 使用）；0 表示關閉取樣。環的大小固定，不會增長。",
  YUNSHU_HISTORY_HOURS:
    "主控台歷史：歷史環保留的小時數（容量 = 小時數 × 3600 ÷ 取樣間隔，啟動時一次配置；5 秒間隔下 12 小時約 0.4 MiB）。",
  YUNSHU_EVALS_DIR:
    "Evals API 本機 JSON 儲存的資料夾；未設定時為 `~/.yunshu/evals`。",
  YUNSHU_VLM_MAX_VIDEO_BYTES: "請求以網址引用的影片大小上限（位元組）。",
  YUNSHU_WEB_SEARCH_PROVIDER_TIMEOUT:
    "聯合搜尋每個來源的期限（秒）；慢的來源不會拖住整個查詢。",
  YUNSHU_WEB_SEARCH_HEALTH_FILE:
    "搜尋來源健康狀態的小型快照檔（不含查詢、搜尋結果或憑證），供 `yunshu config` 讀取。",
  YUNSHU_WEB_MWMBL:
    "在自動聯合搜尋中加入 Mwmbl 開放小型網站索引（資料為 CC-BY-NC-SA 4.0，限非商業）；明確指定 provider=mwmbl 也算同意。",
  YUNSHU_WEB_KEYLESS:
    "允許不需金鑰的 DuckDuckGo（盡力而為，可能被封鎖）與 Wikipedia 搜尋；查詢文字與 IP 會離開這台機器。",
  YUNSHU_MOJEEK_API_KEY:
    "Mojeek 獨立索引的 API 金鑰；設定後會加入自動聯合搜尋。",
  YUNSHU_MARGINALIA_API_KEY:
    "明確選用 Marginalia 小型網站搜尋；公開 API 資料為 CC-BY-NC-SA 4.0（非商業），商業金鑰另有條款，沒有預設的公用金鑰。",
  YUNSHU_SERPER_API_KEY: "Serper（Google 搜尋結果）的 API 金鑰。",
  YUNSHU_PERPLEXITY_API_KEY:
    "Perplexity Search 的 API 金鑰（原始搜尋結果，不是 Sonar）。",
  YUNSHU_WEB_RESEARCH:
    "用來源頁面、不受信任的摘錄與本機排序補強搜尋摘要；穩定的選用功能，品質評估前不預設開啟；擷取的網址會離開這台機器。",
  YUNSHU_WEB_RENDER:
    "進階 Tavily 擷取／爬取／地圖在遇到只有 JavaScript 外殼的頁面時，改用本機 Chromium 後備（需 web-render 套件與已安裝的 Playwright Chromium；只發同源 GET，不帶 cookie，不會自動下載瀏覽器）。",
  YUNSHU_WEB_RESEARCH_BUDGET: "補強搜尋摘要的整體期限（秒，最多 4）。",
  YUNSHU_WEB_RESEARCH_PAGES: "每次補強最多取用的來源頁面數（上限 6）。",
  YUNSHU_WEB_RESEARCH_MODEL:
    "已載入的本機嵌入模型 id；不會自動載入模型，沒有或不可用時只用 BM25。建議 Qwen3-Embedding-0.6B。",
};
export default settingdesc;
