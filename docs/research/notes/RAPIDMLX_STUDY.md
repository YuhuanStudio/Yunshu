# Rapid-MLX 完整對照研究（98a1706c，2026-10-03）

本研究的源碼基準是本機 `reference/Rapid-MLX` 的 `98a1706c568d9b93f6e8c391c37d623dc33b6cf9`（package 0.15.4）。`Rapid-MLX/路徑:行` 皆相對該 checkout；`Yunshu/路徑:行` 相對 codex-rapidmlx worktree。源碼／文件閱讀與實測分開；未跑的cell保持未測，不能以公開headline替代。

公開安裝／比較頁核對：[Homebrew core](https://formulae.brew.sh/formula/rapid-mlx)、[官方比較](https://rapidmlx.com/compare)、[官方原始碼](https://github.com/raullenchai/Rapid-MLX)。本次安裝實際來自上述固定local commit，沒有執行external install.sh、發送issue/PR/comment或上傳benchmark。

## 面向逐項對照

| 面向 | Rapid-MLX 證據 | 它做什麼 | Yunshu 現況與證據 | 缺口／決策 |
|---|---|---|---|---|
| 安裝與 distribution | `Rapid-MLX/pyproject.toml:7`；`Rapid-MLX/docs/getting-started/installation.md:8`；`Rapid-MLX/install.sh:52` | PyPI、uv tool、Homebrew core bottle、guided install.sh；installer 檢查 arm64／Python／完整 venv，重建只清 venv 不清 user state。vision、mtp、dflash extras 分離，runtime upper bounds 防 dependency coherence 漂移。 | `Yunshu/pyproject.toml:31`；uv、核心 text 及 modality extras，Python≥3.13。 | 缺 core brew、signed app 與 guided installer；先補清楚的最短安裝／upgrade／first request 路徑，不能為追 feature 數而增無證據依賴。 |
| 首次體驗 | `Rapid-MLX/install.sh:160`；`Rapid-MLX/README.md:153` | RAM tier 決定 starter，先重用可執行 cached model；chat 無參數可下載並開始 REPL，serve 自選 8000–8009，explicit port 不 fallback。 | `Yunshu/python/yunshu_cli/chat.py:22`；`Yunshu/python/yunshu_cli/serve.py:51`；chat 連既有 server，serve 需要模型／配置；doctor 可在下載前執行。 | 缺一條零配置 chat 的閉環；優先 actionable 失敗、cached starter／選模，保持 download 可控。埠占用 early failure 已做 prototype＋CPU tests。 |
| 桌面 app | `Rapid-MLX/apps/rapid-mac/README.md:3`；`Rapid-MLX/apps/rapid-mac/Tests/GUIGoldenFlows/journeys.yaml:7` | SwiftUI、bundled Python sidecar、download/start/stop 分步、準備／warmup 進度、chat history、files、MCP、voice／image／video；Sparkle signed updater。GUI journeys 與 snapshots 並行驗證。 | 本 worktree 沒有對等 standalone signed Mac app；Yunmo 是一個 consumer，不可代替 Yunshu first-run。 | 高影響／高成本 UX defect；先把 CLI readiness／模型 lifecycle 完整化，app 是另一期可交付產品，不移植 UI 進 decode path。 |
| 模型 mirror／管理 | `Rapid-MLX/rapid_mlx/_mirror.py:1`；`Rapid-MLX/rapid_mlx/byom/preflight.py:2` | R2 per-file fallback HF，寫同 HF snapshot cache，Range resume、size check、LFS SHA256 (`Rapid-MLX/rapid_mlx/_mirror.py:646`)、HF blob symlink、4 workers、慢速 floor fallback。BYOM metadata 先判可證明的 format／arch／RAM 不可行，unknown 不拒絕；suggest alternative、explicit cancellable import。 | `Yunshu/python/yunshu_cli/model.py:223`；`Yunshu/python/yunshu_cli/doctor.py:187`；resume、已有模型不 duplicate download、local/HF cache scan、weights/memory doctor。 | 缺 curated catalog／量測 RAM recommendations／下載前 metadata 判斷／mirror；mirror 要服務維運，不可直接重用對方 private contract 或假設快。先測 metadata refusal＋cached-model guidance。 |
| CLI／設定 | `Rapid-MLX/docs/reference/cli.md:575`；`Rapid-MLX/rapid_mlx/runtime/effective_config.py:109` | 廣泛 CLI，universal context override、recipe／info／agents／launch／service；來源與 constraint trace 可序列化，shell argcomplete，模型 aliases。 | `Yunshu/python/yunshu_cli/config.py:20`；`Yunshu/python/yunshu_engine/settings.py:1`；集中 typed settings、config set/unset、--set、effective value/source、global JSON。 | Yunshu 已有有效配置／JSON，不是空白；缺 aliases completion、recipe、universal context 語法與 fresh first-run。不可新增實驗 flags（cap）。 |
| 文件／compare | `Rapid-MLX/docs/index.md:27`；`Rapid-MLX/README.md:38`；`Rapid-MLX/ROADMAP.md:3` | docs 按 install、server、modalities、agents、reference、bench/release 組織；comparison 將 feature 與測量分開，未知標 —；Roadmap 明確 retired snapshot，release notes 才 authoritative。 | `Yunshu/docs/guides/AGENT_COMPAT.md:11`；`Yunshu/docs/guides/RELEASE_GATE.md:3`；API、settings、client/release gate 文件，有 agent census。 | 缺發佈 docs site 與 first-run navigation 的集中入口；比較要列同 checkpoint／硬體／版本／workload／限制，不能復活 retired whitepaper。 |
| benchmarks／community | `Rapid-MLX/community-benchmarks/README.md:9`；`Rapid-MLX/benchmarks/prompt_host_cache_bench.py:2`；`Rapid-MLX/docs/benchmarks/llm.md:3` | text v1 512/128 和2048/512，每格1 warmup＋5 rounds、synthetic seed corpus；TTFT／decode duration／active memory raw rows，decode=(N−1)/duration；run local then explicit share；immutable protocol digests，量化／runtime／thermal／power unknown 明示。microbench 不等 HTTP。 | `Yunshu/docs/guides/RELEASE_GATE.md:26`；`Yunshu/scripts/research/bench_reference_http.py:1`；quiet gpuq、interleaved≥3、digest、HTTP/in-process gate。 | 缺一般用戶可比較 immutable benchmark schema 與 local archive；先統一 token accounting、完整 request/SSE 與機器 conditions，community leaderboard 可後做。 |
| harness／eval 方法 | `Rapid-MLX/harness/README.md:33`；`Rapid-MLX/evals/run_eval.py:11`；`Rapid-MLX/evals/perf_gate.py:23` | fresh server benchmark BEFORE stress/SDK；schema baseline 綁 checkpoint revision、chip、toolchain、run medians／threshold。tool scenarios、code、math、general deterministic graders；docstring說30但當前tool_calling.json實有31個（含tc31-weather-explicit），其中1個有expected_facts；perf floor 人工 review，advisory 無 floor 不等勝出。 | `Yunshu/docs/guides/RELEASE_GATE.md:9`；`Yunshu/docs/guides/AGENT_COMPAT.md:15`；real SDK、census/replay、MMLU300、capability matrix。 | 兩邊都有 gate，但不可把 PASS schema／parser 當 real agent task correctness。補對等 Rapid eval＋Claude Messages／Codex Responses shapes，錯誤／skip／xfail分開。 |
| tool calling／repair | `Rapid-MLX/rapid_mlx/tool_parsers/hermes_tool_parser.py:128`；`Rapid-MLX/rapid_mlx/api/tool_calling.py:120`；`Rapid-MLX/rapid_mlx/routes/chat.py:2834`；`Rapid-MLX/rapid_mlx/api/tool_grammar.py:5` | 27 parser modules（README count），native history 不降成文字；XML positional scan 保留 literal delimiters、schema declared sibling checks；safe literals、schema-guided numeric/bool conversion、forced-call recovery、llguidance grammar core default（仍要確認 model eligibility）。 | `Yunshu/python/yunshu_engine/tool_format.py:440`；`Yunshu/python/yunshu_gateway/routers/chat.py:660`；tokenizer native formats＋fallback、JSON schema both serving paths、per-request parser/state；main合入的 `Yunshu/python/yunshu_engine/tool_arguments.py:15` 支援schema型別與union/enum，`Yunshu/python/yunshu_engine/tool_thinking.py:1` 恢復client丟掉的tool reasoning（full-prefix match，ambiguity拒絕）。 | 不盲移植 parser 數量；需要 schema scalar／undeclared delimiter／parallel／truncated tool regression corpus。tool_format 與 OpenAI/Anthropic history 目前他人持有，這次先測與列具體 port，避免碰撞。 |
| agent coverage | `Rapid-MLX/docs/agents/matrix.md:1`；`Rapid-MLX/.github/workflows/auto-release.yml:260` | 12 agents×families 文件分 PASS／arch-format-Ultra XFAIL；Copilot/Droid/Kimi 是 wire-smoke，不代表真 CLI；release 另外 gate Claude/Codex/Aider/Hermes/DSH real sessions。 | `Yunshu/docs/guides/AGENT_COMPAT.md:16`；`Yunshu/python/yunshu_cli/integrations/agent_config.py:73`；Claude、Codex、opencode context／reasoning/model metadata；真 CLI census 與 e2e。 | 缺同 release artifact 的 Tier-1 mandatory result matrix，缺小型 family smoke 可持續 refresh；model capability 與 agent route capability 分開列。 |
| API surface | `Rapid-MLX/rapid_mlx/routes/anthropic.py:88`；`Rapid-MLX/rapid_mlx/routes/responses.py:135`；`Rapid-MLX/docs/reference/configuration.md:248` | OpenAI chat/completions/responses、Anthropic Messages/count tokens，model/status/cache/runtime/audio/image/video/MCP；backend contracts不同，native MTP 僅 serial text 且不支援 APC/schema/logprobs 等。 | Yunshu routers 提供 OpenAI/Anthropic、Responses WS、MCP、Realtime、modalities；`Yunshu/docs/guides/API_SURFACE.md:1` | 不能只打 common chat 後宣稱 Codex／Claude parity；逐 endpoint shape 查 streaming ordering、error、cancellation、usage、cache fields。保留 Yunshu constrained decoding。 |
| scheduler／continuous batch | `Rapid-MLX/rapid_mlx/engine_core.py:37`；`Rapid-MLX/rapid_mlx/scheduler.py:899` | 單 GPU worker stream，BatchGenerator queue，memory admission；continuous MTP 改 next() boundary 可對各 uid 返回不同 accepted tokens；min_batch_lanes=2 保留成熟 singleton，quant/window constraints fail closed。 | `Yunshu/python/yunshu_engine/vlm_batch_runner.py:50`；VLM shared BatchGenerator per-row sampler；text mlx-lm serialize fast path。 | multi-row spec 是實際 engine defect，但 round_driver 已有 owner；移植 coordinator 前要 parity dynamic join/cancel/prefill/APC/grammar，8-way throughput只量測，不改產品成 throughput race。 |
| prefix／host cache | `Rapid-MLX/rapid_mlx/memory_cache.py:2042`；`Rapid-MLX/rapid_mlx/prompt_host_cache.py:20`；`Rapid-MLX/rapid_mlx/prefix_cache.py:338` | radix lookup、hybrid boundary snapshots、persist/restart、pin/ref counts、safe trim；host render+tokenize cache independent KV，64 entries／64MiB、order-preserving digest、immutable values、unknown miss。 | `Yunshu/python/yunshu_engine/paths.py:32`；VLM APC exact hybrid/media salts、RAM+bounded internal SSD、turn2 checkpoints；已有 `_VLMTextPromptCache` (`Yunshu/python/yunshu_engine/vlm_engine.py:197`) cache render＋token IDs，256 entries；`Yunshu/python/yunshu_engine/vlm_batch_runner.py:6` | host cache不是缺席；缺更嚴格byte cap／immutable values／order-sensitive identity。加固需 template identity／tools ordering／kwargs＋revision key、bounded memory；APC/restore由 cachesf/prefill owners處理，不能套 legacy trim 到 GDN。 |
| MTP／self-MTP #4071 | `Rapid-MLX/rapid_mlx/spec_decode/mtp/mlx_backend.py:774`；`Rapid-MLX/rapid_mlx/spec_decode/mtp/continuous_engine.py:104`；`Rapid-MLX/rapid_mlx/scheduler.py:975` | draft token arrays留GPU；first draft僅每lane last-valid hidden做vocab projection；target forward後一次materialize selected IDs＋logprobs，CPU longest-prefix accept，processor lanes保留舊path。commit 98a1706c offline M4 Pro：9B B1/2/4 58.7/67.5/74.1→62.0/74.6/81.2；35B 86.9/106/127.2→92.4/118/142.5 tok/s。 | `Yunshu/python/yunshu_engine/mtp_lane.py:163`；已有packed selection、copy/DFlash lanes與batch-invariant verify；多row工作在round driver。 | lossless候選：避免per-lane .item、split lm_head最後位置、single readback；是既有round work重疊，不在本線直接改。離線收益不直接當HTTP；digest＋APC parity＋200 paired score差≤1再promotion。 |
| spec／kernels／memory | `Rapid-MLX/docs/reference/configuration.md:205`；`Rapid-MLX/rapid_mlx/kernels/lane_matmul/installer.py:47`；`Rapid-MLX/rapid_mlx/memory_budget.py:142`；`Rapid-MLX/docs/specdecoding-validation-notes.md:25` | MTP auto-K依accept/cost ratio，不總更快；DFlash/DSpark/suffix/native per-alias gates；TensorFold/MLX2 row invariant lane matmul，M5 mpp vs M1–4 simd不同law id須入cache key；measured weight footprint＋headroom決定Metal cap與admission同值。spec可接受stock AR numerical fork，不宣稱byte equal。 | Yunshu singleton invariant+NAX packed；lossless預設、lossy KV量化user option，memory guard/settings；`Yunshu/python/yunshu_engine/memory_guard.py:41` | 不把Rapid近似lossless contract直接升Yunshu預設；同checkpoint digest gate更強。Memory preflight可CPU改善；kernel port要roofline、HTTP收益、cache numerical-law隔離。 |
| release／CI／community | `Rapid-MLX/RELEASE.md:9`；`Rapid-MLX/.github/workflows/ci.yml:11`；`Rapid-MLX/CONTRIBUTING.md:38`；`Rapid-MLX/README.md:23` | version bump→exact-ref dryrun→Tier1 real agents＋signed Desktop candidate＋main/blocker證據＋protected reviewer→PyPI/Brew/DMG。CI merge queue tree evidence認證後reuse，otherwise full；install patch subprocess測production import。Discord/DeepWiki降低學習門檻。 | `Yunshu/docs/guides/RELEASE_GATE.md:4`；wheel clean install＋SDK＋families＋soak/service gate，現階段小版本。 | 缺分發automation、real install artifact上mandatory agent gate、docs/community discoverability；先加first-run checks與release evidence table，不移植multi-tenant／sharing control plane。 |

## 公開數字的適用範圍

- README:68 的 3× 是 **M2 Pro 32GB、Rapid 0.12.11 vs Ollama 0.32.7、Qwen3.6-35B-A3B、8 streams** 的 aggregate decode 82.9 vs 27.2 tok/s；whole-batch包含prefill約1.6×，single stream約1.5×，dense12B沒有更快。沒有Yunshu arm，不能證明對Yunshu全面勝出。
- docs/benchmarks/llm.md:3 明確legacy；其35B DFlash 194.8–235.75 tok/s是歷史single-run community結果，現今MoE alias沒有通過eligibility gate。不能拿它當這次best supported configuration。
- docs/reference/configuration.md:215 的M2 Pro35B MTP高acceptance仍慢20%；native M3 Ultra serial backend 83.32→130.93 tok/s另有greedy byte parity，但沒有APC/schema/logprobs／batch。採用前必須交代能力代價。
- docs/specdecoding-validation-notes.md:26 與Gemma requalification承認q_len numerical fork；Rapid target-verified不自動等於Yunshu「spec on==off」強契約。跨engine可不同digest；同engine port on/off的digest必須相等。

## 優先移植清單（impact × effort；5最高）

| 順位 | 項目 | Impact | Effort | 類型 | 驗收 |
|---|---|---:|---:|---|---|
| P0 | 同checkpoint實測＋request/usage/digest可追溯 | 5 | 3 | 測量 | quiet、3 interleaved reps、cold/warm/turn2、1K/8K/32K、8way、peak、完整receipt |
| P0 | serve埠占用在model load前actionable refusal | 4 | 1 | UX/CLI | occupied endpoint rc2，既有listener仍活，UDS不走TCPprobe |
| P0 | first-run最短閉環與base vs extras說明 | 5 | 1 | UX/docs | clean install→doctor→cached/pull→serve→curl→launch，copy/paste命令全部可解析 |
| P1 | BYOM metadata format/arch/memory preflight | 5 | 2 | UX/CLI | 明確不可行拒絕weights下載；unknown保留原行為；resume不duplicate |
| P1 | chat fresh session／clear system instruction | 3 | 1 | UX/CLI | 不把上次session history送新model；clear不drop system |
| P1 | model completion＋cached starter picker | 4 | 2 | UX/CLI | aliases/local/HF同解析，no implicit huge download，RAM fit證據 |
| P1 | tool malformed/typed scalar/marker corpus | 5 | 2 | API可靠性 | Rapid eval＋Claude/Codex shapes；schema-correct args與literal losslessness；與glmspark/bigmoe協調 |
| P1 | 加固既有render/tokenization host cache | 4 | 2 | lossless engine原型 | 完整identity、order-sensitive key、cache hit==miss IDs；HTTP TTFT≥3 reps |
| P1 | split first-draft projection＋single sync | 5 | 3 | lossless engine原型 | owned round driver先整合；on/off tokens/logprobs/cache相等，200 paired差≤1 |
| P2 | multi-row self-MTP joins/cancel/rollback | 5 | 5 | lossless engine原型 | dynamic membership、per-row sampler、grammar、APC correctness先於8-wayspeed |
| P2 | release artifact×Tier1agent必過表 | 4 | 3 | release/UX | exact wheel+SHA，PASS／FAIL／SKIP／XFAIL，clean install含tool loop |
| P2 | immutable local benchmark schema/archive | 3 | 2 | UX/measurement | model/revision/quant/hardware/conditions/protocol，不能unreviewed自動覆蓋baseline |
| P3 | signed desktop／Homebrew core／docs site | 5 | 5 | UX/distribution | sidecar lifecycle、download/start/warm進度、signed update與real journeys |

## 本線原創候選（尚未宣稱收益）

1. **Host cache與APC的雙層receipt**：render/tokenize hit不等於KV hit。每請求分別記錄兩層miss原因與耗時，精確定位cached TTFT剩餘時間；模板修訂時只invalidate host plane，不清無關模型的APC。
2. **模型可行性離線receipt**：以config＋weight index stat fingerprint保存format、arch與memory bounds；共享CLI／service／doctor判定，不需先import MLX或下載weights；unknown不硬拒絕。
3. **First-run準備工作避開GPUqueue**：tokenizer/template/config預檢與agent profile生成CPU完成，只有load/prefill進GPU；透過首request breakdown確認這不是把load時間藏掉。

## M5 Max head-to-head 執行狀態

環境：`/Volumes/P5Plus/yunshu-test-envs/rapid-mlx/.venv`；uv重建9/28的venv，保留舊的非venv證據；extras `[mtp,dflash,test]`。安裝log在 `/Volumes/P5Plus/yunshu-build/codex/rapidmlx/install.log`。Yunshu使用main既有venv＋此worktree PYTHONPATH，工具鏈差異會明示，same checkpoint不代表same runtime。

Harness：`scripts/research/rapidmlx/head_to_head.py`；固定local checkpoint、每rep/size fresh process、ARMs順序counterbalance；HTTP decode=(completion_tokens−1)/(last−first text/reasoning event)，TTFT從POST起到第一非空delta；cold是fresh-process第一generation（OS filesystem cache並非強制cold）。warm重複同body，turn2包含前回答與新user。8way使用獨立prefix，分開報aggregate **end-to-end** throughput，不假扮純decode。客戶端SSE事件可能合併多token，短response不能可靠decode；completion usage缺失時保留null。

記憶體採100ms process-treeRSS，另每arm讀macOS `footprint -p PID` 的process-lifetime `phys_footprint_peak`；raw輸出存footprint.txt。RSS／physical peak分開，不把RSS冒充Metal allocator。APC保持Yunshubounded internal product default，其他scratch／cache external；Rapid隔離HOME在P5Plus，disable telemetry與update以免研究產生外部事件（harness已設定）。

CPU dry-run PASS。Tiny correctness已通過，再做27B first-arm smoke後才提交quiet sweep。全部job與results最終append於此。3.5/3.6展示MoE：followlinks掃描config的qwen3_5_moe／num_experts沒有matches（moe-inventory.json），不下載額外checkpoint替代條件。

## 最近200筆已合併PR（git first-parent）

定義：沿固定HEAD的first-parent歷史，取subject帶 `(#N)` 或 `Merge pull request #N` 的最近200筆，排除普通支線commits；這是merge records不是聲稱200個issue或~4000全是merge。完整題目如下。keyword分類可重疊：UX/model/install 102、perf 23、validation/release 44、telemetry39、docs10。重心明顯在first-start/BYOM/agents/Desktop／telemetry與release可靠性，同時持續self-MTP／kernels工作；大量telemetry不等於decode優化。

```text
98a1706c perf(mtp): one host sync per continuous self-MTP cycle (#4071)
4e51b93d Merge pull request #4057 from raullenchai/harbor/open-pr-sweep-consolidation
70692fa9 feat(agents): pi profile, dsh 0.2 setup, opencode headless, and named telemetry callers for codex/opencode/pi/qwen-code/dsh (#4056)
d3f42d62 fix(agents): Claude Code prefix cache, number ints, undeclared tool markup (#4036, #4037, #4038) (#4058)
e9594a39 fix(serve): first-start failure sweep — classify, gate and name the fix for every remaining startup failure class (#4035)
82bc4ca6 feat(byom): rapid-mlx import — explicit, cancel-safe MLX quantization (#4012)
27035167 fix(serve): degrade text-capable VLM checkpoints to the text lane on a base wheel (#4001)
1e3c6d5f feat(api): SillyTavern sampler contract — DRY, repetition range, explicit refusals (#4014)
8db5a959 feat(byom): opt-in support request after a preflight refusal (#4008)
c4e220fb feat(bench): bench --submit no longer submits; points to benchmark run + share (#4010)
388a7473 feat(byom): suggest a runnable alternative when the preflight refuses (#4004)
01c0a534 feat(byom): refuse unrunnable uncataloged models before download (#4002)
1f998f74 feat(links): point install, recipe and benchmark share at the leaderboard views (#4003)
3366d3b6 test: replace blanket warning ignores with a targeted policy (#3988)
f4ffa41f fix(ci): serialize Mergify integration candidates (#4055)
9221ff77 fix(telemetry): sample ledger time after lock wait (#3986)
fdd5f5ad fix(mac): remove telemetry launch banner (#3680)
8fd693dd test(telemetry): wait for loopback event flush (#3916)
e57693c8 feat(cli): add universal --context-length override (#3968)
14d66d70 test(mac): make live progress capture synchronous (#3981)
d15aadb7 docs(release): note restart telemetry deduplication (#3975)
6aaf981b fix(telemetry): dedupe restart-loop server_start_state and app_opened (#3947)
a94821ea docs: document GLM Desktop compatibility in 0.15.4 (#3970)
f4ad2c7b fix: make GLM TensorFold Desktop chat compatible (#3966)
d51ee1e0 feat: add qualified GLM-5.3 TensorFold profile (#3944)
d6a2e4ec fix: restore documented verification evidence (#3958)
24daa7d3 chore: remove one-off scripts superseded by tests, docs, or maintained tools (#3951)
aa769532 chore: purge stale benchmark data (158 unreferenced result files) (#3949)
c2605c19 fix(security): pin SA3 weight downloads to reviewed upstream commit (#3946)
28348b1c chore: remove verified dead code (orphan modules, unreferenced functions) (#3942)
b6359ae4 fix(release): reject direct dependency references (#3937)
be745a7c fix(ci): bound managed GUI shard lifetime (#3941)
cd98bc45 test(mac): derive subprocess timeout ceiling (#3939)
249b9bd0 fix: cap qwen27 accelerated admission at one (#3934)
b6759640 feat(mac): productize qualified Qwen 27B acceleration (#3930)
436b5d53 feat: add qualified Qwen3.8-27B accelerated text profile (#3929)
ec498685 perf(kernels): integrate opt-in row-invariant lane matmul (#3931)
b309f263 fix(mac): complete Share Compute live acceptance and keychain copy (#3927)
8a9e46fc feat(mac): integrate Share Compute Desktop experience (#3923)
e9f4a899 docs(readme): link the /compare head-to-head page (#3921)
fed0eb98 feat(cua): integrate supervised computer use for 0.15.x (#3909)
e07e625b fix(server): preserve startup failure exits with uvicorn 0.42 (#3917)
778781e5 test(desktop): make PDF OCR attachment tests deterministic (#3912)
1987cfd2 fix(desktop): bypass OCR for blank PDF pages (#3911)
d7f864c0 fix(api): accept strict structured output with stream/tools; clamp completion budget to context (#3871)
dd75aa45 fix(serve): route curated repo ids like their alias; classify missing local paths (#3863)
f37ee8f2 feat(cli): first-run welcome for bare rapid-mlx (#3861)
ef8d5e50 fix(serve): make missing optional extras recoverable for every install method (#3831)
adffa4eb feat(mac): anonymous first-run funnel milestones (#3829)
1ee78feb Cap QuickSilver prompt admission before prefill (#3835)
be6dae55 fix(cache): anchor hybrid checkpoints at message boundaries (#3836)
44ae0f65 fix(telemetry): reject public proof through network overrides (#3834)
6be91bab fix(cua): accept cloud brain base URLs from settings (#3822)
88770a19 feat(cua): user-configurable cloud brains (settings UI + consent + degradation) (#3820)
a1bd4782 feat: effective runtime config consumers + resolver parity (#3810)
cf1051c9 Fix Bonsai 2 runtime profile contract (#3809)
b42ae82c Native agent-task panel for the CUA server API (#3807)
ae312bd5 Server CUA API: create/monitor/approve/cancel computer-use runs (#3805)
70b38ae5 CUA decision records and handoff docs (#3803)
964c22ff POC reference tools: GUI-verifier flows and flow tooling (#3804)
dcd617d0 CUA agent loop: local fast thinking, user-configurable slow thinking (#3801)
456e7ecb Computer-use tool layer: native AX execution (model-agnostic) (#3800)
bd9705af fix: release allocator memory before low-memory workloads (#3798)
1394f169 fix(cache): keep agent-session prefix resident on 16-32 GB Macs (#3794)
f754ff05 fix(reasoning): apply user stop= only to the answer on <think> models (#3792)
8058596b test: prove effective runtime resolver parity (#3771)
b4d906e1 docs: one canonical project sentence, sourced speed claim, rapidmlx.com metadata (#3779)
1879f6da test(telemetry): make bench served event deterministic (#3711)
8bae4cdf dev: one local verification command from report or diff to evidence (#3769)
bc4a0dcf feat: run Qwen Image 2.1 on low-memory Macs (#3762)
0b22e92a feat: define effective runtime config contract (#3770)
cb377f3f feat(telemetry): classify failed inferences and fix JS caller buckets (#3773)
380b08e3 Deduplicate repeated model serve failure telemetry (#3763)
dffe05fa fix: explain why image input is rejected and what to serve instead (#3772)
c327ec95 feat: add telemetry capability context (#3765)
f66620a1 fix: guide non-interactive optional extra installs (#3760)
9d5132c1 fix: harden engine-start classification boundaries (#3759)
68486bd0 feat: persist crash diagnostics and unterminated serve state (#3723)
6352c6cc feat: classify engine-start load failures (#3748)
b1eeb2c1 CLI serve auto-falls back to a free port (#3724)
10db2e55 feat(server): add qualified LFM2.5-VL DSpark companion (#3741)
8ea43247 serve: render typed Hugging Face Hub errors (#3725)
7df85408 feat(cli): offer to install missing optional extras (#3726)
87e3d880 feat(mac): make Agent connection setup obvious (#3739)
d6d8cfc0 chore(deps): align mlx-vlm 0.7.2 coherence (#3740)
887ec6f8 feat(server): add Laya and CLM System One support (#3728)
bc3ccb4a fix(reasoning): GLM-5.3 effort coercion detection + serve --default-reasoning-effort (#3714) (#3718)
a0bcc7e3 feat(mirror): track intentionally unmirrored repositories (#3727)
2b996f7c feat(qwen4): load validated PLE rows from a bounded sidecar (#3729)
68f610dd test(telemetry): deflake bench model served event (#3731)
0c049254 feat(share): serve glm-5.3-flash in the QuickSilver pool (#3704)
99892cf4 fix(ci): keep fresh-install golden flow independent of the live update manifest (#3720)
dd405a27 fix(qwen4): normalize direct-gamma RMSNorm checkpoints (#3713)
fd1bbb25 fix(telemetry): reserve served lane before dispatch (#3715)
21c545e0 feat(serve): classify missing optional extras as one actionable failure (#3697)
4bcdf8e0 fix(models): match Cohere2-MoE reference RoPE and norms (#3705)
200076da feat(mirror): detect mirror drift against HuggingFace (#3692)
6999529b feat(telemetry): add server_start_state event (#3693)
49b1b94e fix(telemetry): emit model_served for every serving lane (#3687)
2420c924 fix(desktop): show the real reason when the engine cannot start (#3690)
24ce8f03 fix(serve): print the ready banner only after the listener binds (#3689)
03de3c50 fix(telemetry): classify serve failures through the exception chain (#3686)
19417128 fix(telemetry): seed first_run_date from pre-existing install evidence (#3682)
de1c696d fix(telemetry): report surface=desktop for sidecar-served events (#3681)
ec98a7ba refactor(mllm): vendor the speculative-decode coordinator core (step 3b) (#3578)
b1df2cad refactor(mllm): vendor the text-AR generation core (step 3a) (#3575)
87a8744e feat(mac): telemetry v2 launch notice, merging consent writer and sidecar role (desktop lite) (#3672)
f7ca3a6d feat(telemetry): inference bucket, active-day and capability-rejected events (#3665)
4ef0c5ec feat(telemetry): retire the v1 collector wire; v2 status/preview and on/off/reset-id verbs (#3670)
781fecb3 feat(telemetry): model pull and serve events (#3664)
4e5a9f80 feat(telemetry): agent setup and consent-change events (#3663)
4fb71ed6 fix(image): expose Qwen-Image 2.1 editing in Desktop catalog (#3667)
d6fbc70c fix(telemetry): escape surrogate example for Python 3.14 (#3666)
e0f1eac3 feat(telemetry): v2 track() helper and lifecycle events (#3646)
06bba007 fix(image): accept shared HF blob cache for Qwen 2.1 (#3659)
bff7b55d feat(models): add Xiaomi MiMo-V2.6 Flash alias with Desktop tool-use support (#3658)
0cd131ba feat(image): support Qwen-Image 2.1 in Server and Rapid Mac (#3652)
0ca1e4cc docs: surface Community Benchmark submission in README (#3655)
c14ccd7f fix(desktop): stamp the bundled engine in official Desktop builds so telemetry v2 can transmit (#3647)
b49fb887 fix(image): reject Qwen-Image 2.x before 1.x dispatch (#3649)
6a98c5c4 docs(benchmarks): accept M5 Air community result (#3650)
5025a3b3 feat(release): write and verify the telemetry v2 release stamp in the publish workflow (#3639)
19626fee feat(telemetry): wire the v2 consent decision into CLI and headless server startup (#3640)
a24c9e8a feat(telemetry): PostHog sender with burst caps, gated on official builds (#3637)
375c917b feat(telemetry): build the v2 common-properties block; make the cohort stamp optional (#3635)
73faeae5 feat(telemetry): the default-on consent decision table as one pure function (#3633)
7336e7c3 feat(telemetry): transmit only from official release builds (#3628)
7388a592 feat(telemetry): build PostHog batch items from validated registry events (#3627)
42342695 feat(telemetry): map the chip brand string onto the closed registry enum (#3626)
a336d39e fix(ci): keep stale-ready comments advisory (#3621)
95cd87ae fix(telemetry): bound the public-repo proof and capture model identity at request start (#3606)
66e89385 feat(telemetry): cross-process SQLite state store (v2 block 3) (#3597)
955233de fix(mac): retire idle animation loop, keep straight quotes, gate tool caption on roster, single instance (#3568)
9a676d64 fix(engine,mac): carry stable error codes for the remaining major server failures (#3571)
4d5fb3bb fix(telemetry): close three review findings on the registry and feedback command (#3604)
c776acb9 test(mac): wait for complete star helper PID record (#3615)
48e333ab fix(mirror): report warm HF relinks as cached (#3612)
c9977b8e fix(mtp): preserve sliding-window rollback (#3609)
14540529 feat(mtp): qualify explicit Gemma 4 assistant sidecars (#3596)
e63f60ca feat: `rapid-mlx feedback` + desktop Help item → community Discord (#3598)
1543465e feat(telemetry): privacy-safe telemetry_model_id + X-Rapid-Client (v2 block 2) (#3600)
ec3f72a9 telemetry(v2): shared event registry + strict validators (block 1) (#3599)
9189277a test(mac): recreate drag source before bounded retry (#3594)
acae0bff ci(mac): build GUI artifact on Manzanita (#3591)
06c85973 fix(ci): validate installed test dependency versions (#3589)
b5da1294 fix(ci): surface stale merge-ready authorization (#3586)
8074e1a7 fix(mac,community): keep the run progress bar clear of the mascot; dogfood hardening (#3574)
17cfbb0e refactor(mllm): vendor the prepare_inputs surface (step 2c) (#3582)
5ea04ed5 refactor(mllm): dual-namespace recognition for the vendored APC engine (step 2b-3) (#3563)
d4be0a2b refactor(mllm): vendor the mlx-vlm APC engine (step 2b-2) (#3558)
b0be8995 refactor(mllm): vendor the mlx-vlm APC support surface (step 2b-1) (#3554)
fb273ca2 refactor(mllm): vendor mlx-vlm cache types as the phase-B types+seam slice (step 2a) (#3545)
3123b05b refactor(mllm): deprecate the legacy mlx-vlm generation surface (#3534)
e1bd4390 fix(mllm,mac): faithfully reflect server-side engine errors to the GUI (#3564) (#3565)
023b906c test(mac): refresh update-state personal intelligence baseline (#3561)
2c953e33 fix(desktop,agent): harden Personal Intelligence and Community Benchmark dogfood (#3550)
e16c575c feat(models): productize Bonsai 2 Hadamard — alias, catalog, GUI image support + lane guard (#3555)
a93a484a feat(models): support Bonsai 2 Hadamard MLX packs (#3556)
65d4a677 feat(mac): let Personal Intelligence work with local files and code (#3538)
f79a1f61 fix(rapid-mac): ship a flat-plist DMG Finder layout so the install window opens (#3468) (#3543)
318b86c6 feat(benchmark): add community benchmark experience (#3544)
e1144388 fix(eval): reground conv-10/conv-16 media-prefix prompts with their tightened checkers (#3541)
8f4a184c perf(mllm): media-aware prior-turn boundary cache on the serialized lane (#3505)
cf2585d0 refactor(mac): centralize audio readiness state (#2982)
b72f11bb fix(speculative): bound GLM native MTP prefill memory (#3520)
9f311eb6 perf(mllm): skip singleton cache repack on the serialized lane (#3502)
62c4041a fix(rapid-mac): pin the continuation type in IPPinnedHTTPTransport (#3531)
b4a40bb1 test(rapid-mac): refresh chat-depth AX baselines for residency redesign + PI toggle (#3530)
a68134d7 ci: remove one-off #3495 repro workflow (#3529)
ee7b0400 fix(cli): skip interpreter finalization after graceful serve shutdown (#3495) (#3525)
53a329d9 ci: one-off macOS 15 repro workflow for #3495 finalization race (#3528)
86dfe13f feat(agent): qualify remaining personal intelligence builds (#3523)
813a2f6c fix(mllm): honor seed/top_k/min_p on the media lane (#3496)
1e3cda7c refactor(packaging): rename Python package to rapid_mlx with compatibility shim (#3517)
0e3e3e80 fix(mac): hop to the main actor in openURL completions (#3521)
54c72e43 feat(macos): introduce Personal Intelligence control (#3490)
2c157f10 fix(engine): load heterogeneous DeepSeek V4 quantization (#3513)
c75c5498 fix(benchmark): support DeepSeek V4.1 serial runtime (#3508)
2020b3c6 ci: remove subscription-gated queue scopes (#3514)
36d650bd ci: switch Mergify to immediate singleton queues (#3510)
f9823948 fix(benchmark): route architecture-owned text models (#3506)
87c164ab docs(models): correct K2 Horizon throughput qualification (#3493)
132e4fee feat(models): expose experimental K2 Horizon 7B (#3486)
a76e5c98 fix(agent): polish compact-model task UX (#3484)
9567a507 feat(models): add Rapid-native K2 Horizon runtime (#3483)
c651acff fix(mac): apply GLM native MTP sampling profile (#3487)
9b0ab0d3 perf(glm5): activate owned MTP on tagged runtime (#3481)
84c8074c feat(desktop): add opt-in Agent mode to Chat (#3480)
6f14994d feat(desktop): own agent session lifecycle (#3474)
b403ebff fix(mllm): promote an exact entry only after its snapshot served (#3476)
7e93964c fix(chat_template): decide live tool-loop rows the way Qwen3.5's template does (#3475)
2a1f2e34 feat(desktop): add agent runtime transport (#3470)
0ffaa1f7 docs(glm5): record release activation boundary (#3472)
332c2628 perf(template): keep Qwen3.5 tool-loop think blocks stable across turns (#3464)
c547059e perf(mllm): resume Qwen3.5 hybrid prompts from recurrent-state checkpoints (#3463)
04db4b8c feat(agent): add bounded server adapter (#3456)
3e1e380a perf(glm5): activate qualified native MTP pair (#3467)
d45c58e8 perf(glm5): own cache-safe native MTP transactions (#3462)
12fbee03 fix(server): honor thinking budgets in serial speculative paths (#3457)
9b6a5ef0 fix(mlx): install stream shim before compiled decode imports (#3458)
```

## 實測前確認的支援邊界

- `Rapid-MLX/rapid_mlx/speculative/tensorfold_qwen27.py:21` 綁TensorFold0.5.0／commit9cd52ab4／MLX0.32.3，`:32` target是Vontra/Qwen3.8-27B-MLX-4bit，validate_pair拒絕其他target/drafter revisions。`:77`拒tools、grammar、response_format、media。Jundot oQ4e不是合法arm，不下載另一target偷換same-checkpoint比較。
- 一般local path沒有alias sidecar preset（`Rapid-MLX/rapid_mlx/cli.py:3539`）。`{"method":"mtp"}`在本checkpoint首次啟動rc3，source/precedingwarning表示sidecar=None拒random-init。這是配置失敗，不直接當成MTP速度結果。
- 更深入CPU檢查：原始head有29tensors、都在model-00004-of-00004；`fc.weight`是BF16 [5120,10240]，decoder q_proj.weight是U32 [12288,640]／BF16 scales [12288,80]。`Rapid-MLX/rapid_mlx/spec_decode/mtp/qwen3_5_inject.py:394`只找standalone named files，沒有indexed-shard reader；`:656`明示fc packing統一套全head，BF16 fc＋量化decoder的mixed layout會refuse。不能以uniform量化改寫原checkpoint來稱lossless。
- 為確認格式，曾以safetensors header＋raw byte slices提取299.7MiB head；29個tensor逐一SHA256完全相同，manifest retained（native-head/manifest.json），臨時head已刪除。這不是engine port，沒有更改target或tensor數值；明知mixed layout拒絕後不提交其GPU sweep。
- 本checkpoint的Rapid可量測模式是普通AR：default auto lane及documented `--no-mllm --enable-auto-tool-choice --tool-call-parser hermes --hybrid-cache-entries 8 --no-spec-decode` text lane。Yunshu default實際log為checkpoint MTP、block6、copy_rows16、lane kernels；沒有DFlash隱藏arm。
- Tiny上Rapid auto lane及27B default會走serialized hybrid MLLM compatibility，不等於所有hybrid checkpoint都能8-row continuous batch。source gate與actual mode必須分開。
- `.github/workflows/pages.yml:1` 只把release tag的install.sh同步GitHub Pages／Cloudflare代理，並非完整docs site build；docs內容與網站的其他build實作不能混為同一repo已驗證能力。

## 本線已提交與CPU驗證

| Commit | 變更 | 驗證 |
|---|---|---|
|24efa51a|serve occupied TCP早退，actionable alternate port|8526 unit PASS、20 skipped；focused30|
|f9f7dda5|chat invocation隔離，/clear保留system|8529 unit PASS、20 skipped；focused8|
|27b93405|first-run guide與README/docs入口|9 CLI help surfaces及relative links PASS；docs-only沿用8529 unit|
|40013bb4|HTTP comparison、raw request/SSE、memory、shape graders|8537 unit PASS、20 skipped；harness8|
|dfafc2ae|逐一檢查全部resolved bind addresses|focused14；merged gate8621 PASS|
|695cc46e|real census replay、JSON integral-number grading|focused14；merged gate8621 PASS|
|d6447610|僅一次main refresh，納入最新tool-loop／release evidence|8621 unit PASS、20 skipped；ruff/mypy PASS|
|57dac02f|startup readiness與TTFT分列，profiles共用prompt identity|8621 unit PASS、20 skipped|
|1ecfa2ff|serve MODEL，options可前後放，-m保留且拒絕衝突|8625 unit PASS、20 skipped；focused35|
|1ceab3eb|unique result stem namespace避免raw artifacts互覆|8626 unit PASS、20 skipped；harness9|

另：Rapid的tool_calling＋forced-call repair＋Hermes regression83 PASS；完整tests/parsers114 PASS／6 XFAIL。XFAIL包含Gemma bare-word compound與legacy Harmony routers的known gaps；openai-harmony production router有獨立passing regression，不能把legacy XFAIL當成production全部壞，也不能把XFAIL算PASS。

未新增YUNSHU_*設定／experimental flags，未更改serving kernels／sampling／APC arithmetic，未push或merge main，也沒有任何external issue/PR/comment／benchmark share。uv wheel install在reference checkout產生的internal build/已刪除；測量runtime、uv/HF cache、scratch及raw artifacts在P5Plus。產品bounded APC維持internal default。

## GPU receipts（工作中，不使用smoke數字宣稱性能）

| Job | 結果／目的 |
|---|---|
|1003-210646-00-rapidmlx-tiny-2108|rc1、complete/failures1；harness錯用無__main__的package入口，已修production rapid_mlx.cli；Yunshu tiny各case成功|
|1003-211728-00-rapidmlx-tiny-cli-2118|cancel=true，錯誤版本未再消耗GPU|
|1003-211758-00-rapidmlx-tiny-entry-2119|rc0、complete/failures0，兩engine smoke；CPU clean|
|1003-213644-00-rapidmlx-27b-mtp-smoke-2137|rc1、complete/failures1；Rapid缺sidecar明確拒絕，Yunshu27B MTP成功；CPU clean|
|1003-220555-00-rapidmlx-27b-ar-default-smoke-2207|rc0、complete/failures0；Rapid AR/default兩arm成功；CPU clean|
|1003-220835-00-rapidmlx-27b-mtp-local-head-2211|cancel=true；CPU查明indexed/mixed head不符injector契約，避免已知必敗job|
|1003-223727-00-rapidmlx-27b-matrix-r3-2237|p0 --quiet；3reps交錯Rapid default/text AR/Yunshu default、1K/8K/32K、256tokens、cold/warm/turn2/8way/physical peak＋Rapid31tools＋own6shapes＋real8request census；等待收割|

Smoke不是timing：1rep、32tokens、未quiet，不報性能勝負。早期standalone smoke的raw arm directory只按engine/rep/size命名，後一次smoke覆蓋了同名log/SSE；其JSONL及gpuq log完整保留，修補1ceab3eb已加result stem隔離，正式matrix使用獨立profile/result namespace，不受此錯誤影響。

## 重現正式矩陣

實際發出的job使用P5Plus/frozen wrapper；repo等價wrapper（已移除CPU確認不支援的MTP profile）為：

```bash
GPUQ_OWNER=rapidmlx /Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq submit \
  --quiet --priority 0 --mem-gb 64 --timeout 150 --label rapidmlx-NEW-UNIQUE-LABEL \
  --out /Volumes/P5Plus/yunshu-build/codex/rapidmlx/NEW/results.jsonl --expect-complete -- \
  /Volumes/P5Plus/yunshu-test-envs/rapid-mlx/.venv/bin/python scripts/research/rapidmlx/matrix.py \
  --model /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp \
  --output /Volumes/P5Plus/yunshu-build/codex/rapidmlx/NEW/results.jsonl \
  --reps 3 --sizes 1024 8192 32768 --tokens 256 --profiles rapid-default rapid-ar yunshu-default --tool-eval
```

結果收割需job rc0、output存在、final complete/failures0且quiet contention乾淨；`summarize.py`只對≥3reps產生median，decode還需每次≥64completion tokens。actual prompt_tokens另記，不把synthetic content 1K假稱整個chat template恰1K。cold/warm同body且profiles共用nonce；turn2接各engine實際前回答，因此若前回答不同，其turn2body也會不同，不能隱藏此限制。8wayprefix每row不同，aggregate decode與aggregate end-to-end分列。model-ready startup另報；OS filesystem cache沒有強制cold。

CPU最終gate：8630 passed、20 skipped、12 warnings；ruff check/format、mypy baseline gate PASS。serve --json errors與repo matrix/summarizer另有獨立commit，正式timing待收割。
