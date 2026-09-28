<div align="center">

# Yunshu

**為 Apple Silicon 打造的快速本地 LLM / VLM 推理引擎。**

單一程序,相容 OpenAI 與 Anthropic API,透過 MLX 在裝置端執行。為單機低延遲而設計:首 token 快、
無損解碼快,並以前綴重用跳過已經算過的部分。第一個完整調校的模型是 **Qwen3.8-27B**。

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/YuhuanStudio/Yunshu/actions/workflows/ci.yml)

[English](./README.md) · [简体中文](./README.zh-CN.md) · **繁體中文**

</div>

---

## 與眾不同之處

- **無損推測解碼。** 使用 checkpoint 自帶的 MTP 頭,或外部 DFlash drafter。會推測的請求,其所有
  解碼與驗證矩陣乘都走同一顆 batch-invariant kernel,所以 greedy 下開推測與不開推測的輸出逐 token
  相同 —— 即 Splash 所說的無損。
- **並發請求各列獨立的 KV。** 共用批次裡每一列保有自己的 KV 長度,短請求不會讀到長請求的補齊
  (Qwen3.5 家族;8 路並發的 MMLU-Pro 88 → 139 tok/s,131K 上下文解碼 24 → 47 tok/s)。
- **有損的只在你要求時才開。** 所有預設都是無損的。會改變輸出的省記憶體選項(int8 KV、KV 量化、
  4-bit 快取前綴、int8 SSD 快取)都是需要手動開啟的設定。
- **混合架構模型的前綴快取。** Qwen3.5 家族把注意力層和遞迴的 GatedDeltaNet 層混在一起,一般的
  KV 快取切不開。Yunshu 保存精確的 checkpoint,以文字與圖片像素共同作為鍵,預設 8 GiB 記憶體,
  可再加一層 SSD。重複或只改尾巴的長 prompt 不必重新 prefill。
- **以輸出驗證過的驗證 kernel。** GatedDeltaNet、注意力、5-bit 矩陣乘的驗證 kernel,部分取自
  oMLX,每一顆都經過同 checkpoint A/B 才採用。
- **快速路徑上有完整 API。** 工具呼叫、JSON-schema 約束、停止序列、logprobs、串流推理/內容分離、
  `reasoning_effort` 直接傳給支援它的 chat template(Qwen3.8),以及客戶端斷線時取消生成。

## 快速開始

需要 Apple Silicon 的 Mac(macOS 14 以上)與 [uv](https://docs.astral.sh/uv/)。

```bash
# 安裝。vision extra 涵蓋 Qwen3.5 / 3.6 / 3.8 系列和所有 VLM。
uv tool install "yunshu[vision]"

yunshu doctor                                   # 檢查這台 Mac,並列出修正方法
yunshu pull mlx-community/Qwen3.5-9B-MLX-4bit   # 下載到 ~/.yunshu/models/
yunshu serve -m mlx-community/Qwen3.5-9B-MLX-4bit
```

其他安裝方式:`pipx install "yunshu[vision]"`、Homebrew(`brew install yuhuanstudio/tap/yunshu`),
或最新的 `main`(`uv tool install "yunshu[vision] @ git+https://github.com/YuhuanStudio/Yunshu"`)。

`yunshu serve -m org/name` 會直接使用模型目錄或 Hugging Face 快取裡已有的模型,兩邊都沒有才下載。
模型預設放在 `~/.yunshu/models`;要放到別處,執行 `yunshu config set models_dir /path/to/models`
(存在 `~/.yunshu/config.toml`)。

伺服器監聽 `http://127.0.0.1:8000`,任何 OpenAI 用戶端都能直接使用:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")  # 任意 key 都可以

# 單模型模式:模型名稱只是佔位,伺服器提供的是你載入的那個模型。
r = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "用一句話解釋 MLX。"}],
    extra_body={"reasoning_effort": "medium"},
)
print(r.choices[0].message.content)
```

要在登入時於背景執行:`yunshu service install -m <model>`
([服務指南](docs/guides/SERVICE.md))。每個指令都有 `--help`。`yunshu model list`
會列出本機模型,包含 Hugging Face 快取。

**從原始碼**(開發用):複製儲存庫,執行 `uv sync --extra vision`(或 `--all-extras`),
然後 `uv run yunshu serve -m <model>`。`uv.lock` 固定了確切版本(MLX 0.32、`mlx-vlm` 0.7.3+)。

**不收集遙測。** Yunshu 不會把任何資料送到任何地方。唯一的對外連線是你要求的模型下載,以及你設定的
MCP 伺服器。

**文件:**
- [連接用戶端](docs/guides/CLIENTS.md)(OpenAI / Anthropic SDK、程式代理、Open WebUI)
- [疑難排解](docs/guides/TROUBLESHOOTING.md)
- [API 參考](docs/API.md)
- [設定參考](docs/CONFIGURATION.md)

## 效能

在 M5 Max(128 GB)上以 Qwen3.8-27B 量測,2026-09-28/29。除非另有註明,都用同一個 Jundot
`oQ4e-mtp` checkpoint。各表的量測方法與腳本見
[docs/BENCHMARKS.md](docs/BENCHMARKS.md);原始數據由維護者保存,不隨 repo 發布。

| 引擎 | 能力檢查 | 對話 TTFT(熱) | 8K prompt:冷 / 重複 / 改尾巴 | 解碼 tok/s |
|---|---|---|---|---|
| **Yunshu 0.1.1**(預設:MTP block 6、batch-invariant、ragged KV) | 34/34 | 0.192 s | 8.42 / 0.115 / 0.259 s | 73 |
| mlx-vlm 0.7.3 server(APC) | 27/28 | 0.212 s | 8.60 / 0.108 / 0.265 s | 32 |
| oMLX.app 0.7(MTP + 快取) | 31/31 | 0.312 s | 8.60 / 0.361 / 0.376 s | 85 |
| Splash 1.1(自家量化模型 + DFlash2) | 31/31 | 0.206 s | 7.88 / 0.131 / 7.88 s | 119 |
| TensorFold 0.3.6.1(MTP,parallel 8) | 23/34 | — | — | 28 |

Yunshu 的檢查項比舊的量測多(logprobs、串流推理分離);TensorFold 在圖片、工具、JSON schema、
logprobs 幾項沒有通過。

單一請求的無損解碼,依輸出類型(同一 checkpoint、程式內、greedy、384 token;tok/s):

| 上下文 | 程式碼 | 散文 | 類 JSON | 開推測 == 關推測 |
|---|---|---|---|---|
| 1K | 82.1 | 57.8 | 69.0 | 是(已測)¹ |
| 32K | 75.4 | 51.2 | 64.2 | 是(已測)¹ |
| 131K | 59.7 | 43.8 | 46.0 | 是(已測)¹ |

¹ 上面每個上下文、每個任務,開推測與不開推測的 greedy 輸出都逐 token 相同。矩陣乘與列數無關;
解碼與驗證的注意力走同一顆逐列 kernel,一個 token 的結果不會因為一起驗證的 token 數而改變。

MMLU-Pro,300 題,8 路並行,上限 16384 token,`reasoning_effort=medium`(同時檢驗準確率與長時間
穩定性;所有引擎設定相同):

| 引擎 | 答對 | 耗時 | 總吞吐 tok/s | 記憶體峰值 |
|---|---|---|---|---|
| **Yunshu**(ragged KV) | 249 / 300 | 28.3 分 | 139 | 33.7 GiB |
| Yunshu 0.1.0 時期的共用批次(補齊 KV) | 250 / 300 | 46.5 分 | 88 | 45 GiB |
| TensorFold 0.3.6.1(MTP,parallel 8) | 250 / 300 | 24.9 分 | 159 | 35.1 GiB |
| Splash 1.1 | 252 / 300 | 17.2 分 | 223 | 67 GiB |
| oMLX.app | 229 / 300(27 題被它的 prefill 記憶體保護拒絕) | 29.2 分 | 120 | 75 GiB |

開啟 `YUNSHU_KV_PRECISION=int8`(有損,需手動開啟)時 Yunshu 答對 251 / 300,峰值 28.1 GiB
(在較早版本的 ragged 快取上量測,34.6 分)。

速度掃描(每個 prompt 都不同,不命中快取;生成 128 token;未註明單位者為 tok/s):

| | Yunshu | oMLX | Splash | TensorFold(MTP) |
|---|---|---|---|---|
| 8K / 131K token 的 TTFT | 8.6 / 207 s | 8.5 / 214 s | 7.9 / 207 s | 9.8 / 293 s |
| 1K / 32K / 131K 之後的解碼 | 59² / 59 / 47 | 71 / 60 / 38 | 101 / 48 / 68 | 26 / 57 / 19 |
| 8 個 1K prompt 並發,總吞吐 | 64 | 53 | 70 | 65 |

² 單一請求的解碼速度取決於草稿被接受多少,會隨 prompt 變動;Yunshu 的 1K 數字是 8 次的平均
(單次介於 40–70)。其他格都是單次量測。

Yunshu 目前的位置:
- 前綴重用與熱 TTFT 是量到最好的;冷 prefill 已達硬體上限(各引擎相差約 10% 以內)。
- 準確率與其他引擎相當;每次長時間執行都是 0 錯誤。
- **落後 Splash** 的地方:長上下文解碼(131K:47 vs 68 tok/s)與並發長輸出(MMLU-Pro:139 vs
  223 tok/s)。TensorFold 在後者也領先(159),因為它每一列都起草;Yunshu 只在請求單獨執行時起草。
  多列推測解碼開發中(`YUNSHU_ROUND_DRIVER`,實驗性)。
- 其他請求解碼時若有長 prompt 進來,prefill 期間其他請求的解碼會停住;量過的每個引擎都是如此。
- 2026-09-28 版本的 60 分鐘混合壓測(對話、長文件、圖片、工具、JSON schema、思考、斷線)完成
  699 個請求,伺服器 0 錯誤,記憶體沒有成長(17–26 GiB)。

## 支援的模型

| 層級 | 模型 | 路徑 | 可得到的功能 |
|---|---|---|---|
| 1 —— 已調校並量測 | Qwen3.5 / 3.6 / 3.8 家族(文字 + 圖片) | VLM batch runner | 前綴快取(記憶體 + SSD)、MTP / DFlash 無損推測解碼、上述所有 API 功能 |
| 2 —— 支援 | 任何 `mlx-lm` 文字模型 | 單請求快速路徑(`generate_step`) | KV 前綴快取、工具、JSON schema、logprobs;可開啟 n-gram 推測、KV 量化 |
| 2 —— 支援 | 其他 `mlx-vlm` 模型(GLM、Qwen-VL、Gemma-4、Qwen3-Omni、Nemotron-Omni 等) | 同一個 VLM batch runner | 連續批次、前綴快取(使用 sliding window 的模型除外)、圖片 / 音訊 / 影片、上述所有 API 功能;無推測解碼 |

2026-09-28 這一輪只重新量測了第 1 層;第 2 層的 VLM 之後才改走 runner,仍需實機冒煙測試。

## 其他模態

以下功能在同一個伺服器裡,透過可選 extras 安裝。**2026-09-28 這一輪都沒有重新驗證**,
這一輪只涵蓋 LLM/VLM。

| 模態 | 端點 | 後端 | Extra |
|---|---|---|---|
| 原生語音對語音(Qwen3-Omni Thinker→Talker,串流) | `POST /v1/omni/speech/stream` | `mlx-vlm` | `omni` |
| 即時語音 | `WS /v1/realtime` | omni,或 ASR → LLM → TTS | `audio` |
| ASR | `/v1/audio/transcriptions` | `mlx-audio` / Whisper | `audio` |
| TTS | `/v1/audio/speech` | `mlx-audio` | `audio` |
| 圖像生成 | `/v1/images/generations` | 擴散模型 | `generation` |
| 影片生成(Wan 2.x / LTX-2) | `/v1/video/generations` | `mlx-video` | `video` |
| Embeddings / rerank(文字 + 多模態) | `/v1/embeddings`、`/v1/rerank` | `mlx-lm` / `mlx-embeddings` | `embeddings` |

語音對語音:服務一個 Qwen3-Omni 模型(`uv sync --extra omni`),試試
[`examples/talk.py`](examples/talk.py)(麥克風)或 [`examples/quickstart.py`](examples/quickstart.py)
(輸出 WAV,不需音訊硬體)。已確認上游 `mlx-vlm` 0.7.3 多輪 omni 輸出正確
(維護者於 2026-09-28 確認);伺服器的 Realtime 路徑尚未確認。

影片生成需要從 git 安裝 `mlx-video`:PyPI 上的版本(0.1.0)只有前處理,所以從 PyPI 裝
`yunshu[video]` 會得到不完整的後端,`yunshu[all]` 也不含影片。要加上它:
`uv tool install "yunshu[vision]" --with "mlx-video @ git+https://github.com/Blaizzy/mlx-video.git"`
(原始碼 checkout 用 `uv sync --extra video` 即可)。

另外:MCP 伺服器/客戶端,以及相容 Anthropic 的 `/v1/messages`。

## 架構

```
  客戶端(任意 OpenAI / Anthropic SDK)
        │   OpenAI / Anthropic / MCP / Realtime-WS / SSE
  ┌─────┴───────────────────────────────────────────────┐
  │  閘道(FastAPI)       路由 + 中介層                   │
  ├─────────────────────────────────────────────────────┤
  │  引擎                                                 │
  │   · VLM batch runner(所有 mlx-vlm 模型)             │
  │       連續批次 · 前綴快取(記憶體 + SSD)             │
  │       Qwen3.5 家族:MTP / DFlash + batch-invariant    │
  │   · LLM 快速路徑(mlx-lm generate_step)              │
  │       KV 前綴快取 · 約束解碼                          │
  │   · 其他模態:omni、ASR/TTS、圖像、影片、embeddings   │
  └─────────────────────────────────────────────────────┘
        單一 MLX 執行緒 · 經由 Apple MLX 在裝置端執行
```

## 服務模型

所有 GPU 工作都在一條 MLX 執行緒上。VLM(mlx-vlm)模型的並行請求共用一個連續批次,每列有自己的
取樣設定;單獨一個請求時會使用推測解碼(Qwen3.5 家族),期間進來的請求則加入共用批次、不做推測。
純文字的 mlx-lm 模型走單請求快速路徑,並行請求會依序執行。每個回應在它自己的生成結束時就立即返回。

## 設定

所有設定都是 [設定參考](docs/CONFIGURATION.md) 列出的 `YUNSHU_*` 名稱(由同一份登錄表產生)。
可用環境變數、TOML 檔(`yunshu serve --config yunshu.toml`)或 `yunshu serve --set KEY=VALUE`
設定;`yunshu config` 顯示每項的生效值與來源。值無法解析會中止啟動,拼錯的名稱會收到警告。

## 建構於

[MLX](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) ·
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) · [mlx-audio](https://github.com/Blaizzy/mlx-audio)。
部分驗證 kernel 取自 [oMLX](https://github.com/jundot/omlx)(Apache-2.0),見
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 授權

Apache 2.0 —— 見 [LICENSE](LICENSE)。
