"""Generate the configuration guide from the settings registry (CPU only).

Same source as docs/CONFIGURATION.md (scripts/gen_config_docs.py), rendered as three MDX pages:
site/content/docs/guides/configuration{,.zh-TW,.zh-CN}.mdx. The output is gitignored; run
`just docs-gen` (also part of `just docs-build`).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
OUT = ROOT / "site" / "content" / "docs" / "guides"

spec = importlib.util.spec_from_file_location("gen_config_docs", ROOT / "scripts" / "gen_config_docs.py")
assert spec and spec.loader
gcd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gcd)

HEADINGS = {
    "## Stable settings": ("## Stable settings", "## 穩定設定", "## 稳定设置"),
    "## Experimental settings": ("## Experimental settings", "## 實驗性設定", "## 实验性设置"),
    "## Internal settings": ("## Internal settings", "## 內部設定", "## 内部设置"),
}
HEADERS = {
    "| Setting | Type | Default | Description |": (
        "| Setting | Type | Default | Description |",
        "| 設定 | 型別 | 預設值 | 說明 |",
        "| 设置 | 类型 | 默认值 | 说明 |",
    ),
    "| Setting | Type | Default | Description | Decided by | Added |": (
        "| Setting | Type | Default | Description | Decided by | Added |",
        "| 設定 | 型別 | 預設值 | 說明 | 決定依據 | 加入日期 |",
        "| 设置 | 类型 | 默认值 | 说明 | 决定依据 | 加入日期 |",
    ),
}
NOTE = (
    "The descriptions below come straight from the settings registry and are in English in every language.",
    "下表的說明直接來自設定登錄檔，所有語言都以英文呈現。",
    "下表的说明直接来自设置注册表，所有语言都以英文呈现。",
)
INTRO_ZH = {
    1: """\
每個 Yunshu 設定都是登錄在 `python/yunshu_engine/settings.py` 的 `YUNSHU_*` 名稱，各有唯一的型別、預設值與解析規則。
除此之外沒有任何程式碼會讀取 `YUNSHU_*` 環境變數。

## 設定值的來源

優先順序由高到低：

1. `yunshu serve` 的旗標與 `--set KEY=VALUE`（可重複；`KEY` 可省略 `YUNSHU_` 前綴，例如 `--set mtp_block_size=4`），
2. 行程的環境變數，
3. TOML 設定檔：`yunshu serve --config yunshu.toml` 或 `YUNSHU_CONFIG=yunshu.toml`，
4. 下表列出的預設值。

設定檔是一組 `KEY = value`，可以平放，也可以放在任意表格（table）之下（表格名稱會被忽略，鍵可用短名或全名）：

```toml
model = "~/models/Qwen3.8-27B-4bit"

[server]
default_max_tokens = 1024

[voice]
YUNSHU_REALTIME_SILENCE_MS = 400
```

布林值接受 `1/true/yes/on` 與 `0/false/no/off`。無法解析的值會在啟動時當成錯誤，並列出所有有問題的設定。
未登錄的 `YUNSHU_*` 名稱會被忽略，並以警告指出最接近的已登錄設定。

`yunshu config` 會印出每個穩定設定的有效值與來源（`cli`、`env`、`file`、`default`）；`--all` 會加上實驗性與內部設定，
`--json` 輸出 JSON，`--config FILE` 會納入設定檔。

## 穩定性

- **stable**：受支援的部署設定。
- **experimental**：暫時性。每一項都註明由哪一項量測決定去留與加入時間；量測完成後，勝出者成為預設，旗標與落敗的程式路徑會被刪除。同一時間最多 {MAX} 項。
- **internal**：除錯輔助。
""",
    2: """\
每个 Yunshu 设置都是注册在 `python/yunshu_engine/settings.py` 的 `YUNSHU_*` 名称，各有唯一的类型、默认值与解析规则。
除此之外没有任何代码会读取 `YUNSHU_*` 环境变量。

## 设置值的来源

优先级由高到低：

1. `yunshu serve` 的参数与 `--set KEY=VALUE`（可重复；`KEY` 可省略 `YUNSHU_` 前缀，例如 `--set mtp_block_size=4`），
2. 进程的环境变量，
3. TOML 配置文件：`yunshu serve --config yunshu.toml` 或 `YUNSHU_CONFIG=yunshu.toml`，
4. 下表列出的默认值。

配置文件是一组 `KEY = value`，可以平铺，也可以放在任意表（table）之下（表名会被忽略，键可用短名或全名）：

```toml
model = "~/models/Qwen3.8-27B-4bit"

[server]
default_max_tokens = 1024

[voice]
YUNSHU_REALTIME_SILENCE_MS = 400
```

布尔值接受 `1/true/yes/on` 与 `0/false/no/off`。无法解析的值会在启动时作为错误，并列出所有有问题的设置。
未注册的 `YUNSHU_*` 名称会被忽略，并以警告指出最接近的已注册设置。

`yunshu config` 会打印每个稳定设置的有效值与来源（`cli`、`env`、`file`、`default`）；`--all` 会加上实验性与内部设置，
`--json` 输出 JSON，`--config FILE` 会纳入配置文件。

## 稳定性

- **stable**：受支持的部署设置。
- **experimental**：临时性。每一项都注明由哪一项测量决定去留与加入时间；测量完成后，胜出者成为默认，标志与落败的代码路径会被删除。同一时间最多 {MAX} 项。
- **internal**：调试辅助。
""",
}
TITLES = (
    ("Configuration", "Every YUNSHU_* setting, its type and default, generated from the settings registry."),
    ("設定", "所有 YUNSHU_* 設定的型別與預設值，由設定登錄檔產生。"),
    ("配置", "所有 YUNSHU_* 设置的类型与默认值，由设置注册表生成。"),
)
SUFFIX = ("", ".zh-TW", ".zh-CN")


def mdx_safe(line: str) -> str:
    """Escape characters MDX would read as JSX/expressions, outside inline code."""
    parts = line.split("`")
    for i in range(0, len(parts), 2):
        parts[i] = (
            parts[i].replace("{", "&#123;").replace("}", "&#125;").replace("<", "&lt;").replace(">", "&gt;")
        )
    return "`".join(parts)


def sanitize(text: str) -> str:
    out, fenced = [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
            out.append(line)
        elif fenced:
            out.append(line)
        else:
            out.append(mdx_safe(line))
    return "\n".join(out)


def main() -> None:
    full = gcd.render()
    head, tail = full.split("## Stable settings", 1)
    tail = "## Stable settings" + tail
    # English intro: drop the H1 and the generator comment.
    head = re.sub(r"^# Configuration\n+<!--.*?-->\n+", "", head, flags=re.S)
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (title, desc) in enumerate(TITLES):
        body_head = head if i == 0 else INTRO_ZH[i].replace("{MAX}", str(gcd.MAX_EXPERIMENTAL))
        body_tail = tail
        for k, v in HEADINGS.items():
            body_tail = body_tail.replace(k, v[i])
        for k, v in HEADERS.items():
            body_tail = body_tail.replace(k, v[i])
        page = f"---\ntitle: {title}\ndescription: {desc}\n---\n\n{sanitize(body_head.strip())}\n\n{NOTE[i]}\n\n{sanitize(body_tail.strip())}\n"
        (OUT / f"configuration{SUFFIX[i]}.mdx").write_text(page)
    print("wrote configuration pages (en, zh-TW, zh-CN)")


if __name__ == "__main__":
    main()
