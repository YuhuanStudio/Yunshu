"""Keep public documentation aligned with the registered local API and settings."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from yunshu_engine import settings

ROOT = Path(__file__).resolve().parents[2]
READMES = [ROOT / f"README{suffix}.md" for suffix in ("", ".zh-TW", ".zh-CN")]
DOCS = (
    READMES
    + sorted((ROOT / "docs").glob("*.md"))
    + sorted((ROOT / "docs/guides").glob("*.md"))
)

# Each position is a translated heading, not merely the same heading count.
HEADINGS = [
    ("Yunshu", "Yunshu", "Yunshu"),
    ("Highlights", "亮點", "亮点"),
    ("Quickstart", "快速開始", "快速开始"),
    ("Local console", "本地 console", "本地 console"),
    ("Qwen3.8-27B", "Qwen3.8-27B", "Qwen3.8-27B"),
    ("From source", "從原始碼執行", "从原代码运行"),
    ("How it works", "運作方式", "运作方式"),
    ("Speculative decoding", "推測解碼", "推测解码"),
    ("Prefix cache (APC)", "前綴快取（APC）", "前缀缓存（APC）"),
    ("Structured output", "結構化輸出", "结构化输出"),
    ("API compatibility", "API 相容性", "API 兼容性"),
    ("Coding agents", "程式碼 agent", "编程 agent"),
    ("Performance", "效能", "性能"),
    ("Models", "模型", "模型"),
    ("Other capabilities", "其他能力", "其他能力"),
    ("Command line", "命令列", "命令行"),
    ("Configuration", "設定", "设置"),
    ("Docs", "文件", "文档"),
    ("Privacy", "隱私", "隐私"),
    ("Built on and license", "基礎與授權", "基础与授权"),
]


def readme_structure(text):
    """Heading identity/level and code language/section order, excluding fenced headings."""
    headings, blocks = [], []
    fence = None
    for line in text.splitlines():
        if line.startswith("```"):
            if fence is None:
                fence = line[3:]
                blocks.append((len(headings), fence))
            else:
                fence = None
        elif fence is None and (m := re.match(r"^(#{1,6}) (.+)$", line)):
            headings.append((len(m[1]), m[2]))
    assert fence is None, "unclosed code fence"
    return headings, blocks


def test_readme_translated_structure_matches():
    structures = [readme_structure(p.read_text()) for p in READMES]
    for lang, (headings, blocks) in enumerate(structures):
        assert [h for _, h in headings] == [h[lang] for h in HEADINGS]
        assert [level for level, _ in headings] == [
            level for level, _ in structures[0][0]
        ]
        assert blocks == structures[0][1], READMES[lang].name
        assert re.findall(
            r"(?ms)^```[^\n]*\n.*?^```", READMES[lang].read_text()
        ) == re.findall(r"(?ms)^```[^\n]*\n.*?^```", READMES[0].read_text()), (
            f"{READMES[lang].name}: code example drift"
        )


def test_structure_detects_reordered_sections_and_missing_code():
    base = "## A\n```bash\necho ok\n```\n## B\n"
    assert readme_structure(base) != readme_structure(base.replace("## A", "## B", 1))
    assert readme_structure(base) != readme_structure("## A\n## B\n")


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_public_settings_are_registered(path):
    names = set(re.findall(r"\bYUNSHU_[A-Z][A-Z_0-9]+\b", path.read_text()))
    assert not names - settings.REGISTRY.keys(), (
        path,
        names - settings.REGISTRY.keys(),
    )


@pytest.fixture(scope="module")
def routes():
    spec = importlib.util.spec_from_file_location(
        "doc_api_coverage", ROOT / "scripts/dev/api_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses needs the module registered while executing it.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return {r.split(" ", 1)[1] for r in module.implemented_routes()}


def endpoint_mentions(text):
    # Full local API prefixes only: relative Tavily paths are covered by its guide/tests.
    return set(
        re.findall(
            r"(?<![\w/])/(?:v1|api|tavily)/[\w/*{}:.-]+",
            re.sub(r"https?://[^\s)]+", "", text),
        )
    )


def route_matches(mention, routes):
    mention = re.sub(r"\{[^}]*\}", "{}", mention.rstrip("/.,"))
    # '*' is documentation shorthand for a registered route family.
    pattern = re.escape(mention).replace(r"\*", ".*")
    # Concrete example IDs must fit an existing parameterized route.
    if any(
        re.fullmatch(re.escape(r).replace(r"\{\}", "[^/]+"), mention) for r in routes
    ):
        return True
    return any(re.fullmatch(pattern, r) for r in routes)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_documented_endpoints_exist(path, routes):
    text = path.read_text()
    # The matrix explicitly preserves retired routes so users can migrate.
    if path.name == "API_SURFACE.md":
        text = re.sub(r"(?ms)^## Removed\n.*?(?=^## |\Z)", "", text)
    exceptions = {
        "API_EXTENSIONS.md": {"/api/v0": "LM Studio", "/api/v0/*": "LM Studio"},
        "API_SURFACE.md": {
            "/v1/fine_tuning": "not applicable",
            "/v1/moderations": "not applicable",
            "/v1/assistants": "not applicable",
            "/v1/vector_stores": "not applicable",
            "/v1/uploads": "not applicable",
            "/api/v1/gw/monitoring/prometheus": "Was",
        },
        "PROMPT_CACHING_APIS.md": {"/v1/cachedContents": "no longer offers"},
    }.get(path.name, {})
    mentions = endpoint_mentions(text)
    for endpoint, explanation in exceptions.items():
        assert endpoint in mentions and explanation in text
        assert not route_matches(endpoint, routes), (
            f"stale exclusion: {endpoint} is now served"
        )
    unknown = {
        p for p in mentions if not route_matches(p, routes) and p not in exceptions
    }
    assert not unknown, (str(path.relative_to(ROOT)), sorted(unknown))


def test_unknown_endpoint_is_rejected(routes):
    assert not route_matches("/v1/docs015_invented_endpoint", routes)
    assert route_matches("/v1/chat/completions/{completion_id}", routes)


@pytest.mark.parametrize("name", ["API_SURFACE.md", "API_EXTENSIONS.md"])
def test_memory_unit_contract_is_documented(name):
    from yunshu_engine.units import GIB

    text = (ROOT / "docs/guides" / name).read_text()
    assert GIB == 1024**3
    assert "1 GiB = 1024^3 bytes" in text
    assert "`*_bytes`" in text
    assert "128.0" in text
