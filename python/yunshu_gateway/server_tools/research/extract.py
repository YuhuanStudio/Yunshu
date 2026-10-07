"""Main-content extraction, with hidden HTML removed before either extractor."""

import logging
import re

from lxml import html
from prometheus_client import Counter

from ..webfetch import html_to_text

logger = logging.getLogger(__name__)
_INSTRUCTION = re.compile(
    r"(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior|system)\s+instructions|忽略.{0,12}(?:指令|規則)",
    re.I,
)
_instruction_pages = Counter(
    "yunshu_web_research_instruction_pages_total",
    "Extracted pages containing suspected instruction text (retained as untrusted data).",
)
_INVISIBLE = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]|\x1b\[[0-?]*[ -/]*[@-~]"
)


def extract(body: str, url: str) -> tuple[str, str]:
    try:
        root = html.fromstring(body)
        hidden_selectors: list[str] = []
        for css in root.xpath("//style/text()"):
            for selectors, declarations in re.findall(r"([^{}]+)\{([^{}]+)\}", css):
                declarations = re.sub(r"\s+", "", declarations).lower()
                if (
                    "display:none" in declarations
                    or "visibility:hidden" in declarations
                ):
                    hidden_selectors.extend(s.strip() for s in selectors.split(","))
        for node in root.xpath(
            "//script|//style|//noscript|//template|//iframe|//svg|//meta|//comment()"
        ):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)
        for node in list(root.iter()):
            if not isinstance(node.tag, str):
                continue
            style = re.sub(r"\s+", "", node.get("style", "")).lower()
            css_hidden = any(
                (
                    selector.startswith(".")
                    and selector[1:] in node.get("class", "").split()
                )
                or (selector.startswith("#") and selector[1:] == node.get("id"))
                or selector == node.tag
                for selector in hidden_selectors
            )
            if (
                css_hidden
                or node.get("aria-hidden", "").lower() == "true"
                or "hidden" in node.attrib
                or "display:none" in style
                or "visibility:hidden" in style
            ):
                parent = node.getparent()
                if parent is not None:
                    parent.remove(node)
                else:
                    return "", ""
        safe = html.tostring(root, encoding="unicode")
    except (ValueError, TypeError):
        return "", ""  # fail closed: never run a fallback over unsanitized HTML
    title, fallback = html_to_text(safe, url)
    try:
        import trafilatura

        text = (
            trafilatura.extract(
                safe,
                output_format="markdown",
                include_comments=False,
                include_formatting=True,
                include_tables=True,
                include_links=False,
            )
            or fallback
        )
    except Exception:
        logger.debug(
            "Main-content extraction fell back to sanitized HTML", exc_info=True
        )
        text = fallback
    # Trafilatura may flatten short pages; restore surviving heading/code spans from
    # the sanitized DOM, without reintroducing text it deliberately discarded.
    for node in root.xpath("//h1|//h2|//h3|//h4|//h5|//h6|//pre"):
        original = node.text_content().strip()
        flat = re.sub(r"\s+", " ", original)
        if not flat:
            continue
        if node.tag == "pre":
            replacement = "\n```\n" + original + "\n```\n"
        else:
            replacement = "\n" + "#" * int(node.tag[1]) + " " + flat + "\n"
        if flat in text and not (
            node.tag != "pre" and re.search(r"(?m)^#+ " + re.escape(flat) + r"$", text)
        ):
            text = text.replace(flat, replacement, 1)
    title, text = _INVISIBLE.sub("", title), clean_text(text)
    if _INSTRUCTION.search(text):
        _instruction_pages.inc()  # diagnostic only: never discard visible instructions
    return title, text


def clean_text(text: str) -> str:
    text = _INVISIBLE.sub("", text)
    return re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)


def extract_with_metadata(body: str, url: str) -> tuple[str, str, str | None]:
    title, text = extract(body, url)
    date = None
    try:
        root = html.fromstring(body)
        values = root.xpath(
            '//meta[@property="article:published_time"]/@content | //meta[@name="date"]/@content | //time/@datetime'
        )
        date = next(
            (
                v
                for v in values
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T[0-9:.+Z-]+)?", v)
            ),
            None,
        )
    except (ValueError, TypeError):
        pass
    return title, text, date
