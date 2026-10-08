"""Static content metadata and deterministic IR; no inference in this module."""

import re
from urllib.parse import urljoin, urlsplit

from lxml import html

from yunshu_gateway.server_tools.research.chunk import chunks
from yunshu_gateway.server_tools.research.extract import extract_with_metadata
from yunshu_gateway.server_tools.research.rank import bm25


def page_content(body, url):
    title, text, published = extract_with_metadata(body, url)
    metadata = {"links": [], "images": [], "favicon": None, "language": None}
    try:
        root = html.fromstring(body)
        metadata["js_shell"] = bool(root.xpath("//script")) and len(text.strip()) < 200
        metadata["language"] = root.get("lang")
        for key, values in (
            ("links", root.xpath("//a/@href")),
            ("images", root.xpath('//img/@src|//meta[@property="og:image"]/@content')),
        ):
            seen = set()
            for value in values:
                absolute = urljoin(url, value)
                if (
                    urlsplit(absolute).scheme in ("http", "https")
                    and absolute not in seen
                ):
                    seen.add(absolute)
                    metadata[key].append(absolute)
                if len(metadata[key]) >= (500 if key == "links" else 100):
                    break
        icons = root.xpath(
            '//link[contains(concat(" ", normalize-space(@rel), " "), " icon ")]/@href'
        )
        if icons:
            icon = urljoin(url, icons[0])
            if urlsplit(icon).scheme in ("http", "https"):
                metadata["favicon"] = icon
    except (ValueError, TypeError):
        pass
    return title, text, published, metadata


def plain_text(markdown):
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", markdown)
    text = re.sub(r"(?m)^\s*(?:#{1,6}\s+|```[^\n]*|[-*+]\s+)", "", text)
    return text.replace("**", "").replace("__", "").replace("`", "").strip()


def top_chunks(query, text, count):
    passages = chunks(text, size=500, overlap=80)
    order = bm25(query, passages)
    chosen = []
    for index in order:
        p = passages[index]
        if any(
            min(p.end, old.end) - max(p.start, old.start) > len(p.text) // 2
            for old in chosen
        ):
            continue
        chosen.append(p)
        if len(chosen) >= count:
            break
    return " [...] ".join(p.text for p in chosen), max(
        (p.score for p in chosen), default=0
    )


def favicon(url, found=None):
    parts = urlsplit(url)
    return found or f"{parts.scheme}://{parts.netloc}/favicon.ico"
