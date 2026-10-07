"""Server-side ``web_fetch``: SSRF-guarded download with size / time limits and HTML to text.

No provider is needed. Errors carry Anthropic's ``web_fetch_tool_result_error`` codes:
``invalid_tool_input``, ``url_too_long``, ``url_not_allowed``, ``url_not_accessible``,
``too_many_requests``, ``unsupported_content_type``, ``max_uses_exceeded``, ``unavailable``.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin

import httpx

from yunshu_engine import settings
from yunshu_engine.netguard import (
    UrlNotAllowedError,
    domain_matches,
    join_url,
    parse_url,
    pin_request,
    resolve_target,
)

MAX_URL_LEN = 250
MAX_REDIRECTS = 5
USER_AGENT = (
    "Mozilla/5.0 (compatible; YunshuFetch/1.0; +https://github.com/yuhuanowo/yunshu)"
)
_TEXT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml",
    "application/javascript",
)


class FetchError(Exception):
    def __init__(
        self,
        code: str,
        message: str = "",
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass
class FetchResult:
    url: str
    title: str
    text: str
    media_type: str
    retrieved_at: str
    truncated: bool = False
    bytes_read: int = 0
    redirects: list[str] = field(default_factory=list)
    etag: str | None = None
    last_modified: str | None = None
    not_modified: bool = False
    published_at: str | None = None


# ── HTML to text ──────────────────────────────────────────────────────────────
_SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "canvas", "head"}
_BLOCK = {
    "p",
    "div",
    "section",
    "article",
    "header",
    "footer",
    "main",
    "nav",
    "aside",
    "ul",
    "ol",
    "table",
    "tr",
    "pre",
    "blockquote",
    "form",
    "figure",
    "figcaption",
    "dl",
    "dt",
    "dd",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "br",
    "hr",
    "li",
    "details",
    "summary",
}


class _Extract(HTMLParser):
    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False
        self._href: list[str | None] = []
        self._pre = 0

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in _SKIP and tag != "head":
            self._skip += 1
            return
        if self._skip:
            return
        if tag in _BLOCK:
            self.parts.append("\n")
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("#" * int(tag[1]) + " ")
        elif tag == "li":
            self.parts.append("- ")
        elif tag == "pre":
            self._pre += 1
        elif tag == "a":
            href = dict(attrs).get("href")
            self._href.append(
                urljoin(self.base, href)
                if href and not href.startswith(("#", "javascript:"))
                else None
            )
        elif tag == "img":
            alt = dict(attrs).get("alt")
            if alt:
                self.parts.append(f"[image: {alt}]")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _SKIP and tag != "head":
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "a" and self._href:
            href = self._href.pop()
            if href:
                self.parts.append(f" ({href})")
        elif tag == "pre":
            self._pre = max(0, self._pre - 1)
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip:
            return
        self.parts.append(data if self._pre else re.sub(r"\s+", " ", data))


def html_to_text(html: str, base_url: str = "") -> tuple[str, str]:
    """Return (title, text). Block structure becomes newlines, links keep their target."""
    p = _Extract(base_url)
    try:
        p.feed(html)
        p.close()
    except Exception:
        pass
    text = "".join(p.parts)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", text)
    text = re.sub(r"\n{2,}(?=- )", "\n", text)
    return re.sub(r"\s+", " ", p.title).strip(), text.strip()


# ── fetch ─────────────────────────────────────────────────────────────────────
def _decode(raw: bytes, ctype: str) -> str:
    m = re.search(r"charset=([\w\-]+)", ctype, re.I)
    for enc in ([m.group(1)] if m else []) + ["utf-8"]:
        try:
            return raw.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", "replace")


def _retry_after(value: str | None) -> float | None:
    from email.utils import parsedate_to_datetime

    if value is None:
        return None
    try:
        return max(0, float(value))
    except (ValueError, TypeError):
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return None


async def fetch_url(
    url: str,
    *,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    max_content_chars: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> FetchResult:
    """Download ``url`` and return its text. Raises :class:`FetchError` with a spec error code.

    One wall-clock deadline (``YUNSHU_WEB_FETCH_TIMEOUT``) covers DNS, every redirect hop, the
    body and the HTML parse."""
    try:
        async with asyncio.timeout(float(settings.get("YUNSHU_WEB_FETCH_TIMEOUT"))):
            return await _fetch_url(
                url,
                allowed_domains=allowed_domains,
                blocked_domains=blocked_domains,
                max_content_chars=max_content_chars,
                client=client,
            )
    except TimeoutError as e:
        raise FetchError("url_not_accessible", "timed out") from e


async def _fetch_url(
    url: str,
    *,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    max_content_chars: int | None = None,
    client: httpx.AsyncClient | None = None,
    extractor=None,
    request_headers: dict[str, str] | None = None,
    before_redirect=None,
) -> FetchResult:
    if not settings.get("YUNSHU_WEB_FETCH"):
        raise FetchError("unavailable", "web_fetch is disabled (YUNSHU_WEB_FETCH=0)")
    if not isinstance(url, str) or not url.strip():
        raise FetchError("invalid_tool_input", "url is required")
    if len(url) > MAX_URL_LEN:
        raise FetchError("url_too_long", f"url longer than {MAX_URL_LEN} characters")
    allow_private = bool(settings.get("YUNSHU_WEB_FETCH_ALLOW_PRIVATE"))
    max_bytes = int(settings.get("YUNSHU_WEB_FETCH_MAX_BYTES"))
    timeout = float(settings.get("YUNSHU_WEB_FETCH_TIMEOUT"))
    cap = int(settings.get("YUNSHU_WEB_FETCH_MAX_TEXT_CHARS"))
    if max_content_chars:
        cap = min(cap, int(max_content_chars))
    deadline = time.monotonic() + timeout
    cur = url.strip()
    hops: list[str] = []
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
    try:
        for _ in range(MAX_REDIRECTS + 1):
            try:
                host = parse_url(cur).hostname or ""
                if allowed_domains and not domain_matches(host, allowed_domains):
                    raise FetchError(
                        "url_not_allowed", f"{host} is not in allowed_domains"
                    )
                if blocked_domains and domain_matches(host, blocked_domains):
                    raise FetchError("url_not_allowed", f"{host} is in blocked_domains")
                tgt = await resolve_target(cur, allow_private=allow_private)
            except UrlNotAllowedError as e:
                raise FetchError("url_not_allowed", str(e)) from e
            pinned, pin_headers, ext = pin_request(tgt)
            headers = {
                **(request_headers or {}),
                **pin_headers,
                "User-Agent": USER_AGENT,
                "Accept": "text/html,text/plain,*/*;q=0.5",
            }
            remaining = max(0.5, deadline - time.monotonic())
            try:
                async with client.stream(
                    "GET", pinned, headers=headers, extensions=ext, timeout=remaining
                ) as resp:
                    if resp.status_code in (
                        301,
                        302,
                        303,
                        307,
                        308,
                    ) and resp.headers.get("location"):
                        try:
                            cur = join_url(cur, resp.headers["location"])
                        except UrlNotAllowedError as e:
                            raise FetchError("url_not_allowed", str(e)) from e
                        if before_redirect is not None:
                            await before_redirect(cur)
                        hops.append(cur)
                        continue
                    if resp.status_code == 304:
                        return FetchResult(
                            cur, "", "", "text/plain", "", not_modified=True
                        )
                    if resp.status_code == 429:
                        raise FetchError(
                            "too_many_requests",
                            "the site rate-limited the fetch",
                            status_code=429,
                            retry_after=_retry_after(resp.headers.get("retry-after")),
                        )
                    if resp.status_code >= 400:
                        raise FetchError(
                            "url_not_accessible",
                            f"HTTP {resp.status_code}",
                            status_code=resp.status_code,
                        )
                    ctype = (
                        resp.headers.get("content-type", "text/html")
                        .split(";")[0]
                        .strip()
                        .lower()
                    )
                    if (
                        not ctype.startswith(_TEXT_TYPES)
                        and "+xml" not in ctype
                        and "+json" not in ctype
                    ):
                        raise FetchError(
                            "unsupported_content_type", f"cannot read {ctype}"
                        )
                    buf = bytearray()
                    truncated = False
                    async for chunk in resp.aiter_bytes():
                        buf += chunk
                        if len(buf) > max_bytes:
                            del buf[max_bytes:]
                            truncated = True
                            break
                        if time.monotonic() > deadline:
                            truncated = True
                            break
                    raw_ctype = resp.headers.get("content-type", "")
            except FetchError:
                raise
            except httpx.TimeoutException as e:
                raise FetchError("url_not_accessible", "timed out") from e
            except httpx.HTTPError as e:
                raise FetchError(
                    "url_not_accessible", f"{type(e).__name__}: {e}"
                ) from e
            text = _decode(bytes(buf), raw_ctype)
            title = ""
            published_at = None
            media = "text/plain"
            if ctype in ("text/html", "application/xhtml+xml") or (
                "<html" in text[:2000].lower()
            ):
                # off the event loop: parsing a large page must not stall other streams
                extracted = await asyncio.to_thread(
                    extractor or html_to_text, text, cur
                )
                title, text = extracted[:2]
                if len(extracted) > 2:
                    published_at = extracted[2]
            elif ctype == "text/markdown":
                media = "text/markdown"
            if len(text) > cap:
                text, truncated = text[:cap], True
            return FetchResult(
                url=cur,
                title=title,
                text=text,
                media_type=media,
                retrieved_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                truncated=truncated,
                bytes_read=len(buf),
                redirects=hops,
                published_at=published_at,
                etag=resp.headers.get("etag"),
                last_modified=resp.headers.get("last-modified"),
            )
        raise FetchError("url_not_accessible", "too many redirects")
    finally:
        if own:
            await client.aclose()
