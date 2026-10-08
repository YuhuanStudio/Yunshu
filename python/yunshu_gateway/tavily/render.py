"""Bounded opt-in Chromium replay, with no direct browser network connections.

Only same-origin GET documents/scripts/fetches are replayed through the existing
guard. Redirected resources are refused because fulfilling their body under the
old browser URL would assign the wrong origin/base URL. No bot-protection bypass.
"""

import asyncio
import time

from yunshu_engine import settings
from yunshu_engine.netguard import UrlNotAllowedError, parse_url
from yunshu_gateway.server_tools.research.fetcher import page
from yunshu_gateway.server_tools.webfetch import FetchError

from .content import page_content

_slots = asyncio.Semaphore(1)


def origin(url):
    parts = parse_url(url)
    return (
        parts.scheme,
        parts.hostname,
        parts.port or (443 if parts.scheme == "https" else 80),
    )


class Resources:
    """The browser receives fulfilled text, never a route.continue_ connection."""

    def __init__(self, url, automated, timeout, fetcher=None):
        self.origin = origin(url)
        self.automated = automated
        self.deadline = time.monotonic() + timeout
        self.fetcher = fetcher or page
        self.count = self.bytes = 0
        self.slots = asyncio.Semaphore(2)

    async def handle(self, route):
        request = route.request
        try:
            if (
                request.method != "GET"
                or request.resource_type not in ("document", "script", "xhr", "fetch")
                or origin(request.url) != self.origin
            ):
                await route.abort()
                return
            self.count += 1
            if self.count > 32 or self.bytes >= 4 * 1024 * 1024:
                await route.abort()
                return
            async with self.slots:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError

                def reject_redirect(target):
                    # Reject before the next DNS resolution/connection, not after
                    # receiving a body under the wrong browser origin/base URL.
                    raise FetchError(
                        "url_not_allowed", "Rendered resource redirect refused"
                    )

                result = await self.fetcher(
                    request.url,
                    automated=self.automated,
                    timeout=remaining,
                    timeout_seconds=remaining,
                    max_url_length=2048,
                    extractor=lambda body, url: ("", body),
                    cache_namespace="tavily-render-raw",
                    redirect_guard=reject_redirect,
                )
                size = len(result.text.encode("utf-8"))
                self.bytes += size
                if (
                    result.truncated
                    or self.bytes > 4 * 1024 * 1024
                    or result.url != request.url
                    or origin(result.url) != self.origin
                ):
                    await route.abort()
                    return
                content_type = {
                    "document": "text/html",
                    "script": "application/javascript",
                }.get(request.resource_type, "text/plain")
                await route.fulfill(
                    status=200, body=result.text, content_type=content_type
                )
        except (FetchError, UrlNotAllowedError, TimeoutError, ValueError):
            await route.abort()


async def render(url, *, automated, timeout):
    try:
        from playwright.async_api import Error, async_playwright
    except ImportError as exc:
        raise FetchError(
            "unavailable", "Install the web-render extra and Chromium"
        ) from exc
    resources = Resources(url, automated, timeout)
    try:
        async with asyncio.timeout(timeout), _slots, async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                # Defense in depth: any request escaping interception hits a dead
                # local proxy; include loopback instead of Chromium's implicit bypass.
                proxy={"server": "http://127.0.0.1:9", "bypass": "<-loopback>"},
                args=[
                    "--disable-gpu",
                    "--disable-background-networking",
                    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                ],
            )
            try:
                context = await browser.new_context(
                    service_workers="block", accept_downloads=False
                )
                await context.route("**/*", resources.handle)
                await context.route_web_socket("**/*", lambda ws: ws.close())
                await context.add_init_script("""
                    Object.defineProperty(globalThis, 'RTCPeerConnection', {value: undefined});
                    Object.defineProperty(globalThis, 'webkitRTCPeerConnection', {value: undefined});
                """)
                tab = await context.new_page()
                await tab.goto(
                    url, wait_until="networkidle", timeout=max(1, int(timeout * 1000))
                )
                # Capture only a bounded DOM, before bringing it across the driver pipe.
                body = await tab.evaluate(
                    "document.documentElement.outerHTML.slice(0, 1000000)"
                )
                final = tab.url
                if origin(final) != resources.origin:
                    raise FetchError("url_not_allowed", "Rendered page changed origin")
                title, text, published, metadata = await asyncio.to_thread(
                    page_content, body, final
                )
                metadata["rendered"] = True
                cap = int(settings.get("YUNSHU_WEB_FETCH_MAX_TEXT_CHARS"))
                truncated = len(body) >= 1000000 or len(text) > cap
                from yunshu_gateway.server_tools.webfetch import FetchResult

                return FetchResult(
                    final,
                    title,
                    text[:cap],
                    "text/html",
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    published_at=published,
                    metadata=metadata,
                    bytes_read=resources.bytes,
                    truncated=truncated,
                )
            finally:
                await asyncio.shield(browser.close())
    except Error as exc:
        raise FetchError("unavailable", "Chromium rendering is unavailable") from exc
