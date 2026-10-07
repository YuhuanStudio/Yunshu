import { expect, test, type Page } from "@playwright/test";

/**
 * Cheap visual lint over every page against an engine that lacks the newer admin routes (404),
 * the state that looked worst: ellipsis in stat cards, bordered chips in the status band,
 * narrow empty-state columns, reserved horizontal gaps inside one text line, and pages
 * without the standard frame and title.
 */
const PAGES = [
  "overview",
  "requests",
  "logs",
  "diagnostics",
  "models",
  "downloads",
  "cache",
  "playground",
  "api",
  "keys",
  "settings",
];

const status = {
  object: "yunshu.status",
  version: "0.1.3",
  state: "running",
  uptime_s: 8000,
  load_error: null,
  models: [
    {
      id: "Qwen3.8-27B-oQ4e-mtp",
      type: "VLMEngine",
      loaded: true,
      loading: false,
      pinned: true,
      size_gb: 15.7,
    },
  ],
  memory: { active_gb: 23.2, cache_gb: 2, peak_gb: 31.9, total_gb: 137.4 },
  requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
  last: {
    request_id: "req_25ab1b3f793ab",
    prompt_tokens: 32812,
    completion_tokens: 2048,
    cached_tokens: 0,
    prefill_tps: 954.8,
    decode_tps: 32.2,
    ttft_ms: 34403,
    t: 1.79e9,
  },
  throughput: {
    window_s: 60,
    requests: 1,
    prompt_tokens: 32812,
    completion_tokens: 2048,
    live_decode_tps: null,
    mean_prefill_tps: null,
    mean_decode_tps: null,
  },
};

async function install(page: Page) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status")
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(status),
      });
    return route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Not Found" }),
    });
  });
}

async function lint(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const out: string[] = [];
    const vis = (e: Element) => {
      const r = e.getBoundingClientRect();
      return r.width > 0 && r.height > 0;
    };
    // 1. no ellipsis in stat cards
    for (const el of document.querySelectorAll("[data-stat-grid] *"))
      if (vis(el) && getComputedStyle(el).textOverflow === "ellipsis")
        out.push(`ellipsis in a stat card: ${el.className}`);
    // 2. status band: dot and plain text, no bordered chips
    for (const li of document.querySelectorAll("ul[aria-label] > li[data-tone]"))
      if (vis(li) && parseFloat(getComputedStyle(li).borderTopWidth) > 0)
        out.push(`bordered chip in the status band: ${li.textContent}`);
    // 3. unavailable notices keep a readable measure
    for (const el of document.querySelectorAll(
      '[data-testid$="-unsupported"], [data-testid$="-unavailable"]',
    ))
      if (vis(el) && el.getBoundingClientRect().width < 320)
        out.push(`narrow notice (${Math.round(el.getBoundingClientRect().width)}px)`);
    // 4. no reserved gap inside one line of text
    for (const p of document.querySelectorAll("main p, main [role=status]")) {
      const kids = [...p.children].filter(vis);
      for (let i = 1; i < kids.length; i++) {
        const a = kids[i - 1].getBoundingClientRect();
        const b = kids[i].getBoundingClientRect();
        if (Math.abs(a.top - b.top) < 4 && b.left - a.right > 32)
          out.push(`gap ${Math.round(b.left - a.right)}px in: ${p.textContent?.slice(0, 40)}`);
      }
    }
    // 5. standard frame and title
    const h1 = document.querySelector("h1");
    if (!h1 || !vis(h1)) out.push("page has no visible h1");
    else if (!h1.closest(".max-w-7xl")) out.push("h1 is outside the max-w-7xl frame");
    return out;
  });
}

for (const p of PAGES)
  test(`visual lint: ${p}`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await install(page);
    await page.goto(`/console/#/${p}`, { waitUntil: "domcontentloaded" });
    await page.getByRole("heading", { level: 1 }).first().waitFor();
    await page.waitForTimeout(1500);
    expect(await lint(page)).toEqual([]);
  });
