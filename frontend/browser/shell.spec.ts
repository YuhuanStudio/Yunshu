import { expect, test, type Page } from "@playwright/test";

async function install(page: Page) {
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    const ok = path === "/v1/yunshu/status";
    return route.fulfill({
      status: ok ? 200 : 404,
      contentType: "application/json",
      body: JSON.stringify(
        ok
          ? {
              object: "yunshu.status",
              version: "fixture-1.0",
              state: "ready",
              uptime_s: 100,
              load_error: null,
              models: [],
              memory: { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 64 },
              requests: {
                active: 0,
                queued: 0,
                prefill: 0,
                decode: 0,
                items: [],
              },
              last: null,
              throughput: {
                window_s: 60,
                requests: 0,
                prompt_tokens: 0,
                completion_tokens: 0,
                live_decode_tps: null,
                mean_prefill_tps: 800,
                mean_decode_tps: 35,
              },
            }
          : { detail: "fixture endpoint not found" },
      ),
    });
  });
}

const geometry = (page: Page) =>
  page.evaluate(() => {
    // Scroll every scroll container (and the document) to its end.
    for (const e of document.querySelectorAll("*"))
      if (
        e.scrollHeight > e.clientHeight + 1 &&
        /auto|scroll/.test(getComputedStyle(e).overflowY)
      )
        e.scrollTop = 1e6;
    window.scrollTo(0, 1e6);
    const bar = document.querySelector("ul[aria-label='最近一筆請求']");
    const doc = document.scrollingElement!;
    return {
      barBottom: bar?.getBoundingClientRect().bottom ?? null,
      inner: window.innerHeight,
      docScroll: doc.scrollHeight,
      scrollY: window.scrollY,
    };
  });

test.describe("shell scroll containment", () => {
  test("long page: status band stays pinned to the viewport bottom and the document never scrolls", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 640 });
    await install(page);
    await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("overview")).toBeVisible();
    await page.waitForTimeout(1500);
    const g = await geometry(page);
    expect(g.barBottom).not.toBeNull();
    expect(Math.round(g.barBottom!)).toBe(g.inner);
    expect(g.scrollY).toBe(0);
    expect(g.docScroll).toBeLessThanOrEqual(g.inner);
  });

  test("short page: no blank scroll height and the band sits at the viewport bottom", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 1800 });
    await install(page);
    await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("requests")).toBeVisible();
    await page.waitForTimeout(1500);
    const g = await geometry(page);
    expect(Math.round(g.barBottom!)).toBe(g.inner);
    expect(g.docScroll).toBeLessThanOrEqual(g.inner);
  });
});
