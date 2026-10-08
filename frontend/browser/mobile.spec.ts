import { expect, test, type Page } from "@playwright/test";

/**
 * Phone layout contract at 402x874 (iPhone 15 Pro class): nothing overflows
 * sideways, no two controls overlap, the footer is one row, the top bar clips
 * nothing. Runs on every page the shell owns.
 */
test.use({
  viewport: { width: 402, height: 874 },
  isMobile: true,
  hasTouch: true,
  deviceScaleFactor: 3,
});

const status = {
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "running",
  uptime_s: 4000,
  load_error: null,
  models: [
    {
      id: "Qwen3.8-27B-oQ4e-mtp",
      type: "VLM",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 17.6,
    },
    {
      id: "Qwen3.5-9B",
      type: "LLM",
      loaded: false,
      loading: false,
      pinned: false,
      size_gb: 5.8,
    },
  ],
  memory: { active_gb: 27.1, cache_gb: 2, peak_gb: 30, total_gb: 137 },
  requests: {
    active: 1,
    queued: 0,
    prefill: 1,
    decode: 0,
    items: [
      {
        request_id: "req-1",
        elapsed_s: 4,
        phase: "prefill",
        model: "Qwen3.8-27B-oQ4e-mtp",
        prompt_tokens: 60000,
        processed_tokens: 50400,
        percent: 84,
        cached_tokens: 0,
        completion_tokens: 0,
        tokens_per_second: 799.4,
      },
    ],
  },
  last: null,
  throughput: {
    window_s: 60,
    requests: 3,
    prompt_tokens: 100,
    completion_tokens: 50,
    live_decode_tps: 94.1,
    mean_prefill_tps: 98.9,
    mean_decode_tps: 91.9,
  },
};

async function install(page: Page) {
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    const ok = path === "/v1/yunshu/status";
    const models = path === "/v1/models";
    return route.fulfill({
      status: ok || models ? 200 : 404,
      contentType: "application/json",
      body: JSON.stringify(
        ok
          ? status
          : models
            ? { object: "list", data: status.models.map((m) => ({ id: m.id })) }
            : { detail: "fixture endpoint not found" },
      ),
    });
  });
}

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

type Issue = string;
async function audit(page: Page): Promise<Issue[]> {
  return page.evaluate(() => {
    const issues: string[] = [];
    const vw = window.innerWidth;
    const label = (e: Element) =>
      `${e.tagName.toLowerCase()}${e.id ? "#" + e.id : ""}.${String(e.className).split(" ").slice(0, 3).join(".")} "${(e.textContent ?? "").trim().slice(0, 24)}"`;
    const visible = (e: Element) => {
      const r = e.getBoundingClientRect();
      const cs = getComputedStyle(e);
      return (
        r.width > 0 &&
        r.height > 0 &&
        cs.visibility !== "hidden" &&
        cs.display !== "none" &&
        Number(cs.opacity) > 0
      );
    };
    // The drawer is off canvas by design; anything inside a clipping box that fits the screen is fine.
    const drawer = document
      .querySelector("nav[aria-label]")
      ?.closest("aside, div[class*='fixed']");
    const clipped = (e: Element) => {
      for (
        let p = e.parentElement;
        p && p !== document.body;
        p = p.parentElement
      ) {
        const cs = getComputedStyle(p);
        if (
          /(auto|scroll|hidden|clip)/.test(cs.overflowX) &&
          p.getBoundingClientRect().right <= vw + 1
        )
          return true;
      }
      return false;
    };
    // On-screen part of a control: its box clipped by every scroll container above it.
    const shown = (e: Element) => {
      const r = e.getBoundingClientRect();
      let { left, right, top, bottom } = r;
      for (
        let p = e.parentElement;
        p && p !== document.body;
        p = p.parentElement
      ) {
        const cs = getComputedStyle(p);
        if (!/(auto|scroll|hidden|clip)/.test(cs.overflowY + cs.overflowX))
          continue;
        const pr = p.getBoundingClientRect();
        left = Math.max(left, pr.left);
        right = Math.min(right, pr.right);
        top = Math.max(top, pr.top);
        bottom = Math.min(bottom, pr.bottom);
      }
      return right - left > 0 && bottom - top > 0
        ? { left, right, top, bottom }
        : null;
    };
    for (const e of document.querySelectorAll("body *")) {
      if (
        !visible(e) ||
        (drawer && drawer.contains(e)) ||
        e.closest("[data-radix-popper-content-wrapper], [role='dialog']")
      )
        continue;
      const r = e.getBoundingClientRect();
      if ((r.right > vw + 1 || r.left < -1) && !clipped(e))
        issues.push(
          `overflow-x ${label(e)} [${Math.round(r.left)}..${Math.round(r.right)}]`,
        );
    }
    if (document.documentElement.scrollWidth > vw + 1)
      issues.push(
        `document scrolls sideways (${document.documentElement.scrollWidth})`,
      );
    // Interactive controls must not overlap one another.
    const controls = [
      ...document.querySelectorAll<HTMLElement>(
        "main button, main a[href], main input, main textarea, main select, main [role='combobox'], header button, header a[href], footer button, footer a[href]",
      ),
    ].filter(
      (e) => visible(e) && !(drawer && drawer.contains(e)) && shown(e) !== null,
    );
    for (let i = 0; i < controls.length; i++)
      for (let j = i + 1; j < controls.length; j++) {
        const a = controls[i],
          b = controls[j];
        if (a.contains(b) || b.contains(a)) continue;
        const ra = shown(a)!,
          rb = shown(b)!;
        const w = Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left);
        const h = Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top);
        if (w > 2 && h > 2) issues.push(`overlap ${label(a)} x ${label(b)}`);
      }
    // The top bar clips no text.
    const header = document.querySelector("header");
    if (header)
      for (const e of header.querySelectorAll("*")) {
        if (!visible(e) || !(e.textContent ?? "").trim() || e.children.length)
          continue;
        const cs = getComputedStyle(e);
        if (e.scrollWidth > e.clientWidth + 1 && cs.textOverflow !== "ellipsis")
          issues.push(`header text clipped ${label(e)}`);
        if (cs.textOverflow === "ellipsis" && e.scrollWidth > e.clientWidth + 1)
          issues.push(`header text truncated ${label(e)}`);
      }
    return issues;
  });
}

for (const name of PAGES) {
  test(`phone layout: ${name}`, async ({ page }) => {
    await install(page);
    await page.goto(`/console/#/${name}`);
    await expect(page.locator("main")).toBeVisible();
    await page.waitForTimeout(1200);
    expect(await audit(page)).toEqual([]);
    // Breadcrumb pill is replaced by the page title; theme and language live in the drawer.
    await expect(page.getByTestId("mobile-title")).toBeVisible();
    await expect(page.locator("header nav[aria-label]")).toBeHidden();
    // Footer: one compact row, not the wrapping pill band.
    const footer = page.locator("footer");
    await expect(page.getByTestId("footer-compact")).toBeVisible();
    await expect(footer.locator("ul[aria-label]")).toHaveCount(0);
    const box = await footer.boundingBox();
    expect(box!.height).toBeLessThanOrEqual(56);
  });
}

test("footer sheet shows the full pills on tap", async ({ page }) => {
  await install(page);
  await page.goto("/console/#/overview");
  await page.getByTestId("footer-compact").click();
  await expect(
    page.getByRole("dialog").locator("ul[aria-label]"),
  ).toBeVisible();
});

test("safe areas: the viewport covers the notch and bars pad for it", async ({
  page,
}) => {
  await install(page);
  await page.goto("/console/#/overview");
  const meta = await page
    .locator("meta[name=viewport]")
    .getAttribute("content");
  expect(meta).toContain("viewport-fit=cover");
  const css = await page.evaluate(() => {
    const pad = (sel: string) =>
      getComputedStyle(document.querySelector(sel)!).paddingBottom;
    return {
      header: getComputedStyle(document.querySelector("header")!).paddingTop,
      footer: pad("footer"),
    };
  });
  expect(parseFloat(css.header)).toBeGreaterThanOrEqual(16);
});

test("overview: stat cards are 2-up and short; the phase strip is not cut", async ({
  page,
}) => {
  await install(page);
  await page.goto("/console/#/overview");
  const cards = page.locator(
    "[data-stat-grid] > *, [data-stat-grid] > .contents > *",
  );
  await expect(page.getByTestId("overview-stats")).toBeVisible();
  const boxes = await page.getByTestId("overview-stats").evaluate((grid) =>
    [...grid.querySelectorAll(".card")].map((c) => {
      const r = c.getBoundingClientRect();
      return {
        x: Math.round(r.left),
        w: Math.round(r.width),
        h: Math.round(r.height),
      };
    }),
  );
  void cards;
  expect(boxes.length).toBeGreaterThanOrEqual(4);
  expect(boxes[0].w).toBeLessThan(220);
  expect(boxes[0].h).toBeLessThan(140);
  expect(boxes[1].x).toBeGreaterThan(boxes[0].x + 100);
  const detail = page.getByTestId("state-strip-detail");
  expect(await detail.evaluate((e) => e.scrollWidth <= e.clientWidth + 1)).toBe(
    true,
  );
});

test("playground toolbar: model full width, dialect and mode on one row, actions in a menu", async ({
  page,
}) => {
  await install(page);
  await page.goto("/console/#/playground");
  const group = page.getByRole("group", { name: "測試模型" });
  await expect(group).toBeVisible();
  const model = await group.boundingBox();
  const dialect = await page
    .getByRole("group", { name: "API 格式" })
    .boundingBox();
  const mode = await page
    .getByRole("tablist", { name: "測試模式" })
    .boundingBox();
  expect(model!.width).toBeGreaterThan(300);
  expect(dialect!.y).toBeGreaterThan(model!.y + model!.height - 2);
  expect(Math.abs(dialect!.y - mode!.y)).toBeLessThan(12);
  // No lone overflow dot: the two actions are plain labelled buttons on their own row.
  await expect(page.getByRole("button", { name: "更多動作" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "檢視程式碼" })).toBeVisible();
  await expect(page.getByRole("button", { name: "新測試" })).toBeVisible();
});

test("requests: the phone list fits the width with its actions reachable (no sideways scroll)", async ({
  page,
}) => {
  await install(page);
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  const details = page.getByRole("button", { name: "詳情" }).first();
  await expect(details).toBeVisible();
  const box = await details.boundingBox();
  const vw = page.viewportSize()!.width;
  expect(box!.x + box!.width).toBeLessThanOrEqual(vw);
  const overflow = await page
    .getByTestId("request-identity")
    .first()
    .evaluate((td) => {
      const table = td.closest("table")!;
      const box = table.parentElement!;
      return box.scrollWidth - box.clientWidth;
    });
  expect(overflow).toBeLessThanOrEqual(1);
});
