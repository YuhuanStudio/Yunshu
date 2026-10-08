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
    for (const li of document.querySelectorAll(
      "ul[aria-label] > li[data-tone]",
    ))
      if (vis(li) && parseFloat(getComputedStyle(li).borderTopWidth) > 0)
        out.push(`bordered chip in the status band: ${li.textContent}`);
    // 3. unavailable notices keep a readable measure
    for (const el of document.querySelectorAll(
      '[data-testid$="-unsupported"], [data-testid$="-unavailable"]',
    ))
      if (vis(el) && el.getBoundingClientRect().width < 320)
        out.push(
          `narrow notice (${Math.round(el.getBoundingClientRect().width)}px)`,
        );
    // 4. no reserved gap inside one line of text
    for (const p of document.querySelectorAll("main p, main [role=status]")) {
      const kids = [...p.children].filter(vis);
      for (let i = 1; i < kids.length; i++) {
        const a = kids[i - 1].getBoundingClientRect();
        const b = kids[i].getBoundingClientRect();
        if (Math.abs(a.top - b.top) < 4 && b.left - a.right > 32)
          out.push(
            `gap ${Math.round(b.left - a.right)}px in: ${p.textContent?.slice(0, 40)}`,
          );
      }
    }
    // 6. no hairline rules inside cards: only table rows and list items may separate content
    const inCard = (e: Element) =>
      e.closest('.card, [data-yunui="card"], [data-slot="card"]');
    for (const el of document.querySelectorAll("main *")) {
      if (!vis(el) || !inCard(el) || el === inCard(el)) continue;
      if (el.closest("table, li, svg, [role=tablist], [role=progressbar]"))
        continue;
      if (/^(BUTTON|INPUT|TEXTAREA|SELECT|A)$/.test(el.tagName)) continue;
      if (
        el.closest(
          "button, [role=combobox], [role=switch], [data-radix-popper-content-wrapper]",
        )
      )
        continue;
      const cs = getComputedStyle(el);
      const w = el.getBoundingClientRect().width;
      // YunUI parts with their own structure: heatmap rows, a code block's filled header bar,
      // and a bar chart's baseline (the box that holds the bars).
      if (el.closest("[role=row]")) continue;
      if (cs.backgroundColor !== "rgba(0, 0, 0, 0)") continue;
      if (el.querySelector("[role=img][aria-label], [role=button][aria-label]"))
        continue;
      // A rule is a top or bottom edge only; a box with all four borders is a field, not a rule.
      const line =
        (parseFloat(cs.borderTopWidth) > 0 ||
          parseFloat(cs.borderBottomWidth) > 0) &&
        parseFloat(cs.borderLeftWidth) === 0;
      if (line && w >= 200)
        out.push(
          `hairline inside a card: ${el.tagName.toLowerCase()}.${String(el.className).split(" ").slice(0, 4).join(".")}`,
        );
    }
    // 5. standard frame and title
    const h1 = document.querySelector("h1");
    if (!h1 || !vis(h1)) out.push("page has no visible h1");
    else if (!h1.closest(".max-w-7xl"))
      out.push("h1 is outside the max-w-7xl frame");
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

// ---- Round 7: measurable limits for the defects the review kept finding ----

test("idle overview: no giant empty cards, and Idle is said once outside the status band", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await install(page);
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.getByTestId("live-panel").waitFor();
  await page.waitForTimeout(1000);
  // The idle live card is one compact card, not a half-empty chart plus list.
  const live = await page.getByTestId("live-panel").boundingBox();
  expect(live!.height).toBeLessThan(220);
  expect(await page.getByTestId("live-panel").getAttribute("data-idle")).toBe(
    "true",
  );
  // The status block, stat cards and live card never say the idle word (the top pill and the band do).
  for (const id of ["state-strip", "overview-stats", "live-panel"])
    expect(await page.getByTestId(id).innerText(), id).not.toContain("閒置");
  // One status block: the verdict row and the phase strip share a card.
  const strip = page.getByTestId("state-strip");
  await expect(strip.getByTestId("health-verdict")).toBeVisible();
});

test("every stat card has the same inner structure: value, label, one meta line", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await install(page);
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.getByTestId("overview-stats").waitFor();
  await page.waitForTimeout(800);
  const shape = await page.getByTestId("overview-stats").evaluate((grid) =>
    [...grid.querySelectorAll(".card")].map((card) => {
      const box = card.getBoundingClientRect();
      const sub = card.querySelector(".yunui-stat-sub");
      const value = card.querySelector(".yunui-stat-value");
      return {
        h: Math.round(box.height),
        // The meta line is plain muted text: no foreground-coloured run inside it.
        mutedOnly: sub
          ? [...sub.querySelectorAll("*")].every(
              (e) => getComputedStyle(e).color === getComputedStyle(sub).color,
            )
          : false,
        hasValue: !!value,
      };
    }),
  );
  expect(shape.length).toBeGreaterThanOrEqual(4);
  expect(new Set(shape.map((s) => s.h)).size).toBe(1);
  for (const s of shape) {
    expect(s.hasValue).toBe(true);
    expect(s.mutedOnly).toBe(true);
  }
});

test("phone: stat cards are compact, 2-up, at most 88px tall", async ({
  page,
}) => {
  await page.setViewportSize({ width: 402, height: 874 });
  await install(page);
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.getByTestId("overview-stats").waitFor();
  await page.waitForTimeout(800);
  const heights = await page
    .getByTestId("overview-stats")
    .evaluate((grid) =>
      [...grid.querySelectorAll(".card")].map((c) =>
        Math.round(c.getBoundingClientRect().height),
      ),
    );
  for (const h of heights) expect(h).toBeLessThanOrEqual(88);
});

test("the top status pill hugs its content", async ({ page }) => {
  // The phone top bar carries no pill (the footer band owns the state), so only the wide bar.
  for (const width of [1440]) {
    await page.setViewportSize({ width, height: 874 });
    await install(page);
    await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
    const pill = page.getByTestId("live-phase");
    await pill.waitFor();
    await page.waitForTimeout(800);
    const m = await pill.evaluate((el) => {
      const cs = getComputedStyle(el);
      const kids = [...el.children].filter(
        (c) => c.getBoundingClientRect().width > 0,
      );
      const inner = kids.reduce(
        (n, c) => n + c.getBoundingClientRect().width,
        0,
      );
      const gaps =
        Math.max(0, kids.length - 1) * parseFloat(cs.columnGap || "0");
      const pad = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight);
      return {
        w: el.getBoundingClientRect().width,
        content: inner + gaps + pad,
      };
    });
    expect(m.w).toBeLessThanOrEqual(m.content + 2);
  }
});

test("zh-TW pages carry no untranslated glossary words", async ({ page }) => {
  test.setTimeout(90_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await install(page);
  const WORDS =
    /\b(Decode|Prefill|Queued|Idle|Starting|Unload|Warm ?up|Latest request|Speculative|Prefix hit)\b/;
  for (const p of [
    "overview",
    "requests",
    "diagnostics",
    "models",
    "playground",
    "settings",
  ]) {
    await page.goto(`/console/#/${p}`, { waitUntil: "domcontentloaded" });
    await page.getByRole("heading", { level: 1 }).first().waitFor();
    await page.waitForTimeout(800);
    const text = await page.locator("main").innerText();
    expect(
      text.match(WORDS)?.[0] ?? null,
      `English glossary word on ${p}`,
    ).toBeNull();
  }
});

test("phone top bar is the hamburger and the title; search and the bell live in the menu", async ({
  page,
}) => {
  for (const width of [390, 402]) {
    await page.setViewportSize({ width, height: 874 });
    await install(page);
    await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
    const header = page.locator("header").first();
    await expect(page.getByTestId("mobile-title")).toBeVisible();
    const controls = await header
      .locator("button:visible, a:visible, input:visible")
      .count();
    expect(controls).toBe(1); // the hamburger
    await expect(header.getByTestId("live-phase")).toHaveCount(0);
    await expect(header.getByTestId("bell-badge")).toHaveCount(0);
    const over = await page.evaluate(
      () => document.documentElement.scrollWidth - innerWidth,
    );
    expect(over).toBeLessThanOrEqual(0);
    await page.getByRole("button", { name: "開啟導覽" }).click();
    const sheet = page.getByRole("navigation", { name: /./ }).first();
    await expect(page.getByRole("button", { name: "搜尋" })).toBeVisible();
    await expect(page.getByTestId("bell-badge")).toBeAttached();
    await expect(sheet).toBeVisible();
    await page.reload({ waitUntil: "domcontentloaded" });
  }
});
