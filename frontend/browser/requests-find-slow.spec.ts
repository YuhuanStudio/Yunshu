import { expect, test, type Page, type Route } from "@playwright/test";

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });

const T0 = 1_800_000_000;
const id = (n: string) => `req-${n}-a1b2c3d4e5f6a7b8c9d0e1f2`;

function entry(name: string, i: number, extra: Record<string, unknown> = {}) {
  return {
    request_id: id(name),
    t: T0 + i * 60 + 3,
    t0_wall: T0 + i * 60,
    path: "/v1/chat/completions",
    model: "org/qwen-mlx",
    status: 200,
    finish_reason: "stop",
    stream: true,
    offsets_ms: {
      arrive: 0,
      admit: 40,
      first_token: 340,
      last_token: 2340,
      done: 2360,
    },
    queue_wait_ms: 40,
    ttft_ms: 300,
    prompt_tokens: 1000,
    cached_tokens: 900,
    completion_tokens: 80,
    prefill_tps: 800,
    decode_tps: 40,
    cache: { tier: "ram", reload_ms: null },
    speculative: null,
    cancelled: false,
    ...extra,
  };
}

// 22 ordinary requests (TTFT 200..620 ms) and two outliers: a cold 41K prefill and a long queue.
const ring = [
  ...Array.from({ length: 22 }, (_, i) =>
    entry(String(i).padStart(2, "0"), i, {
      ttft_ms: 200 + i * 20,
      offsets_ms: {
        arrive: 0,
        admit: 40,
        first_token: 200 + i * 20,
        last_token: 2300,
        done: 2320,
      },
    }),
  ),
  entry("cold", 30, {
    ttft_ms: 41000,
    prompt_tokens: 45000,
    cached_tokens: 4000,
    offsets_ms: {
      arrive: 0,
      admit: 100,
      first_token: 41100,
      last_token: 43500,
      done: 43600,
    },
  }),
  entry("queued", 31, {
    ttft_ms: 12600,
    queue_wait_ms: 12000,
    offsets_ms: {
      arrive: 0,
      admit: 12000,
      first_token: 12600,
      last_token: 14000,
      done: 14020,
    },
  }),
];

const quiet = {
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "ready",
  uptime_s: 3600,
  load_error: null,
  models: [
    {
      id: "org/qwen-mlx",
      type: "text",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 8,
      expires_in_s: 900,
    },
  ],
  memory: {
    active_gb: 12,
    cache_gb: 2,
    peak_gb: 14,
    total_gb: 64,
    pressure: 0.2,
  },
  requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
  last: null,
  throughput: {
    window_s: 60,
    requests: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    live_decode_tps: null,
    mean_prefill_tps: null,
    mean_decode_tps: null,
  },
};

async function open(
  page: Page,
  opts: { recent?: boolean; running?: boolean } = {},
) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status")
      return json(
        route,
        opts.running
          ? {
              ...quiet,
              requests: {
                active: 1,
                queued: 0,
                prefill: 0,
                decode: 1,
                items: [
                  {
                    request_id: "live-decode-0123456789abcdef",
                    elapsed_s: 4,
                    phase: "decode",
                    model: "org/qwen-mlx",
                    prompt_tokens: 600,
                    cached_tokens: 200,
                    completion_tokens: 42,
                    tokens_per_second: 28.5,
                  },
                ],
              },
            }
          : quiet,
      );
    if (path === "/v1/yunshu/requests/recent")
      return opts.recent === false
        ? json(route, { detail: "Not Found" }, 404)
        : json(route, {
            object: "list",
            data: ring,
            count: ring.length,
            capacity: 512,
          });
    return json(route, { detail: "Not Found" }, 404);
  });
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  return { errors, requests: page.getByTestId("requests") };
}

test.describe("requests: identity stays readable at every width", () => {
  for (const [width, height] of [
    [390, 844],
    [1024, 768],
    [1440, 900],
    [1920, 1080],
  ] as const) {
    test(`id and model cells are visible and wide at ${width}`, async ({
      page,
    }) => {
      await page.setViewportSize({ width, height });
      const { errors, requests } = await open(page, { running: true });
      // Running request first (auto scope), then the finished list.
      for (const scope of ["進行中", "已結束"]) {
        await requests.getByRole("tab", { name: scope, exact: true }).click();
        const cells = requests.getByTestId("request-identity");
        await expect(cells.first()).toBeVisible();
        const n = Math.min(await cells.count(), 5);
        for (let i = 0; i < n; i++) {
          const box = (await cells.nth(i).boundingBox())!;
          expect(
            box.width,
            `${scope} #${i} at ${width}`,
          ).toBeGreaterThanOrEqual(150);
          await expect(cells.nth(i)).toContainText(/req-|live-/);
        }
        await expect(cells.first()).toContainText("qwen-mlx");
      }
      // With the inspector open the list narrows; the identity must not collapse.
      await requests.getByRole("button", { name: "詳情" }).first().click();
      const box = (await requests
        .getByTestId("request-identity")
        .first()
        .boundingBox())!;
      expect(box.width).toBeGreaterThanOrEqual(150);
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - window.innerWidth,
      );
      expect(overflow).toBeLessThanOrEqual(1);
      expect(errors).toEqual([]);
    });
  }
});

test.describe("requests: find the slow one", () => {
  test("opens on 已結束 when nothing runs, with clock time per row", async ({
    page,
  }) => {
    const { requests } = await open(page);
    await expect(
      requests.getByRole("tab", { name: "已結束", exact: true }),
    ).toHaveAttribute("data-state", "active");
    const row = requests
      .getByRole("row")
      .filter({ hasText: id("cold").slice(0, 9) });
    await expect(row).toContainText(/\d{2}:\d{2}:\d{2}/);
  });

  test("keeps 進行中 as the first tab while something runs", async ({
    page,
  }) => {
    const { requests } = await open(page, { running: true });
    await expect(
      requests.getByRole("tab", { name: "進行中", exact: true }),
    ).toHaveAttribute("data-state", "active");
  });

  test("sorting by TTFT puts the cold prefill first, a second click reverses", async ({
    page,
  }) => {
    const { requests } = await open(page);
    const first = () => requests.getByTestId("request-identity").first();
    await requests
      .getByRole("columnheader", { name: /TTFT/ })
      .getByRole("button")
      .click();
    await expect(first()).toContainText(id("cold").slice(0, 9));
    await expect(
      requests.getByRole("columnheader", { name: /TTFT/ }),
    ).toHaveAttribute("aria-sort", "descending");
    await requests
      .getByRole("columnheader", { name: /TTFT/ })
      .getByRole("button")
      .click();
    await expect(
      requests.getByRole("columnheader", { name: /TTFT/ }),
    ).toHaveAttribute("aria-sort", "ascending");
    await expect(first()).toContainText(id("00").slice(0, 9));
  });

  test("the slow filter keeps only requests above p90 and states its rule", async ({
    page,
  }) => {
    const { requests } = await open(page);
    await requests.getByRole("button", { name: /^速度：/ }).click();
    await page.getByRole("option", { name: "速度：慢請求" }).click();
    const rows = requests.getByTestId("request-identity");
    await expect(rows.filter({ hasText: id("cold").slice(0, 9) })).toHaveCount(
      1,
    );
    await expect(
      rows.filter({ hasText: id("queued").slice(0, 9) }),
    ).toHaveCount(1);
    await expect(rows.filter({ hasText: id("00").slice(0, 9) })).toHaveCount(0);
    expect(await rows.count()).toBeLessThanOrEqual(5);
    await expect(requests.getByTestId("slow-rule")).toContainText("p90");
  });

  test("the detail states the cause in one sentence and links to logs", async ({
    page,
  }) => {
    const { requests } = await open(page);
    await requests
      .getByRole("row")
      .filter({ hasText: id("cold").slice(0, 9) })
      .getByRole("button", { name: "詳情" })
      .click();
    const cause = page.getByTestId("request-cause").first();
    await expect(cause).toContainText("預填 41K 新 token");
    await expect(cause).toContainText("未命中");
    const link = cause.getByRole("link", { name: "查看這段時間的日誌" });
    await expect(link).toHaveAttribute("href", /^#\/logs\/range\/\d+\/\d+$/);
    await page.keyboard.press("Escape");
    await requests
      .getByRole("row")
      .filter({ hasText: id("queued").slice(0, 9) })
      .getByRole("button", { name: "詳情" })
      .click();
    await expect(page.getByTestId("request-cause").first()).toContainText(
      "排隊 12 s",
    );
  });

  test("an older engine without the ring stays calm", async ({ page }) => {
    const { errors, requests } = await open(page, { recent: false });
    await expect(
      requests.getByText(/此引擎版本沒有提供完成記錄/),
    ).toBeVisible();
    expect(errors).toEqual([]);
  });
});

test.describe("requests: phone profile (402x874)", () => {
  test("no overflow, full-width search, tray and sort on one row, actions reachable", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 402, height: 874 });
    const { errors, requests } = await open(page, { running: true });
    await expect(requests.getByTestId("requests-footer")).toBeVisible();
    const search = requests.getByRole("textbox", { name: "搜尋請求" });
    const sbox = (await search.boundingBox())!;
    expect(sbox.width).toBeGreaterThan(402 - 2 * 24);
    const tray = (await requests.getByRole("tablist").first().boundingBox())!;
    const sort = (await requests
      .getByRole("button", { name: /^排序：/ })
      .boundingBox())!;
    expect(Math.abs(tray.y - sort.y)).toBeLessThan(12);
    // Stat cards keep their content: no clipped sub-line, short cards.
    const card = requests.getByTestId("request-stats").locator("> *").first();
    expect((await card.boundingBox())!.height).toBeLessThan(180);
    for (const scope of ["進行中", "已結束"]) {
      await requests.getByRole("tab", { name: scope, exact: true }).click();
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - window.innerWidth,
      );
      expect(overflow).toBeLessThanOrEqual(1);
      const detailBtn = requests.getByRole("button", { name: "詳情" }).first();
      await detailBtn.scrollIntoViewIfNeeded();
      const b = (await detailBtn.boundingBox())!;
      expect(b.x + b.width).toBeLessThanOrEqual(402);
    }
    await requests.getByRole("button", { name: "詳情" }).first().click();
    await expect(page.getByRole("dialog", { name: "請求詳情" })).toBeVisible();
    expect(errors).toEqual([]);
  });
});
