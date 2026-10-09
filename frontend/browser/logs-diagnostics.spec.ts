import { expect, test, type Page, type Route } from "@playwright/test";

// `#/logs` once the shell routes it; R63_LOGS_URL points at a standalone mount while it does not.
const LOGS_URL = process.env.R63_LOGS_URL ?? "/console/#/logs";

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });

const T = 1_800_000_000;
const rec = (id: number, level: string, msg: string) => ({
  id,
  t: T + id,
  level,
  logger: id % 2 ? "yunshu.engine" : "uvicorn.access",
  msg,
});
const base = [
  rec(1, "INFO", "model loaded org/qwen-mlx"),
  rec(2, "WARNING", "memory pressure 0.91"),
  rec(3, "ERROR", "request failed: Authorization: [redacted]"),
  ...Array.from({ length: 60 }, (_, i) => rec(4 + i, "INFO", `step ${i}`)),
];

async function mock(page: Page, opts: { logs?: "ok" | 404 | 403 } = {}) {
  const calls: string[] = [];
  const queue: ReturnType<typeof rec>[] = [];
  const pageErrors: string[] = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));
  await page.addInitScript(() => {
    (window as unknown as { __copied: string[] }).__copied = [];
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (v: string) =>
          (window as unknown as { __copied: string[] }).__copied.push(v),
      },
    });
  });
  await page.route("**/v1/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/yunshu/logs/stream") {
      calls.push(`stream ${url.search}`);
      if (opts.logs === 404) return json(route, { detail: "nf" }, 404);
      const body =
        ": keepalive\n\n" +
        queue
          .splice(0)
          .map((r) => `id: ${r.id}\ndata: ${JSON.stringify(r)}\n\n`)
          .join("");
      return route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body,
      });
    }
    if (url.pathname === "/v1/yunshu/logs") {
      calls.push(`logs ${url.search}`);
      if (opts.logs === 404) return json(route, { detail: "nf" }, 404);
      if (opts.logs === 403) return json(route, { detail: "admin" }, 403);
      const level = url.searchParams.get("level");
      const q = url.searchParams.get("q");
      const rank: Record<string, number> = { INFO: 20, WARNING: 30, ERROR: 40 };
      const rows = base.filter(
        (r) =>
          (!level || rank[r.level] >= rank[level]) && (!q || r.msg.includes(q)),
      );
      return json(route, {
        records: rows,
        next_id: 63,
        dropped: 0,
        capacity: 2000,
        server_time: T + 70,
      });
    }
    return json(route, { detail: "nf" }, 404);
  });
  return { calls, queue, pageErrors };
}

test.describe("logs page", () => {
  test("level and text filters go to the server; redaction is stated; mono is log text only", async ({
    page,
  }) => {
    const m = await mock(page);
    await page.goto(LOGS_URL);
    const logs = page.getByTestId("logs");
    await expect(logs.getByText("model loaded org/qwen-mlx")).toBeVisible();
    await expect(logs.getByTestId("logs-redaction")).toContainText("伺服器端");
    await logs.getByRole("tab", { name: "錯誤", exact: true }).click();
    await expect(logs.getByText("request failed")).toBeVisible();
    await expect(logs.getByText("model loaded")).toHaveCount(0);
    expect(
      m.calls.some((c) => c.startsWith("logs") && c.includes("level=ERROR")),
    ).toBe(true);
    await logs.getByRole("tab", { name: "全部", exact: true }).click();
    await logs.getByRole("textbox", { name: "搜尋日誌" }).fill("pressure");
    await expect(logs.getByText("memory pressure 0.91")).toBeVisible();
    await expect(logs.getByText("model loaded")).toHaveCount(0);
    expect(m.calls.some((c) => c.includes("q=pressure"))).toBe(true);
    const fonts = await logs
      .getByText("memory pressure 0.91")
      .evaluate((e) => [
        getComputedStyle(e).fontFamily,
        getComputedStyle(e.parentElement!.firstElementChild!).fontFamily,
      ]);
    expect(fonts[0]).not.toEqual(fonts[1]);
    expect(m.pageErrors).toEqual([]);
  });

  test("live tail appends, pausing stops it, scrolling up locks and 跳到最新 returns", async ({
    page,
  }) => {
    const m = await mock(page);
    await page.goto(LOGS_URL);
    const logs = page.getByTestId("logs");
    await expect(logs.getByText("step 59")).toBeVisible();
    await expect(logs.getByTestId("logs-live")).toContainText(
      /即時追蹤|重新連線/,
    );
    // The mocked stream ends after each batch, so the client reconnects; a real one stays open.
    m.queue.push(rec(64, "INFO", "live line A"));
    await expect(logs.getByText("live line A")).toBeVisible({ timeout: 8000 });
    // Scroll up: follow unlocks and the jump button appears with a new-record count.
    await logs.getByTestId("logs-scroller").evaluate((e) => (e.scrollTop = 0));
    m.queue.push(rec(65, "INFO", "live line B"));
    const jump = logs.getByRole("button", { name: /跳到最新/ });
    await expect(jump).toBeVisible({ timeout: 8000 });
    await expect(jump).toContainText("1 筆新紀錄");
    await jump.click();
    await expect(logs.getByText("live line B")).toBeInViewport();
    await expect(jump).toHaveCount(0);
    // Pause: nothing new arrives; resume catches up from the cursor.
    await logs.getByRole("button", { name: "暫停" }).click();
    await expect(logs.getByTestId("logs-live")).toContainText("已暫停");
    m.queue.push(rec(66, "INFO", "while paused"));
    await page.waitForTimeout(2500);
    await expect(logs.getByText("while paused")).toHaveCount(0);
    await logs.getByRole("button", { name: "繼續" }).click();
    await expect(logs.getByText("while paused")).toBeVisible({ timeout: 8000 });
  });

  test("copy a line and download the visible lines", async ({ page }) => {
    await mock(page);
    await page.goto(LOGS_URL);
    const logs = page.getByTestId("logs");
    await expect(logs.getByText("memory pressure 0.91")).toBeVisible();
    const row = logs.getByText("memory pressure 0.91").locator("..");
    // Keyboard users reach the per-line copy button: it is in the tab order.
    const copyLine = row.getByRole("button", { name: "複製這一行" });
    await expect(copyLine).not.toHaveAttribute("tabindex", "-1");
    await copyLine.focus();
    await expect(copyLine).toBeFocused();
    await row.hover();
    await copyLine.click();
    const copied = await page.evaluate(
      () => (window as unknown as { __copied: string[] }).__copied,
    );
    expect(copied.at(-1)).toMatch(/WARNING .*memory pressure 0\.91$/);
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      logs.getByRole("button", { name: "下載可見行" }).click(),
    ]);
    expect(download.suggestedFilename()).toMatch(/^yunshu-logs-.*\.log$/);
  });

  test("a time-window link shows a fixed range without live tail", async ({
    page,
  }) => {
    const m = await mock(page);
    await page.goto(LOGS_URL);
    await page.evaluate(
      ([a, b]) => (location.hash = `#/logs/range/${a}/${b}`),
      [T + 2, T + 4],
    );
    const logs = page.getByTestId("logs");
    await expect(logs.getByTestId("logs-live")).toContainText("固定時間範圍");
    await expect(logs.getByText("request failed")).toBeVisible();
    await expect(logs.getByText("step 3")).toHaveCount(0);
    expect(
      m.calls.some((c) => c.startsWith("logs") && c.includes(`since=${T + 2}`)),
    ).toBe(true);
  });

  test("older engines (404) and missing admin (403) show a calm state", async ({
    page,
  }) => {
    for (const [mode, text] of [
      [404, "這個引擎還沒有日誌介面"],
      [403, "需要管理權限"],
    ] as const) {
      const p = await page.context().newPage();
      const m = await mock(p, { logs: mode });
      await p.goto(LOGS_URL);
      await expect(p.getByText(text)).toBeVisible();
      // A missing token is fixed in Settings: the notice links there (and only then).
      await expect(p.getByTestId("unavailable-to-settings")).toHaveCount(
        mode === 403 ? 1 : 0,
      );
      expect(m.pageErrors).toEqual([]);
      await p.close();
    }
  });
});

const status = {
  object: "yunshu.status",
  version: "t-1",
  state: "running",
  uptime_s: 90,
  load_error: null,
  models: [
    {
      id: "org/qwen-mlx",
      type: "LLM",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 5,
    },
  ],
  memory: { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 64 },
  requests: { active: 0, queued: 0, prefill: 0, decode: 0, items: [] },
  last: null,
  throughput: {
    window_s: 60,
    requests: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    live_decode_tps: null,
    mean_prefill_tps: null,
    mean_decode_tps: 42,
  },
};

async function diag(
  page: Page,
  over: Record<string, unknown>,
  bundle: 200 | 404 | 403,
) {
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status")
      return json(route, { ...status, ...over });
    if (path === "/v1/yunshu/bundle")
      return bundle === 200
        ? route.fulfill({
            status: 200,
            contentType: "application/json",
            headers: {
              "content-disposition":
                'attachment; filename="yunshu-bundle-1.json"',
            },
            body: "{}",
          })
        : json(route, { detail: "x" }, bundle);
    return json(route, { detail: "nf" }, 404);
  });
  await page.route("**/debug/**", (route) =>
    json(route, { detail: "off" }, 404),
  );
  await page.goto("/console/#/diagnostics");
}

test.describe("diagnostics", () => {
  test("a healthy engine gets 健康, a queue or memory pressure gets 注意, a load error 異常", async ({
    page,
  }) => {
    await diag(page, {}, 200);
    const v = page.getByTestId("health-verdict");
    await expect(v).toContainText("健康");
    // A healthy engine shows the compact chip only: the logs link keeps its slot but is hidden.
    await expect(v.locator("a[href='#/logs']")).toBeHidden();
    await page.unroute("**/v1/**");
    await diag(
      page,
      { requests: { ...status.requests, active: 4, queued: 4 } },
      200,
    );
    await expect(page.getByTestId("health-verdict")).toContainText("注意");
    await expect(
      page
        .getByTestId("health-verdict")
        .getByRole("link", { name: "查看日誌" }),
    ).toHaveAttribute("href", "#/logs");
    await expect(page.getByTestId("health-checks")).toContainText("排隊");
    await page.unroute("**/v1/**");
    await diag(page, { load_error: "weights missing" }, 200);
    await expect(page.getByTestId("health-verdict")).toContainText("異常");
  });

  test("an unreadable /debug/system is a neutral 無法讀取, never an 異常 verdict", async ({
    page,
  }) => {
    await diag(page, {}, 200);
    await page.unroute("**/debug/**");
    await page.route("**/debug/**", (route) =>
      json(route, { detail: "read-only proxy" }, 403),
    );
    await page.goto("/console/#/diagnostics");
    const v = page.getByTestId("health-verdict");
    await expect(v).toContainText("健康");
    await expect(v).not.toContainText("異常");
    const checks = page.getByTestId("health-checks");
    await expect(checks).toContainText("無法讀取");
    await expect(checks).toContainText("HTTP 403");
  });

  test("下載診斷包 saves the server file; 404 and 403 explain themselves", async ({
    page,
  }) => {
    await diag(page, {}, 200);
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "下載診斷包" }).click(),
    ]);
    expect(download.suggestedFilename()).toBe("yunshu-bundle-1.json");
    await expect(page.getByTestId("bundle-note")).toContainText(
      "yunshu-bundle-1.json",
    );
    await page.unroute("**/v1/**");
    await diag(page, {}, 404);
    await page.getByRole("button", { name: "下載診斷包" }).click();
    await expect(page.getByTestId("bundle-note")).toContainText(
      "還沒有診斷包介面",
    );
    await page.unroute("**/v1/**");
    await diag(page, {}, 403);
    await page.getByRole("button", { name: "下載診斷包" }).click();
    await expect(page.getByTestId("bundle-note")).toContainText("管理權限");
  });
});

test("phone profile (402x874): logs and diagnostics have no horizontal overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 402, height: 874 });
  await mock(page);
  await page.goto(LOGS_URL);
  await expect(page.getByText("step 59")).toBeVisible();
  const over = () =>
    page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
  expect(await over()).toBeLessThanOrEqual(1);
  for (const name of ["暫停", "複製可見行", "下載可見行"])
    await expect(page.getByRole("button", { name })).toBeInViewport();
});
