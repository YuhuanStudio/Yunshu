import { expect, test, type Page, type Route } from "@playwright/test";

const items = [
  {
    request_id: "qa-queue-01",
    elapsed_s: 6.2,
    phase: "queued",
    model: "org/qwen-mlx",
    queue_position: 2,
    queue_est_wait_ms: 1800,
  },
  {
    request_id: "qa-prefill-01",
    elapsed_s: 3.4,
    phase: "prefill",
    model: "org/qwen-mlx",
    prompt_tokens: 4000,
    cached_tokens: 1000,
    processed_tokens: 2500,
    percent: 62.5,
    eta_s: 1.2,
    tokens_per_second: 900,
  },
  {
    request_id: "qa-decode-01",
    elapsed_s: 9,
    phase: "decode",
    model: "org/qwen-mlx",
    prompt_tokens: 600,
    cached_tokens: 200,
    completion_tokens: 42,
    tokens_per_second: 28.5,
  },
];

type Recent = Record<string, unknown>;
function ringEntry(i: number, extra: Recent = {}): Recent {
  return {
    request_id: `ring-req-0123456789abcdef-${String(i).padStart(3, "0")}`,
    t: 1_800_000_000 - i,
    path: "/v1/chat/completions",
    model: "org/qwen-mlx",
    status: 200,
    finish_reason: "stop",
    stream: true,
    t0_wall: 1_799_999_990 - i,
    offsets_ms: {
      arrive: 0,
      admit: 40,
      first_token: 340,
      last_token: 2340,
      done: 2360,
    },
    queue_wait_ms: 40,
    ttft_ms: 300,
    prompt_tokens: 4000,
    cached_tokens: 1000,
    completion_tokens: 80,
    prefill_tps: 800,
    decode_tps: 40,
    cache: { tier: "ram", reload_ms: null },
    speculative: { mode: "mtp", acceptance_rate: 0.8, rounds: 20 },
    cancelled: false,
    ...extra,
  };
}

function installFixture(page: Page, recent: Recent[] | null = null) {
  let sample = 0;
  const detailPaths: string[] = [];
  const recentCalls: string[] = [];
  const unexpected: string[] = [];
  const json = (route: Route, status: number, body: unknown) =>
    route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  const status = () => {
    sample += 1;
    const n = Math.min(sample, 3);
    return {
      object: "yunshu.status",
      version: "fixture-1.0",
      state: "ready",
      uptime_s: 3_600 + sample,
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
      requests: { active: 3, queued: 1, prefill: 1, decode: 1, items },
      last: {
        request_id: `qa-done-${n}`,
        prompt_tokens: 1000,
        completion_tokens: 200 + n,
        cached_tokens: 250,
        prefill_tps: 800,
        decode_tps: 40 + n,
        ttft_ms: 300 + n * 10,
        t: 1_800_000_000 + n,
        speculative: { mode: "mtp", acceptance_rate: 0.82, rounds: 12 },
      },
      throughput: {
        window_s: 60,
        requests: n,
        prompt_tokens: 1000,
        completion_tokens: 200,
        live_decode_tps: 33 + sample,
        mean_prefill_tps: 800,
        mean_decode_tps: 35,
      },
    };
  };
  return {
    detailPaths,
    recentCalls,
    unexpected,
    install: () =>
      page.route("**/v1/**", async (route) => {
        const path = new URL(route.request().url()).pathname;
        if (path === "/v1/yunshu/status") return json(route, 200, status());
        // The shell may also ask for the history ring and the memory ledger; this page uses neither.
        if (path === "/v1/yunshu/history" || path === "/v1/yunshu/memory")
          return json(route, 404, { detail: "Not Found" });
        if (path === "/v1/yunshu/requests/recent") {
          recentCalls.push(route.request().url());
          return recent
            ? json(route, 200, {
                object: "list",
                data: recent,
                count: recent.length,
                capacity: 512,
              })
            : json(route, 404, { detail: "Not Found" });
        }
        const match = /^\/v1\/requests\/(.+)$/.exec(path);
        const item = items.find((i) => i.request_id === match?.[1]);
        if (route.request().method() === "GET" && item) {
          detailPaths.push(path);
          return json(route, 200, item);
        }
        unexpected.push(`${route.request().method()} ${path}`);
        return json(route, 404, { detail: "fixture endpoint not found" });
      }),
  };
}

async function open(page: Page, recent: Recent[] | null = null) {
  const fixture = installFixture(page, recent);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await fixture.install();
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  const requests = page.getByTestId("requests");
  await expect(
    requests.getByRole("row").filter({ hasText: "qa-decode-01" }),
  ).toBeVisible();
  return { fixture, pageErrors, requests };
}

test.describe("requests page trace", () => {
  test("spark strip, live list and distinct observed finished requests with speculative stats", async ({
    page,
  }) => {
    const { fixture, pageErrors, requests } = await open(page);
    await expect(
      requests.getByText("進行中請求", { exact: true }),
    ).toBeVisible();
    await expect(
      requests.getByText("TTFT（已結束請求）", { exact: true }),
    ).toBeVisible();
    await expect(requests.getByText("解碼速度", { exact: true })).toBeVisible();
    await expect(
      requests.getByText("前綴命中率", { exact: true }),
    ).toBeVisible();
    const live = requests.getByRole("row").filter({ hasText: "qa-decode-01" });
    await expect(live).toContainText("28.5");
    await expect(
      requests.getByRole("row").filter({ hasText: "qa-prefill-01" }),
    ).toContainText("預填 63%");

    await requests.getByRole("tab", { name: "已結束", exact: true }).click();
    // Three distinct ids appear over successive 3 s polls; a repeated `last` is deduped.
    await expect(
      requests.getByRole("row").filter({ hasText: "qa-done-3" }),
    ).toBeVisible({ timeout: 15_000 });
    // The same `last` keeps arriving on later polls; it must stay one row.
    await page.waitForTimeout(3_500);
    await expect(
      requests.getByRole("row").filter({ hasText: "qa-done-3" }),
    ).toHaveCount(1);
    const done = requests.getByRole("row").filter({ hasText: "qa-done-3" });
    await expect(done).toContainText("mtp · 接受 82%");
    await expect(done).toContainText("330 ms");
    await expect(
      requests.getByText(/此引擎版本沒有提供完成記錄/),
    ).toBeVisible();
    // The shell also asks for the downloads list (another page's endpoint); not this page's concern.
    expect(
      fixture.unexpected.filter((u) => !u.includes("/yunshu/downloads")),
    ).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test("prefill detail shows stages, token proportions and live progress without invented times", async ({
    page,
  }) => {
    const { fixture, pageErrors, requests } = await open(page);
    await requests
      .getByRole("row")
      .filter({ hasText: "qa-prefill-01" })
      .getByRole("button", { name: "詳情", exact: true })
      .click();
    // Below xl the detail is a Sheet (dialog); from xl it is the inspector column (complementary).
    const dialog = page
      .getByRole("dialog", { name: "請求詳情" })
      .or(page.getByRole("complementary", { name: "請求詳情" }));
    await expect(
      dialog.getByRole("img", {
        name: /Token 組成：快取命中 1000，本次預填 3000，輸出 0/,
      }),
    ).toBeVisible();
    await expect(dialog.getByText("62.5%")).toBeVisible();
    await expect(dialog.getByText("1,000 tok", { exact: true })).toBeVisible();
    await expect(dialog.getByText("3,000 tok", { exact: true })).toBeVisible();
    await expect(
      dialog
        .getByRole("list", { name: "請求階段" })
        .locator('[aria-current="step"]'),
    ).toContainText("預填");
    await expect(dialog.getByText(/不代表耗時/)).toBeVisible();
    expect(fixture.detailPaths.length).toBeGreaterThan(0);
    expect(pageErrors).toEqual([]);
  });

  test("queued detail shows position and the server's wait estimate only", async ({
    page,
  }) => {
    const { pageErrors, requests } = await open(page);
    await requests
      .getByRole("row")
      .filter({ hasText: "qa-queue-01" })
      .getByRole("button", { name: "詳情", exact: true })
      .click();
    // Below xl the detail is a Sheet (dialog); from xl it is the inspector column (complementary).
    const dialog = page
      .getByRole("dialog", { name: "請求詳情" })
      .or(page.getByRole("complementary", { name: "請求詳情" }));
    await expect(dialog.getByText("佇列位置")).toBeVisible();
    await expect(dialog).toContainText("1,800ms");
    await expect(dialog.getByText(/尚未回報 prompt token 數/)).toBeVisible();
    expect(pageErrors).toEqual([]);
  });

  test("finished detail keeps speculative mode, acceptance and rounds", async ({
    page,
  }) => {
    const { fixture, pageErrors, requests } = await open(page);
    await requests.getByRole("tab", { name: "已結束", exact: true }).click();
    await requests
      .getByRole("row")
      .filter({ hasText: "qa-done-3" })
      .getByRole("button", { name: "詳情", exact: true })
      .click();
    // Below xl the detail is a Sheet (dialog); from xl it is the inspector column (complementary).
    const dialog = page
      .getByRole("dialog", { name: "請求詳情" })
      .or(page.getByRole("complementary", { name: "請求詳情" }));
    await expect(dialog.getByText("mtp", { exact: true })).toBeVisible();
    await expect(dialog).toContainText("82%");
    await expect(dialog.getByText("回合")).toBeVisible();
    await expect(dialog.getByText("750 tok", { exact: true })).toBeVisible();
    expect(fixture.detailPaths).toEqual([]); // finished requests are not polled
    expect(pageErrors).toEqual([]);
  });
});

test.describe("requests page finished-request ring", () => {
  const ring = [
    ...Array.from({ length: 118 }, (_, i) => ringEntry(i + 3)),
    ringEntry(1, { status: 504, finish_reason: "error", cancelled: false }),
    ringEntry(2, { cancelled: true, finish_reason: "abort", status: 200 }),
  ];
  const withRing = async (page: Page) => {
    const o = await open(page, ring);
    await o.requests.getByRole("tab", { name: "已結束", exact: true }).click();
    return o;
  };

  test("rows are capped, show more grows them, ids keep both ends and tok/s never overlaps", async ({
    page,
  }) => {
    const { fixture, pageErrors, requests } = await withRing(page);
    const rows = requests.getByRole("row");
    await expect(rows.filter({ hasText: "ring-req-" })).toHaveCount(50);
    await requests.getByRole("button", { name: /顯示更多/ }).click();
    await expect(rows.filter({ hasText: "ring-req-" })).toHaveCount(100);
    const first = rows.filter({ hasText: "ring-req-" }).first();
    const id = first.locator("span[title^='ring-req-']");
    await expect(id).toHaveAttribute(
      "title",
      /ring-req-0123456789abcdef-\d{3}/,
    );
    await expect(id).toContainText("…");
    const boxes = await first.locator("td").evaluateAll((cells) =>
      cells
        .map((c) => c.getBoundingClientRect())
        .filter((r) => r.width > 0)
        .map((r) => [r.left, r.right]),
    );
    for (let i = 1; i < boxes.length; i++)
      expect(boxes[i][0]).toBeGreaterThanOrEqual(
        (boxes[i - 1][1] as number) - 1,
      );
    await expect(first).toContainText("命中 1,000 / 4,000");
    // The shell also asks for the downloads list (another page's endpoint); not this page's concern.
    expect(
      fixture.unexpected.filter((u) => !u.includes("/yunshu/downloads")),
    ).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test("status filter uses the real status and cancel flag; search narrows by id", async ({
    page,
  }) => {
    const { pageErrors, requests } = await withRing(page);
    const rows = requests.getByRole("row");
    const pick = async (label: string) => {
      await requests.getByRole("button", { name: /^結果：/ }).click();
      await page.getByRole("option", { name: label, exact: true }).click();
    };
    await requests.getByRole("tab", { name: "已結束", exact: true }).click();
    await pick("結果：錯誤");
    await expect(rows.filter({ hasText: "ring-req-" })).toHaveCount(1);
    await expect(rows.filter({ hasText: "-001" })).toBeVisible();
    await pick("結果：已取消");
    await expect(rows.filter({ hasText: "ring-req-" })).toHaveCount(1);
    await expect(rows.filter({ hasText: "-002" })).toBeVisible();
    await pick("結果：全部");
    await requests.getByLabel("搜尋請求").fill("0123456789abcdef-017");
    await expect(rows.filter({ hasText: "ring-req-" })).toHaveCount(1);
    expect(pageErrors).toEqual([]);
  });

  test("detail shows the wall-clock breakdown, a time-proportional timeline and no invented text tabs", async ({
    page,
  }) => {
    const { pageErrors, requests } = await withRing(page);
    await requests
      .getByRole("row")
      .filter({ hasText: "-005" })
      .getByRole("button", { name: "詳情", exact: true })
      .click();
    const panel = page
      .getByRole("dialog", { name: "請求詳情" })
      .or(page.getByRole("complementary", { name: "請求詳情" }));
    const breakdown = panel.getByRole("region", { name: "耗時分解" });
    await expect(breakdown.locator("p", { hasText: /^排隊$/ })).toBeVisible();
    await expect(breakdown).toContainText("40 ms");
    await expect(breakdown).toContainText("300 ms");
    await expect(breakdown).toContainText("2 s");
    await expect(breakdown).toContainText("2.4 s");
    await expect(breakdown).toContainText("命中 token");
    await expect(breakdown).toContainText("80%");
    const timeline = panel.getByLabel(
      "請求 ring-req-0123456789abcdef-005 時間軸",
    );
    await expect(timeline).toBeVisible();
    await expect(timeline).toContainText("預填");
    await expect(timeline).toContainText("解碼");
    await expect(panel.getByRole("tab")).toHaveCount(0);
    await expect(panel.getByText(/引擎不保存 prompt 與輸出文字/)).toBeVisible();
    expect(pageErrors).toEqual([]);
  });

  test("footer prints the retention contract and the prompt-token totals", async ({
    page,
  }) => {
    const { requests } = await withRing(page);
    const footer = requests.getByTestId("requests-footer");
    await expect(footer).toContainText("引擎保留最近 512 筆");
    await expect(footer).toContainText(/自 \d{2}:\d{2} 起 120 筆請求/);
    await expect(footer).toContainText(
      "讀取 480,000 個 prompt token（重用 120,000）",
    );
  });

  test("an older server without the endpoint falls back to sampled rows and says so", async ({
    page,
  }) => {
    const { fixture, pageErrors, requests } = await open(page, null);
    await expect(requests.getByTestId("requests-footer")).toContainText(
      "此引擎版本沒有提供完成記錄",
    );
    await expect(requests.getByTestId("requests-footer")).not.toContainText(
      "伺服器保留最近",
    );
    const early = fixture.recentCalls.length; // StrictMode and the shell signals hook may each ask
    expect(early).toBeLessThanOrEqual(4);
    await page.waitForTimeout(6_000);
    // 404 stops this page's polling; the shell signals hook keeps its own slow poll.
    expect(fixture.recentCalls.length).toBeLessThanOrEqual(early + 2);
    // The shell also asks for the downloads list (another page's endpoint); not this page's concern.
    expect(
      fixture.unexpected.filter((u) => !u.includes("/yunshu/downloads")),
    ).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
});
