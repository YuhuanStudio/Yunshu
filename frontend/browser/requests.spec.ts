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

function installFixture(page: Page) {
  let sample = 0;
  const detailPaths: string[] = [];
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
    unexpected,
    install: () =>
      page.route("**/v1/**", async (route) => {
        const path = new URL(route.request().url()).pathname;
        if (path === "/v1/yunshu/status") return json(route, 200, status());
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

async function open(page: Page) {
  const fixture = installFixture(page);
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
    await expect(
      requests.getByText("Decode 速度", { exact: true }),
    ).toBeVisible();
    await expect(
      requests.getByText("前綴快取命中", { exact: true }),
    ).toBeVisible();
    const live = requests.getByRole("row").filter({ hasText: "qa-decode-01" });
    await expect(live).toContainText("28.5");
    await expect(
      requests.getByRole("row").filter({ hasText: "qa-prefill-01" }),
    ).toContainText("Prefill 63%");

    await requests.getByRole("button", { name: "觀測到的已結束請求" }).click();
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
    await expect(done).toContainText("330 ms TTFT");
    await expect(
      requests.getByText(/本頁開啟後觀測到的已完成請求/),
    ).toBeVisible();
    expect(fixture.unexpected).toEqual([]);
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
    ).toContainText("Prefill");
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
    await requests.getByRole("button", { name: "觀測到的已結束請求" }).click();
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
