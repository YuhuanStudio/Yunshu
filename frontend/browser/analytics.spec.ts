import { readFile } from "node:fs/promises";
import { expect, test, type Page, type Route } from "@playwright/test";

type Deferred = { promise: Promise<void>; resolve: () => void };

function deferred(): Deferred {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function makeStatus(sample: number) {
  const ttft = [120, 320, 720, 1_400, 4_000, null][(sample - 1) % 6] ?? null;
  return {
    object: "yunshu.status",
    version: "fixture-1.0",
    state: "ready",
    uptime_s: 3_600 + sample,
    load_error: null,
    models: [
      {
        id: "fixture/Qwen3.8-27B",
        type: "text",
        loaded: true,
        loading: false,
        pinned: false,
        size_gb: 18,
      },
    ],
    memory: {
      active_gb: 10 + sample / 10,
      cache_gb: 1 + sample / 20,
      peak_gb: 12,
      total_gb: 64,
      pressure: 0.2,
    },
    requests: {
      active: 3,
      queued: 1,
      prefill: 1,
      decode: 1,
      items: [
        {
          request_id: "qa-queued-01",
          elapsed_s: 8 + sample,
          phase: "queued",
          queue_position: 1,
        },
        {
          request_id: "qa-prefill-01",
          elapsed_s: 5 + sample,
          phase: "prefill",
          prompt_tokens: 400,
          cached_tokens: 120,
        },
        {
          request_id: "qa-decode-01",
          elapsed_s: 3 + sample,
          phase: "decode",
          prompt_tokens: 600,
          completion_tokens: 42 + sample,
          tokens_per_second: 28 + sample,
        },
      ],
    },
    last: {
      request_id: `qa-completed-${String(sample).padStart(2, "0")}`,
      prompt_tokens: 100 + sample,
      completion_tokens: 30 + sample,
      cached_tokens: 40 + sample,
      prefill_tps: 500 + sample * 10,
      decode_tps: 30 + sample,
      ttft_ms: ttft,
      t: 1_800_000_000 + sample,
    },
    throughput: {
      window_s: 60,
      requests: sample,
      prompt_tokens: 1_000 + sample * 100,
      completion_tokens: 500 + sample * 50,
      live_decode_tps: 25 + sample,
      mean_prefill_tps: 500 + sample * 10,
      mean_decode_tps: 30 + sample,
    },
  };
}

function installStatusFixture(page: Page, holdFirst = false) {
  let calls = 0;
  const started = deferred();
  const release = deferred();
  const unexpected: string[] = [];
  let firstReleased = false;

  const routeHandler = async (route: Route) => {
    const url = new URL(route.request().url());
    if (url.pathname !== "/v1/yunshu/status") {
      unexpected.push(`${route.request().method()} ${url.pathname}`);
      await route.fulfill({
        status: 404,
        contentType: "application/json",
        body: JSON.stringify({ detail: "fixture route not found" }),
      });
      return;
    }
    calls += 1;
    if (holdFirst && !firstReleased) {
      started.resolve();
      await release.promise;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(makeStatus(calls)),
    });
  };

  return {
    calls: () => calls,
    unexpected,
    firstStarted: started.promise,
    releaseFirst: () => {
      firstReleased = true;
      release.resolve();
    },
    install: () => page.route("**/v1/**", routeHandler),
  };
}

async function openDashboard(
  page: Page,
  holdFirst = false,
  virtualTime = true,
) {
  if (virtualTime)
    await page.clock.install({ time: new Date("2025-01-01T12:00:00.000Z") });
  const fixture = installStatusFixture(page, holdFirst);
  await fixture.install();
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("overview")).toBeVisible();
  return { fixture, pageErrors };
}

async function collectSixSamples(
  page: Page,
  fixture: ReturnType<typeof installStatusFixture>,
  virtualTime = true,
) {
  await expect(
    page.getByText("本頁開啟後採樣 · 1 筆 · 中斷期間不補資料", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "暫停更新" }).click();
  await expect(page.getByRole("button", { name: "恢復更新" })).toBeVisible();
  let expectedCalls = fixture.calls();

  for (let sample = 2; sample <= 6; sample += 1) {
    expectedCalls += 1;
    if (virtualTime) await page.clock.fastForward(2 * 60 * 1000);
    else await page.waitForTimeout(10);
    await page.getByRole("button", { name: "更新", exact: true }).click();
    await expect.poll(fixture.calls).toBe(expectedCalls);
    await expect(
      page.getByText(`本頁開啟後採樣 · ${sample} 筆 · 中斷期間不補資料`, {
        exact: true,
      }),
    ).toBeVisible();
  }
}

test.describe("analytics dashboard contracts", () => {
  test("distinguishes empty and hidden charts, syncs keyboard cursor, and exports the selected window", async ({
    page,
  }) => {
    const { fixture, pageErrors } = await openDashboard(page, true);
    await fixture.firstStarted;

    const overview = page.getByTestId("overview");
    const throughput = page.getByTestId("throughput-panel");
    const memory = page.getByTestId("memory-panel");
    await expect(
      throughput.getByText("等待第一筆採樣", { exact: true }),
    ).toBeVisible();
    await expect(
      memory.getByText("等待第一筆採樣", { exact: true }),
    ).toBeVisible();

    fixture.releaseFirst();
    await collectSixSamples(page, fixture);

    await throughput.getByRole("button", { name: "比較", exact: true }).click();
    const seriesGroup = throughput.getByRole("group", {
      name: "吞吐速度時序圖，單位 tok/s 顯示或隱藏序列",
    });
    const decode = seriesGroup.getByRole("button", {
      name: "Decode 平均",
      exact: true,
    });
    const prefill = seriesGroup.getByRole("button", {
      name: "Prefill 平均",
      exact: true,
    });
    await expect(decode).toHaveAttribute("aria-pressed", "true");
    await expect(prefill).toHaveAttribute("aria-pressed", "true");
    await decode.click();
    await prefill.click();
    const chart = throughput.getByRole("group", {
      name: "吞吐速度時序圖，單位 tok/s",
      exact: true,
    });
    await expect(chart.getByRole("status")).toHaveText(
      "序列已隱藏，可點選圖例重新顯示",
    );
    await decode.click();
    await expect(chart.getByRole("status")).toHaveCount(0);
    await prefill.click();

    await chart.focus();
    await page.keyboard.press("ArrowRight");
    const selectedX = await throughput
      .locator('[data-yunui="time-series-chart"]')
      .getAttribute("data-active-x");
    expect(selectedX).not.toBeNull();
    await expect(
      memory.locator('[data-yunui="time-series-chart"]'),
    ).toHaveAttribute("data-active-x", selectedX!);

    await overview.getByRole("button", { name: "5 分鐘", exact: true }).click();
    const downloadPromise = page.waitForEvent("download");
    await overview
      .getByRole("button", { name: "匯出觀測", exact: true })
      .click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe("yunshu-observations-5m.csv");
    const path = await download.path();
    expect(path).not.toBeNull();
    const csv = await readFile(path!, "utf8");
    expect(csv.charCodeAt(0)).toBe(0xfeff);
    expect(csv.trimEnd().split(/\r?\n/)).toHaveLength(4); // header plus the three samples in the selected five-minute window
    expect(csv).toContain('"mean_decode_tps_300s"');
    expect(fixture.unexpected).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test("filters request phases, synchronizes heatmap selections, and restores focus from latency details", async ({
    page,
  }) => {
    const { fixture, pageErrors } = await openDashboard(page, false, false);
    await expect(
      page.getByText("本頁開啟後採樣 · 1 筆 · 中斷期間不補資料", {
        exact: true,
      }),
    ).toBeVisible();
    await collectSixSamples(page, fixture, false);

    const phase = page.getByTestId("phase-panel");
    await phase.getByRole("button", { name: /Prefill/ }).click();
    await expect(
      phase.getByText("qa-prefill-01", { exact: true }),
    ).toBeVisible();
    await expect(phase.getByText("qa-decode-01", { exact: true })).toHaveCount(
      0,
    );
    await phase
      .getByRole("button", { name: "清除階段篩選", exact: true })
      .click();
    await expect(
      phase.getByText("qa-decode-01", { exact: true }),
    ).toBeVisible();

    const activity = page.getByTestId("activity-panel");
    const grid = activity.getByRole("grid", { name: "請求階段活動熱圖" });
    const cells = grid.getByRole("gridcell");
    await expect(cells).toHaveCount(48);
    await cells.nth(0).focus();
    await page.keyboard.press("ArrowRight");
    await expect(cells.nth(1)).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(activity.getByRole("status")).toBeVisible();

    // The final cell is the latest Decode bucket, which has a real fixture sample.
    await cells.last().click();
    const throughputChart = page
      .getByTestId("throughput-panel")
      .locator('[data-yunui="time-series-chart"]');
    const memoryChart = page
      .getByTestId("memory-panel")
      .locator('[data-yunui="time-series-chart"]');
    await expect(throughputChart).toHaveAttribute("data-active-x", /\d+/);
    await expect(memoryChart).toHaveAttribute(
      "data-active-x",
      (await throughputChart.getAttribute("data-active-x")) ?? "",
    );

    const latency = page.getByTestId("latency-panel");
    const firstBucket = latency.getByRole("button").first();
    await expect(firstBucket).toHaveAttribute("aria-label", /<250/);
    await firstBucket.focus();
    await firstBucket.click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText("延遲 <250 ms");
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await expect(firstBucket).toBeFocused();

    expect(fixture.calls()).toBeGreaterThanOrEqual(6);
    expect(fixture.unexpected).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
});
