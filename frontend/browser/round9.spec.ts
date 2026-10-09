import { expect, test, type Page } from "@playwright/test";

/** Round 9 features, each against what the engine reports; absent data shows the honest empty state. */
const model = (id: string, loaded: boolean) => ({
  id,
  type: "LLM",
  loaded,
  loading: false,
  pinned: false,
  size_gb: 5.8,
});

function statusWith(items: unknown[]) {
  return {
    object: "yunshu.status",
    version: "0.1.5",
    state: "running",
    uptime_s: 900,
    load_error: null,
    models: [model("Qwen3.5-9B", true), model("Llama-3.2-3B", true)],
    memory: { active_gb: 12, cache_gb: 1, peak_gb: 14, total_gb: 64 },
    requests: {
      active: items.length,
      queued: 0,
      prefill: 0,
      decode: items.length,
      items,
    },
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
}

async function install(
  page: Page,
  items: unknown[],
  host?: () => unknown,
  recent?: unknown,
) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status")
      return route.fulfill({ json: statusWith(items) });
    if (path === "/v1/yunshu/host" && host)
      return route.fulfill({ json: host() });
    if (path === "/v1/yunshu/requests/recent" && recent)
      return route.fulfill({ json: recent });
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
}

test("unload preview lists the in-flight requests on that model and asks again", async ({
  page,
}) => {
  await install(page, [
    {
      request_id: "req_a1",
      model: "Qwen3.5-9B",
      phase: "decode",
      elapsed_s: 12,
    },
    {
      request_id: "req_b2",
      model: "Llama-3.2-3B",
      phase: "decode",
      elapsed_s: 3,
    },
  ]);
  await page.goto("/console/#/models?action=unload&model=Qwen3.5-9B", {
    waitUntil: "domcontentloaded",
  });
  const impact = page.getByTestId("unload-impact");
  await expect(impact).toBeVisible();
  await expect(impact).toHaveAttribute("data-count", "1");
  await expect(impact).toContainText("req_a1");
  await expect(impact).not.toContainText("req_b2");
  await expect(page.getByRole("button", { name: "仍要卸載" })).toBeVisible();
});

test("unload preview says so when nothing runs on the model", async ({
  page,
}) => {
  await install(page, [
    {
      request_id: "req_b2",
      model: "Llama-3.2-3B",
      phase: "decode",
      elapsed_s: 3,
    },
  ]);
  await page.goto("/console/#/models?action=unload&model=Qwen3.5-9B", {
    waitUntil: "domcontentloaded",
  });
  const impact = page.getByTestId("unload-impact");
  await expect(impact).toContainText("沒有請求使用這個模型");
  await expect(
    page.getByRole("button", { name: "卸載", exact: true }),
  ).toBeVisible();
});

const hostBody = (telemetry: unknown) => ({
  object: "yunshu.host",
  sampled_at: Date.now() / 1000,
  thermal: { state: "normal", cpu_speed_limit_percent: 100 },
  power: { state: "ac", source: "AC Power", battery_percent: null },
  memory_pressure: { state: "normal", level: 1 },
  ...(telemetry ? { telemetry } : {}),
});
const sampleTelemetry = (over: Record<string, unknown> = {}) => ({
  state: "ok",
  sampled_at: Date.now() / 1000,
  interval_s: 1.0,
  watts: { cpu: 5.0, gpu: 30.5, ane: 0.0, dram: 2.0, package: 37.5 },
  gpu: { frequency_mhz: 900.0, active_ratio: 0.75 },
  temperature: {
    state: "ok",
    die_max_c: 70.0,
    die_mean_c: 65.0,
    battery_c: null,
  },
  reasons: {},
  ...over,
});

test("host panel: power, clock, activity, die temperature and OS limits with source and age", async ({
  page,
}) => {
  await install(page, [], () => hostBody(sampleTelemetry()));
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const panel = page.getByTestId("host-panel");
  await expect(panel).toBeVisible();
  const stats = panel.getByTestId("host-stats");
  await expect(stats).toContainText("30.5");
  await expect(stats).toContainText("900");
  await expect(stats).toContainText("MHz");
  await expect(stats).toContainText("70");
  await expect(stats).toContainText("活躍時間 75%");
  await expect(stats).toContainText("記憶體壓力 正常");
  await expect(panel).toContainText("取樣於");
  await expect(panel).not.toHaveAttribute("data-stale", "true");
});

test("host panel: a missing counter is a dash with its reason, never 0", async ({
  page,
}) => {
  await install(page, [], () =>
    hostBody(
      sampleTelemetry({
        state: "partial",
        watts: { cpu: 5.0, gpu: 30.5, ane: null, dram: null, package: null },
        reasons: {
          "watts.dram": "counter missing",
          "watts.ane": "no ANE channel",
        },
      }),
    ),
  );
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const panel = page.getByTestId("host-panel");
  await expect(panel).toBeVisible();
  await panel.getByRole("button", { name: "詳細資訊" }).click();
  const details = panel.getByTestId("host-details");
  await expect(details).toContainText("DRAM 功耗");
  await expect(details.getByText("—").first()).toBeVisible();
  await expect(details).not.toContainText("0 W");
  await expect(panel.getByTestId("host-reasons")).toContainText(
    "counter missing",
  );
  await expect(panel.getByTestId("host-stats")).toContainText("合計未回報");
});

test("host panel: unknown telemetry is one calm notice; an old engine shows no panel at all", async ({
  page,
}) => {
  await install(page, [], () =>
    hostBody({ state: "unknown", reason: "telemetry disabled" }),
  );
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("host-unavailable")).toContainText(
    "telemetry disabled",
  );
  await page.unroute("**/v1/**");
  await install(page, [], () => hostBody(null));
  await page.goto("/console/#/diagnostics");
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.getByTestId("live-panel").waitFor();
  await page.waitForTimeout(2500);
  await expect(page.getByTestId("host-panel")).toHaveCount(0);
});

test("host panel: a sample older than a few seconds is marked stale", async ({
  page,
}) => {
  await install(page, [], () =>
    hostBody(sampleTelemetry({ sampled_at: Date.now() / 1000 - 40 })),
  );
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const panel = page.getByTestId("host-panel");
  await expect(panel).toHaveAttribute("data-stale", "true");
  await expect(panel).toContainText("沒有更新");
});

const ring = (extra: Record<string, unknown>) => ({
  request_id: "ring-req-latency-001",
  t: 1_800_000_000,
  path: "/v1/chat/completions",
  model: "Qwen3.5-9B",
  status: 200,
  finish_reason: "stop",
  stream: true,
  t0_wall: 1_799_999_990,
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
  speculative: null,
  cancelled: false,
  ...extra,
});

async function openRingDetail(page: Page, entry: Record<string, unknown>) {
  await install(page, [], undefined, {
    object: "list",
    data: [entry],
    count: 1,
    capacity: 512,
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  const requests = page.getByTestId("requests");
  await requests.getByRole("tab", { name: "已結束", exact: true }).click();
  await requests
    .getByRole("row")
    .filter({ hasText: "ring-req-latency" })
    .getByRole("button", { name: "詳情", exact: true })
    .click();
  return page
    .getByRole("dialog", { name: "請求詳情" })
    .or(page.getByRole("complementary", { name: "請求詳情" }));
}

test("request detail: stage waterfall lists unreported stages as unreported, and energy is an estimate", async ({
  page,
}) => {
  const detail = await openRingDetail(
    page,
    ring({
      latency: {
        milestones_ms: {
          gateway_admit: 1,
          template_start: 2,
          template_end: 12,
          prefill_start: 14,
          prefill_end: 214,
          first_decode: 230,
          sse_first_flush: 231,
        },
        durations_ms: {
          model_lease: null,
          gateway_admit: 1,
          engine_queue: null,
          template_tokenize: 10,
          apc_lookup_restore: null,
          prefill: 200,
          first_decode: 16,
          sse_first_flush: 1,
        },
      },
      energy: {
        schema: "yunshu.energy.v1",
        prefill: { state: "unknown", reason: "no coverage", joules: null },
        decode: {
          state: "estimated",
          joules: 32,
          joules_per_token: 0.4,
          coverage_ratio: 1,
          extrapolated_s: 0.25,
        },
      },
    }),
  );
  const wf = detail.getByTestId("request-waterfall");
  await expect(wf).toBeVisible();
  await expect(wf.getByTestId("waterfall-null")).toContainText("模型租用");
  await expect(wf.getByTestId("waterfall-null")).toContainText("引擎排隊");
  await expect(wf.getByTestId("waterfall-null")).not.toContainText("預填");
  const energy = detail.getByTestId("request-energy");
  await expect(energy).toContainText("估算");
  await expect(energy).toContainText("0.4 J/token");
  await expect(energy).toContainText("未取得");
  await expect(energy).not.toContainText("0 J ");
});

test("request detail: an engine without stage fields gets one compact notice", async ({
  page,
}) => {
  const detail = await openRingDetail(page, ring({}));
  await expect(detail.getByTestId("waterfall-unsupported")).toBeVisible();
  await expect(detail.getByTestId("request-waterfall")).toHaveCount(0);
});

test("latency distribution: P50/P90 only for groups with at least 20 requests", async ({
  page,
}) => {
  const data = [
    ...Array.from({ length: 25 }, (_, i) =>
      ring({
        request_id: `ring-warm-${i}`,
        ttft_ms: 100 + i,
        prompt_tokens: 1000,
        cached_tokens: 900,
      }),
    ),
    ...Array.from({ length: 5 }, (_, i) =>
      ring({
        request_id: `ring-cold-${i}`,
        ttft_ms: 9000,
        prompt_tokens: 1000,
        cached_tokens: 0,
      }),
    ),
  ];
  await install(page, [], undefined, {
    object: "list",
    data,
    count: data.length,
    capacity: 512,
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  const panel = page.getByTestId("latency-distribution");
  await expect(panel).toBeVisible();
  const warm = panel.locator('tr[data-group="warm"]');
  const cold = panel.locator('tr[data-group="cold"]');
  await expect(warm).toContainText("25");
  await expect(warm).not.toContainText("—");
  await expect(cold).toContainText("5");
  await expect(cold).toContainText("—");
  await expect(panel).not.toContainText("9 s");
});

test("latency distribution on a phone fits without sideways scroll", async ({
  page,
}) => {
  const data = Array.from({ length: 25 }, (_, i) =>
    ring({ request_id: `ring-w-${i}`, ttft_ms: 100 + i * 40 }),
  );
  await install(page, [], undefined, {
    object: "list",
    data,
    count: data.length,
    capacity: 512,
  });
  await page.setViewportSize({ width: 402, height: 874 });
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  const panel = page.getByTestId("latency-distribution");
  await expect(panel).toBeVisible();
  const de = await page.evaluate(() => ({
    sw: document.documentElement.scrollWidth,
    cw: document.documentElement.clientWidth,
  }));
  expect(de.sw).toBeLessThanOrEqual(de.cw + 1);
  const scrolls = await panel.evaluate((el) =>
    [...el.querySelectorAll<HTMLElement>("*")]
      .filter(
        (n) =>
          /auto|scroll/.test(getComputedStyle(n).overflowX) &&
          n.scrollWidth > n.clientWidth + 1,
      )
      .map((n) => `${n.tagName}.${String(n.className).slice(0, 40)}`),
  );
  expect(scrolls).toEqual([]);
});
