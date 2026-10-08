import { expect, test, type Page } from "@playwright/test";

/** Round 10 panels, each against what the engine reports; absent data shows the honest empty state. */
export const status = () => ({
  object: "yunshu.status",
  version: "0.1.5",
  state: "running",
  uptime_s: 900,
  load_error: null,
  models: [
    {
      id: "Qwen3.5-9B",
      type: "LLM",
      loaded: true,
      loading: false,
      pinned: false,
      size_gb: 5.8,
    },
  ],
  memory: { active_gb: 12, cache_gb: 1, peak_gb: 14, total_gb: 64 },
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
});

type Routes = Record<string, unknown | (() => unknown)>;
export async function install(page: Page, routes: Routes = {}) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") return route.fulfill({ json: status() });
    const hit = routes[path];
    if (hit !== undefined)
      return route.fulfill({
        json: typeof hit === "function" ? (hit as () => unknown)() : hit,
      });
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
}

const ring = (extra: Record<string, unknown>) => ({
  request_id: "ring-1",
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
const list = (data: unknown[]) => ({
  object: "list",
  data,
  count: data.length,
  capacity: 512,
});

test("speculation panel: weighted acceptance with denominators, per-depth stated as not reported", async ({
  page,
}) => {
  await install(page, {
    "/v1/yunshu/requests/recent": list([
      ring({
        request_id: "s1",
        speculative: { mode: "mtp", drafted: 100, accepted: 80, rounds: 20 },
      }),
      ring({
        request_id: "s2",
        speculative: { mode: "mtp", drafted: 10, accepted: 0, rounds: 2 },
      }),
      ring({ request_id: "s3", speculative: { mode: "mtp" } }),
      ring({ request_id: "p1" }),
    ]),
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  const panel = page.getByTestId("speculation-panel");
  await expect(panel).toBeVisible();
  const row = panel.locator('tr[data-mode="mtp"]');
  await expect(row).toContainText("72.7%");
  await expect(row).toContainText("80 / 110");
  await expect(panel).toContainText("沒有推測解碼紀錄");
  await expect(panel).toContainText("只回報了模式");
  await expect(panel).toContainText("每個深度的接受率");
});

test("speculation panel says plainly when no request reports speculation", async ({
  page,
}) => {
  await install(page, {
    "/v1/yunshu/requests/recent": list([ring({ request_id: "p1" })]),
  });
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("speculation-panel")).toContainText(
    "都沒有推測解碼紀錄",
  );
});

test("support bundle preview lists included, redacted and excluded for both exports", async ({
  page,
}) => {
  await install(page);
  await page.goto("/console/#/diagnostics", { waitUntil: "domcontentloaded" });
  const preview = page.getByTestId("bundle-preview");
  await preview.getByRole("button", { name: "支援包內容預覽" }).click();
  const engine = preview.locator('[data-bundle="engine"]');
  await expect(engine).toContainText("提示詞、回覆內容、請求本文、模型權重");
  await expect(engine).toContainText("[OMITTED]");
  const copy = preview.locator('[data-bundle="page"]');
  await expect(copy).toContainText("存取權杖、API 金鑰");
  await expect(copy).toContainText("/debug/system");
});

test("speculation counters from /debug/spec-decode appear next to the request summary", async ({
  page,
}) => {
  await install(page, {
    "/v1/yunshu/requests/recent": list([ring({ request_id: "p1" })]),
  });
  await page.route("**/debug/spec-decode", (route) =>
    route.fulfill({
      json: {
        models: [
          {
            model_id: "org/Qwen3.5-9B",
            mtp_stats: { accepts: 800, rejects: 200, total_cycles: 1000 },
          },
        ],
      },
    }),
  );
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  const c = page.getByTestId("spec-counters");
  await expect(c).toContainText("Qwen3.5-9B");
  await expect(c).toContainText("1,000");
});

test("without /debug the counters say they are not provided", async ({
  page,
}) => {
  await install(page, {
    "/v1/yunshu/requests/recent": list([ring({ request_id: "p1" })]),
  });
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("spec-counters")).toContainText("沒有提供");
});

const kv = {
  caches: [
    {
      model_id: "org/Qwen3.5-9B",
      apc: {
        entries: 12,
        resident_bytes: 2_000_000_000,
        warm_bytes: 500_000_000,
        warm_ratio: 2,
        disk_bytes: 9_000_000_000,
        lookups_hit: 30,
        lookups_miss: 10,
        matched_tokens: 90000,
        memory_evictions: 0,
        memory_skips: 3,
        warm_demotions: 7,
      },
    },
  ],
};

test("cache lifecycle: counters by stage, hit requests apart from cached tokens, reported zero stays zero", async ({
  page,
}) => {
  await install(page);
  await page.route("**/debug/kv-cache", (r) => r.fulfill({ json: kv }));
  await page.goto("/console/#/cache", { waitUntil: "domcontentloaded" });
  const life = page.getByTestId("cache-lifecycle");
  await expect(life).toBeVisible();
  await expect(life.locator('[data-counter="memory_evictions"]')).toContainText(
    "0",
  );
  await expect(life.locator('[data-counter="memory_skips"]')).toContainText(
    "3",
  );
  await expect(life.locator('[data-counter="lookups_hit"]')).toContainText(
    "30",
  );
  await expect(life.locator('[data-counter="matched_tokens"]')).toContainText(
    "90,000",
  );
  await expect(life).toContainText("請求命中率 75");
  await expect(life).toContainText("≈ 估算");
  await expect(life.locator('[data-counter="warm_dropped"]')).toHaveCount(0);
});

test("cache lifecycle without /debug says it is not provided", async ({
  page,
}) => {
  await install(page);
  await page.goto("/console/#/cache", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("cache-lifecycle-unavailable")).toContainText(
    "沒有提供",
  );
});

test("request archive is off by default, opt-in keeps finished rows across an engine restart, clear forgets them", async ({
  page,
}) => {
  let ringRows = [
    ring({ request_id: "keep-a", t: 1_800_000_100 }),
    ring({ request_id: "keep-b", t: 1_800_000_200 }),
  ];
  await install(page, {
    "/v1/yunshu/requests/recent": () => list(ringRows),
  });
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  const controls = page.getByTestId("archive-controls");
  const sw = controls.getByRole("switch");
  await expect(sw).toHaveAttribute("aria-checked", "false");
  await expect(page.getByTestId("archive-count")).toHaveCount(0);
  await sw.click();
  await expect(sw).toHaveAttribute("aria-checked", "true");
  // Let the first write land, then "restart": the engine's ring is empty again.
  await page.waitForTimeout(600);
  ringRows = [];
  await page.reload({ waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("archive-count")).toContainText("2");
  await controls.getByRole("button", { name: "清除瀏覽器紀錄" }).click();
  await expect(page.getByTestId("archive-count")).toHaveCount(0);
});
