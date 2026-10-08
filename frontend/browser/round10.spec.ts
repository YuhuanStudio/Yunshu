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
  offsets_ms: { arrive: 0, admit: 40, first_token: 340, last_token: 2340, done: 2360 },
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
