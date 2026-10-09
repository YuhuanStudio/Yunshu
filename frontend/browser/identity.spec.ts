import { expect, test, type Page, type Route } from "@playwright/test";

/**
 * Facts read from one service are never shown for another (memory pressure, swap), and the poll
 * failure count belongs to one connection. A transient 5xx keeps the last numbers and says so.
 */
const status = (uptime: number) => ({
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "running",
  uptime_s: uptime,
  load_error: null,
  models: [
    { id: "M", type: "LLM", loaded: true, loading: false, pinned: false },
  ],
  memory: { active_gb: 10, cache_gb: 1, peak_gb: 11, total_gb: 64 },
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

const ledgerA = {
  object: "yunshu.memory",
  total_gb: 64,
  free_gb: 2,
  host: { pressure_level: "critical", swap_used_gb: 18.5753 },
  mlx: {
    active_gb: 10,
    cache_gb: 1,
    peak_gb: 11,
    recommended_working_set_gb: 50,
  },
  owners: [],
  attribution_overshoot_gb: null,
  limits: { apc_max_gb: null, apc_warm_max_gb: null, guard_margin_pct: null },
};

const json = (route: Route, code: number, body: unknown) =>
  route.fulfill({
    status: code,
    contentType: "application/json",
    body: JSON.stringify(body),
  });

async function connect(
  page: Page,
  handler: (route: Route, path: string) => unknown,
) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) =>
    handler(route, new URL(route.request().url()).pathname),
  );
}

async function switchTo(page: Page, base: string) {
  await page.getByRole("button", { name: /^開啟設定/ }).click();
  const url = page.getByLabel("服務位址");
  await expect(async () => {
    await url.fill(base);
    expect(await url.inputValue()).toBe(base);
  }).toPass();
  await page.getByRole("button", { name: "儲存並連線", exact: true }).click();
}

test("memory facts from service A are not shown after switching to service B", async ({
  page,
}) => {
  await connect(page, (route, path) => {
    const b = path.startsWith("/server-b/");
    const p = b ? path.slice("/server-b".length) : path;
    if (p === "/v1/yunshu/status") return json(route, 200, status(b ? 20 : 10));
    if (p === "/v1/yunshu/memory")
      return b
        ? json(route, 500, { detail: "boom" })
        : json(route, 200, ledgerA);
    return json(route, 404, { detail: "Not Found" });
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  const band = page.getByRole("list", { name: "引擎狀態" });
  await expect(band).toContainText("17.3");
  const origin = new URL(page.url()).origin;
  await switchTo(page, `${origin}/server-b`);
  await expect(
    page.getByTestId("overview").or(page.locator("main")),
  ).toBeVisible();
  // B answers status but its memory route fails: A's swap and pressure must be gone at once and stay gone.
  await expect(band).not.toContainText("17.3", { timeout: 8_000 });
  await page.waitForTimeout(6_000);
  await expect(band).not.toContainText("17.3");
  await expect(band).not.toContainText("嚴重");
});

test("one failed poll keeps the last numbers and says so; recovery clears the note", async ({
  page,
}) => {
  let fail = false;
  await connect(page, (route, path) => {
    if (path === "/v1/yunshu/status")
      return fail
        ? json(route, 500, { detail: "x" })
        : json(route, 200, status(10));
    return json(route, 404, { detail: "Not Found" });
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("overview")).toContainText(
    "Yunshu fixture-1.0",
  );
  await expect(page.getByTestId("retry-stamp")).toHaveCount(0);
  fail = true;
  const note = page.getByTestId("retry-stamp");
  await expect(note).toBeVisible({ timeout: 8_000 });
  await expect(note).toContainText("暫時無法更新");
  // Still the last good reading, not offline.
  await expect(page.getByTestId("overview")).toContainText(
    "Yunshu fixture-1.0",
  );
  fail = false;
  await expect(note).toHaveCount(0, { timeout: 8_000 });
});

test("failures counted against service A do not carry over to service B", async ({
  page,
}) => {
  let aCalls = 0;
  let bCalls = 0;
  await connect(page, (route, path) => {
    const b = path.startsWith("/server-b/");
    const p = b ? path.slice("/server-b".length) : path;
    if (p !== "/v1/yunshu/status")
      return json(route, 404, { detail: "Not Found" });
    if (!b) {
      aCalls += 1;
      // Service A: good once, then failing: two failures in a row, one short of offline.
      return aCalls <= 1 ? json(route, 200, status(10)) : json(route, 500, {});
    }
    bCalls += 1;
    // Service B: its first two polls fail (one short of offline in its own count), then it is fine.
    return bCalls <= 2 ? json(route, 500, {}) : json(route, 200, status(20));
  });
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await expect
    .poll(() => aCalls, { timeout: 15_000 })
    .toBeGreaterThanOrEqual(3);
  // Let the third call (A's second failure) settle before leaving.
  await page.waitForTimeout(700);
  const origin = new URL(page.url()).origin;
  await switchTo(page, `${origin}/server-b`);
  await expect
    .poll(() => bCalls, { timeout: 10_000 })
    .toBeGreaterThanOrEqual(2);
  // B's own count is two: never the offline banner from A's carried failures.
  await page.waitForTimeout(800);
  // A one-shot read, not a retrying assertion: B recovers a few seconds later and would mask the bug.
  expect(await page.getByTestId("live-phase").innerText()).not.toContain(
    "離線",
  );
  // Once B answers, it is simply online.
  await expect(page.getByTestId("live-phase")).toContainText("閒置", {
    timeout: 10_000,
  });
});

test("the notification list belongs to one connection: another service starts empty, the first one's history returns with it", async ({
  page,
}) => {
  let uptimeA = 5000;
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (!path.endsWith("/yunshu/status"))
      return json(route, 404, { detail: "fixture" });
    return json(
      route,
      200,
      status(path.startsWith("/server-b/") ? 9000 : uptimeA),
    );
  });
  await page.goto("/console/#/overview");
  await expect(page.getByTestId("health-verdict")).toHaveAttribute(
    "data-level",
    "ok",
  );
  uptimeA = 7;
  await expect(page.getByTestId("bell-badge")).toBeVisible({ timeout: 20000 });
  const save = async (url: string) => {
    await page.getByRole("button", { name: /^開啟設定/ }).click();
    await page.getByLabel("服務位址", { exact: true }).fill(url);
    await page.getByRole("button", { name: "儲存並連線", exact: true }).click();
  };
  const origin = new URL(page.url()).origin;
  await save(`${origin}/server-b`);
  await expect(page.getByTestId("bell-badge")).toBeHidden();
  await save(origin);
  await expect(page.getByTestId("bell-badge")).toBeVisible();
});
