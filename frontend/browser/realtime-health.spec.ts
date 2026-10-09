import { expect, test, type Page } from "@playwright/test";

const status = {
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "ready",
  uptime_s: 100,
  load_error: null,
  models: [],
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
    mean_decode_tps: null,
  },
};

async function install(page: Page) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") return route.fulfill({ json: status });
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
}

test("realtime health: first event measured by the browser, server rows honest", async ({
  page,
}) => {
  await install(page);
  await page.routeWebSocket(/\/v1\/realtime$/, (ws) => {
    ws.send(JSON.stringify({ type: "session.created", session: {} }));
  });
  await page.goto("/console/#/diagnostics", { waitUntil: "domcontentloaded" });
  const card = page.locator('[data-yunui="stream-health"]');
  await expect(card).toContainText("尚未測試");
  await card.getByTestId("realtime-test").click();
  await expect(card).toContainText("連得上，引擎有回應");
  await expect(card).toContainText("session.created");
  // Nobody measured the server side: an em dash with its reason, never 0.
  const latency = card
    .locator("dt", { hasText: "伺服器延遲" })
    .locator("xpath=following-sibling::dd[1]");
  await expect(latency).toHaveText("—");
  await expect(latency).toHaveAttribute("title", /沒有回報/);
});

test("realtime health: a refused socket is a failure, not a success", async ({
  page,
}) => {
  await install(page);
  await page.routeWebSocket(/\/v1\/realtime$/, (ws) => {
    ws.close({ code: 1008, reason: "no" });
  });
  await page.goto("/console/#/diagnostics", { waitUntil: "domcontentloaded" });
  const card = page.locator('[data-yunui="stream-health"]');
  await card.getByTestId("realtime-test").click();
  await expect(card).toContainText("連線失敗");
  await expect(card).not.toContainText("連得上");
});
