import { expect, test, type Page, type Route } from "@playwright/test";

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });

const item = {
  request_id: "chatcmpl-observe-0001",
  phase: "decode",
  model: "org/qwen-mlx",
  elapsed_s: 4.2,
  prompt_tokens: 900,
  cached_tokens: 0,
  completion_tokens: 40,
  offsets_ms: { admit: 5, first_token: 900 },
};
const status = (observe: boolean) => ({
  object: "yunshu.status",
  version: "t-1",
  state: "running",
  uptime_s: 90,
  observe,
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
  requests: { active: 1, queued: 0, prefill: 0, decode: 1, items: [item] },
  last: null,
  throughput: {
    window_s: 60,
    requests: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    live_decode_tps: 30,
    mean_prefill_tps: null,
    mean_decode_tps: 42,
  },
});

async function mock(page: Page, observe: boolean, engineGenerated = 41) {
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") return json(route, status(observe));
    return json(route, { detail: "nf" }, 404);
  });
  await page.route("**/debug/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/debug/requests")
      return json(route, {
        active: 1,
        in_flight: [
          {
            request_id: item.request_id,
            has_stats: true,
            phase: "decode",
            generated: engineGenerated,
            t_admit_ms: 5,
            t_first_ms: 900,
          },
        ],
      });
    return json(route, {});
  });
}

test.describe("observe mode", () => {
  test("shows the engine's numbers next to the page's, on by default under observe", async ({
    page,
  }) => {
    await mock(page, true);
    await page.goto("/console/#/requests");
    await expect(page.getByTestId("observe-badge")).toBeVisible();
    const row = page.getByTestId("observe-row");
    await expect(row).toHaveCount(1);
    await expect(row).toHaveAttribute("data-differs", "false");
    await expect(row).toContainText("41");
  });

  test("a mismatch is marked, and the toggle hides the section", async ({
    page,
  }) => {
    await mock(page, true, 900);
    await page.goto("/console/#/requests");
    await expect(page.getByTestId("observe-row")).toHaveAttribute(
      "data-differs",
      "true",
    );
    await page.getByTestId("observe-toggle").click();
    await expect(page.getByTestId("observe-row")).toHaveCount(0);
  });

  test("without observe the section starts closed", async ({ page }) => {
    await mock(page, false);
    await page.goto("/console/#/requests");
    await expect(page.getByTestId("observe-panel")).toBeVisible();
    await expect(page.getByTestId("observe-badge")).toHaveCount(0);
    await expect(page.getByTestId("observe-row")).toHaveCount(0);
  });

  test("diagnostics states observe mode", async ({ page }) => {
    await mock(page, true);
    await page.goto("/console/#/diagnostics");
    await expect(page.getByTestId("observe-badge-diag")).toBeVisible();
  });
});
