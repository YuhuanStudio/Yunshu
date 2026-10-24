import { expect, test, type Page } from "@playwright/test";

/**
 * An unreachable engine under `--engine auto` is routine (restart, switch, measurement pause): every page
 * reads it as waiting, never as a health fault. The console's own routes answer; the engine routes 502.
 */
async function away(page: Page, list: Record<string, unknown>) {
  const engines = {
    object: "yunshu.console.engines",
    mode: "auto",
    selected: null,
    pinned: null,
    waiting: true,
    data: [],
    ...list,
  };
  await page.route("**/v1/yunshu/console/engines", (r) =>
    r.fulfill({ json: engines }),
  );
  await page.route("**/console/api/engines", (r) =>
    r.fulfill({ json: engines }),
  );
  await page.route("**/v1/yunshu/console", (r) =>
    r.fulfill({
      json: {
        object: "yunshu.console",
        recording: true,
        up: false,
        since: (Date.now() - 5_000) / 1000,
        outages: { count: 3, median_down_s: 25 },
      },
    }),
  );
  await page.route(/\/v1\/(?!yunshu\/console).*/, (r) =>
    r.fulfill({
      status: 502,
      json: {
        error: { message: "engine_unreachable", code: "engine_unreachable" },
      },
    }),
  );
  await page.route("**/health", (r) => r.fulfill({ status: 502, body: "" }));
}

test("a measurement pause reads as paused: neutral banner, no health fault anywhere", async ({
  page,
}) => {
  await away(page, { pause: { reason: "speed job", until: null } });
  await page.goto("/console/#/diagnostics");
  const calm = page.getByTestId("outage-calm");
  await expect(calm).toBeVisible({ timeout: 25_000 });
  await expect(calm).toHaveAttribute("data-kind", "paused");
  await expect(calm).toContainText("speed job");
  await expect(page.getByTestId("health-verdict")).toHaveAttribute(
    "data-level",
    "waiting",
  );
  await expect(page.getByTestId("offline-line")).toHaveCount(0);
  for (const hash of ["#/overview", "#/models", "#/cache", "#/settings"]) {
    await page.evaluate((h) => (location.hash = h), hash);
    await expect(page.getByTestId("outage-calm")).toBeVisible();
    await expect(page.getByTestId("outage-down")).toHaveCount(0);
  }
});

test("a restart reads as restarting, and only a long outage escalates", async ({
  page,
}) => {
  await away(page, {});
  await page.goto("/console/#/requests");
  const calm = page.getByTestId("outage-calm");
  await expect(calm).toBeVisible({ timeout: 25_000 });
  await expect(calm).toHaveAttribute("data-kind", "restarting");
  await expect(page.getByTestId("outage-down")).toHaveCount(0);
});
