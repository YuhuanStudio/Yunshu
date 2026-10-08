import { expect, test, type Page } from "@playwright/test";

const status = {
  object: "yunshu.status",
  version: "fixture",
  state: "ready",
  uptime_s: 100,
  models: [],
  load_error: null,
  memory: { active_gb: 12, cache_gb: 0, peak_gb: 12, total_gb: 64 },
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
async function switchConnection(page: Page, token = "") {
  await page.getByRole("button", { name: /^開啟設定/ }).click();
  await page
    .getByLabel("服務位址", { exact: true })
    .fill(`${new URL(page.url()).origin}/server-b`);
  await page.getByLabel("存取權杖", { exact: true }).fill(token);
  await page.getByRole("button", { name: "儲存並連線", exact: true }).click();
}

for (const kind of ["url", "token"] as const) {
  test(`switching ${kind} does not announce another service's restart or download`, async ({
    page,
  }) => {
    let aDownloads = 0,
      bDownloads = 0,
      bStatus = 0;
    await page.route("**/v1/**", (route) => {
      const req = route.request(),
        path = new URL(req.url()).pathname;
      const b =
        kind === "url"
          ? path.startsWith("/server-b/")
          : req.headers()["authorization"] === "Bearer service-b";
      if (path.endsWith("/status")) {
        if (b) bStatus++;
        return route.fulfill({ json: { ...status, uptime_s: b ? 2 : 100 } });
      }
      if (path.endsWith("/downloads")) {
        if (b) bDownloads++;
        else aDownloads++;
        return route.fulfill({
          json: {
            downloads: [
              {
                id: "same-id",
                repo: "org/model",
                state: b ? "done" : "running",
                error: null,
              },
            ],
          },
        });
      }
      return route.fulfill({ status: 404, json: { detail: "fixture" } });
    });
    await page.goto("/console/#/overview");
    await expect(page.getByTestId("health-verdict")).toHaveAttribute(
      "data-level",
      "ok",
    );
    await expect.poll(() => aDownloads).toBeGreaterThan(0);
    await expect(page.getByTestId("bell-badge")).toBeHidden();
    if (kind === "url") await switchConnection(page);
    else {
      await page.getByRole("button", { name: /^開啟設定/ }).click();
      await page.getByLabel("存取權杖", { exact: true }).fill("service-b");
      await page
        .getByRole("button", { name: "儲存並連線", exact: true })
        .click();
    }
    await expect.poll(() => bDownloads).toBeGreaterThan(0);
    await page.getByRole("link", { name: "引擎總覽", exact: true }).click();
    await expect
      .poll(() => bStatus, { timeout: 8000 })
      .toBeGreaterThanOrEqual(2);
    await expect(page.getByTestId("bell-badge")).toBeHidden();
    await expect(page.getByText(/引擎已重新啟動/)).toHaveCount(0);
  });
}
