import { expect, test, type Page } from "@playwright/test";

const status = (tps: number | null) => ({
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "ready",
  uptime_s: 100,
  load_error: null,
  models: [
    {
      id: "org/Qwen-VL",
      type: "VLMEngine",
      loaded: true,
      loading: false,
      pinned: false,
    },
  ],
  memory: { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 137.438953472 },
  requests: {
    active: tps ? 1 : 0,
    queued: 0,
    prefill: 0,
    decode: tps ? 1 : 0,
    items: [],
  },
  last: null,
  throughput: {
    window_s: 60,
    requests: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    live_decode_tps: tps,
    mean_prefill_tps: null,
    mean_decode_tps: null,
  },
});

async function install(
  page: Page,
  withHost: boolean,
  tps: () => number | null = () => null,
) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status")
      return route.fulfill({ json: status(tps()) });
    if (path === "/v1/yunshu/host" && withHost)
      return route.fulfill({
        json: {
          object: "yunshu.host",
          telemetry: {
            state: "ok",
            sampled_at: Date.now() / 1000,
            interval_s: 1,
            watts: { gpu: 10, package: 15 },
            gpu: { frequency_mhz: 1200, active_ratio: 0.5 },
            temperature: { state: "ok", die_max_c: 60 },
            reasons: {},
          },
          thermal: { state: "normal" },
          memory_pressure: { state: "normal" },
          power: { state: "ac" },
        },
      });
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
}

const trigger = (page: Page) =>
  page.getByTestId(
    page.viewportSize()!.width < 640 ? "footer-compact" : "footer-trigger",
  );

test("island opens from the pill, closes on Esc and outside press, and returns focus", async ({
  page,
}) => {
  await install(page, true);
  await page.goto("/console/#/overview");
  await trigger(page).click();
  const island = page.getByRole("dialog", { name: "引擎狀態" });
  await expect(island).toBeVisible();
  await expect(island).toContainText("Qwen-VL");
  await page.keyboard.press("Escape");
  await expect(island).toBeHidden();
  await expect(trigger(page)).toBeFocused();
  await trigger(page).click();
  await expect(island).toBeVisible();
  await page.mouse.click(700, 120);
  await expect(island).toBeHidden();
});

test("machine card exists only when the engine serves host telemetry", async ({
  page,
}) => {
  await install(page, true);
  await page.goto("/console/#/overview");
  await trigger(page).click();
  await expect(page.getByTestId("island-machine")).toBeVisible();
  await expect(page.getByTestId("island-memory")).toBeVisible();
});

test("machine card is absent without /host and the island is content-sized", async ({
  page,
}) => {
  await install(page, false);
  await page.goto("/console/#/overview");
  await trigger(page).click();
  const island = page.getByRole("dialog", { name: "引擎狀態" });
  await expect(island).toBeVisible();
  await expect(page.getByTestId("island-machine")).toHaveCount(0);
  const box = await island.boundingBox();
  expect(box!.height).toBeLessThan(page.viewportSize()!.height * 0.71);
  const overflow = await page.evaluate(
    () =>
      document.documentElement.scrollWidth -
      document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});

test("live values do not move the card while they update", async ({ page }) => {
  let n = 0;
  await install(page, true, () => 40 + (n++ % 5) * 11);
  await page.goto("/console/#/overview");
  await trigger(page).click();
  const island = page.getByRole("dialog", { name: "引擎狀態" });
  await expect(page.getByTestId("island-tps")).toBeVisible();
  const a = await island.boundingBox();
  await page.waitForTimeout(7000);
  const b = await island.boundingBox();
  expect(b!.x).toBe(a!.x);
  expect(b!.width).toBe(a!.width);
  expect(Math.abs(b!.height - a!.height)).toBeLessThanOrEqual(1);
});

test("a nav row goes to the page and closes the island", async ({ page }) => {
  await install(page, false);
  await page.goto("/console/#/overview");
  await trigger(page).click();
  await page
    .getByRole("dialog", { name: "引擎狀態" })
    .getByRole("link", { name: "診斷" })
    .click();
  await expect(page.getByRole("dialog", { name: "引擎狀態" })).toBeHidden();
  await expect(page).toHaveURL(/#\/diagnostics/);
});

test("a 128 GiB machine shows 128 GB, never 137 GB (binary units, as macOS)", async ({
  page,
}) => {
  await install(page, false);
  await page.goto("/console/#/overview");
  await expect(page.getByTestId("status-band")).toContainText("128 GB");
  await expect(page.getByTestId("status-band")).not.toContainText("137");
  await trigger(page).click();
  const memory = page.getByTestId("island-memory");
  await expect(memory).toContainText("/128");
  await expect(memory).not.toContainText("137");
});
