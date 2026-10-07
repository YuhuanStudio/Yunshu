import { expect, test, type Page, type Route } from "@playwright/test";

// Palette verbs and notification links are deep links such as `#/models?action=load&model=<id>`.
// Each owning page must act on its intent once (and then drop it from the address).

const model = (id: string, loaded: boolean) => ({
  id,
  type: "LLM",
  loaded,
  loading: false,
  pinned: false,
  size_gb: 4,
});

async function install(page: Page, calls: string[]) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", async (route: Route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    const json = (status: number, body: unknown) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    if (req.method() !== "GET") calls.push(`${req.method()} ${path}`);
    if (path === "/v1/yunshu/status")
      return json(200, {
        object: "yunshu.status",
        version: "fixture",
        state: "running",
        uptime_s: 5000,
        load_error: null,
        models: [model("org/loaded", true), model("org/idle", false)],
        memory: { active_gb: 30, total_gb: 137, cache_gb: 1, peak_gb: 30 },
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
    if (path === "/v1/models/load") return json(200, { status: "loaded" });
    if (path === "/v1/yunshu/cache")
      return json(200, {
        enabled: true,
        caches: [
          {
            model: "org/loaded",
            tiers: [
              {
                name: "ram",
                used_bytes: 1e9,
                cap_bytes: 8e9,
                entries: 3,
                hits: 4,
              },
            ],
            lookups: { hit: 4, miss: 1, by_tier: { ram: 4 } },
            entries: [],
          },
        ],
      });
    if (path === "/v1/yunshu/downloads")
      return json(200, {
        downloads: [],
        active: 0,
        free_bytes: 2e11,
        models_dir: "/models",
      });
    if (path === "/v1/yunshu/keys") return json(200, { data: [] });
    return json(404, { detail: "fixture endpoint not found" });
  });
}

test("load with a model id loads it once and drops the intent from the address", async ({
  page,
}) => {
  const calls: string[] = [];
  await install(page, calls);
  await page.goto("/console/#/models?action=load&model=org%2Fidle");
  await expect.poll(() => calls).toContain("POST /v1/models/load");
  expect(page.url()).not.toContain("action=");
  expect(calls.filter((c) => c === "POST /v1/models/load")).toHaveLength(1);
});

test("unload with a model id opens the confirm dialog without unloading", async ({
  page,
}) => {
  const calls: string[] = [];
  await install(page, calls);
  await page.goto("/console/#/models?action=unload&model=org%2Floaded");
  await expect(page.getByRole("dialog")).toContainText("org/loaded");
  expect(calls.filter((c) => c.includes("unload"))).toHaveLength(0);
});

test("unload without a model narrows the list to loaded models and focuses search", async ({
  page,
}) => {
  await install(page, []);
  await page.goto("/console/#/models?action=unload");
  await expect(page.locator("#models-search")).toBeFocused();
});

test("clear cache opens the confirm dialog for the first clearable tier", async ({
  page,
}) => {
  const calls: string[] = [];
  await install(page, calls);
  await page.goto("/console/#/cache?action=clear");
  await expect(page.getByRole("dialog")).toBeVisible();
  expect(calls).not.toContain("POST /v1/yunshu/cache/clear");
  expect(page.url()).not.toContain("action=");
});

test("download focuses the repo field and prefills the model", async ({
  page,
}) => {
  await install(page, []);
  await page.goto("/console/#/downloads?action=new&model=org%2Fnew-model");
  await expect(page.locator("#dl-repo")).toBeFocused();
  await expect(page.locator("#dl-repo")).toHaveValue("org/new-model");
});

test("create key opens the create form", async ({ page }) => {
  await install(page, []);
  await page.goto("/console/#/keys?action=create");
  await expect(page.getByRole("dialog")).toBeVisible();
});
