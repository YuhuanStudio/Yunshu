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

async function install(page: Page, items: unknown[]) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status")
      return route.fulfill({ json: statusWith(items) });
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
