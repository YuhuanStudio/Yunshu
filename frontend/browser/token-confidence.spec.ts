import { expect, test, type Page } from "@playwright/test";

const status = {
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "ready",
  uptime_s: 100,
  load_error: null,
  models: [
    {
      id: "Qwen3.8-27B",
      type: "LLM",
      loaded: true,
      loading: false,
      pinned: true,
      size_gb: 16,
    },
  ],
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

const chunk = (o: unknown) => `data: ${JSON.stringify(o)}\n\n`;
const withLogprobs =
  chunk({
    choices: [
      {
        delta: { content: "你好" },
        logprobs: {
          content: [
            { token: "你好", logprob: Math.log(0.95), top_logprobs: [] },
          ],
        },
      },
    ],
  }) +
  chunk({
    choices: [
      {
        delta: { content: "世界" },
        finish_reason: "stop",
        logprobs: {
          content: [
            {
              token: "世界",
              logprob: Math.log(0.1),
              top_logprobs: [
                { token: "世界", logprob: Math.log(0.1) },
                { token: "朋友", logprob: Math.log(0.7) },
              ],
            },
          ],
        },
      },
    ],
    usage: { prompt_tokens: 5, completion_tokens: 2 },
  }) +
  "data: [DONE]\n\n";
const withoutLogprobs =
  chunk({
    choices: [{ delta: { content: "你好世界" }, finish_reason: "stop" }],
    usage: { prompt_tokens: 5, completion_tokens: 2 },
  }) + "data: [DONE]\n\n";

async function install(
  page: Page,
  stream: string,
  bodies: Record<string, unknown>[],
) {
  await page.addInitScript(() =>
    localStorage.setItem("yunshu.console.url", location.origin),
  );
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") return route.fulfill({ json: status });
    if (path === "/v1/models")
      return route.fulfill({
        json: { object: "list", data: [{ id: "Qwen3.8-27B" }] },
      });
    if (path === "/v1/chat/completions") {
      bodies.push(route.request().postDataJSON());
      return route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: stream,
      });
    }
    return route.fulfill({ status: 404, json: { detail: "Not Found" } });
  });
}

async function ask(page: Page, logprobs: boolean) {
  await page.goto("/console/#/playground", { waitUntil: "domcontentloaded" });
  if (logprobs) {
    await page.getByRole("button", { name: "生成參數" }).click();
    await page.getByRole("switch", { name: "Token 信心（logprobs）" }).click();
    await page.keyboard.press("Escape");
  }
  await page.locator("textarea").first().fill("hi");
  await page.keyboard.press("Enter");
}

test("token confidence: logprobs requested, tokens shaded, alternatives in the tooltip", async ({
  page,
}) => {
  const bodies: Record<string, unknown>[] = [];
  await install(page, withLogprobs, bodies);
  await ask(page, true);
  const panel = page.getByTestId("token-confidence");
  await expect(panel).toBeVisible();
  await expect(panel).toContainText("2 個 token");
  const low = panel.locator('[data-band="veryLow"]');
  await expect(low).toHaveText("世界");
  await expect(low).toHaveAttribute("title", /10%.*朋友 70%/);
  expect(bodies[0]).toMatchObject({ logprobs: true, top_logprobs: 5 });
});

test("token confidence: off by default and never sent", async ({ page }) => {
  const bodies: Record<string, unknown>[] = [];
  await install(page, withoutLogprobs, bodies);
  await ask(page, false);
  await expect(page.getByText("你好世界")).toBeVisible();
  await expect(page.getByTestId("token-confidence")).toHaveCount(0);
  await expect(page.getByTestId("conf-unavailable")).toHaveCount(0);
  expect(bodies[0]).not.toHaveProperty("logprobs");
});

test("token confidence: asked for but not returned says so, honestly", async ({
  page,
}) => {
  const bodies: Record<string, unknown>[] = [];
  await install(page, withoutLogprobs, bodies);
  await ask(page, true);
  await expect(page.getByTestId("conf-unavailable")).toBeVisible();
  await expect(page.getByTestId("token-confidence")).toHaveCount(0);
});
