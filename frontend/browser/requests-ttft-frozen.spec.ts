import { expect, test, type Route } from "@playwright/test";

// A decoding request's TTFT column used to show its elapsed time, so it kept counting up through
// decode. TTFT freezes at the first token; the running time sits in the Total column.

test("a decoding row keeps its TTFT while its running total grows", async ({
  page,
}) => {
  let sample = 0;
  const json = (route: Route, status: number, body: unknown) =>
    route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  await page.route("**/v1/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") {
      sample += 1;
      return json(route, 200, {
        object: "yunshu.status",
        version: "fixture-1.0",
        state: "ready",
        uptime_s: 3_600 + sample,
        load_error: null,
        models: [
          { id: "org/qwen-mlx", type: "text", loaded: true, loading: false, pinned: false, size_gb: 8, expires_in_s: 900 },
        ],
        memory: { active_gb: 12, cache_gb: 2, peak_gb: 14, total_gb: 64, pressure: 0.2 },
        requests: {
          active: 1,
          queued: 0,
          prefill: 0,
          decode: 1,
          items: [
            {
              request_id: "ttft-decode-01",
              phase: "decode",
              model: "org/qwen-mlx",
              elapsed_s: 20 + sample * 30,
              completion_tokens: 400 * sample,
              tokens_per_second: 41.5,
              offsets_ms: { arrive: 0, admit: 40, first_token: 1234 },
            },
          ],
        },
        last: null,
        throughput: {
          window_s: 60,
          requests: 0,
          prompt_tokens: 0,
          completion_tokens: 0,
          live_decode_tps: 41.5,
          mean_prefill_tps: 800,
          mean_decode_tps: 35,
        },
      });
    }
    return json(route, 404, { detail: "Not Found" });
  });
  await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
  const requests = page.getByTestId("requests");
  const row = requests.getByRole("row").filter({ hasText: "ttft-decode-01" });
  await expect(row).toBeVisible();
  const header = requests.getByRole("columnheader");
  const names = await header.allInnerTexts();
  const ttftCol = names.findIndex((n) => n.trim() === "TTFT");
  const totalCol = names.findIndex((n) => n.trim() === "總計" || n.trim() === "Total");
  expect(ttftCol).toBeGreaterThanOrEqual(0);
  const ttft = () => row.getByRole("cell").nth(ttftCol);
  await expect(ttft()).toHaveText("1.2 s");
  const total0 = totalCol >= 0 ? await row.getByRole("cell").nth(totalCol).innerText() : "";
  // Several polls later the elapsed time has grown by minutes; TTFT has not moved.
  await page.waitForTimeout(7_500);
  expect(sample).toBeGreaterThan(2);
  await expect(ttft()).toHaveText("1.2 s");
  if (totalCol >= 0)
    expect(await row.getByRole("cell").nth(totalCol).innerText()).not.toBe(total0);
});
