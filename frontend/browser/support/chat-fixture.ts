import type { Page } from "@playwright/test";

/** A same-origin engine stand-in for the chat surfaces: status with two chat models, a streaming
 *  /v1/chat/completions whose pace and length the test sets, and the docs "Ask AI" routes. */
export const MODELS = [
  { id: "Qwen3.8-27B", type: "LLM" },
  { id: "Qwen3.5-9B", type: "LLM" },
].map((m) => ({
  ...m,
  loaded: true,
  loading: false,
  pinned: false,
  size_gb: 8.4,
  idle_s: 0,
  keep_alive_s: null,
  expires_in_s: null,
}));

export const status = {
  object: "yunshu.status",
  version: "fixture-1.0",
  state: "ready",
  uptime_s: 100,
  load_error: null,
  models: MODELS,
  memory: { active_gb: 12, cache_gb: 1, peak_gb: 14, total_gb: 64 },
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

export type ReplyPlan = {
  /** Words streamed as the answer. */
  words: number;
  /** Milliseconds between chunks. */
  every: number;
  reasoning?: string;
  /** Text streamed after the words (a long token, a code block). */
  tail?: string;
};

/** Install the engine routes. `plan` can change between turns. */
export async function installEngine(
  page: Page,
  plan: { current: ReplyPlan } = {
    current: { words: 40, every: 20, reasoning: "Let me think about it." },
  },
) {
  await page.addInitScript(() => {
    localStorage.setItem("yunshu.console.url", location.origin);
  });
  await page.addInitScript((planned: ReplyPlan) => {
    const w = window as unknown as { __plan: ReplyPlan };
    w.__plan = planned;
    const nativeFetch = window.fetch.bind(window);
    const encoder = new TextEncoder();
    const event = (data: unknown) => `data: ${JSON.stringify(data)}\n\n`;
    window.fetch = async (input, init = {}) => {
      const raw =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.toString()
            : input.url;
      const url = new URL(raw, location.href);
      if (!url.pathname.endsWith("/v1/chat/completions"))
        return nativeFetch(input, init);
      const p = w.__plan;
      const signal = init.signal as AbortSignal | undefined;
      return new Response(
        new ReadableStream({
          async start(controller) {
            const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
            try {
              if (p.reasoning) {
                controller.enqueue(
                  encoder.encode(
                    event({
                      choices: [{ delta: { reasoning_content: p.reasoning } }],
                    }),
                  ),
                );
                await wait(60);
              }
              for (let i = 0; i < p.words; i++) {
                if (signal?.aborted) {
                  controller.close();
                  return;
                }
                const piece =
                  (i === 0 ? "" : " ") +
                  (i % 17 === 16 ? "\n\n" : "") +
                  `word${i}`;
                controller.enqueue(
                  encoder.encode(
                    event({ choices: [{ delta: { content: piece } }] }),
                  ),
                );
                await wait(p.every);
              }
              if (p.tail)
                controller.enqueue(
                  encoder.encode(
                    event({ choices: [{ delta: { content: p.tail } }] }),
                  ),
                );
              controller.enqueue(
                encoder.encode(
                  event({ choices: [{ delta: {}, finish_reason: "stop" }] }),
                ),
              );
              controller.enqueue(
                encoder.encode(
                  event({
                    choices: [],
                    usage: { prompt_tokens: 64, completion_tokens: p.words },
                  }),
                ),
              );
              controller.enqueue(encoder.encode("data: [DONE]\n\n"));
              controller.close();
            } catch {
              /* the reader went away */
            }
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } },
      );
    };
  }, plan.current);
  await page.route("**/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/v1/yunshu/status") return route.fulfill({ json: status });
    return route.fulfill({ status: 404, json: { error: { message: "n/a" } } });
  });
}

export async function setPlan(page: Page, plan: ReplyPlan) {
  await page.evaluate((p) => {
    (window as unknown as { __plan: ReplyPlan }).__plan = p;
  }, plan);
}

/** Ask AI answers a canned markdown reply, optionally with sources. */
export async function installAsk(page: Page, answer = "Use **APC**. [1]") {
  await page.route("**/console/api/docs/state", (route) =>
    route.fulfill({
      json: { ready: true, model: "local", question: 2000, history: 8 },
    }),
  );
  await page.route("**/console/api/docs/ask", (route) =>
    route.fulfill({
      contentType: "text/event-stream",
      body: `event: sources\ndata: {"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","model":"local","sources":[{"id":"/docs/guides/prompt-caching#apc","slug":"guides/prompt-caching","anchor":"apc","title":"Prompt caching","heading":"APC"}]}\n\nevent: token\ndata: ${JSON.stringify({ text: answer })}\n\nevent: done\ndata: {"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\n\n`,
    }),
  );
  await page.route("**/console/api/docs/feedback", (route) =>
    route.fulfill({ json: { saved: true } }),
  );
}
