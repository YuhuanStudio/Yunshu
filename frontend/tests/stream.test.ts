import assert from "node:assert/strict";
import test from "node:test";
import {
  chatCompletionsUrl,
  streamCompletion,
  type CompletionDelta,
} from "../src/stream.ts";

const connection = { baseUrl: "http://127.0.0.1:8000/", token: "local-token" };
const body = {
  model: "local-model",
  messages: [{ role: "user", content: "Hello" }],
  temperature: 0.7,
  max_tokens: 128,
};

function responseWithChunks(chunks: Uint8Array[], status = 200): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
  return new Response(stream, {
    status,
    headers: { "Content-Type": "text/event-stream" },
  });
}

function splitEveryByte(text: string): Uint8Array[] {
  const bytes = new TextEncoder().encode(text);
  return Array.from({ length: bytes.length }, (_, index) =>
    bytes.slice(index, index + 1),
  );
}

function installFetch(t: test.TestContext, implementation: typeof fetch): void {
  const original = globalThis.fetch;
  globalThis.fetch = implementation;
  t.after(() => {
    globalThis.fetch = original;
  });
}

test("normalizes base URLs without duplicating /v1", () => {
  assert.equal(
    chatCompletionsUrl("http://127.0.0.1:8000"),
    "http://127.0.0.1:8000/v1/chat/completions",
  );
  assert.equal(
    chatCompletionsUrl("http://127.0.0.1:8000/v1/"),
    "http://127.0.0.1:8000/v1/chat/completions",
  );
  assert.equal(
    chatCompletionsUrl("http://localhost:8000/api/v1"),
    "http://localhost:8000/api/v1/chat/completions",
  );
});

test("parses fragmented UTF-8 SSE, CRLF, comments, content and reasoning deltas", async (t) => {
  let requestUrl = "";
  let requestInit: RequestInit | undefined;
  const events = [
    ": keepalive comment\r\n",
    "event: message\r\n",
    `data: ${JSON.stringify({ choices: [{ delta: { content: "你好🙂", reasoning_content: "先分析" } }] })}\r\n\r\n`,
    `data: ${JSON.stringify({ choices: [{ delta: { reasoning: "再檢查" } }] })}\r\n\r\n`,
    "data: [DONE]\r\n\r\n",
  ].join("");
  installFetch(t, (async (input, init) => {
    requestUrl = String(input);
    requestInit = init;
    return responseWithChunks(splitEveryByte(events));
  }) as typeof fetch);

  const deltas: Array<{ content?: string; reasoning?: string }> = [];
  const controller = new AbortController();
  await streamCompletion(
    connection,
    { ...body, stream: false },
    (delta) => deltas.push(delta),
    controller.signal,
  );

  assert.equal(requestUrl, "http://127.0.0.1:8000/v1/chat/completions");
  assert.equal(requestInit?.method, "POST");
  assert.equal(requestInit?.signal, controller.signal);
  const headers = new Headers(requestInit?.headers);
  assert.equal(headers.get("authorization"), "Bearer local-token");
  assert.equal(headers.get("accept"), "text/event-stream");
  assert.match(headers.get("x-request-id") ?? "", /^[0-9a-f-]{36}$/i);
  assert.equal(JSON.parse(String(requestInit?.body)).stream, true);
  assert.deepEqual(deltas, [
    { content: "你好🙂", reasoning: "先分析" },
    { reasoning: "再檢查" },
  ]);
});

test("surfaces structured HTTP errors", async (t) => {
  installFetch(
    t,
    (async () =>
      new Response(
        JSON.stringify({ error: { message: "invalid API token" } }),
        {
          status: 401,
          statusText: "Unauthorized",
          headers: { "Content-Type": "application/json" },
        },
      )) as typeof fetch,
  );

  await assert.rejects(
    streamCompletion(connection, body, () => {}, new AbortController().signal),
    /401 Unauthorized.*invalid API token/,
  );
});

test("surfaces SSE error events and malformed event JSON", async (t) => {
  const cases = [
    {
      payload:
        'event: error\ndata: {"error":{"message":"model is unavailable"}}\n\n',
      message: /model is unavailable/,
    },
    {
      payload: "data: {not-json}\n\n",
      message: /Invalid JSON in OpenAI stream/,
    },
    {
      payload: 'data: {"error":"quota exceeded"}\n\n',
      message: /quota exceeded/,
    },
  ];

  const original = globalThis.fetch;
  try {
    for (const item of cases) {
      globalThis.fetch = (async () =>
        responseWithChunks([
          new TextEncoder().encode(item.payload),
        ])) as typeof fetch;
      await assert.rejects(
        streamCompletion(
          connection,
          body,
          () => {},
          new AbortController().signal,
        ),
        item.message,
      );
    }
  } finally {
    globalThis.fetch = original;
  }
});

test("rejects EOF before [DONE], including a finish_reason-only terminal event", async (t) => {
  const payload = `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\r\n\r\n`;
  installFetch(t, (async () =>
    responseWithChunks([new TextEncoder().encode(payload)])) as typeof fetch);
  await assert.rejects(
    streamCompletion(connection, body, () => {}, new AbortController().signal),
    /ended before the \[DONE\] event/,
  );
});

test("passes cancellation through the caller's AbortSignal", async (t) => {
  let startedResolve!: () => void;
  const started = new Promise<void>((resolve) => {
    startedResolve = resolve;
  });
  installFetch(t, (async (_input, init) => {
    const signal = init?.signal;
    startedResolve();
    return await new Promise<Response>((_resolve, reject) => {
      if (!signal) return reject(new Error("request signal was not passed"));
      signal.addEventListener(
        "abort",
        () => reject(new DOMException("Aborted", "AbortError")),
        { once: true },
      );
    });
  }) as typeof fetch);

  const controller = new AbortController();
  const pending = streamCompletion(
    connection,
    body,
    () => {},
    controller.signal,
  );
  await started;
  controller.abort();
  await assert.rejects(
    pending,
    (error) => (error as Error).name === "AbortError",
  );
});

test("cancels a pending stream reader when the caller aborts", async (t) => {
  let readStarted!: () => void;
  const started = new Promise<void>((resolve) => {
    readStarted = resolve;
  });
  let readerCancelled = false;
  const stream = new ReadableStream<Uint8Array>(
    {
      pull() {
        readStarted();
      },
      cancel() {
        readerCancelled = true;
      },
    },
    { highWaterMark: 0 },
  );
  installFetch(
    t,
    (async () =>
      new Response(stream, {
        headers: { "Content-Type": "text/event-stream" },
      })) as typeof fetch,
  );

  const controller = new AbortController();
  const pending = streamCompletion(
    connection,
    body,
    () => {},
    controller.signal,
  );
  await started;
  controller.abort();
  await assert.rejects(
    pending,
    (error) => (error as Error).name === "AbortError",
  );
  assert.equal(readerCancelled, true);
});

test("preserves VLM image parts and thinking/output options, and exposes length termination", async (t) => {
  let sent: Record<string, unknown> | undefined;
  installFetch(t, (async (_input, init) => {
    sent = JSON.parse(String(init?.body));
    return responseWithChunks([
      new TextEncoder().encode(
        'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n',
      ),
    ]);
  }) as typeof fetch);
  const deltas: unknown[] = [];
  const image = "data:image/png;base64,dGVzdA==";
  await streamCompletion(
    connection,
    {
      ...body,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Describe this." },
            { type: "image_url", image_url: { url: image } },
          ],
        },
      ],
      enable_thinking: false,
      response_format: { type: "json_object" },
    },
    (delta) => deltas.push(delta),
    new AbortController().signal,
  );
  assert.equal(sent?.enable_thinking, false);
  assert.deepEqual(sent?.response_format, { type: "json_object" });
  assert.equal(
    (sent?.messages as { content: { image_url?: { url: string } }[] }[])[0]
      .content[1].image_url?.url,
    image,
  );
  assert.deepEqual(deltas, [{ finishReason: "length" }]);
});

function sse(events: Array<[string | null, unknown]>) {
  const text = events
    .map(
      ([name, data]) =>
        `${name ? `event: ${name}\n` : ""}data: ${JSON.stringify(data)}\n\n`,
    )
    .join("");
  return new Response(text, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

test("responses dialect posts /v1/responses and reads deltas, usage and x_yunshu", async (t) => {
  let seen: { url: string; body: Record<string, unknown> } | undefined;
  t.mock.method(globalThis, "fetch", async (url: string, init: RequestInit) => {
    seen = { url, body: JSON.parse(String(init.body)) };
    return sse([
      [
        "response.reasoning_summary_text.delta",
        { type: "response.reasoning_summary_text.delta", delta: "hm" },
      ],
      [
        "response.output_text.delta",
        { type: "response.output_text.delta", delta: "Hel" },
      ],
      [
        "response.output_text.delta",
        { type: "response.output_text.delta", delta: "lo" },
      ],
      [
        "response.completed",
        {
          type: "response.completed",
          response: {
            usage: {
              input_tokens: 10,
              output_tokens: 4,
              input_tokens_details: { cached_tokens: 6 },
              x_yunshu: { ttft_ms: 33 },
            },
          },
        },
      ],
    ]);
  });
  const deltas: CompletionDelta[] = [];
  await streamCompletion(
    { baseUrl: "http://h:1", token: "" },
    {
      model: "m",
      messages: [
        { role: "system", content: "S" },
        { role: "user", content: "U" },
      ],
      temperature: 0.2,
      max_tokens: 9,
      response_format: { type: "json_object" },
    },
    (d) => deltas.push(d),
    new AbortController().signal,
    "responses",
  );
  assert.equal(seen?.url, "http://h:1/v1/responses");
  assert.equal(seen?.body.instructions, "S");
  assert.equal(seen?.body.max_output_tokens, 9);
  assert.deepEqual(seen?.body.text, { format: { type: "json_object" } });
  assert.equal(
    deltas
      .filter((d) => d.content)
      .map((d) => d.content)
      .join(""),
    "Hello",
  );
  assert.equal(deltas.find((d) => d.reasoning)?.reasoning, "hm");
  const usage = deltas.find((d) => d.usage)?.usage;
  assert.equal(usage?.promptTokens, 10);
  assert.equal(usage?.completionTokens, 4);
  assert.equal(usage?.cachedTokens, 6);
  assert.equal(usage?.ttftMs, 33);
});

test("messages dialect posts /v1/messages and reads text, thinking and usage", async (t) => {
  let seen: { url: string; body: Record<string, any> } | undefined;
  t.mock.method(globalThis, "fetch", async (url: string, init: RequestInit) => {
    seen = { url, body: JSON.parse(String(init.body)) };
    return sse([
      [
        "message_start",
        {
          type: "message_start",
          message: {
            usage: {
              input_tokens: 4,
              cache_read_input_tokens: 6,
              output_tokens: 1,
            },
          },
        },
      ],
      [
        "content_block_delta",
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "thinking_delta", thinking: "t" },
        },
      ],
      [
        "content_block_delta",
        {
          type: "content_block_delta",
          index: 1,
          delta: { type: "text_delta", text: "ok" },
        },
      ],
      [
        "message_delta",
        {
          type: "message_delta",
          delta: { stop_reason: "max_tokens" },
          usage: { output_tokens: 7 },
        },
      ],
      ["message_stop", { type: "message_stop" }],
    ]);
  });
  const deltas: CompletionDelta[] = [];
  await streamCompletion(
    { baseUrl: "http://h:1/v1", token: "tok" },
    {
      model: "m",
      messages: [
        { role: "system", content: "S" },
        { role: "user", content: "U" },
      ],
      temperature: 0.2,
      max_tokens: 40,
      enable_thinking: true,
    },
    (d) => deltas.push(d),
    new AbortController().signal,
    "messages",
  );
  assert.equal(seen?.url, "http://h:1/v1/messages");
  assert.equal(seen?.body.system, "S");
  assert.equal(seen?.body.messages.length, 1);
  assert.equal(seen?.body.thinking.type, "enabled");
  assert.equal(deltas.find((d) => d.content)?.content, "ok");
  assert.equal(deltas.find((d) => d.reasoning)?.reasoning, "t");
  const usage = deltas.find((d) => d.usage)?.usage;
  assert.equal(usage?.promptTokens, 10);
  assert.equal(usage?.cachedTokens, 6);
  assert.equal(usage?.completionTokens, 7);
  assert.equal(deltas.find((d) => d.finishReason)?.finishReason, "length");
});

test("non-chat streams fail closed without a terminal event or on error events", async (t) => {
  t.mock.method(globalThis, "fetch", async () =>
    sse([["message_start", { type: "message_start", message: {} }]]),
  );
  await assert.rejects(
    streamCompletion(
      { baseUrl: "http://h:1", token: "" },
      { model: "m", messages: [], temperature: 0, max_tokens: 1 },
      () => {},
      new AbortController().signal,
      "messages",
    ),
    /message_stop/,
  );
  t.mock.method(globalThis, "fetch", async () =>
    sse([["error", { type: "error", error: { message: "boom" } }]]),
  );
  await assert.rejects(
    streamCompletion(
      { baseUrl: "http://h:1", token: "" },
      { model: "m", messages: [], temperature: 0, max_tokens: 1 },
      () => {},
      new AbortController().signal,
      "responses",
    ),
    /boom/,
  );
});
