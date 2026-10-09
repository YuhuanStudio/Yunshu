import assert from "node:assert/strict";
import test from "node:test";
import { buildSnippets } from "../src/playground-code.ts";
import type { CompletionBody } from "../src/stream.ts";

const body: CompletionBody = {
  model: "Qwen3.8-27B",
  messages: [
    { role: "system", content: "Be brief." },
    { role: "user", content: "it's 'quoted'" },
  ],
  temperature: 0.7,
  max_tokens: 512,
  enable_thinking: false,
  stream_options: { include_usage: true },
};

test("snippets carry the exact URL and payload of each dialect", () => {
  for (const [dialect, path] of [
    ["chat", "/v1/chat/completions"],
    ["responses", "/v1/responses"],
    ["messages", "/v1/messages"],
  ] as const) {
    const r = buildSnippets({
      dialect,
      baseUrl: "http://127.0.0.1:8000",
      body,
    });
    assert.equal(r.url, `http://127.0.0.1:8000${path}`);
    for (const code of Object.values(r.snippets)) {
      assert.ok(code.includes(path));
      assert.ok(code.includes("Qwen3.8-27B"));
    }
  }
});

test("curl body is shell-safe and python uses python literals", () => {
  const r = buildSnippets({
    dialect: "chat",
    baseUrl: "http://h:1/v1",
    body,
  });
  assert.ok(r.snippets.curl.includes(`'\\''quoted'\\''`));
  assert.ok(r.snippets.python.includes("False"));
  assert.ok(!r.snippets.python.includes("false"));
  assert.ok(r.snippets.curl.includes("$YUNSHU_AUTH_TOKEN"));
});

test("messages dialect maps system, thinking and omits json mode", () => {
  const r = buildSnippets({
    dialect: "messages",
    baseUrl: "http://h:1",
    body: { ...body, response_format: { type: "json_object" } },
  });
  assert.ok(r.snippets.curl.includes('"system": "Be brief."'));
  assert.ok(r.snippets.curl.includes('"type": "disabled"'));
  assert.ok(!r.snippets.curl.includes("json_object"));
});

test("long image payloads are shortened and flagged", () => {
  const url = `data:image/png;base64,${"A".repeat(5000)}`;
  const r = buildSnippets({
    dialect: "chat",
    baseUrl: "http://h:1",
    body: {
      ...body,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "see" },
            { type: "image_url", image_url: { url } },
          ],
        },
      ],
    },
  });
  assert.equal(r.shortened, true);
  assert.ok(r.snippets.curl.length < 1500);
});
