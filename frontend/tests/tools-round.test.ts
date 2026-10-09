import assert from "node:assert/strict";
import test from "node:test";
import {
  unsupportedKeywords,
  validateSchema,
} from "../src/json-schema-lite.ts";
import {
  checkStructured,
  checkToolDefs,
  nextMessages,
  readRound,
} from "../src/tools-round.ts";

const tool = {
  type: "function",
  function: {
    name: "get_weather",
    parameters: {
      type: "object",
      required: ["city"],
      additionalProperties: false,
      properties: { city: { type: "string" }, unit: { enum: ["c", "f"] } },
    },
  },
};

test("schema check names the path and rule of every violation", () => {
  const issues = validateSchema(tool.function.parameters, {
    unit: "k",
    extra: 1,
  });
  const text = issues.map((i) => `${i.path} ${i.message}`).join("\n");
  assert.match(text, /\$\.city required/);
  assert.match(text, /\$\.unit enum/);
  assert.match(text, /\$\.extra additionalProperties/);
  assert.deepEqual(
    validateSchema(tool.function.parameters, { city: "Taipei" }),
    [],
  );
  assert.equal(validateSchema({ type: "integer" }, 1.5).length, 1);
});

test("keywords the checker cannot judge are listed, never silently passed", () => {
  assert.deepEqual(
    unsupportedKeywords({
      type: "object",
      properties: { a: { format: "email" } },
      $ref: "#/x",
    }),
    ["$ref", "format"],
  );
});

test("tool definitions are checked before send", () => {
  assert.equal(checkToolDefs(JSON.stringify([tool])).ok, true);
  assert.deepEqual(
    (checkToolDefs("[") as { errors: string[] }).errors[0].startsWith("json:"),
    true,
  );
  assert.deepEqual((checkToolDefs("[]") as { errors: string[] }).errors, [
    "array",
  ]);
  const dup = checkToolDefs(JSON.stringify([tool, tool])) as {
    errors: string[];
  };
  assert.deepEqual(dup.errors, ["duplicate:get_weather"]);
  const bad = checkToolDefs(
    JSON.stringify([{ type: "function", function: { name: "bad name" } }]),
  ) as { errors: string[] };
  assert.deepEqual(bad.errors, ["name:1"]);
});

const reply = (calls: unknown[]) => ({
  choices: [
    {
      message: { role: "assistant", content: null, tool_calls: calls },
      finish_reason: "tool_calls",
    },
  ],
  usage: { prompt_tokens: 10, completion_tokens: 5 },
});

test("a round keeps the calls verbatim, validates arguments, and flags unknown tools and broken JSON", () => {
  const r = readRound(
    reply([
      {
        id: "c1",
        type: "function",
        function: { name: "get_weather", arguments: '{"unit":"k"}' },
      },
      {
        id: "c2",
        type: "function",
        function: { name: "nope", arguments: "{}" },
      },
      {
        id: "c3",
        type: "function",
        function: { name: "get_weather", arguments: "{oops" },
      },
    ]),
    [tool] as never,
  )!;
  assert.equal(r.finishReason, "tool_calls");
  assert.equal(r.toolCalls[0].issues.length, 2);
  assert.equal(r.toolCalls[1].known, false);
  assert.ok(r.toolCalls[2].parseError);
  assert.equal(r.usage.prompt, 10);
  assert.equal(readRound({}, []), null);
});

test("tool results go back as tool messages with the call id, never as assistant text", () => {
  const r = readRound(
    reply([
      {
        id: "c1",
        type: "function",
        function: { name: "get_weather", arguments: '{"city":"T"}' },
      },
    ]),
    [tool] as never,
  )!;
  const msgs = nextMessages([{ role: "user", content: "hi" }], r, {
    c1: '{"t":20}',
  });
  assert.deepEqual(
    msgs.map((m) => m.role),
    ["user", "assistant", "tool"],
  );
  assert.equal(msgs[2].tool_call_id, "c1");
  assert.equal(msgs[2].content, '{"t":20}');
});

test("structured output: not JSON, wrong shape and fine are told apart", () => {
  const schema = JSON.stringify({
    type: "object",
    required: ["a"],
    properties: { a: { type: "integer" } },
  });
  assert.equal(checkStructured("", schema).state, "empty");
  assert.equal(checkStructured("hello", schema).state, "notJson");
  assert.equal(checkStructured('{"a":"x"}', schema).state, "bad");
  assert.equal(checkStructured('{"a":1}', schema).state, "ok");
});
