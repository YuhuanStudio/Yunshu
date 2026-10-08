import assert from "node:assert/strict";
import { test } from "node:test";
import {
  MAX_HISTORY,
  deleteHistory,
  deletePreset,
  libraryKey,
  parseLibrary,
  savePreset,
  titleOf,
  upsertHistory,
  type HistoryEntry,
  type Preset,
} from "../src/playground-library.ts";

const preset = (id: string, name = id): Preset => ({
  id,
  name,
  system: "",
  temperature: 0.7,
  maxTokens: 512,
  thinking: "auto",
  format: "text",
  model: "",
});
const entry = (id: string, at = 1): HistoryEntry => ({
  id,
  at,
  title: id,
  model: "m",
  messages: [{ role: "user", content: "hi" }],
});

test("the key is per service address and never holds a token", () => {
  assert.equal(
    libraryKey("http://a:1"),
    "yunshu.console.playground:http://a:1",
  );
  assert.notEqual(libraryKey("http://a:1"), libraryKey("http://b:1"));
});

test("unreadable storage is an empty library, bad rows are dropped, values are clamped", () => {
  assert.deepEqual(parseLibrary("{not json"), { presets: [], history: [] });
  assert.deepEqual(parseLibrary(null), { presets: [], history: [] });
  const lib = parseLibrary(
    JSON.stringify({
      presets: [
        { ...preset("a"), temperature: 99, maxTokens: -5, thinking: "weird" },
        { id: "x", name: "   " },
        7,
      ],
      history: [
        { id: "h", at: 1, messages: [{ role: "system", content: "no" }] },
      ],
    }),
  );
  assert.equal(lib.presets.length, 1);
  assert.equal(lib.presets[0].temperature, 2);
  assert.equal(lib.presets[0].maxTokens, 1);
  assert.equal(lib.presets[0].thinking, "auto");
  assert.equal(lib.history.length, 0);
});

test("saving a preset replaces the same id and puts it first; delete removes it", () => {
  let list = savePreset([], preset("a"));
  list = savePreset(list, preset("b"));
  list = savePreset(list, { ...preset("a"), name: "renamed" });
  assert.deepEqual(
    list.map((p) => p.id),
    ["a", "b"],
  );
  assert.equal(list[0].name, "renamed");
  assert.deepEqual(
    deletePreset(list, "a").map((p) => p.id),
    ["b"],
  );
});

test("history upserts by id, newest first, capped; an emptied conversation leaves the list", () => {
  let h: HistoryEntry[] = [];
  for (let i = 0; i < MAX_HISTORY + 5; i++)
    h = upsertHistory(h, entry(`c${i}`, i));
  assert.equal(h.length, MAX_HISTORY);
  assert.equal(h[0].id, `c${MAX_HISTORY + 4}`);
  h = upsertHistory(h, {
    ...entry("c10"),
    messages: [{ role: "user", content: "again" }],
  });
  assert.equal(h[0].id, "c10");
  assert.equal(h.filter((x) => x.id === "c10").length, 1);
  assert.equal(
    upsertHistory(h, { ...entry("c10"), messages: [] }).some(
      (x) => x.id === "c10",
    ),
    false,
  );
  assert.equal(
    deleteHistory(h, "c10").some((x) => x.id === "c10"),
    false,
  );
});

test("the title is the first user line, one line, shortened", () => {
  assert.equal(titleOf([{ role: "user", content: "  a\n\n b  " }]), "a b");
  assert.equal(
    titleOf([{ role: "user", content: "x".repeat(100) }]).length,
    60,
  );
  assert.equal(titleOf([]), "");
});
