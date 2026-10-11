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

import {
  branchEntry,
  exportLibrary,
  mergeLibraries,
  parseExport,
  reconcile,
  updatePreset,
} from "../src/playground-library.ts";

test("export round-trips through a schema check; another schema or kind is refused", () => {
  const lib = { presets: [preset("p")], history: [entry("h", 5)] };
  const round = parseExport(JSON.parse(JSON.stringify(exportLibrary(lib, 0))));
  assert.equal(round?.presets[0].id, "p");
  assert.equal(round?.history[0].id, "h");
  assert.equal(parseExport({ ...exportLibrary(lib), schema: 2 }), null);
  assert.equal(parseExport({ ...exportLibrary(lib), kind: "other" }), null);
  assert.equal(parseExport("nope"), null);
});

test("import merges by id with the imported copy winning, newest history first, within the caps", () => {
  const current = { presets: [preset("a", "old")], history: [entry("h1", 1)] };
  const imported = {
    presets: [preset("a", "new"), preset("b")],
    history: [entry("h2", 9)],
  };
  const merged = mergeLibraries(current, imported);
  assert.deepEqual(
    merged.presets.map((p) => p.name),
    ["new", "b"],
  );
  assert.deepEqual(
    merged.history.map((h) => h.id),
    ["h2", "h1"],
  );
});

test("a branch keeps the exchange up to the chosen point and gets its own id", () => {
  const e: HistoryEntry = {
    ...entry("h"),
    messages: [
      { role: "user", content: "1" },
      { role: "assistant", content: "a" },
      { role: "user", content: "2" },
      { role: "assistant", content: "b" },
    ],
  };
  const b = branchEntry(e, 3);
  assert.equal(b.messages.length, 2);
  assert.notEqual(b.id, e.id);
  assert.equal(e.messages.length, 4);
  assert.equal(branchEntry(e, 99).messages.length, 4);
});

test("renaming a preset trims, caps and never blanks the name", () => {
  const out = updatePreset([preset("a", "x")], "a", { name: "  " });
  assert.equal(out[0].name, "x");
  assert.equal(
    updatePreset([preset("a")], "a", { name: " Hi " })[0].name,
    "Hi",
  );
});

test("after upgrading, the synchronous copy is migrated once; the store wins when it has data", () => {
  const empty = { presets: [], history: [] };
  const sync = { presets: [preset("a")], history: [] };
  assert.deepEqual(reconcile(sync, empty), { library: sync, migrate: true });
  assert.equal(reconcile(sync, undefined).migrate, false);
  const stored = { presets: [preset("z")], history: [] };
  assert.deepEqual(reconcile(sync, stored), {
    library: stored,
    migrate: false,
  });
});
