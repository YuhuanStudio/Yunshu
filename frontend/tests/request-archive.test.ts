import assert from "node:assert/strict";
import test from "node:test";
import {
  archiveScope,
  exportEnvelope,
  memoryStore,
  mergeArchived,
  parseEnvelope,
  retain,
} from "../src/request-archive.ts";

const row = (id: string, t: number) => ({ id, t, phase: "complete" });

test("retention drops rows past the age limit and past the row cap, newest kept", () => {
  const now = 1_000_000;
  const rows = [
    row("old", now - 100),
    row("a", now - 3),
    row("b", now - 2),
    row("c", now - 1),
  ];
  const { keep, drop } = retain(rows as never, now, {
    maxRows: 2,
    maxAgeS: 50,
  });
  assert.deepEqual(
    keep.map((r) => r.id),
    ["c", "b"],
  );
  assert.deepEqual(drop.sort(), ["a", "old"]);
});

test("the engine ring wins over an archived copy; archived-only rows are marked", () => {
  const merged = mergeArchived(
    [{ ...row("x", 20), ttft_ms: 5 }] as never,
    [{ ...row("x", 20), ttft_ms: 99 }, row("gone", 10)] as never,
  );
  assert.deepEqual(
    merged.map((r) => r.id),
    ["gone", "x"],
  );
  assert.equal(merged[0].source, "archive");
  assert.equal(merged[1].ttft_ms, 5);
});

test("the archive is keyed by address, not token, and is cleared per service", async () => {
  assert.equal(archiveScope("http://127.0.0.1:8000/"), "http://127.0.0.1:8000");
  const s = memoryStore();
  await s.put("a", [row("1", 1)] as never);
  await s.put("b", [row("2", 2)] as never);
  await s.clear("a");
  assert.equal((await s.all("a"))!.length, 0);
  assert.equal((await s.all("b"))!.length, 1);
});

test("export envelope round-trips and a foreign file is rejected", () => {
  const env = exportEnvelope([row("1", 1)] as never, "http://h:8000/", 0);
  assert.equal(env.schema, 1);
  assert.equal(parseEnvelope(JSON.parse(JSON.stringify(env)))!.length, 1);
  assert.equal(parseEnvelope({ schema: 2, rows: [] }), null);
  assert.equal(parseEnvelope("x"), null);
});
