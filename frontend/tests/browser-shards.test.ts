import assert from "node:assert/strict";
import test from "node:test";
// @ts-expect-error plain .mjs helper
import { shardArgs, shardEnv } from "../scripts/browser-shards.mjs";

test("each shard gets its own slice, port and output directory", () => {
  assert.deepEqual(shardArgs(2, 4, ["--project=chromium"]), [
    "exec",
    "playwright",
    "test",
    "--shard=2/4",
    "--project=chromium",
  ]);
  const a = shardEnv(1, 20001, {});
  const b = shardEnv(2, 20002, {});
  assert.notEqual(a.PLAYWRIGHT_PORT, b.PLAYWRIGHT_PORT);
  assert.notEqual(a.PLAYWRIGHT_OUTPUT_DIR, b.PLAYWRIGHT_OUTPUT_DIR);
});

test("a bad shard index or count is refused", () => {
  assert.throws(() => shardArgs(0, 4));
  assert.throws(() => shardArgs(5, 4));
  assert.throws(() => shardArgs(1, 0));
});
