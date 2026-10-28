import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
// @ts-expect-error plain .mjs helper
import {
  nodeModulesProblem,
  shardArgs,
  shardEnv,
} from "../scripts/browser-shards.mjs";

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

test("a node_modules outside the tree, or a missing one, is refused", () => {
  const tree = mkdtempSync(join(tmpdir(), "shards-"));
  assert.match(nodeModulesProblem(tree) ?? "", /no node_modules/);
  const other = mkdtempSync(join(tmpdir(), "other-"));
  mkdirSync(join(other, "node_modules"));
  symlinkSync(join(other, "node_modules"), join(tree, "node_modules"));
  assert.match(nodeModulesProblem(tree) ?? "", /outside/);
  const own = mkdtempSync(join(tmpdir(), "own-"));
  mkdirSync(join(own, "node_modules"));
  assert.equal(nodeModulesProblem(own), null);
});
