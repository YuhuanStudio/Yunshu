import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { type Connection } from "../src/api.ts";
import {
  diskShortfall,
  getFit,
  listDownloads,
  parseDownloads,
  parseFit,
  parseLocal,
  parsePatterns,
  startDownload,
  UNSUPPORTED,
  REPO_RE,
} from "../src/admin-models-api.ts";
import {
  clearableTier,
  getCache,
  hitRate,
  parseCache,
} from "../src/admin-cache-api.ts";
import { splitBytes, rateText } from "../src/byte-format.ts";
import { ApiError } from "../src/api.ts";

const c: Connection = { baseUrl: "http://127.0.0.1:8000/v1/", token: "" };
const original = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = original;
});
const reply = (status: number, body: unknown) => async () =>
  new Response(JSON.stringify(body), { status });

test("a download job parses tolerantly: missing numbers stay null, never 0", () => {
  const d = parseDownloads({
    downloads: [{ id: "a", repo: "o/n", state: "running", bytes_done: 5 }],
    free_bytes: null,
  });
  assert.equal(d.jobs[0].bytesTotal, null);
  assert.equal(d.jobs[0].etaS, null);
  assert.equal(d.freeBytes, null);
  assert.equal(d.active, 1);
});

test("an unknown job state reads as failed, not as a success", () => {
  assert.equal(
    parseDownloads({ downloads: [{ state: "weird" }] }).jobs[0].state,
    "failed",
  );
});

test("the 507 body becomes a shortfall; other errors do not", () => {
  const e = new ApiError("x", 507, {
    needed_bytes: 9e9,
    free_bytes: 1e9,
    path: "/m",
  });
  assert.deepEqual(diskShortfall(e), { needed: 9e9, free: 1e9, path: "/m" });
  assert.equal(diskShortfall(new ApiError("x", 502)), null);
});

test("older servers: 404 on any new route is 'unsupported'", async () => {
  globalThis.fetch = reply(404, { detail: "Not Found" });
  assert.equal(await listDownloads(c), UNSUPPORTED);
  assert.equal(await getCache(c), UNSUPPORTED);
  assert.equal(await getFit(c, "m"), UNSUPPORTED);
});

test("fit on a single-model server (400) is unsupported, a 500 is an error", async () => {
  globalThis.fetch = reply(400, { detail: "needs multi-model" });
  assert.equal(await getFit(c, "m"), UNSUPPORTED);
  globalThis.fetch = reply(500, { detail: "boom" });
  await assert.rejects(getFit(c, "m"));
});

test("startDownload posts the body and surfaces the 507 detail", async () => {
  let sent = "";
  globalThis.fetch = async (_u, init) => {
    sent = String(init?.body);
    return new Response(
      JSON.stringify({
        detail: { message: "x", needed_bytes: 5, free_bytes: 1, path: "/p" },
      }),
      { status: 507 },
    );
  };
  await assert.rejects(
    startDownload(c, { repo: "o/n", allow_patterns: ["*.json"] }),
    (e) => diskShortfall(e)?.needed === 5,
  );
  assert.deepEqual(JSON.parse(sent), {
    repo: "o/n",
    allow_patterns: ["*.json"],
  });
});

test("fit result: verdict, evictions and the estimate flag", () => {
  const f = parseFit({
    model: "m",
    verdict: "tight",
    needed_bytes: 10,
    free_bytes: 3,
    would_evict: ["a", "b"],
    basis: { estimated: true },
  });
  assert.equal(f.verdict, "tight");
  assert.deepEqual(f.wouldEvict, ["a", "b"]);
  assert.equal(f.budgetBytes, null);
});

test("local inventory keeps unknown quant/context as null", () => {
  const l = parseLocal({
    models: [
      {
        id: "x",
        path: "/x",
        size_bytes: 10,
        complete: false,
        registered_as: null,
      },
    ],
  });
  assert.equal(l.models[0].quantBits, null);
  assert.equal(l.models[0].contextLength, null);
  assert.equal(l.models[0].complete, false);
});

test("cache overview: hit rate needs lookups; only ram/warm/ssd are clearable", () => {
  const o = parseCache({
    caches: [
      {
        model: "m",
        tiers: [
          { name: "ram", used_bytes: 4, cap_bytes: 8, entries: 2, hits: 1 },
        ],
        lookups: { hit: 3, miss: 1, by_tier: { ram: 3 } },
        entries: [
          {
            key: "ab12",
            tokens: 10,
            bytes: 4,
            tier: "ram",
            hits: 2,
            last_hit_age_s: null,
          },
        ],
      },
    ],
  });
  assert.equal(o.enabled, true);
  assert.equal(hitRate(o.caches[0]), 0.75);
  assert.equal(hitRate({ hit: 0, miss: 0 }), null);
  assert.equal(o.caches[0].entries[0].lastHitAgeS, null);
  assert.equal(clearableTier("ssd2"), null);
  assert.equal(clearableTier("warm"), "warm");
});

test("patterns split on commas and newlines; the repo id must be org/name", () => {
  assert.deepEqual(parsePatterns("*.json, *.safetensors\n tok*"), [
    "*.json",
    "*.safetensors",
    "tok*",
  ]);
  assert.deepEqual(parsePatterns("  "), []);
  assert.equal(REPO_RE.test("mlx-community/Qwen3-4B-4bit"), true);
  assert.equal(REPO_RE.test("no-slash"), false);
});

test("bytes: unit split off the number, unknown is a dash", () => {
  assert.deepEqual(splitBytes(1.5 * 1024 ** 3), { value: "1.5", unit: "GB" });
  assert.deepEqual(splitBytes(null), { value: "—", unit: "" });
  assert.equal(rateText(null), "—");
});
