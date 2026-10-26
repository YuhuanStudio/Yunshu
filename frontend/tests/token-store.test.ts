import assert from "node:assert/strict";
import test from "node:test";
import { rememberIfApplied, rememberedToken } from "../src/token-store.ts";

function fakeStorage() {
  const m = new Map<string, string>();
  (globalThis as { localStorage?: unknown }).localStorage = {
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => void m.set(k, v),
    removeItem: (k: string) => void m.delete(k),
  };
}

test("a token is remembered only when the connection took it and the user asked", () => {
  fakeStorage();
  const url = "http://h:8000";
  rememberIfApplied(false, true, url, "tok");
  assert.equal(rememberedToken(url), "");
  rememberIfApplied(true, false, url, "tok");
  assert.equal(rememberedToken(url), "");
  rememberIfApplied(true, true, url, "tok");
  assert.equal(rememberedToken(url), "tok");
});
