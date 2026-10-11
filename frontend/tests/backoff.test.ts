import assert from "node:assert/strict";
import test from "node:test";
import { retryDelayMs } from "../src/backoff.ts";

const mid = () => 0.5;

test("doubles from one second and caps near ten", () => {
  assert.deepEqual(
    [1, 2, 3, 4, 5, 6, 20].map((n) => retryDelayMs(n, mid)),
    [1000, 2000, 4000, 8000, 10000, 10000, 10000],
  );
});

test("jitter stays within plus or minus 25 percent and never passes 12.5 s", () => {
  for (const n of [1, 2, 3, 4, 5, 9]) {
    const lo = retryDelayMs(n, () => 0);
    const hi = retryDelayMs(n, () => 1);
    const base = retryDelayMs(n, mid);
    assert.ok(lo >= base * 0.74 && lo <= base * 0.76, `${n}: ${lo}`);
    assert.ok(hi >= base * 1.24 || hi === 12500, `${n}: ${hi}`);
    assert.ok(hi <= 12500);
  }
});

test("a bad failure count is treated as the first", () => {
  assert.equal(retryDelayMs(0, mid), 1000);
  assert.equal(retryDelayMs(-3, mid), 1000);
  assert.equal(retryDelayMs(2.9, mid), 2000);
});
