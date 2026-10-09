import assert from "node:assert/strict";
import test from "node:test";
import {
  boxDeltas,
  clsOf,
  summarize,
  topMovers,
} from "../scripts/review-analyze.mjs";

const frame = (
  t: number,
  scrollTop: number,
  boxes: Record<
    string,
    { top: number; height: number; width?: number; key?: string }
  >,
) => ({
  t,
  scrollTop,
  boxes: Object.fromEntries(
    Object.entries(boxes).map(([id, b]) => [
      id,
      { left: 0, width: 100, key: "k" + id, ...b },
    ]),
  ),
});

test("CLS ignores shifts that follow user input", () => {
  assert.equal(
    clsOf([
      { value: 0.02, hadRecentInput: false },
      { value: 0.5, hadRecentInput: true },
      { value: 0.01, hadRecentInput: false },
    ]),
    0.03,
  );
});

test("a row pushed down or a card that grew is a move; scrolling is not", () => {
  const frames = [
    frame(0, 0, { 1: { top: 100, height: 40 }, 2: { top: 140, height: 40 } }),
    frame(250, 0, { 1: { top: 100, height: 40 }, 2: { top: 180, height: 40 } }), // pushed down by 40
    frame(500, 0, { 1: { top: 100, height: 60 }, 2: { top: 180, height: 40 } }), // grew by 20
    frame(750, 50, { 1: { top: 50, height: 60 }, 2: { top: 130, height: 40 } }), // scrolled 50: nothing moved
  ];
  const d = boxDeltas(frames);
  assert.deepEqual(
    d.map((x) => [x.id, x.dTop, x.dH]),
    [
      ["2", 40, 0],
      ["1", 0, 20],
    ],
  );
});

test("sub-pixel noise and elements that appear or vanish are not moves", () => {
  const frames = [
    frame(0, 0, { 1: { top: 100.2, height: 40 } }),
    frame(250, 0, {
      1: { top: 100.9, height: 40.4 },
      2: { top: 10, height: 10 },
    }),
  ];
  assert.deepEqual(boxDeltas(frames), []);
});

test("top movers rank by total movement and carry the element's key", () => {
  const d = [
    { t: 1, id: "a", key: "row", dTop: 40, dH: 0, dW: 0 },
    { t: 2, id: "b", key: "num", dTop: 0, dH: 0, dW: 3 },
    { t: 3, id: "a", key: "row", dTop: 20, dH: 0, dW: 0 },
  ];
  assert.deepEqual(topMovers(d, 2), [
    { key: "row", moves: 2, total: 60 },
    { key: "num", moves: 1, total: 3 },
  ]);
});

test("the summary joins CLS and the moves", () => {
  const s = summarize(
    [{ t: 1, value: 0.0123, hadRecentInput: false }],
    [
      frame(0, 0, { 1: { top: 0, height: 20 } }),
      frame(250, 0, { 1: { top: 30, height: 20 } }),
    ],
  );
  assert.equal(s.cls, 0.0123);
  assert.equal(s.moves, 1);
});
