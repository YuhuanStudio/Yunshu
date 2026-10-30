import { useRef } from "react";
import { mayAnimate } from "./motion/burst-core.ts";

/**
 * Which rows arrived after the list was first shown. The first non-empty batch is the baseline
 * (a page that just loaded does not fade row by row); a row that appears later gets the one-shot
 * `fresh-row` class, an opacity-only mount animation, so it never moves its neighbours. Only
 * while the screen is calm (see BurstGovernor): rows that arrive in a burst are drawn in place.
 * Each row is decided once, when it is first seen, so a re-render never starts or restarts it.
 */
export function useFreshIds(ids: readonly string[]): (id: string) => boolean {
  const baseline = useRef<Set<string> | null>(null);
  const born = useRef(0);
  const decided = useRef(new Map<string, boolean>());
  if (baseline.current === null && ids.length > 0) {
    baseline.current = new Set(ids);
    born.current = Date.now();
  } else if (
    baseline.current !== null &&
    Date.now() - born.current < SETTLE_MS
  ) {
    // The first page of rows can arrive in several batches (history, then the live ring): all of
    // them are "what the page opened with", not arrivals to animate one by one.
    for (const id of ids) baseline.current.add(id);
  }
  const base = baseline.current;
  return (id) => {
    if (base === null || base.has(id)) return false;
    let animate = decided.current.get(id);
    if (animate === undefined) {
      animate = mayAnimate();
      decided.current.set(id, animate);
    }
    return animate;
  };
}

/** Rows that arrive within this long of the first batch belong to the baseline. */
const SETTLE_MS = 1500;

/** Opacity only (`.fresh-row` in styles.css): no transform or blur on table rows. */
export const FRESH_ROW = "fresh-row";
