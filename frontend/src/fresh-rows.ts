import { useRef } from "react";

/**
 * Which rows arrived after the list was first shown. The first non-empty batch is the baseline
 * (a page that just loaded does not fade row by row); a row that appears later gets the one-shot
 * `yunui-fade-in` class, an opacity-only mount animation, so it never moves its neighbours.
 */
export function useFreshIds(ids: readonly string[]): (id: string) => boolean {
  const baseline = useRef<Set<string> | null>(null);
  if (baseline.current === null && ids.length > 0)
    baseline.current = new Set(ids);
  const base = baseline.current;
  return (id) => base !== null && !base.has(id);
}

export const FRESH_ROW = "yunui-fade-in";
