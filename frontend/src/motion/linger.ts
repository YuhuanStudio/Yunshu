import { useEffect, useRef, useState } from "react";
import { reducedMotion } from "./ticker.ts";

export interface Lingering<T> {
  row: T;
  leaving: boolean;
}

/**
 * The rows to draw: the current ones plus, for `ms` after they disappear, the rows that just left,
 * kept at their old position and flagged `leaving` so they can fade out and collapse instead of
 * vanishing. Under reduced motion nothing lingers.
 */
export function useLinger<T>(
  rows: readonly T[],
  keyOf: (row: T) => string,
  ms = 340,
): Lingering<T>[] {
  const [, bump] = useState(0);
  const prev = useRef<T[]>([]);
  const leaving = useRef(
    new Map<
      string,
      { row: T; index: number; timer: ReturnType<typeof setTimeout> }
    >(),
  );
  const keys = new Set(rows.map(keyOf));

  if (!reducedMotion()) {
    prev.current.forEach((row, index) => {
      const k = keyOf(row);
      if (!keys.has(k) && !leaving.current.has(k)) {
        const timer = setTimeout(() => {
          leaving.current.delete(k);
          bump((n) => n + 1);
        }, ms);
        leaving.current.set(k, { row, index, timer });
      }
    });
    // a row that came back stops leaving
    for (const [k, v] of leaving.current)
      if (keys.has(k)) {
        clearTimeout(v.timer);
        leaving.current.delete(k);
      }
  }
  prev.current = [...rows];

  useEffect(
    () => () => {
      for (const v of leaving.current.values()) clearTimeout(v.timer);
      leaving.current.clear();
    },
    [],
  );

  const out: Lingering<T>[] = rows.map((row) => ({ row, leaving: false }));
  [...leaving.current.values()]
    .sort((a, b) => a.index - b.index)
    .forEach((v) =>
      out.splice(Math.min(v.index, out.length), 0, {
        row: v.row,
        leaving: true,
      }),
    );
  return out;
}
