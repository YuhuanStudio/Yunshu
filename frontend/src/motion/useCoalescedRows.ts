import { useEffect, useRef, useState } from "react";
import { RowPacer } from "./coalesce-core.ts";
import { reducedMotion } from "./ticker.ts";

/**
 * The rows of a live table, with rows arriving, leaving or re-ordering at most once per `gapMs` (see
 * RowPacer). `scope` is everything the reader chose (filter, sort, search): a change of it shows the
 * new list at once. Rows that stay keep their newest data. Under reduced motion nothing is paced.
 */
export function useCoalescedRows<T extends { id: string }>(
  rows: readonly T[],
  scope: string,
  gapMs = 1500,
): T[] {
  const pacer = useRef(new RowPacer(gapMs));
  const scopeRef = useRef(scope);
  const seen = useRef(new Map<string, T>());
  const [, wake] = useState(0);
  const now = performance.now();
  const ids = rows.map((r) => r.id);
  for (const r of rows) seen.current.set(r.id, r);
  const jump = scopeRef.current !== scope || reducedMotion();
  scopeRef.current = scope;
  const drawn = pacer.current.offer(ids, now, jump);
  const wait = pacer.current.dueIn(ids, now);
  useEffect(() => {
    if (wait == null) return;
    const id = setTimeout(() => wake((n) => n + 1), wait + 5);
    return () => clearTimeout(id);
  });
  // a row that left the data but is still drawn keeps its last values until the table lets it go
  const out = drawn.flatMap((id) => {
    const row = seen.current.get(id);
    return row ? [row] : [];
  });
  if (seen.current.size > drawn.length + 200)
    for (const id of seen.current.keys())
      if (!drawn.includes(id) && !ids.includes(id)) seen.current.delete(id);
  return out;
}
