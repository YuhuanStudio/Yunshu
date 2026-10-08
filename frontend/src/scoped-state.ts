import { useCallback, useState } from "react";
import type { Connection } from "./api.ts";

/**
 * Identity of one engine connection for state that must never outlive it: the address and the
 * token (kept only in this in-memory string, never stored). Facts read from service A (memory
 * pressure, finished requests, downloads) are not facts about service B.
 */
export const connectionScope = (c: Connection) =>
  `${c.baseUrl}\u0000${c.token}`;

/**
 * State tied to a scope. When the scope changes, the value reads as `initial` at once (in the same
 * render), so a later poll failure on the new service can never show the previous service's data.
 */
export function useScopedState<T>(
  scope: string,
  initial: T,
): [T, (next: T | ((current: T) => T)) => void] {
  const [held, setHeld] = useState<{ scope: string; value: T }>({
    scope,
    value: initial,
  });
  const value = held.scope === scope ? held.value : initial;
  const set = useCallback(
    (next: T | ((current: T) => T)) =>
      setHeld((h) => {
        const base = h.scope === scope ? h.value : initial;
        return {
          scope,
          value:
            typeof next === "function" ? (next as (c: T) => T)(base) : next,
        };
      }),
    // `initial` is a constant shape at every call site.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [scope],
  );
  return [value, set];
}
