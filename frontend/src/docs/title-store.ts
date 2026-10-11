import { useSyncExternalStore } from "react";

// The shell's breadcrumb shows the open docs page's title; the page itself learns it from the
// table of contents, so it reports it here.
let title: string | null = null;
const listeners = new Set<() => void>();

export function setDocTitle(next: string | null) {
  if (next === title) return;
  title = next;
  listeners.forEach((l) => l());
}

export function useDocTitle(): string | null {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => title,
    () => title,
  );
}
