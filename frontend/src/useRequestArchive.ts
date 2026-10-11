import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Row } from "./RequestTrace";
import {
  archiveScope,
  indexedDbStore,
  memoryStore,
  mergeArchived,
  retain,
  type ArchiveStore,
} from "./request-archive";

const FLAG = "yunshu.console.requestArchive";

export function readArchiveFlag(): boolean {
  try {
    return localStorage.getItem(FLAG) === "1";
  } catch {
    return false;
  }
}
function writeArchiveFlag(on: boolean) {
  try {
    if (on) localStorage.setItem(FLAG, "1");
    else localStorage.removeItem(FLAG);
  } catch {
    /* a per-viewer convenience */
  }
}

export interface RequestArchive {
  enabled: boolean;
  setEnabled: (on: boolean) => void;
  /** Ring rows plus archived-only rows, oldest first. Equals `ring` while the archive is off. */
  rows: Row[];
  /** Rows only the archive still has (from before the engine's last restart or past the ring). */
  archivedOnly: number;
  /** False when the browser refused storage: the toggle then says why instead of pretending. */
  available: boolean;
  clear: () => Promise<void>;
}

/** Keeps the finished-request metadata of the engine ring in this browser, when the viewer opts in. */
export function useRequestArchive(
  baseUrl: string,
  ring: readonly Row[],
  store?: ArchiveStore,
): RequestArchive {
  const [enabled, setEnabledState] = useState(readArchiveFlag);
  const [archived, setArchived] = useState<Row[]>([]);
  const [available, setAvailable] = useState(true);
  const scope = archiveScope(baseUrl);
  const backend = useRef<ArchiveStore | null>(null);
  if (!backend.current)
    backend.current =
      store ??
      (typeof indexedDB === "undefined" ? memoryStore() : indexedDbStore());
  const setEnabled = useCallback((on: boolean) => {
    writeArchiveFlag(on);
    setEnabledState(on);
  }, []);

  useEffect(() => {
    if (!enabled) {
      setArchived([]);
      return;
    }
    let live = true;
    const s = backend.current!;
    (async () => {
      const ok = await s.put(scope, ring as Row[]);
      const all = await s.all(scope);
      if (!live) return;
      if (!ok || all == null) {
        setAvailable(false);
        return;
      }
      setAvailable(true);
      const { keep, drop } = retain(all, Date.now() / 1000);
      if (drop.length) await s.remove(scope, drop);
      if (live) setArchived(keep);
    })();
    return () => {
      live = false;
    };
  }, [enabled, scope, ring]);

  const rows = useMemo(
    () => (enabled ? mergeArchived(ring, archived) : (ring as Row[])),
    [enabled, ring, archived],
  );
  const archivedOnly = useMemo(() => {
    const ids = new Set(ring.map((r) => r.id));
    return enabled ? archived.filter((r) => !ids.has(r.id)).length : 0;
  }, [enabled, ring, archived]);
  const clear = useCallback(async () => {
    await backend.current!.clear(scope);
    setArchived([]);
  }, [scope]);
  return { enabled, setEnabled, rows, archivedOnly, available, clear };
}
