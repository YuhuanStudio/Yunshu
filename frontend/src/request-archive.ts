import type { Row } from "./RequestTrace";

/**
 * Opt-in, browser-side archive of finished-request METADATA (ids, model, token counts, timings,
 * outcome: the same numbers the engine's ring reports, never a prompt or a reply). The engine's
 * ring lives in memory and is empty after a restart, so this is the only history that survives
 * one. It is scoped to the service address (never the token), bounded, and clearable.
 */
export const ARCHIVE_MAX_ROWS = 2000;
export const ARCHIVE_MAX_AGE_S = 30 * 86400;
export const ARCHIVE_SCHEMA = 1;

export interface ArchiveStore {
  /** Add or replace rows by id; resolves false when storage is unavailable. */
  put(scope: string, rows: Row[]): Promise<boolean>;
  all(scope: string): Promise<Row[] | null>;
  remove(scope: string, ids: string[]): Promise<void>;
  clear(scope: string): Promise<void>;
}

/** Service identity for the archive: the address only. A token is never stored. */
export const archiveScope = (baseUrl: string) =>
  baseUrl.trim().replace(/\/+$/, "").toLowerCase();

const when = (r: Row) => (typeof r.t === "number" ? r.t : 0);

/**
 * Rows to keep: newest first by finish time, older than the age limit dropped, at most
 * `maxRows`. `drop` lists the ids that fell out, so the store can forget them.
 */
export function retain(
  rows: readonly Row[],
  nowS: number,
  limits = { maxRows: ARCHIVE_MAX_ROWS, maxAgeS: ARCHIVE_MAX_AGE_S },
): { keep: Row[]; drop: string[] } {
  const sorted = [...rows].sort((a, b) => when(b) - when(a));
  const keep: Row[] = [];
  const drop: string[] = [];
  for (const r of sorted) {
    const old = when(r) > 0 && nowS - when(r) > limits.maxAgeS;
    if (old || keep.length >= limits.maxRows) drop.push(r.id);
    else keep.push(r);
  }
  return { keep, drop };
}

/** The engine's rows win over archived copies of the same id; the result is oldest first. */
export function mergeArchived(
  ring: readonly Row[],
  archived: readonly Row[],
): Row[] {
  const ids = new Set(ring.map((r) => r.id));
  const old = archived
    .filter((r) => !ids.has(r.id))
    .map((r) => ({ ...r, source: "archive" as const }));
  return [...old, ...ring].sort((a, b) => when(a) - when(b));
}

/** Filtered JSON export: a versioned envelope around the rows as shown. */
export function exportEnvelope(
  rows: readonly Row[],
  service: string,
  nowMs = Date.now(),
) {
  return {
    schema: ARCHIVE_SCHEMA,
    kind: "yunshu.requests",
    exported_at: new Date(nowMs).toISOString(),
    service: archiveScope(service),
    note: "Metadata only: no prompts, replies or secrets.",
    rows,
  };
}

/** Parse an export back; anything else (wrong schema, not rows) is rejected with null. */
export function parseEnvelope(raw: unknown): Row[] | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (o.schema !== ARCHIVE_SCHEMA || !Array.isArray(o.rows)) return null;
  return o.rows.filter(
    (r): r is Row =>
      !!r && typeof r === "object" && typeof (r as Row).id === "string",
  );
}

const DB = "yunshu-console";
const STORE = "request-archive";

/** IndexedDB-backed store; every call is guarded, so a blocked or private-mode browser degrades to "unavailable". */
export function indexedDbStore(): ArchiveStore {
  const open = () =>
    new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open(DB, 1);
      req.onupgradeneeded = () => {
        const store = req.result.createObjectStore(STORE, { keyPath: "key" });
        store.createIndex("scope", "scope");
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  const run = async <T>(
    mode: IDBTransactionMode,
    fn: (s: IDBObjectStore) => IDBRequest<T> | void,
  ): Promise<T | undefined> => {
    const db = await open();
    try {
      return await new Promise<T | undefined>((resolve, reject) => {
        const tx = db.transaction(STORE, mode);
        const r = fn(tx.objectStore(STORE));
        tx.oncomplete = () => resolve(r ? (r.result as T) : undefined);
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error);
      });
    } finally {
      db.close();
    }
  };
  return {
    async put(scope, rows) {
      try {
        await run("readwrite", (s) => {
          for (const row of rows)
            s.put({ key: `${scope}\u0000${row.id}`, scope, row });
        });
        return true;
      } catch {
        return false;
      }
    },
    async all(scope) {
      try {
        const found = await run<{ row: Row }[]>("readonly", (s) =>
          s.index("scope").getAll(scope),
        );
        return (found ?? []).map((x) => x.row);
      } catch {
        return null;
      }
    },
    async remove(scope, ids) {
      try {
        await run("readwrite", (s) => {
          for (const id of ids) s.delete(`${scope}\u0000${id}`);
        });
      } catch {
        /* storage is a convenience */
      }
    },
    async clear(scope) {
      try {
        const rows = (await this.all(scope)) ?? [];
        await this.remove(
          scope,
          rows.map((r) => r.id),
        );
      } catch {
        /* storage is a convenience */
      }
    },
  };
}

/** In-memory store for tests and as the fallback when IndexedDB is missing. */
export function memoryStore(): ArchiveStore {
  const data = new Map<string, Map<string, Row>>();
  const of = (scope: string) => {
    let m = data.get(scope);
    if (!m) data.set(scope, (m = new Map()));
    return m;
  };
  return {
    async put(scope, rows) {
      for (const r of rows) of(scope).set(r.id, r);
      return true;
    },
    async all(scope) {
      return [...of(scope).values()];
    },
    async remove(scope, ids) {
      for (const id of ids) of(scope).delete(id);
    },
    async clear(scope) {
      data.delete(scope);
    },
  };
}
