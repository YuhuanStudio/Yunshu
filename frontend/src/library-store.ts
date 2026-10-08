import { parseLibraryData, type Library } from "./playground-library.ts";

/**
 * IndexedDB home of the Playground library. localStorage caps near 5 MB, which a few long saved
 * conversations exceed; IndexedDB does not. Every call is guarded: `undefined` means the browser
 * has no usable store (private mode, blocked site data), and the caller keeps its other copy.
 */
const DB = "yunshu-playground";
const STORE = "library";

const open = () =>
  new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error("blocked"));
  });

export async function readStoredLibrary(
  baseUrl: string,
): Promise<Library | undefined> {
  try {
    if (typeof indexedDB === "undefined") return undefined;
    const db = await open();
    try {
      const value = await new Promise<unknown>((resolve, reject) => {
        const r = db.transaction(STORE).objectStore(STORE).get(baseUrl);
        r.onsuccess = () => resolve(r.result);
        r.onerror = () => reject(r.error);
      });
      return parseLibraryData(value);
    } finally {
      db.close();
    }
  } catch {
    return undefined;
  }
}

export async function writeStoredLibrary(
  baseUrl: string,
  library: Library,
): Promise<boolean> {
  try {
    if (typeof indexedDB === "undefined") return false;
    const db = await open();
    try {
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).put(library, baseUrl);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error);
      });
      return true;
    } finally {
      db.close();
    }
  } catch {
    return false;
  }
}
