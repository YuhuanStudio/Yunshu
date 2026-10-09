/**
 * Playground presets and local conversation history. Browser-only (no server copy), kept per
 * connection identity (the service address; the token is never stored), every storage call is
 * wrapped, and nothing depends on it: when storage is unavailable the lists are simply empty.
 */
export interface Preset {
  id: string;
  name: string;
  system: string;
  temperature: number;
  maxTokens: number;
  thinking: "auto" | "on" | "off";
  format: "text" | "json";
  /** The model it was saved with; applying it only switches when that model exists. */
  model: string;
}
export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}
export interface HistoryEntry {
  id: string;
  title: string;
  /** Epoch ms of the last change. */
  at: number;
  model: string;
  messages: HistoryMessage[];
}
export interface Library {
  presets: Preset[];
  history: HistoryEntry[];
}

export const MAX_PRESETS = 30;
export const MAX_HISTORY = 40;
/** A conversation longer than this is cut when stored (the newest messages are kept). */
export const MAX_HISTORY_MESSAGES = 60;
const MAX_TEXT = 20_000;

export const emptyLibrary = (): Library => ({ presets: [], history: [] });
export const libraryKey = (baseUrl: string) =>
  `yunshu.console.playground:${baseUrl}`;

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const str = (v: unknown, max = MAX_TEXT) =>
  typeof v === "string" ? v.slice(0, max) : "";

function parsePreset(v: unknown): Preset | null {
  if (!isRecord(v) || typeof v.id !== "string" || !str(v.name).trim())
    return null;
  const temperature = Number(v.temperature);
  const maxTokens = Number(v.maxTokens);
  return {
    id: v.id,
    name: str(v.name, 80),
    system: str(v.system),
    temperature: Number.isFinite(temperature)
      ? Math.max(0, Math.min(2, temperature))
      : 0.7,
    maxTokens: Number.isFinite(maxTokens)
      ? Math.max(1, Math.min(32768, Math.round(maxTokens)))
      : 512,
    thinking: v.thinking === "on" || v.thinking === "off" ? v.thinking : "auto",
    format: v.format === "json" ? "json" : "text",
    model: str(v.model, 200),
  };
}
function parseEntry(v: unknown): HistoryEntry | null {
  if (!isRecord(v) || typeof v.id !== "string" || !Array.isArray(v.messages))
    return null;
  const messages = v.messages.flatMap((m): HistoryMessage[] =>
    isRecord(m) &&
    (m.role === "user" || m.role === "assistant") &&
    typeof m.content === "string"
      ? [{ role: m.role, content: m.content.slice(0, MAX_TEXT) }]
      : [],
  );
  if (!messages.length || typeof v.at !== "number") return null;
  return {
    id: v.id,
    title: str(v.title, 120),
    at: v.at,
    model: str(v.model, 200),
    messages,
  };
}

/** Anything unreadable is dropped, never thrown. */
export function parseLibrary(raw: string | null): Library {
  try {
    return parseLibraryData(raw ? JSON.parse(raw) : null);
  } catch {
    return emptyLibrary();
  }
}

export function parseLibraryData(data: unknown): Library {
  try {
    if (!isRecord(data)) return emptyLibrary();
    return {
      presets: (Array.isArray(data.presets) ? data.presets : [])
        .map(parsePreset)
        .filter((p): p is Preset => p !== null)
        .slice(0, MAX_PRESETS),
      history: (Array.isArray(data.history) ? data.history : [])
        .map(parseEntry)
        .filter((e): e is HistoryEntry => e !== null)
        .slice(0, MAX_HISTORY),
    };
  } catch {
    return emptyLibrary();
  }
}

export function loadLibrary(baseUrl: string): Library {
  try {
    return parseLibrary(localStorage.getItem(libraryKey(baseUrl)));
  } catch {
    return emptyLibrary();
  }
}
export function removeLocalLibrary(baseUrl: string) {
  try {
    localStorage.removeItem(libraryKey(baseUrl));
  } catch {
    /* nothing to remove */
  }
}
/** false when the browser refused (private window, quota): the caller keeps working in memory. */
export function saveLibrary(baseUrl: string, library: Library): boolean {
  try {
    localStorage.setItem(libraryKey(baseUrl), JSON.stringify(library));
    return true;
  } catch {
    return false;
  }
}

export function savePreset(presets: Preset[], preset: Preset): Preset[] {
  const without = presets.filter((p) => p.id !== preset.id);
  return [preset, ...without].slice(0, MAX_PRESETS);
}
export const deletePreset = (presets: Preset[], id: string) =>
  presets.filter((p) => p.id !== id);

/** The first user line, one line, as the entry's title. */
export function titleOf(messages: readonly HistoryMessage[]): string {
  const first = messages.find((m) => m.role === "user")?.content ?? "";
  const line = first.replace(/\s+/g, " ").trim();
  return line.length > 60 ? `${line.slice(0, 59)}…` : line;
}

/** Insert or update one conversation (newest first); an empty conversation is not stored. */
export function upsertHistory(
  history: HistoryEntry[],
  entry: Omit<HistoryEntry, "title"> & { title?: string },
): HistoryEntry[] {
  const messages = entry.messages.slice(-MAX_HISTORY_MESSAGES);
  if (!messages.length) return history.filter((h) => h.id !== entry.id);
  const stored: HistoryEntry = {
    ...entry,
    messages,
    title: entry.title ?? titleOf(messages),
  };
  return [stored, ...history.filter((h) => h.id !== entry.id)].slice(
    0,
    MAX_HISTORY,
  );
}
export const deleteHistory = (history: HistoryEntry[], id: string) =>
  history.filter((h) => h.id !== id);

export const newId = () =>
  `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;

/** Version of the export file; an import of any other version is refused, not guessed at. */
export const EXPORT_SCHEMA = 1;

export function exportLibrary(library: Library, nowMs = Date.now()) {
  return {
    schema: EXPORT_SCHEMA,
    kind: "yunshu.playground",
    exported_at: new Date(nowMs).toISOString(),
    presets: library.presets,
    history: library.history,
  };
}

/** A parsed export, or null when it is not a Playground export of this schema. */
export function parseExport(raw: unknown): Library | null {
  if (!isRecord(raw) || raw.kind !== "yunshu.playground") return null;
  if (raw.schema !== EXPORT_SCHEMA) return null;
  return parseLibraryData(raw);
}

/**
 * Merge an imported library into the current one. Same id: the imported copy wins. The caps
 * apply afterwards (newest history first), so an import can never push the list past them.
 */
export function mergeLibraries(current: Library, imported: Library): Library {
  const presetIds = new Set(imported.presets.map((p) => p.id));
  const historyIds = new Set(imported.history.map((h) => h.id));
  return {
    presets: [
      ...imported.presets,
      ...current.presets.filter((p) => !presetIds.has(p.id)),
    ].slice(0, MAX_PRESETS),
    history: [
      ...imported.history,
      ...current.history.filter((h) => !historyIds.has(h.id)),
    ]
      .sort((a, b) => b.at - a.at)
      .slice(0, MAX_HISTORY),
  };
}

/**
 * A new conversation that shares the first `keep` messages of `entry`: the point to continue
 * from a different answer. `keep` is clamped to what exists and cut back to end on an
 * assistant reply (a branch point is a finished exchange).
 */
export function branchEntry(
  entry: HistoryEntry,
  keep: number,
  at = Date.now(),
): HistoryEntry {
  let n = Math.max(1, Math.min(keep, entry.messages.length));
  while (n > 1 && entry.messages[n - 1].role !== "assistant") n--;
  return {
    id: newId(),
    at,
    model: entry.model,
    title: entry.title,
    messages: entry.messages.slice(0, n).map((m) => ({ ...m })),
  };
}

/** Rename a preset or overwrite its parameters with the current ones, keeping id and name. */
export function updatePreset(
  presets: Preset[],
  id: string,
  change: Partial<Omit<Preset, "id">>,
): Preset[] {
  return presets.map((p) =>
    p.id === id
      ? {
          ...p,
          ...change,
          name: (change.name ?? p.name).trim().slice(0, 80) || p.name,
        }
      : p,
  );
}

/**
 * Which copy to trust after the async store answers: the store when it has anything,
 * else the synchronous one (a first run after upgrading), which then gets migrated.
 */
export function reconcile(
  sync: Library,
  stored: Library | null | undefined,
): { library: Library; migrate: boolean } {
  const has = (l: Library) => l.presets.length + l.history.length > 0;
  if (stored && has(stored)) return { library: stored, migrate: false };
  return { library: sync, migrate: stored !== undefined && has(sync) };
}
