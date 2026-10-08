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
    const data: unknown = raw ? JSON.parse(raw) : null;
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
