import { t, tr } from "./i18n/index.ts";

/** Rows of GET /v1/yunshu/config, normalised for the 有效設定 table. */
export interface ConfigRow {
  name: string;
  value: unknown;
  default: unknown;
  source: string;
  stability: string;
  category: string;
  description: string;
  secret: boolean;
  /** Registry type: bool, int, float, gb, enum, str, path, list, json. */
  type: string;
  choices: string[];
  /** When a change takes effect: live, reload or restart ("" from an older server). */
  applies: string;
  minimum: number | null;
}

export interface ConfigPayload {
  rows: ConfigRow[];
  warnings: string[];
  experimentalCount: number;
  experimentalMax: number;
}

const SECRET_NAME = /(TOKEN|SECRET|PASSWORD|API_KEY|_KEY$)/i;

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** Never show a secret even if a server forgot to mask it. */
export function maskedValue(
  name: string,
  value: unknown,
  secret = SECRET_NAME.test(name),
): unknown {
  if (value == null || value === "" || value === "***") return value;
  return secret ? "***" : value;
}

export function parseConfig(payload: unknown): ConfigPayload | null {
  if (!isRecord(payload) || !Array.isArray(payload.settings)) return null;
  const rows: ConfigRow[] = [];
  for (const item of payload.settings) {
    if (!isRecord(item) || typeof item.name !== "string") continue;
    const secret = item.secret === true || SECRET_NAME.test(item.name);
    rows.push({
      name: item.name,
      value: maskedValue(item.name, item.value, secret),
      default: maskedValue(item.name, item.default, secret),
      source: typeof item.source === "string" ? item.source : "default",
      stability: typeof item.stability === "string" ? item.stability : "stable",
      category: typeof item.category === "string" ? item.category : "",
      description: typeof item.description === "string" ? item.description : "",
      secret,
      type: typeof item.type === "string" ? item.type : "",
      choices: Array.isArray(item.choices)
        ? item.choices.filter((c): c is string => typeof c === "string")
        : [],
      applies: typeof item.applies === "string" ? item.applies : "",
      minimum: typeof item.minimum === "number" ? item.minimum : null,
    });
  }
  return {
    rows,
    warnings: Array.isArray(payload.warnings)
      ? payload.warnings.filter((w): w is string => typeof w === "string")
      : [],
    experimentalCount:
      typeof payload.experimental_count === "number"
        ? payload.experimental_count
        : 0,
    experimentalMax:
      typeof payload.experimental_max === "number"
        ? payload.experimental_max
        : 8,
  };
}

export function formatConfigValue(value: unknown): string {
  if (value === null || value === undefined)
    return t("diagnostics.config.value.unset");
  if (typeof value === "string")
    return value === "" ? t("diagnostics.config.value.empty") : value;
  return JSON.stringify(value);
}

/** A row differs from its default when it is not sourced from the default. */
export function isChanged(row: ConfigRow): boolean {
  if (row.source !== "default") {
    return JSON.stringify(row.value) !== JSON.stringify(row.default);
  }
  return false;
}

export function filterConfig(
  rows: readonly ConfigRow[],
  query: string,
  changedOnly: boolean,
): ConfigRow[] {
  const q = query.trim().toLowerCase();
  return rows.filter(
    (row) =>
      (!changedOnly || isChanged(row)) &&
      (!q ||
        `${row.name} ${formatConfigValue(row.value)} ${row.category} ${row.description}`
          .toLowerCase()
          .includes(q)),
  );
}

const SOURCES = ["default", "env", "cli", "file"];
const STABILITIES = ["stable", "experimental", "internal"];
export const sourceLabel = (source: string) =>
  SOURCES.includes(source) ? tr(`diagnostics.config.source.${source}`) : source;
export const stabilityLabel = (stability: string) =>
  STABILITIES.includes(stability)
    ? tr(`diagnostics.config.stability.${stability}`)
    : stability;

// ── editing ─────────────────────────────────────────────────────────

export type FieldKind = "bool" | "number" | "enum" | "text" | "secret";

/** The control a row needs. Choices win over the type; secrets are always write-only. */
export function fieldKind(row: ConfigRow): FieldKind {
  if (row.secret) return "secret";
  if (row.choices.length) return "enum";
  if (row.type === "bool") return "bool";
  if (["int", "float", "gb"].includes(row.type)) return "number";
  return "text";
}

/** A draft maps NAME to the new value; null resets the setting to its default. */
export type Drafts = Record<string, unknown>;

const same = (a: unknown, b: unknown) =>
  JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/** Set or clear one draft; a draft equal to the current value is not a change. */
export function withDraft(
  drafts: Drafts,
  row: ConfigRow,
  value: unknown,
): Drafts {
  const next = { ...drafts };
  const kind = fieldKind(row);
  // An empty text box over an unset value is not a change.
  const blank = kind === "text" && value === "" && (row.value ?? "") === "";
  const unchanged =
    value !== null && kind !== "secret" && (blank || same(value, row.value));
  if (unchanged) delete next[row.name];
  else next[row.name] = value;
  return next;
}

/** A row can be reset when it is not already at its default. */
export const canReset = (row: ConfigRow) => row.source !== "default";

/** The value a row shows: its draft when edited, otherwise the effective value. */
export const shownValue = (row: ConfigRow, drafts: Drafts): unknown =>
  row.name in drafts
    ? drafts[row.name] === null
      ? row.default
      : drafts[row.name]
    : row.value;

export const experimentalNames = (
  rows: readonly ConfigRow[],
  drafts: Drafts,
): string[] =>
  rows
    .filter((r) => r.name in drafts && r.stability !== "stable")
    .map((r) => r.name);

const APPLIES = ["live", "reload", "restart"];
export const appliesLabel = (applies: string) =>
  APPLIES.includes(applies) ? tr(`settings.config.applies.${applies}`) : "";

/** Text drafts of list / json types stay strings; a number draft must be finite. */
export function draftIsValid(row: ConfigRow, value: unknown): boolean {
  if (value === null) return true;
  switch (fieldKind(row)) {
    case "number":
      return (
        typeof value === "number" &&
        Number.isFinite(value) &&
        (row.minimum == null || value >= row.minimum)
      );
    case "secret":
      return typeof value === "string" && value.trim() !== "";
    default:
      return true;
  }
}
