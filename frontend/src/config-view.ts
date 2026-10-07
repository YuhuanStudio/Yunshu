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
export function maskedValue(name: string, value: unknown): unknown {
  if (value == null || value === "" || value === "***") return value;
  return SECRET_NAME.test(name) ? "***" : value;
}

export function parseConfig(payload: unknown): ConfigPayload | null {
  if (!isRecord(payload) || !Array.isArray(payload.settings)) return null;
  const rows: ConfigRow[] = [];
  for (const item of payload.settings) {
    if (!isRecord(item) || typeof item.name !== "string") continue;
    const secret = SECRET_NAME.test(item.name);
    rows.push({
      name: item.name,
      value: maskedValue(item.name, item.value),
      default: maskedValue(item.name, item.default),
      source: typeof item.source === "string" ? item.source : "default",
      stability: typeof item.stability === "string" ? item.stability : "stable",
      category: typeof item.category === "string" ? item.category : "",
      description: typeof item.description === "string" ? item.description : "",
      secret,
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
