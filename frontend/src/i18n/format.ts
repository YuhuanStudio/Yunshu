import { getLocale } from "./index.ts";
import type { Locale } from "./types.ts";

// Intl formatters are expensive to build and a poll formats hundreds of
// values (every chart tick), so each distinct format is built once per locale.
const cache = new Map<string, unknown>();
function memo<T>(key: string, make: () => T): T {
  let v = cache.get(key) as T | undefined;
  if (!v) {
    v = make();
    cache.set(key, v);
  }
  return v;
}
const L = (): Locale => getLocale();

const numberFormat = (min: number, max: number) => {
  const l = L();
  return memo(
    `n:${l}:${min}:${max}`,
    () =>
      new Intl.NumberFormat(l, {
        minimumFractionDigits: min,
        maximumFractionDigits: max,
      }),
  );
};
const missing = (v: number | null | undefined): v is null | undefined =>
  v == null || !Number.isFinite(v);

export const number = (v: number | null | undefined, digits = 1) =>
  missing(v) ? "—" : numberFormat(0, digits).format(v);
/** Fixed decimals, so a polled value keeps the same number of characters. */
export const fixed = (v: number | null | undefined, digits = 1) =>
  missing(v) ? "—" : numberFormat(digits, digits).format(v);
/** `0.42` -> `42%`, without locale-specific spacing surprises. */
export const percent = (ratio: number | null | undefined, digits = 0) => {
  if (missing(ratio)) return "—";
  const l = L();
  return memo(
    `p:${l}:${digits}`,
    () =>
      new Intl.NumberFormat(l, {
        style: "percent",
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      }),
  ).format(ratio);
};
/** Gigabytes with a fixed decimal count: `12.3 GB`. */
export const gb = (v: number | null | undefined, digits = 1) =>
  missing(v) ? "—" : `${fixed(v, digits)} GB`;
/** Binary byte sizes: 1536 -> `1.5 KB`. */
export function bytes(v: number | null | undefined): string {
  if (missing(v)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let n = Math.abs(v);
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${numberFormat(0, i === 0 ? 0 : 1).format(v < 0 ? -n : n)} ${units[i]}`;
}

const dateFormat = (key: string, opts: Intl.DateTimeFormatOptions) => {
  const l = L();
  return memo(`d:${l}:${key}`, () => new Intl.DateTimeFormat(l, opts));
};
export const clock = (t: number) =>
  dateFormat("clock", {
    hourCycle: "h23",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(t);
/** HH:MM, for events where the second adds nothing (a restart, a stale stamp). */
export const clockShort = (t: number) =>
  dateFormat("clockShort", {
    hourCycle: "h23",
    hour: "2-digit",
    minute: "2-digit",
  }).format(t);
export const dateTime = (t: number | Date) =>
  dateFormat("dt", {
    dateStyle: "medium",
    timeStyle: "medium",
    hourCycle: "h23",
  }).format(t);

/** "3 seconds ago" in the active locale; `now` is injectable for tests. */
export function relative(seconds: number): string {
  const l = L();
  const rtf = memo(
    `r:${l}`,
    () => new Intl.RelativeTimeFormat(l, { numeric: "auto", style: "short" }),
  );
  const s = Math.round(seconds);
  if (Math.abs(s) < 60) return rtf.format(-s, "second");
  if (Math.abs(s) < 3600) return rtf.format(-Math.round(s / 60), "minute");
  if (Math.abs(s) < 86400) return rtf.format(-Math.round(s / 3600), "hour");
  return rtf.format(-Math.round(s / 86400), "day");
}

const unitFormat = (unit: "hour" | "minute" | "second", digits = 0) => {
  const l = L();
  return memo(
    `u:${l}:${unit}:${digits}`,
    () =>
      new Intl.NumberFormat(l, {
        style: "unit",
        unit,
        unitDisplay: "narrow",
        maximumFractionDigits: digits,
      }),
  );
};
/** Elapsed time as `1h 2m`, `3m 4s` or `5.2s` (narrow units from Intl). */
export function elapsed(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  if (seconds >= 3600)
    return `${unitFormat("hour").format(Math.floor(seconds / 3600))} ${unitFormat("minute").format(Math.floor((seconds % 3600) / 60))}`;
  if (seconds >= 60)
    return `${unitFormat("minute").format(Math.floor(seconds / 60))} ${unitFormat("second").format(Math.floor(seconds % 60))}`;
  return unitFormat("second", 1).format(seconds);
}

export function list(items: string[]): string {
  const l = L();
  return memo(
    `l:${l}`,
    () => new Intl.ListFormat(l, { style: "narrow", type: "conjunction" }),
  ).format(items);
}
