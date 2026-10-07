import { fixed, number } from "./i18n/format.ts";

const UNITS = ["B", "KB", "MB", "GB", "TB"] as const; // i18n-ignore
const MISSING = Object.freeze({ value: "—", unit: "" });

/** Decimal byte size split into number and unit, so the unit can sit smaller after the number. */
export function splitBytes(v: number | null | undefined): {
  value: string;
  unit: string;
} {
  if (v == null || !Number.isFinite(v) || v < 0) return MISSING;
  let n = v,
    i = 0;
  while (n >= 1000 && i < UNITS.length - 1) {
    n /= 1000;
    i++;
  }
  return {
    value: i < 2 ? number(n, 0) : fixed(n, n >= 100 ? 0 : 1),
    unit: UNITS[i],
  };
}

/** Plain text form, for titles and aria labels: `12.3GB`. */
export function bytesText(v: number | null | undefined): string {
  const s = splitBytes(v);
  return s.value + s.unit;
}

/** Transfer rate, `45MB/s`. */
export function rateText(bps: number | null | undefined): string {
  if (bps == null || !Number.isFinite(bps) || bps <= 0) return "—";
  return `${bytesText(bps)}/s`; // i18n-ignore
}
