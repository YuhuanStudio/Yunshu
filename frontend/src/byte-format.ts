import { fixed, number } from "./i18n/format.ts";

const UNITS = ["B", "KB", "MB", "GB", "TB"] as const; // i18n-ignore
const MISSING = Object.freeze({ value: "—", unit: "" });

/**
 * Memory and file sizes use binary units labelled "GB"/"MB" (1024-based), as macOS, Activity Monitor and
 * Apple spec sheets do: a 128 GiB machine reads "128 GB", never "137 GB". The engine reports
 * decimal gigabytes in its `*_gb` fields; `binaryGb` converts them once, at the parse boundary.
 */
export const BINARY_GB_PER_DECIMAL_GB = 1e9 / 2 ** 30;
export const binaryGb = (decimalGb: number): number =>
  decimalGb * BINARY_GB_PER_DECIMAL_GB;
export const BYTES_PER_GB = 2 ** 30;

/**
 * One memory figure from an engine record, for both engine generations. A newer engine reports
 * binary GB plus an exact `<base>_bytes`: the bytes win. An older engine (0.1.4 and earlier)
 * reports only decimal `<base>_gb`: convert it once. Anything else is unknown (null), never 0.
 */
export function readGb(
  record: Record<string, unknown>,
  base: string,
): number | null {
  const bytes = record[`${base}_bytes`];
  if (typeof bytes === "number" && Number.isFinite(bytes))
    return bytes / BYTES_PER_GB;
  const gb = record[`${base}_gb`];
  return typeof gb === "number" && Number.isFinite(gb) ? binaryGb(gb) : null;
}

/** Byte size (1024-based, labelled KB/MB/GB) split into number and unit, so the unit can sit smaller after the number. */
export function splitBytes(v: number | null | undefined): {
  value: string;
  unit: string;
} {
  if (v == null || !Number.isFinite(v) || v < 0) return MISSING;
  let n = v,
    i = 0;
  while (n >= 1024 && i < UNITS.length - 1) {
    n /= 1024;
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

/**
 * One rule for "used / total" memory everywhere (band, island, sidebar, Diagnostics, Models):
 * used with one decimal, a total that is a whole number of GB without decimals, spaces around
 * the slash, unit once at the end: `23.2 / 128 GB`.
 */
export const gbTotalText = (total: number): string =>
  Math.abs(total - Math.round(total)) < 0.05
    ? number(total, 0)
    : fixed(total, 1);
export const memoryPairText = (used: number, total: number): string =>
  `${fixed(used, 1)} / ${gbTotalText(total)} GB`; // i18n-ignore
