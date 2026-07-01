/** Formatting helpers shared across the Yunshu webui. */

/** Human-readable bytes: 1536 → "1.5 KB". */
export function fmtBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "0 B";
  if (bytes === 0) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), u.length - 1);
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${u[i]}`;
}

/** Compact integer with thousands separators: 48213 → "48,213". */
export function fmtNumber(n: number): string {
  if (!Number.isFinite(n)) return "0";
  return Math.round(n).toLocaleString("en-US");
}

/** Percentage to one decimal, clamped 0–100: 72.345 → "72.3%". */
export function fmtPct(n: number): string {
  const v = Math.max(0, Math.min(100, Number.isFinite(n) ? n : 0));
  return `${v.toFixed(1)}%`;
}

/** Seconds → "1h 03m", "3m 12s", or "45s". */
export function fmtDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0s";
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m.toString().padStart(2, "0")}m`;
  if (m > 0) return `${m}m ${sec.toString().padStart(2, "0")}s`;
  return `${sec}s`;
}

/** mm:ss clock for media time. */
export function fmtClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const sec = Math.floor(seconds % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
