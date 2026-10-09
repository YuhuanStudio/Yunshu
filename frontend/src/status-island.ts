import type { EngineModelStatus } from "./api.ts";
import type { SeriesPoint } from "./series.ts";

/** Fewer decode samples than this is a calm "collecting" state: no line is drawn. */
export const SPARK_MIN_POINTS = 5;
export const SPARK_WINDOW_MS = 60_000;

/**
 * The live decode total over the last minute, oldest first. Anything the engine did not report
 * (null decode: nothing decoding, or an outage marker) is a 0 tok/s sample, never interpolated;
 * the series is cut at an outage marker so a line never bridges the time the engine was away.
 * Returns [] while there are fewer than SPARK_MIN_POINTS samples.
 */
export function decodeSparkPoints(
  series: readonly SeriesPoint[],
  windowMs = SPARK_WINDOW_MS,
): { t: number; v: number }[] {
  const last = series.at(-1);
  if (!last) return [];
  let start = series.length;
  while (
    start > 0 &&
    !series[start - 1].gap &&
    series[start - 1].at >= last.at - windowMs
  )
    start -= 1;
  const rows = series.slice(start);
  if (rows.length < SPARK_MIN_POINTS) return [];
  return rows.map((p) => ({
    t: p.at,
    v: typeof p.decode === "number" && p.decode > 0 ? p.decode : 0,
  }));
}

/** The same samples as bare values (oldest first), for charts that do not need the time. */
export const decodeSparkline = (
  series: readonly SeriesPoint[],
  windowMs = SPARK_WINDOW_MS,
): number[] => decodeSparkPoints(series, windowMs).map((p) => p.v);

/** "VLM" for an mlx-vlm engine, "LLM" for a text-only one; null when the type says nothing. */
export function modelKind(
  model: EngineModelStatus | undefined,
): "VLM" | "LLM" | null {
  if (!model?.type) return null;
  return /vlm|vision|omni/i.test(model.type) ? "VLM" : "LLM";
}

export const shortModelName = (id: string): string =>
  id.split("/").filter(Boolean).at(-1) ?? id;
