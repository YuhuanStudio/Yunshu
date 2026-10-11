import { useEffect, useState } from "react";
import type { Connection } from "./api.ts";
import { fetchMetricsHistory } from "./history-api.ts";
import { gapPoint, type SeriesPoint } from "./series.ts";

/** The chart ranges: label key, window in seconds. Up to an hour the live series (backfilled) is enough. */
export const RANGES = [
  { id: "15m", seconds: 15 * 60 },
  { id: "1h", seconds: 3600 },
  { id: "6h", seconds: 6 * 3600 },
  { id: "24h", seconds: 24 * 3600 },
  { id: "7d", seconds: 7 * 86400 },
  { id: "30d", seconds: 30 * 86400 },
] as const;
export type RangeId = (typeof RANGES)[number]["id"];

/** Beyond this the in-page series (2,500 ms samples, 2,400 of them) no longer reaches back far enough. */
export const LIVE_RANGE_MAX_S = 3600;
const TARGET_POINTS = 600;

export const rangeSeconds = (id: string): number =>
  RANGES.find((r) => r.id === id)?.seconds ?? 900;

/** The step to ask the server for so a range comes back as about TARGET_POINTS rows. */
export const stepFor = (seconds: number): number =>
  Math.max(1, Math.round((seconds / TARGET_POINTS) * 10) / 10);

export interface RangeHistory {
  /** True while this hook, not the live series, supplies the chart (ranges beyond an hour). */
  active: boolean;
  points: SeriesPoint[];
  loading: boolean;
  /** The server answered without the history route, or the history is switched off. */
  unavailable: boolean;
  /** Spans (epoch ms) with no samples: the engine, or the console, was not running. */
  gaps: [number, number][];
  resolutionS: number | null;
}

/**
 * The recorded history for the long ranges (6 h to 30 d) from the console process, refreshed about
 * as often as a point is wide. Gaps come back as outage markers, so a chart breaks where the engine
 * was down and never interpolates across it.
 */
export function useRangeHistory(
  connection: Connection,
  range: string,
  /** The answers come through the console process: only then is there recorded history to ask for. */
  recorded: boolean,
): RangeHistory {
  const seconds = rangeSeconds(range);
  const active = seconds > LIVE_RANGE_MAX_S && recorded;
  const [state, setState] = useState<RangeHistory>({
    active,
    points: [],
    loading: active,
    unavailable: false,
    gaps: [],
    resolutionS: null,
  });
  useEffect(() => {
    if (!active) {
      setState((s) => ({ ...s, active: false, loading: false }));
      return;
    }
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const step = stepFor(seconds);
    const load = async () => {
      const now = Date.now();
      const loaded = await fetchMetricsHistory(connection, {
        signal: controller.signal,
        since: now - seconds * 1000,
        until: now,
        step,
      }).catch(() => null);
      if (controller.signal.aborted) return;
      if (loaded) {
        const points: SeriesPoint[] = [...loaded.points];
        for (const [from] of loaded.gaps) points.push(gapPoint(from));
        points.sort((a, b) => a.at - b.at);
        setState({
          active: true,
          points,
          loading: false,
          unavailable: false,
          gaps: loaded.gaps,
          resolutionS: loaded.intervalS,
        });
      } else
        setState((s) => ({
          ...s,
          active: true,
          loading: false,
          unavailable: s.points.length === 0,
        }));
      timer = setTimeout(() => void load(), Math.max(10_000, step * 1000));
    };
    setState((s) => ({ ...s, active: true, loading: s.points.length === 0 }));
    void load();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [connection.baseUrl, connection.token, active, seconds]);
  return { ...state, active };
}
