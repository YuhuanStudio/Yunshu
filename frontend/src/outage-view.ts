import type { EngineList, EngineStatus } from "./api.ts";

/** With no live engine for this long a restart is no longer "normal": the page escalates to an error. */
export const ESCALATE_AFTER_S = 60;

export type OutageView =
  | { kind: "none" }
  /** The followed engine is away and nothing else is running: a restart, as far as anyone can tell. */
  | {
      kind: "restarting";
      downS: number;
      /** How long the engine had been up when it went away, or null when never seen. */
      wasUpS: number | null;
      /** Median of the recent outages the console recorded, or null. */
      usualS: number | null;
    }
  /** Another engine is running: the console is about to follow it. */
  | { kind: "switching"; to: string; port: number | null; downS: number }
  /** Sustained outage with no live engine: a real error, with the engines the console knows about. */
  | { kind: "down"; downS: number; engines: EngineList["data"] };

export interface OutageInput {
  phase: "connecting" | "online" | "offline" | "unauthorized";
  /** Epoch ms of the first failed poll of this outage. */
  offlineSince: number | null;
  updatedAt: number | null;
  status: Pick<EngineStatus, "uptime_s"> | null;
  /** The console's engine list; null when it follows one fixed engine (or there is no console process). */
  list: EngineList | null;
  usualS: number | null;
  now: number;
  escalateAfterS?: number;
}

/**
 * What an unreachable engine means to the person looking. With `--engine auto` restarts are routine,
 * so a short gap is a calm "restarting" state, a running second engine is "switching", and only a
 * sustained outage with no live engine is an error.
 */
export function outageView(input: OutageInput): OutageView {
  const { phase, list } = input;
  if (phase !== "offline" || !list || list.mode !== "auto")
    return { kind: "none" };
  const since = input.offlineSince ?? input.updatedAt ?? input.now;
  const downS = Math.max(0, (input.now - since) / 1000);
  const running = list.data.filter((e) => e.state === "running");
  const live = running.find((e) => e.selected) ?? running[0];
  if (live) return { kind: "switching", to: live.url, port: live.port, downS };
  if (downS < (input.escalateAfterS ?? ESCALATE_AFTER_S)) {
    const lastSeenUp = input.status?.uptime_s;
    const wasUpS =
      lastSeenUp == null
        ? null
        : lastSeenUp +
          Math.max(
            0,
            ((input.offlineSince ?? input.now) - (input.updatedAt ?? 0)) / 1000,
          );
    return { kind: "restarting", downS, wasUpS, usualS: input.usualS };
  }
  return { kind: "down", downS, engines: list.data };
}
