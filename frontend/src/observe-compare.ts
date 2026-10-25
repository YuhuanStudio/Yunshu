import type { Row } from "./RequestTrace";

/** One in-flight request as `GET /debug/requests` (`in_flight`) reports it from the engine's own counters. */
export interface DebugInFlight {
  request_id: string;
  path?: string;
  stream?: boolean;
  elapsed_ms?: number;
  has_stats?: boolean;
  phase?: string;
  prompt_tokens?: number;
  cached_tokens?: number;
  prefill_done?: number;
  prefill_total?: number;
  prefill_known?: boolean;
  generated?: number;
  t_admit_ms?: number | null;
  t_first_ms?: number | null;
  t_last_ms?: number | null;
}

export type CompareState = "match" | "differs" | "missing";

export interface CompareCell {
  key: "phase" | "prefill" | "tokens" | "admit" | "first";
  ui: string;
  debug: string;
  state: CompareState;
}

/** Pull `in_flight` out of a `/debug/requests` body; anything else is an empty list. */
export function parseInFlight(raw: unknown): DebugInFlight[] {
  const list = (raw as { in_flight?: unknown } | null)?.in_flight;
  if (!Array.isArray(list)) return [];
  return list.filter(
    (r): r is DebugInFlight =>
      typeof r === "object" &&
      r !== null &&
      typeof (r as DebugInFlight).request_id === "string",
  );
}

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;
const show = (v: number | null, unit = ""): string =>
  v == null ? "-" : `${Math.round(v * 10) / 10}${unit}`;

/** Counters read a moment apart may differ by a few tokens; ms stamps are the same instant on both sides. */
const near = (a: number, b: number, abs: number): boolean =>
  Math.abs(a - b) <= abs;

function cell(
  key: CompareCell["key"],
  ui: number | null,
  dbg: number | null,
  abs: number,
  unit = "",
): CompareCell {
  const state: CompareState =
    ui == null && dbg == null
      ? "match"
      : ui == null || dbg == null
        ? "missing"
        : near(ui, dbg, abs)
          ? "match"
          : "differs";
  return { key, ui: show(ui, unit), debug: show(dbg, unit), state };
}

/**
 * Side-by-side of what the console shows for a request (the /v1/yunshu/status row) and what the engine's
 * /debug/requests says. `dbg` undefined means /debug has no such request (finished in between, or not tracked).
 */
export function compareRequest(
  ui: Row,
  dbg: DebugInFlight | undefined,
  skewS = 0,
): CompareCell[] {
  if (!dbg)
    return (["phase", "prefill", "tokens", "admit", "first"] as const).map(
      (key) => ({ key, ui: "", debug: "-", state: "missing" as const }),
    );
  const uiOff = (ui.offsets_ms ?? {}) as Record<string, unknown>;
  const dPhase = dbg.has_stats === false ? "starting" : (dbg.phase ?? "-");
  const phaseCell: CompareCell = {
    key: "phase",
    ui: ui.phase,
    debug: dPhase,
    state: ui.phase === dPhase ? "match" : "differs",
  };
  const dPre =
    dbg.prefill_known === false
      ? null
      : (num(dbg.prefill_done) ?? 0) + (num(dbg.cached_tokens) ?? 0);
  const uiPre = ui.phase === "prefill" ? num(ui.processed_tokens) : null;
  const prefill =
    ui.phase === "prefill" || dbg.phase === "prefill"
      ? cell(
          "prefill",
          uiPre,
          dPre,
          64 + skewS * (num(ui.tokens_per_second) ?? 0),
        )
      : cell("prefill", null, null, 0);
  const uiGen = num(ui.completion_tokens);
  // The two reads are `skewS` apart; at the request's own speed that many tokens are expected to differ.
  const tokens = cell(
    "tokens",
    uiGen,
    num(dbg.generated),
    16 + skewS * (num(ui.tokens_per_second) ?? 0),
  );
  return [
    phaseCell,
    prefill,
    tokens,
    cell("admit", num(uiOff.admit), num(dbg.t_admit_ms), 250, " ms"),
    cell("first", num(uiOff.first_token), num(dbg.t_first_ms), 250, " ms"),
  ];
}

export const anyDiffers = (cells: readonly CompareCell[]): boolean =>
  cells.some((c) => c.state === "differs");
