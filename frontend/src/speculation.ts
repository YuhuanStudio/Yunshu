import type { Row } from "./RequestTrace";

const count = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;

export interface ModeSummary {
  mode: string;
  /** Requests that reported this mode. */
  requests: number;
  /** Requests whose drafted AND accepted counts were both reported (the denominator rows). */
  counted: number;
  drafted: number;
  accepted: number;
  rounds: number;
  /** Requests that reported a round count, and how many rounds / copy rounds they had. */
  roundsReported: number;
  copyRounds: number;
  copyTokens: number;
  /** accepted / drafted over the counted requests; null when nothing was drafted. */
  acceptance: number | null;
}

export interface SpeculationSummary {
  /** Finished requests looked at. */
  total: number;
  /** Requests that reported a speculative block (the mode actually engaged). */
  engaged: number;
  /** Requests with no speculative block: the engine ran plain decode or did not say. */
  plain: number;
  modes: ModeSummary[];
  /** Requests whose mode was reported but whose drafted/accepted counters were not. */
  unattributed: number;
}

/**
 * What the engine reported about speculation across finished requests. Acceptance is the
 * token-weighted ratio (sum accepted / sum drafted) over requests that reported both counters;
 * a request with a mode but no counters is counted as unattributed, never filled in.
 */
export function summarizeSpeculation(
  rows: readonly Pick<Row, "speculative">[],
): SpeculationSummary {
  const by = new Map<string, ModeSummary>();
  let engaged = 0;
  let unattributed = 0;
  for (const r of rows) {
    const s = r.speculative;
    if (!s) continue;
    engaged++;
    const mode = s.mode || "—";
    let m = by.get(mode);
    if (!m) {
      m = {
        mode,
        requests: 0,
        counted: 0,
        drafted: 0,
        accepted: 0,
        rounds: 0,
        roundsReported: 0,
        copyRounds: 0,
        copyTokens: 0,
        acceptance: null,
      };
      by.set(mode, m);
    }
    m.requests++;
    const d = count(s.drafted);
    const a = count(s.accepted);
    if (d != null && a != null) {
      m.counted++;
      m.drafted += d;
      m.accepted += a;
    } else unattributed++;
    const rounds = count(s.rounds);
    if (rounds != null) {
      m.rounds += rounds;
      m.roundsReported++;
    }
    m.copyRounds += count(s.copy?.rounds) ?? 0;
    m.copyTokens += count(s.copy?.tokens) ?? 0;
  }
  const modes = [...by.values()].map((m) => ({
    ...m,
    acceptance: m.drafted > 0 ? m.accepted / m.drafted : null,
  }));
  modes.sort((x, y) => y.requests - x.requests);
  return {
    total: rows.length,
    engaged,
    plain: rows.length - engaged,
    modes,
    unattributed,
  };
}
