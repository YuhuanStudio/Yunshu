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

/** One loaded model's cumulative speculation counters from `GET /debug/spec-decode`. */
export interface EngineSpecCounters {
  model: string;
  mtp: {
    cycles: number;
    accepts: number;
    rejects: number;
    cooldowns: number | null;
  } | null;
  adaptive: {
    currentK: number | null;
    drafted: number;
    accepted: number;
  } | null;
}

const rec = (v: unknown): Record<string, unknown> | null =>
  v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : null;

/** Only counters the engine actually sent; a model with none of them is left out. */
export function parseSpecCounters(raw: unknown): EngineSpecCounters[] {
  const models = rec(raw)?.models;
  if (!Array.isArray(models)) return [];
  const out: EngineSpecCounters[] = [];
  for (const m of models) {
    const r = rec(m);
    if (!r) continue;
    const mt = rec(r.mtp_stats);
    const cycles = count(mt?.total_cycles);
    const accepts = count(mt?.accepts);
    const rejects = count(mt?.rejects);
    const ad = rec(r.adaptive_spec);
    const drafted = count(ad?.total_draft_tokens);
    const accepted = count(ad?.total_accepted_tokens);
    const mtp =
      cycles != null && accepts != null && rejects != null
        ? { cycles, accepts, rejects, cooldowns: count(mt?.cooldowns) }
        : null;
    const adaptive =
      drafted != null && accepted != null
        ? { currentK: count(ad?.current_k), drafted, accepted }
        : null;
    if (mtp || adaptive)
      out.push({
        model: typeof r.model_id === "string" ? r.model_id : "",
        mtp,
        adaptive,
      });
  }
  return out;
}
