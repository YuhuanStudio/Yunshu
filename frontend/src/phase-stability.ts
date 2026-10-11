import type { ActivityPhase } from "./engineView.ts";

/**
 * A steady phase for the display. The engine's phase is exact per request and never goes back
 * (queued, prefill, decode, done), but a busy console sees MANY requests: an agent's turns each prefill
 * briefly and then decode, so a sampler that reads "any request prefilling" flips 解碼 / 預填 / 解碼 as often
 * as it polls. The display therefore follows two rules, both about time and neither about guessing:
 *   - going forward is immediate (idle to anything, queued to prefill, prefill to decode);
 *   - going back (decode to prefill, anything to idle) needs the new phase to hold for a moment, so a short
 *     prefill between two decodes, or the gap between two requests, does not flash.
 * A prefill that lasts (a long prompt) still shows once it has held for `backHoldMs`.
 */
export interface StablePhaseState {
  phase: ActivityPhase;
  /** A different raw phase seen continuously since this time, or null. */
  candidate: ActivityPhase | null;
  candidateSince: number;
}

const RANK: Record<ActivityPhase, number> = {
  idle: 0,
  queued: 1,
  prefill: 2,
  decode: 3,
};

export const BACK_HOLD_MS = 900;
export const IDLE_HOLD_MS = 1500;

export const initialStablePhase = (raw: ActivityPhase): StablePhaseState => ({
  phase: raw,
  candidate: null,
  candidateSince: 0,
});

export function nextStablePhase(
  prev: StablePhaseState,
  raw: ActivityPhase,
  now: number,
  hold: { back?: number; idle?: number } = {},
): StablePhaseState {
  if (raw === prev.phase)
    return { phase: prev.phase, candidate: null, candidateSince: 0 };
  // Forward moves (a higher phase) apply at once; everything else is a step back and has to hold.
  if (RANK[raw] > RANK[prev.phase])
    return { phase: raw, candidate: null, candidateSince: 0 };
  const needed =
    raw === "idle" ? (hold.idle ?? IDLE_HOLD_MS) : (hold.back ?? BACK_HOLD_MS);
  const since = prev.candidate === raw ? prev.candidateSince : now;
  if (now - since >= needed)
    return { phase: raw, candidate: null, candidateSince: 0 };
  return { phase: prev.phase, candidate: raw, candidateSince: since };
}

/** Exponential moving average over irregular samples: `tauMs` is the time constant. */
export function emaStep(
  prev: number | null,
  sample: number | null,
  dtMs: number,
  tauMs = 1000,
): number | null {
  if (sample == null) return null;
  if (prev == null) return sample;
  const a = 1 - Math.exp(-Math.max(0, dtMs) / tauMs);
  return prev + (sample - prev) * a;
}
