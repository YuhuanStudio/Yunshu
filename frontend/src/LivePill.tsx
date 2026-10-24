import { StatusIndicator } from "@yuhuanowo/yunui";
import type { EngineStatus } from "./api";
import { Slot } from "./ui";
import { livePill } from "./engineView";
import type { EngineConnectionPhase } from "./useEngine";
import type { OutageView } from "./outage-view";

/**
 * The always-visible live pill (top bar, every page): the connection when it is
 * not online, otherwise the engine phase with its own number. It never shows a
 * bare tok/s that is not live. Its text is not a live region: the phase is
 * spoken only when the user reaches it.
 */
export function LivePill({
  phase,
  status,
  outage,
}: {
  phase: EngineConnectionPhase;
  status: EngineStatus | null;
  outage?: OutageView["kind"];
}) {
  const pill = livePill(phase, status, outage);
  return (
    <span
      className="card inline-flex h-8 items-center gap-2 rounded-full px-3 text-xs text-muted-foreground"
      data-testid="live-phase"
      data-phase={pill.phase}
    >
      <StatusIndicator status={pill.tone} />
      <Slot ch={6} className="leading-4 text-foreground">
        {pill.phase}
      </Slot>
      {/* Sized by its text (tabular digits, so a changing number does not jitter); a reserved
          30ch slot left most of the pill empty. */}
      {pill.detail ? (
        <span className="hidden whitespace-nowrap leading-4 tabular-nums xl:inline-block">
          {pill.detail}
        </span>
      ) : null}
    </span>
  );
}
