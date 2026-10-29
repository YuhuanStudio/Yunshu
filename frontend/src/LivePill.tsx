import { StatusIndicator } from "@yuhuanowo/yunui";
import type { EngineStatus } from "./api";
import { HugWidth } from "./HugWidth";
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
      <HugWidth>
        <span className="leading-4 text-foreground">{pill.phase}</span>
        {pill.detail ? (
          <span className="ml-2 hidden leading-4 tabular-nums xl:inline">
            {pill.detail}
          </span>
        ) : null}
      </HugWidth>
    </span>
  );
}
