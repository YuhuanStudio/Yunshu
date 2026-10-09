import { StatusIndicator } from "@yuhuanowo/yunui";
import type { EngineStatus } from "./api";
import { Slot } from "./ui";
import { livePill } from "./engineView";
import type { EngineConnectionPhase } from "./useEngine";

/**
 * The always-visible live pill (top bar, every page): the connection when it is
 * not online, otherwise the engine phase with its own number. It never shows a
 * bare tok/s that is not live. Its text is not a live region: the phase is
 * spoken only when the user reaches it.
 */
export function LivePill({
  phase,
  status,
}: {
  phase: EngineConnectionPhase;
  status: EngineStatus | null;
}) {
  const pill = livePill(phase, status);
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
      {/* As wide as the longest "prefill 100% · decode 123.4 tok/s", idle or busy: the pill and the
          cluster around it keep their width while phase and numbers change. */}
      <Slot ch={32} className="hidden leading-4 sm:inline-block">
        {pill.detail}
      </Slot>
    </span>
  );
}
