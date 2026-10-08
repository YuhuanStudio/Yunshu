import { lazy, Suspense, useMemo, useRef, useState } from "react";
import { Button, StatusIndicator } from "@yuhuanowo/yunui";
import {
  StatusIsland,
  StatusPill,
  StatusPillBar,
} from "@yuhuanowo/yunui/patterns";
import { ChevronUp } from "lucide-react";
import type { Connection } from "./api";
import { useMinWidth, type Engine } from "./ui";
import { footerPills, gpuBusyFraction, type FooterPill } from "./footer-status";
import { t } from "./i18n/index.ts";
import { useSignals } from "./signals";
import { livePill } from "./engineView";
import type { IslandDot } from "./StatusIslandContent";

// Loaded on first open: the island carries the meter, sparkline and host card.
const StatusIslandContent = lazy(() =>
  import("./StatusIslandContent").then((m) => ({
    default: m.StatusIslandContent,
  })),
);
import { useHostTelemetry } from "./host-hook";

/**
 * The status band: engine, what it does now, the machine. Current state only.
 * At phone width it is one row (dot and the current phase) that opens a sheet
 * with every pill, so it never wraps into a third of the screen.
 */
export function FooterStatus({
  engine,
  connection,
}: {
  engine: Engine;
  connection: Connection;
}) {
  const online = engine.phase === "online";
  const { ledger } = useSignals();
  const wide = useMinWidth(640);
  const [open, setOpen] = useState(false);
  const anchorRef = useRef<HTMLElement | null>(null);
  const host = useHostTelemetry(connection, online && open);
  const history = engine.history;
  const gpuBusy = useMemo(() => {
    const latest = history.at(-1);
    if (!latest) return null;
    // Compare with the sample about 9 s earlier (three polls) to smooth single ticks.
    const earlier =
      [...history].reverse().find((p) => latest.at - p.at >= 8_000) ??
      history.at(-2);
    return earlier && earlier !== latest
      ? gpuBusyFraction(earlier.status, latest.status)
      : null;
  }, [history]);
  const pills = footerPills({
    phase: engine.phase,
    errorStatus: engine.errorStatus,
    status: engine.status,
    gpuBusy: online ? gpuBusy : null,
    ledger: online ? ledger : null,
  });
  const lead = pills[0];
  const dot: IslandDot =
    lead.tone === "success"
      ? engine.status?.models.some((m) => m.loading)
        ? "away"
        : "online"
      : lead.tone === "danger"
        ? "offline"
        : lead.tone === "warning"
          ? "away"
          : "neutral";
  const toggle = () => setOpen((v) => !v);
  const island = (
    <StatusIsland
      open={open}
      onClose={() => setOpen(false)}
      anchorRef={anchorRef}
      label={t("shell.footer.sheetTitle")}
    >
      <Suspense fallback={null}>
        <StatusIslandContent
          engine={engine}
          ledger={online ? ledger : null}
          host={host}
          dot={dot}
          now={Date.now()}
          onNavigate={() => setOpen(false)}
        />
      </Suspense>
    </StatusIsland>
  );
  if (wide)
    return (
      <>
        <div className="shrink-0 px-4 lg:px-6">
          <div className="relative w-fit max-w-full">
            <StatusPillBar
              ariaLabel={t("shell.footer.ariaLabel")}
              className="px-0"
            >
              {pills.map((p) => (
                <StatusPill
                  key={p.key}
                  label={p.label}
                  value={p.value}
                  valueMinCh={p.minCh}
                  tone={p.tone}
                  dot={p.dot}
                  help={p.help}
                />
              ))}
            </StatusPillBar>
            <button
              type="button"
              ref={(el) => {
                anchorRef.current = el;
              }}
              aria-haspopup="dialog"
              aria-expanded={open}
              aria-label={t("shell.island.trigger")}
              data-testid="footer-trigger"
              className="absolute inset-0 rounded-2xl outline-none transition-colors hover:bg-foreground/[0.04] focus-visible:outline-2 focus-visible:outline-(--color-accent)"
              onClick={toggle}
            />
          </div>
        </div>
        {island}
      </>
    );
  // The collapsed pill leads with the ENGINE state (same tone as the first pill
  // of the band), then the phase as secondary text: 「運作中 · 閒置」.
  const live = livePill(engine.phase, engine.status);
  const secondary = online
    ? [live.phase, live.detail].filter(Boolean).join(" ")
    : (lead.value ?? "");
  return (
    <>
      <div className="px-4 pb-3 pt-1">
        <Button
          variant="outline"
          ref={(el: HTMLButtonElement | null) => {
            anchorRef.current = el;
          }}
          className="h-8 max-w-full gap-2 rounded-full bg-(--bg-elevated) px-3 text-[11px]"
          aria-haspopup="dialog"
          aria-expanded={open}
          data-testid="footer-compact"
          onClick={toggle}
        >
          <StatusIndicator status={dot} />
          <span className="truncate tabular-nums">
            {[lead.label, secondary].filter(Boolean).join(" · ")}
          </span>
          <ChevronUp size={13} className="shrink-0 text-muted-foreground" />
        </Button>
      </div>
      {island}
    </>
  );
}
