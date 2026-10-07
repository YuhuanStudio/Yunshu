import { useMemo, useState } from "react";
import { Button, Sheet, StatusIndicator } from "@yuhuanowo/yunui";
import { StatusPill, StatusPillBar } from "@yuhuanowo/yunui/patterns";
import { ChevronUp } from "lucide-react";
import type { Connection } from "./api";
import { useMinWidth, type Engine } from "./ui";
import { footerPills, gpuBusyFraction, type FooterPill } from "./footer-status";
import { t } from "./i18n/index.ts";
import { useSignals } from "./signals";
import { livePill } from "./engineView";

/**
 * The status band: engine, what it does now, the machine. Current state only.
 * At phone width it is one row (dot and the current phase) that opens a sheet
 * with every pill, so it never wraps into a third of the screen.
 */
export function FooterStatus({
  engine,
}: {
  engine: Engine;
  connection?: Connection;
}) {
  const online = engine.phase === "online";
  const { ledger } = useSignals();
  const wide = useMinWidth(640);
  const [open, setOpen] = useState(false);
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
  const bar = (
    <StatusPillBar
      ariaLabel={t("shell.footer.ariaLabel")}
      className="shrink-0 gap-x-5 px-4 lg:px-6"
    >
      {pills.map((p) => (
        <StatusPill
          key={p.key}
          flat
          label={p.label}
          value={p.value}
          valueMinCh={p.minCh}
          tone={p.tone}
          dot={p.dot}
          help={p.help}
        />
      ))}
    </StatusPillBar>
  );
  if (wide) return bar;
  // The same state-to-colour map as the top-bar pill: idle is grey, decode green, work in between amber.
  const live = livePill(engine.phase, engine.status);
  return (
    <>
      <Button
        variant="ghost"
        className="h-10 w-full justify-between rounded-none px-4"
        aria-haspopup="dialog"
        aria-expanded={open}
        data-testid="footer-compact"
        onClick={() => setOpen(true)}
      >
        <span className="flex min-w-0 items-center gap-2 text-xs">
          <StatusIndicator status={live.tone} />
          <span className="truncate tabular-nums">
            {[live.phase, live.detail].filter(Boolean).join(" ")}
          </span>
        </span>
        <ChevronUp size={14} className="shrink-0 text-muted-foreground" />
      </Button>
      <Sheet
        open={open}
        onClose={() => setOpen(false)}
        title={t("shell.footer.sheetTitle")}
        closeLabel={t("shell.nav.close")}
      >
        <div className="pb-[env(safe-area-inset-bottom)]">{bar}</div>
      </Sheet>
    </>
  );
}
