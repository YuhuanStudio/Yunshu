import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
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
import { idleDetail, livePill, phaseLabels } from "./engineView";
import type { IslandDot } from "./StatusIslandContent";

// Loaded on first open: the island carries the meter, sparkline and host card.
const StatusIslandContent = lazy(() =>
  import("./StatusIslandContent").then((m) => ({
    default: m.StatusIslandContent,
  })),
);
import { LiveNumber } from "./LiveNumber";
import { HeldText } from "./HeldText";
import { useHostTelemetry } from "./host-hook";

const TONE_COLOR: Record<FooterPill["tone"], string> = {
  neutral: "var(--text-primary)",
  success: "var(--success)",
  warning: "var(--warning)",
  danger: "var(--error)",
};

/**
 * A pill's text: the label, then the value in YunUI's StatusPill value style (mono, medium, tabular;
 * the pill's own value slot cannot share a box with a label that changes, YunUI change proposed: a
 * `hug` prop on StatusPill). The pill is as wide as its text. Every number holds its own width
 * (HeldText, LiveNumber), so a digit count flickering at a boundary moves the pill by one digit,
 * never by the room of a whole sentence, and nothing animates its width.
 */
function PillText({ pill: p }: { pill: FooterPill }) {
  const value = p.tween ? (
    <>
      <LiveNumber value={p.tween.value} digits={p.tween.digits} />
      {p.tween.suffix}
    </>
  ) : p.value != null && p.value !== "" ? (
    p.plain ? (
      p.value
    ) : (
      <HeldText text={p.value} />
    )
  ) : null;
  return (
    <span className="inline-block whitespace-nowrap">
      {p.label}
      {value != null ? (
        <span
          className="ml-1.5 font-mono font-medium tabular-nums"
          style={{ color: TONE_COLOR[p.tone] }}
        >
          {value}
        </span>
      ) : null}
    </span>
  );
}

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
    calm:
      engine.outage.kind === "restarting" ||
      engine.outage.kind === "switching" ||
      engine.outage.kind === "paused"
        ? engine.outage.kind
        : undefined,
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
  // Which capsule the card grew from. Tapping another capsule while open glides the card to it (no
  // close and reopen): the island's outside-press close is skipped for a press that lands on a capsule,
  // and a resize event makes the island re-measure against its new anchor.
  const [openKey, setOpenKey] = useState<string | null>(null);
  const pressOnCapsule = useRef(false);
  useEffect(() => {
    const down = (e: PointerEvent) => {
      pressOnCapsule.current = !!(e.target as Element | null)?.closest?.(
        "[data-island-capsule]",
      );
    };
    window.addEventListener("pointerdown", down, true);
    return () => window.removeEventListener("pointerdown", down, true);
  }, []);
  const close = () => {
    if (pressOnCapsule.current) return;
    setOpen(false);
  };
  const openFrom = (key: string, el: HTMLElement) => {
    pressOnCapsule.current = false;
    anchorRef.current = el;
    if (open && openKey === key) return setOpen(false);
    setOpenKey(key);
    if (open)
      requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    else setOpen(true);
  };
  const island = (
    <StatusIsland
      open={open}
      onClose={close}
      anchorRef={anchorRef}
      label={t("shell.footer.sheetTitle")}
    >
      <Suspense fallback={null}>
        {/* keyed by capsule: switching capsules cross-fades the content (styles.css, island-swap) */}
        <div key={openKey ?? ""} className="island-swap flex flex-col gap-2">
          <StatusIslandContent
            engine={engine}
            connection={connection}
            ledger={online ? ledger : null}
            host={host}
            dot={dot}
            now={Date.now()}
            focus={wide ? openKey : null}
            onNavigate={() => setOpen(false)}
          />
        </div>
      </Suspense>
    </StatusIsland>
  );
  if (wide)
    return (
      <>
        <div className="w-fit max-w-full px-4 pb-3 lg:px-6">
          {/* Islands, like the iPhone's Dynamic Island: each reading is its own capsule floating over the
              page (no band behind them), its width following its content. Each capsule is its own trigger and the
              card grows out of the capsule that was tapped. */}
          <StatusPillBar
            ariaLabel={t("shell.footer.ariaLabel")}
            className="gap-2 px-0 pb-0 pt-0"
          >
            {pills.map((p, i) => (
              <StatusPill
                key={p.key}
                label={
                  <>
                    <PillText pill={p} />
                    <button
                      type="button"
                      data-island-capsule=""
                      aria-haspopup="dialog"
                      aria-expanded={open && openKey === p.key}
                      aria-label={`${t("shell.island.trigger")}: ${p.label}`}
                      data-testid={
                        i === 0 ? "footer-trigger" : `footer-trigger-${p.key}`
                      }
                      className="absolute inset-0 cursor-pointer rounded-full outline-none transition-colors duration-(--dur-fast) ease-(--ease-standard) hover:bg-foreground/[0.04] focus-visible:ring-2 focus-visible:ring-(--ring)"
                      onClick={(e) => openFrom(p.key, e.currentTarget)}
                    />
                  </>
                }
                tone={p.tone}
                dot={p.dot}
                help={p.help}
                className={`relative px-3 py-1.5 shadow-[0_6px_20px_-8px_rgb(0_0_0/0.35)] transition-[translate,box-shadow] duration-(--dur-fast) ease-(--ease-standard) hover:-translate-y-px hover:shadow-[0_10px_24px_-8px_rgb(0_0_0/0.4)] ${open && openKey === p.key ? "-translate-y-px" : ""}`}
              />
            ))}
          </StatusPillBar>
        </div>
        {island}
      </>
    );
  // The collapsed pill leads with the ENGINE state (same tone as the first pill
  // of the band), then the phase as secondary text: 「運作中 · 閒置」.
  const live = livePill(engine.phase, engine.status, engine.outage.kind);
  // Narrow collapsed pill: the idle detail is the short form ("last 254 tok/s · 5s"), never cut off mid-word.
  const idleShort =
    online && engine.status && live.phase === phaseLabels.idle
      ? idleDetail(engine.status, Date.now(), true)
      : null;
  const secondary = online
    ? [live.phase, idleShort ?? live.detail]
        .filter(Boolean)
        .join(idleShort ? " · " : " ")
    : (lead.value ?? "");
  return (
    <>
      <div className="flex justify-center px-4 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
        <Button
          variant="outline"
          ref={(el: HTMLButtonElement | null) => {
            anchorRef.current = el;
          }}
          className="h-8 w-80 max-w-full gap-2 rounded-full bg-(--bg-elevated) px-3 text-[11px] shadow-[0_6px_20px_-8px_rgb(0_0_0/0.35)]"
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
