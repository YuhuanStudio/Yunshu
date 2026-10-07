import {
  Gauge,
  SegmentedBar,
  StatusIndicator,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import type { BarMark, BarSegment, SegmentTone } from "@yuhuanowo/yunui";
import type { Connection } from "./api";
import { gb, number } from "./i18n/format.ts";
import { has, t, tr, useLocale } from "./i18n/index.ts";
import {
  useMemoryLedger,
  type MemoryLedgerData,
  type MemoryOwner,
} from "./memory-api";

/** Capacity tones, as in Strata: warning from 80%, danger above 92% of unified memory. */
export const LEDGER_WARN = 0.8,
  LEDGER_DANGER = 0.92;

const KIND_TONE: Record<string, SegmentTone> = {
  weights: "accent",
  apc_ram: "info",
  apc_warm: "info",
  live_kv: "success",
  mlx_cache: "neutral",
  other: "warning",
};

const gbText = (v: number | null, digits = 1) =>
  v == null ? t("overview.ledger.unknown") : gb(v, digits);

export function ownerLabel(o: MemoryOwner) {
  const key = `overview.ledger.kind.${o.kind}`;
  // i18n-keys: overview.ledger.kind.
  const base = has(key) ? tr(key) : o.kind;
  return o.id ? `${base} · ${o.id.split("/").pop()}` : base;
}

/** Fraction of unified memory held by MLX; null when either number is unknown. */
export function ledgerUsage(d: MemoryLedgerData): number | null {
  const a = d.mlx.active_gb,
    t = d.total_gb;
  return a != null && t != null && t > 0 ? a / t : null;
}

export function ledgerTone(usage: number | null): SegmentTone {
  if (usage == null) return "neutral";
  return usage > LEDGER_DANGER
    ? "error"
    : usage > LEDGER_WARN
      ? "warning"
      : "accent";
}

function ledgerBar(d: MemoryLedgerData) {
  const segments: BarSegment[] = d.owners
    .filter((o) => o.gb != null && o.gb > 0)
    .map((o) => ({
      value: o.gb as number,
      tone: KIND_TONE[o.kind] ?? "neutral",
      label: `${ownerLabel(o)}${o.estimated ? t("overview.ledger.estimated") : ""}`,
    }));
  const marks: BarMark[] = [];
  if (d.mlx.peak_gb != null)
    marks.push({
      value: d.mlx.peak_gb,
      label: t("overview.ledger.peakMark", { gb: gbText(d.mlx.peak_gb) }),
    });
  if (d.mlx.recommended_working_set_gb != null)
    marks.push({
      value: d.mlx.recommended_working_set_gb,
      tone: "warning",
      label: t("overview.ledger.recommendedMark", {
        gb: gbText(d.mlx.recommended_working_set_gb),
      }),
    });
  if (d.host.wired_limit_gb != null)
    marks.push({
      value: d.host.wired_limit_gb,
      tone: "error",
      label: t("overview.ledger.wiredMark", {
        gb: gbText(d.host.wired_limit_gb),
      }),
    });
  return { segments, marks };
}

/** Compact ledger: usage gauge, one bar by owner, one summary line. Safe to embed on the overview. */
export function MemoryLedgerView({
  data,
  compact = false,
}: {
  data: MemoryLedgerData;
  compact?: boolean;
}) {
  useLocale();
  const usage = ledgerUsage(data),
    tone = ledgerTone(usage);
  const { segments, marks } = ledgerBar(data);
  const unknownOwners = data.owners.filter((o) => o.gb == null);
  return (
    <div className="space-y-4" data-testid="memory-ledger">
      <div className="flex flex-wrap items-center gap-5">
        {usage == null ? (
          <p className="text-sm text-muted-foreground">
            {t("overview.ledger.metalUnknown")}
          </p>
        ) : (
          <Gauge
            value={usage * 100}
            size={compact ? 72 : 96}
            thickness={compact ? 7 : 8}
            tone={tone}
            label={`${number(usage * 100, 0)}%`}
            ariaLabel={t("overview.ledger.gaugeAria", {
              pct: number(usage * 100, 0),
            })}
          />
        )}
        <div className="min-w-0 flex-1 basis-56 space-y-1 text-sm tabular-nums">
          <p>
            {t("overview.ledger.usage", { used: gbText(data.mlx.active_gb) })}
            <span className="text-muted-foreground">
              {" "}
              {t("overview.ledger.ofTotal", {
                total: gbText(data.total_gb, 0),
              })}
            </span>
            {tone === "error" && (
              <StatusIndicator
                status="offline"
                className="ml-2 gap-1.5 text-xs text-muted-foreground"
              >
                {t("overview.ledger.nearLimit")}
              </StatusIndicator>
            )}
          </p>
          <p className="text-xs text-muted-foreground">
            {t("overview.ledger.free", {
              gb: gbText(data.free_gb),
              peak: gbText(data.mlx.peak_gb),
            })}
            {data.host.pressure_level
              ? t("overview.ledger.pressure", {
                  level: data.host.pressure_level,
                })
              : ""}
            {data.host.swap_used_gb != null
              ? t("overview.ledger.swap", {
                  gb: gbText(data.host.swap_used_gb, 2),
                })
              : ""}
          </p>
        </div>
      </div>
      {data.total_gb != null && data.total_gb > 0 && segments.length > 0 ? (
        <SegmentedBar
          segments={segments}
          marks={marks}
          total={data.total_gb}
          legend={!compact}
          height={compact ? 6 : 10}
          formatValue={(v) => gb(v, 1)}
          label={t("overview.ledger.barLabel")}
        />
      ) : (
        <p className="text-xs text-muted-foreground">
          {t("overview.ledger.noOwners")}
        </p>
      )}
      {data.attribution_overshoot_gb != null && (
        <p className="text-xs text-muted-foreground">
          {t("overview.ledger.overshoot", {
            gb: gbText(data.attribution_overshoot_gb, 2),
          })}
        </p>
      )}
      {!compact && (
        <div className="overflow-x-auto">
          <Table
            aria-label={t("overview.ledger.tableAria")}
            scrollLabel={t("overview.ledger.tableLabel")}
          >
            <Thead>
              <Tr>
                <Th>{t("overview.ledger.colOwner")}</Th>
                <Th className="text-right">{t("overview.ledger.colSize")}</Th>
                <Th>{t("overview.ledger.colReclaimable")}</Th>
              </Tr>
            </Thead>
            <Tbody>
              {data.owners.map((o, i) => (
                <Tr key={`${o.kind}-${o.id}-${i}`}>
                  <Td>{ownerLabel(o)}</Td>
                  <Td
                    className="text-right tabular-nums"
                    title={o.source ?? t("overview.ledger.noCounter")}
                  >
                    {o.gb == null ? (
                      t("overview.ledger.unknown")
                    ) : (
                      <>
                        {o.estimated ? "≈ " : ""}
                        {gb(o.gb, 2)}
                        {o.estimated && (
                          <span className="ml-1 text-xs text-muted-foreground">
                            {t("overview.ledger.estimatedShort")}
                          </span>
                        )}
                      </>
                    )}
                  </Td>
                  <Td className="text-muted-foreground">
                    {o.reclaimable
                      ? t("overview.ledger.reclaimable")
                      : t("overview.ledger.notReclaimable")}
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
          {unknownOwners.length > 0 && (
            <p className="mt-2 text-xs text-muted-foreground">
              {t("overview.ledger.unknownNote")}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/** Fetches and renders the ledger; an older server without the route gets a one-line note. */
export function MemoryLedger({
  connection,
  enabled = true,
  compact = false,
}: {
  connection: Connection;
  enabled?: boolean;
  compact?: boolean;
}) {
  useLocale();
  const { data, unsupported, error } = useMemoryLedger(connection, enabled);
  if (unsupported)
    return (
      <p
        className="text-sm text-muted-foreground"
        data-testid="memory-ledger-unsupported"
      >
        {t("overview.ledger.unsupported")}
      </p>
    );
  if (!data)
    return (
      <p role="status" className="text-sm text-muted-foreground">
        {error
          ? t("overview.ledger.error", { error })
          : t("overview.ledger.loading")}
      </p>
    );
  return <MemoryLedgerView data={data} compact={compact} />;
}
