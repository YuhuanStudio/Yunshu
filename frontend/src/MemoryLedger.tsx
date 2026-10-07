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
import {
  useMemoryLedger,
  type MemoryLedgerData,
  type MemoryOwner,
} from "./memory-api";

/** Capacity tones, as in Strata: warning from 80%, danger above 92% of unified memory. */
export const LEDGER_WARN = 0.8,
  LEDGER_DANGER = 0.92;

const KIND_LABEL: Record<string, string> = {
  weights: "模型權重",
  apc_ram: "前綴快取（記憶體）",
  apc_warm: "前綴快取（暖層）",
  live_kv: "進行中請求的 KV",
  mlx_cache: "記憶體保留池",
  other: "其他",
};
const KIND_TONE: Record<string, SegmentTone> = {
  weights: "accent",
  apc_ram: "info",
  apc_warm: "info",
  live_kv: "success",
  mlx_cache: "neutral",
  other: "warning",
};

const gbText = (v: number | null, digits = 1) =>
  v == null ? "未知" : `${v.toFixed(digits)} GB`;

export function ownerLabel(o: MemoryOwner) {
  const base = KIND_LABEL[o.kind] ?? o.kind;
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
      label: `${ownerLabel(o)}${o.estimated ? "（估算）" : ""}`,
    }));
  const marks: BarMark[] = [];
  if (d.mlx.peak_gb != null)
    marks.push({
      value: d.mlx.peak_gb,
      label: `自啟動峰值 ${gbText(d.mlx.peak_gb)}`,
    });
  if (d.mlx.recommended_working_set_gb != null)
    marks.push({
      value: d.mlx.recommended_working_set_gb,
      tone: "warning",
      label: `Metal 建議上限 ${gbText(d.mlx.recommended_working_set_gb)}`,
    });
  if (d.host.wired_limit_gb != null)
    marks.push({
      value: d.host.wired_limit_gb,
      tone: "error",
      label: `Wired 上限 ${gbText(d.host.wired_limit_gb)}`,
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
  const usage = ledgerUsage(data),
    tone = ledgerTone(usage);
  const { segments, marks } = ledgerBar(data);
  const unknownOwners = data.owners.filter((o) => o.gb == null);
  return (
    <div className="space-y-4" data-testid="memory-ledger">
      <div className="flex flex-wrap items-center gap-5">
        {usage == null ? (
          <p className="text-sm text-muted-foreground">Metal 記憶體：未知</p>
        ) : (
          <Gauge
            value={usage * 100}
            size={compact ? 72 : 96}
            thickness={compact ? 7 : 8}
            tone={tone}
            label={`${(usage * 100).toFixed(0)}%`}
            ariaLabel={`Metal 記憶體佔統一記憶體 ${(usage * 100).toFixed(0)}%`}
          />
        )}
        <div className="min-w-0 flex-1 basis-56 space-y-1 text-sm tabular-nums">
          <p>
            Metal 記憶體 {gbText(data.mlx.active_gb)}
            <span className="text-muted-foreground">
              {" "}
              / {gbText(data.total_gb, 0)} 統一記憶體
            </span>
            {tone === "error" && (
              <StatusIndicator
                status="offline"
                className="ml-2 gap-1.5 text-xs text-muted-foreground"
              >
                接近上限
              </StatusIndicator>
            )}
          </p>
          <p className="text-xs text-muted-foreground">
            可用 {gbText(data.free_gb)} · 自服務啟動峰值{" "}
            {gbText(data.mlx.peak_gb)}
            {data.host.pressure_level
              ? ` · 系統壓力 ${data.host.pressure_level}`
              : ""}
            {data.host.swap_used_gb != null
              ? ` · swap ${gbText(data.host.swap_used_gb, 2)}`
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
          formatValue={(v) => `${v.toFixed(1)} GB`}
          label="Metal 記憶體依持有者分佈"
        />
      ) : (
        <p className="text-xs text-muted-foreground">
          尚無可顯示的持有者分佈（沒有載入的模型，或服務沒有回報）。
        </p>
      )}
      {data.attribution_overshoot_gb != null && (
        <p className="text-xs text-muted-foreground">
          各持有者加總超過 Metal 活躍記憶體{" "}
          {gbText(data.attribution_overshoot_gb, 2)}
          ，「其他」顯示為 0。
        </p>
      )}
      {!compact && (
        <div className="overflow-x-auto">
          <Table aria-label="記憶體持有者" scrollLabel="記憶體持有者表格">
            <Thead>
              <Tr>
                <Th>持有者</Th>
                <Th className="text-right">大小</Th>
                <Th>可回收</Th>
              </Tr>
            </Thead>
            <Tbody>
              {data.owners.map((o, i) => (
                <Tr key={`${o.kind}-${o.id}-${i}`}>
                  <Td>{ownerLabel(o)}</Td>
                  <Td
                    className="text-right tabular-nums"
                    title={o.source ?? "服務沒有這項的計數"}
                  >
                    {o.gb == null ? (
                      "未知"
                    ) : (
                      <>
                        {o.estimated ? "≈ " : ""}
                        {o.gb.toFixed(2)} GB
                        {o.estimated && (
                          <span className="ml-1 text-xs text-muted-foreground">
                            估算
                          </span>
                        )}
                      </>
                    )}
                  </Td>
                  <Td className="text-muted-foreground">
                    {o.reclaimable ? "可回收" : "否"}
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
          {unknownOwners.length > 0 && (
            <p className="mt-2 text-xs text-muted-foreground">
              「未知」表示服務沒有這項的計數，不代表 0。
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
  const { data, unsupported, error } = useMemoryLedger(connection, enabled);
  if (unsupported)
    return (
      <p
        className="text-sm text-muted-foreground"
        data-testid="memory-ledger-unsupported"
      >
        此引擎版本沒有提供記憶體持有者明細（需要較新的 Yunshu）。
      </p>
    );
  if (!data)
    return (
      <p role="status" className="text-sm text-muted-foreground">
        {error ? `無法讀取記憶體明細：${error}` : "讀取記憶體明細…"}
      </p>
    );
  return <MemoryLedgerView data={data} compact={compact} />;
}
