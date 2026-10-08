import { useMemo, useState } from "react";
import {
  BarChart,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import { ChartCard } from "./AnalyticsPanels";
import { MIN_PERCENTILE_SAMPLES } from "./analytics";
import {
  distribution,
  histogram,
  type Dimension,
  type Metric,
} from "./latency-distribution";
import { t, tr, useLocale } from "./i18n/index.ts";
import { formatMs } from "./RequestTimeline";
import type { Row } from "./RequestTrace";
import { SegmentedTray } from "./SegmentedTray";
import { number } from "./ui";

// i18n-keys: requests.dist.group., requests.dist.dim.
const k = (ms: number) => (ms >= 1000 ? `${ms / 1000}s` : String(ms));
/** Bucket labels in the unit-free style of the overview histogram: "<100", "100–250", "≥10s". */
const binLabel = (min: number, max: number) =>
  min === 0
    ? `<${k(max)}`
    : max === Infinity
      ? `≥${k(min)}`
      : `${k(min)}–${k(max)}`;

const groupLabel = (id: string) =>
  id === "all"
    ? t("requests.dist.group.all")
    : id === "warm"
      ? t("requests.dist.group.warm")
      : id === "cold"
        ? t("requests.dist.group.cold")
        : id.startsWith("ctx")
          ? tr(`requests.dist.group.${id}`)
          : id.replace(/^model:/, "");

/**
 * TTFT or queue-wait distribution of the engine's recent finished requests, split by cache warmth,
 * context size or model. A percentile appears only for a group with at least 20 samples; smaller
 * groups show their count so nobody reads a P90 off five requests.
 */
export function LatencyDistribution({ rows }: { rows: readonly Row[] }) {
  useLocale();
  const [metric, setMetric] = useState<Metric>("ttft");
  const [dim, setDim] = useState<Dimension>("warmth");
  const dist = useMemo(
    () => distribution(rows, dim, metric),
    [rows, dim, metric],
  );
  const bins = useMemo(() => histogram(rows, metric), [rows, metric]);
  const total = bins.reduce((n, b) => n + b.count, 0);
  const ms = (v: number | null) => (v == null ? "—" : formatMs(v));
  if (total === 0) return null;
  return (
    <ChartCard
      data-testid="latency-distribution"
      title={t("requests.dist.title")}
      description={t("requests.dist.desc", { n: number(total, 0) })}
      action={
        <div className="flex flex-wrap items-center gap-2">
          <SegmentedTray
            aria-label={t("requests.dist.metric")}
            value={metric}
            onChange={setMetric}
            options={[
              { value: "ttft", label: t("requests.dist.ttft") },
              { value: "queue", label: t("requests.dist.queue") },
            ]}
          />
          <Select value={dim} onValueChange={(v) => setDim(v as Dimension)}>
            <SelectTrigger
              aria-label={t("requests.dist.dimension")}
              className="w-36"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(["all", "warmth", "context", "model"] as const).map((d) => (
                <SelectItem key={d} value={d}>
                  {tr(`requests.dist.dim.${d}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      }
    >
      <BarChart
        data={bins.map((b) => ({
          id: `b${b.min}`,
          label: binLabel(b.min, b.max),
          value: b.count,
          tone: "neutral" as const,
        }))}
        height={150}
        ariaLabel={t("requests.dist.aria")}
        formatValue={(v) => number(v, 0)}
      />
      <Table scrollLabel={t("requests.dist.tableAria")} className="mt-4">
        <Thead>
          <Tr>
            <Th>{t("requests.dist.col.group")}</Th>
            <Th>{t("requests.dist.col.n")}</Th>
            <Th>P50</Th>
            <Th>P90</Th>
          </Tr>
        </Thead>
        <Tbody>
          {dist.groups.map((g) => (
            <Tr key={g.id} data-group={g.id}>
              <Td
                className="max-w-32 truncate text-sm"
                title={groupLabel(g.id)}
              >
                {groupLabel(g.id)}
              </Td>
              <Td className="tabular-nums">{number(g.n, 0)}</Td>
              <Td
                className="whitespace-nowrap tabular-nums"
                title={
                  g.p50 == null
                    ? t("requests.dist.tooFew", { min: MIN_PERCENTILE_SAMPLES })
                    : undefined
                }
              >
                {ms(g.p50)}
              </Td>
              <Td
                className="whitespace-nowrap tabular-nums"
                title={
                  g.p90 == null
                    ? t("requests.dist.tooFew", { min: MIN_PERCENTILE_SAMPLES })
                    : undefined
                }
              >
                {ms(g.p90)}
              </Td>
            </Tr>
          ))}
        </Tbody>
      </Table>
      <p className="mt-3 text-xs leading-5 text-muted-foreground">
        {t("requests.dist.note", { min: MIN_PERCENTILE_SAMPLES })}
        {dist.unplaced > 0 &&
          ` ${t("requests.dist.unplaced", { n: number(dist.unplaced, 0) })}`}
      </p>
    </ChartCard>
  );
}
