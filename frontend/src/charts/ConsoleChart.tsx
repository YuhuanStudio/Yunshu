import { MetricChart, type MetricChartProps } from "@yuhuanowo/yunui/charts";
import { t, useLocale } from "../i18n/index.ts";

/**
 * The console's time-series chart: the YunUI MetricChart with this page's wording. Loaded lazily (see
 * AnalyticsPanels), so the chart code stays out of every route's first load.
 */
export default function ConsoleChart(props: Omit<MetricChartProps, "labels">) {
  useLocale();
  return (
    <MetricChart
      {...props}
      labels={{
        legend: t("overview.chart.legend"),
        reset: t("overview.chart.reset"),
        expand: t("overview.chart.expand"),
        smooth: t("overview.chart.smooth"),
        logScale: t("overview.chart.logScale"),
        linearScale: t("overview.chart.linearScale"),
        zoomHint: t("overview.chart.zoomHint"),
        empty: t("overview.chart.empty"),
        collecting: (have, need) =>
          t("overview.chart.collectingSamples", { have, need }),
        missing: t("overview.chart.missing"),
        raw: t("overview.chart.raw"),
        noData: t("overview.chart.noDataHere"),
        keyboardHint: t("overview.chart.keyboardHint"),
        isolateHint: t("overview.chart.isolateHint"),
        zoomed: t("overview.chart.zoomed"),
      }}
    />
  );
}
