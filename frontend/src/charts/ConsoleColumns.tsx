import { ColumnChart, type ColumnChartProps } from "@yuhuanowo/yunui/charts";

/** Columns (counts per bucket or per day) from YunUI, loaded lazily with the rest of the chart code. */
export default function ConsoleColumns(props: ColumnChartProps) {
  return <ColumnChart {...props} />;
}
