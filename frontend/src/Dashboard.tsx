import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  AnimatedNumber,
  Badge,
  Button,
  Card,
  EmptyState,
  IconButton,
  Progress,
  ScrollFade,
  SegmentedBar,
  SegmentedSelect,
  Skeleton,
  Sparkline,
  StatusIndicator,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  TimeSeriesChart,
  Tr,
} from "@yuhuanowo/yunui";
import {
  ArrowRight,
  Check,
  Copy,
  Download,
  Gauge,
  HardDrive,
  Link2,
  Pause,
  Play,
  RefreshCw,
  Server,
  Stethoscope,
  Timer,
  Zap,
} from "lucide-react";
import {
  CodeBlock,
  DashboardPage,
  DetailList,
  DetailRow,
  HoverRow,
  PageHeader,
  SectionRow,
  StatCard,
  StatGrid,
} from "@yuhuanowo/yunui/patterns";
import {
  ActivityPanel,
  ChartCard,
  LatencyPanel,
  PhasePanel,
} from "./AnalyticsPanels";
import { observationCsv } from "./analytics";
import { chartRows, windowPoints } from "./series";
import { decodeHeadline, totalsFrom, windowLabel } from "./engineView";
import { SpeedPair, StateStrip, TotalsLine } from "./OverviewParts";
import type { RequestRow } from "./api";
import { buildIntegrations, serviceRoot } from "./integrations";
import {
  clock,
  elapsed,
  fixed,
  modelLabel,
  number,
  phaseDot,
  Slot,
  sizeGb,
  supportsChat,
  type Engine,
} from "./ui";
const chartLabels = {
  emptyLabel: "等待第一筆採樣",
  unavailableLabel: "目前沒有可用的數值",
  missingValueLabel: "未回報",
  hiddenLabel: "序列已隱藏，可點選圖例重新顯示",
  legendLabel: "顯示或隱藏序列",
  minSamples: 5,
  collectingLabel: (have: number, need: number) =>
    `收集採樣中 ${have} / ${need}`,
  keyboardHint: "使用左右方向鍵、Home 或 End 查看採樣",
};
const memorySeries = [
  { key: "active", label: "活躍配置", tone: "accent" as const },
  {
    key: "cache",
    label: "記憶體保留池",
    tone: "neutral" as const,
    dashed: true,
  },
];
const requestSeries = [
  { key: "requests", label: "全部", tone: "accent" as const },
  { key: "queued", label: "排隊", tone: "neutral" as const, dashed: true },
  { key: "prefillRequests", label: "預填", tone: "neutral" as const },
  {
    key: "decodeRequests",
    label: "解碼",
    tone: "accent" as const,
    dashed: true,
  },
];
const rateSeries = {
  decode: [{ key: "decode", label: "解碼 即時合計", tone: "accent" as const }],
  prefill: [
    { key: "prefill", label: "預填 即時合計", tone: "neutral" as const },
  ],
  both: [
    { key: "decode", label: "解碼 即時合計", tone: "accent" as const },
    { key: "prefill", label: "預填 即時合計", tone: "neutral" as const },
  ],
};
const formatNumber = (value: number) => number(value);
const formatCount = (value: number) => number(value, 0);
const phaseText: Record<string, string> = {
  queued: "排隊",
  starting: "準備中",
  prefill: "預填",
  decode: "解碼",
  running: "執行",
};
function RequestLane({ row }: { row: RequestRow }) {
  const phase = String(row.phase);
  const prompt = row.prompt_tokens ?? 0,
    cached = row.cached_tokens ?? 0;
  const progress =
    phase === "prefill" || phase === "starting"
      ? (row.percent ??
        (prompt > 0 && row.processed_tokens != null
          ? (row.processed_tokens / prompt) * 100
          : null))
      : null;
  return (
    <li className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 py-3">
      <div className="flex min-w-0 items-center gap-2.5">
        <StatusIndicator className="shrink-0" status={phaseDot(phase)} />
        <Slot ch={7} className="shrink-0 text-sm font-medium">
          {phaseText[phase] ?? phase}
        </Slot>
        <span
          title={row.request_id}
          className="min-w-0 max-w-[9rem] shrink truncate font-mono text-xs text-muted-foreground"
        >
          {row.request_id}
        </span>
        {row.model && (
          <span
            title={row.model}
            className="hidden min-w-0 flex-1 truncate text-xs text-muted-foreground sm:inline"
          >
            · {modelLabel(row.model)}
          </span>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-4 text-xs tabular-nums text-muted-foreground">
        <Slot ch={9} align="right">
          {phase === "decode"
            ? `${number(row.completion_tokens, 0)} tok`
            : `${number(prompt, 0)} tok`}
        </Slot>
        <Slot ch={12} align="right" className="text-foreground">
          {row.tokens_per_second == null
            ? "—"
            : `${fixed(row.tokens_per_second)} tok/s`}
        </Slot>
        <Slot ch={7} align="right">
          {elapsed(row.elapsed_s)}
        </Slot>
      </div>
      {/* One fixed-height line for either the prefill bar or the cache hint, so a
          phase change never adds or removes a row. */}
      <div className="col-span-2 flex h-4 items-center">
        {progress != null ? (
          <Progress
            className="h-1 w-full"
            value={Math.max(0, Math.min(100, progress))}
            label={`預填 ${number(progress, 0)}%`}
          />
        ) : phase !== "prefill" && cached > 0 && prompt > 0 ? (
          <p className="truncate text-xs text-muted-foreground">
            前綴命中 {number(cached, 0)} / {number(prompt, 0)} tokens
          </p>
        ) : null}
      </div>
    </li>
  );
}

function savedBaseUrl() {
  try {
    return localStorage.getItem("yunshu.console.url") || location.origin;
  } catch {
    return location.origin;
  }
}

function QuickAction({
  icon,
  title,
  caption,
  onClick,
}: {
  icon: ReactNode;
  title: string;
  caption: string;
  onClick: () => void;
}) {
  return (
    <Card className="min-w-0 p-1">
      <HoverRow
        onClick={onClick}
        className="flex w-full items-center gap-3 px-3 py-3 text-left"
      >
        <span className="text-muted-foreground">{icon}</span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium">{title}</span>
          <span className="block truncate text-xs text-muted-foreground">
            {caption}
          </span>
        </span>
        <ArrowRight size={14} className="shrink-0 text-muted-foreground/60" />
      </HoverRow>
    </Card>
  );
}

export function Dashboard({
  engine,
  navigate,
}: {
  engine: Engine;
  navigate: (page: string) => void;
}) {
  const [range, setRange] = useState("15m"),
    [metric, setMetric] = useState("decode"),
    [table, setTable] = useState(false),
    [activeX, setActiveX] = useState<number | null>(null),
    [copied, setCopied] = useState<string | null>(null);
  const status = engine.status,
    online = engine.phase === "online";
  const end = engine.updatedAt ?? Date.now(),
    start = end - (range === "5m" ? 300 : range === "15m" ? 900 : 3600) * 1000;
  // Charts read the slim series (engine history first, then live polls). The
  // window is a binary-search slice and the chart gets at most 300 rows.
  const points = useMemo(
    () => windowPoints(engine.series, start, end),
    [engine.series, start, end],
  );
  const data = useMemo(() => chartRows(points), [points]);
  const heroPoints = useMemo(
    () => windowPoints(engine.series, end - 300_000, end),
    [engine.series, end],
  );
  const throughputSeries = rateSeries[metric as keyof typeof rateSeries];
  useEffect(() => {
    if (activeX !== null && !points.some((point) => point.at === activeX))
      setActiveX(null);
  }, [points, activeX]);
  const last = status?.last;
  const activePoint =
    activeX == null
      ? null
      : (points.find((sample) => sample.at === activeX) ?? null);
  const items = status?.requests.items ?? [];
  const memory = status?.memory;
  const loaded = (status?.models ?? []).filter((m) => m.loaded);
  // Trends compare the later half of this window's samples with the earlier half.
  const observed = useMemo(
    () => engine.finished.filter((r) => r.firstObservedAt >= start),
    [engine.finished, start],
  );
  const totals = useMemo(() => totalsFrom(engine.finished), [engine.finished]);
  // Prefix reuse is weighted over the observed finished requests (cached /
  // prompt tokens); the latest request is a secondary line, as on Requests.
  const promptSum = observed.reduce((n, r) => n + r.prompt_tokens, 0),
    cachedSum = observed.reduce((n, r) => n + r.cached_tokens, 0);
  const cache = promptSum > 0 ? (cachedSum / promptSum) * 100 : null;
  const lastHit =
    last && last.prompt_tokens > 0
      ? (last.cached_tokens / last.prompt_tokens) * 100
      : null;
  const heroData = useMemo(() => chartRows(heroPoints), [heroPoints]);
  const baseUrl = serviceRoot(savedBaseUrl());
  const quickModel = (loaded.find(supportsChat) ?? loaded[0])?.id ?? "";
  const curl =
    buildIntegrations(baseUrl, quickModel).find((i) => i.id === "curl")?.code ??
    "";
  const copy = (id: string, text: string) => {
    void navigator.clipboard?.writeText(text).then(
      () => {
        setCopied(id);
        setTimeout(() => setCopied((c) => (c === id ? null : c)), 2000);
      },
      () => undefined,
    );
  };
  const chooseRange = (next: string) => {
    setRange(next);
    setActiveX(null);
  };
  function exportData() {
    const rows = engine.history.filter((s) => s.at >= start && s.at <= end);
    const url = URL.createObjectURL(
        new Blob([observationCsv(rows)], { type: "text/csv;charset=utf-8" }),
      ),
      a = document.createElement("a");
    a.href = url;
    a.download = `yunshu-observations-${range}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  // First-run onboarding only when nothing was ever configured or the token is
  // refused. A restart (502, refused, timeout) keeps the skeleton and the banner.
  const neverConfigured = (() => {
    try {
      return localStorage.getItem("yunshu.console.url") === null;
    } catch {
      return true;
    }
  })();
  if (
    !status &&
    (engine.phase === "unauthorized" ||
      (engine.phase === "offline" && neverConfigured))
  )
    return (
      <DashboardPage width="7xl" data-testid="overview">
        <PageHeader
          title="引擎總覽"
          description="模型、請求與效能，都從你的本機引擎開始。"
        />
        <Card className="p-4">
          <EmptyState
            size="inline"
            icon={<Server size={22} strokeWidth={1.5} />}
            title="連接本機推理引擎"
            description="填入 Yunshu 服務位址與存取權杖，取得真實模型、資源和請求狀態。"
            action={
              <Button onClick={() => navigate("settings")}>
                開啟連線設定
                <ArrowRight size={14} />
              </Button>
            }
          />
          <ol className="mt-3 grid gap-4 border-t border-border/60 pt-4 sm:grid-cols-3">
            {[
              ["01", "連接服務", "使用現有的 Yunshu HTTP 服務"],
              ["02", "載入模型", "管理權重與閒置保留時間"],
              ["03", "觀察與驗證", "查看圖表、請求與推理結果"],
            ].map(([step, title, description]) => (
              <li key={step}>
                <span className="font-mono text-xs text-muted-foreground">
                  {step}
                </span>
                <h2 className="yunui-section-title mt-2 text-base font-semibold">
                  {title}
                </h2>
                <p className="mt-1 text-caption">{description}</p>
              </li>
            ))}
          </ol>
        </Card>
        <SectionRow title="連線後可直接呼叫" />
        <CodeBlock code={curl} language="bash" filename={baseUrl} />
      </DashboardPage>
    );
  return (
    <DashboardPage width="7xl" data-testid="overview">
      <PageHeader
        title="引擎總覽"
        description="觀察這台 Mac 如何處理每一次推理。"
        actions={
          <div className="flex items-center gap-1.5">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => engine.setPolling(!engine.polling)}
            >
              {engine.polling ? <Pause size={13} /> : <Play size={13} />}
              {engine.polling ? "暫停更新" : "恢復更新"}
            </Button>
            <IconButton
              icon={<RefreshCw size={14} />}
              label="更新"
              onClick={() => void engine.refresh()}
            />
            <Button
              size="sm"
              disabled={
                !online ||
                !status?.models.some(
                  (model) => model.loaded && supportsChat(model),
                )
              }
              onClick={() => navigate("playground")}
            >
              <Play size={13} />
              開始測試
            </Button>
          </div>
        }
      />
      <p className="-mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
        <StatusIndicator status={online ? "online" : "offline"}>
          <span className="text-foreground">
            {online
              ? status?.state === "running"
                ? "運行中"
                : (status?.state ?? "已連線")
              : "未連線"}
          </span>
        </StatusIndicator>
        <span>Yunshu {status?.version ?? "—"}</span>
        <Slot ch={13}>· 運行 {elapsed(status?.uptime_s)}</Slot>
        <Slot ch={16}>
          ·{" "}
          {engine.updatedAt
            ? `更新於 ${clock(engine.updatedAt)}`
            : "等待服務回應"}
        </Slot>
        <Slot ch={6}>{!engine.polling ? "（已暫停）" : ""}</Slot>
        {status?.load_error && (
          <span className="text-error">{status.load_error}</span>
        )}
      </p>

      <StateStrip status={status ?? null} />

      {!status ? (
        <StatGrid>
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-[104px] w-full rounded-lg" />
          ))}
        </StatGrid>
      ) : (
        <div
          className="grid gap-3 sm:grid-cols-2 xl:grid-cols-[minmax(0,2fr)_repeat(3,minmax(0,1fr))]"
          data-testid="overview-stats"
        >
          <SpeedPair status={status} points={heroPoints} />
          <StatCard
            compact
            valueFirst
            icon={Timer}
            label="首 token 延遲 (TTFT)"
            value={
              last?.ttft_ms == null ? "—" : `${number(last.ttft_ms, 0)} ms`
            }
            subtext={
              last ? `最近一筆 · ${clock(last.t * 1000)}` : "尚無完成請求"
            }
          />
          <StatCard
            compact
            valueFirst
            icon={Gauge}
            label="前綴命中率"
            value={cache == null ? "—" : `${number(cache, 0)}%`}
            subtext={
              cache == null
                ? "尚無完成請求"
                : `本頁觀測 ${number(observed.length, 0)} 筆加權${lastHit == null ? "" : ` · 最近一筆 ${number(lastHit, 0)}%`}`
            }
          />
          <StatCard
            compact
            valueFirst
            icon={HardDrive}
            label="Metal 記憶體"
            value={`${number(memory?.active_gb)} GB`}
            subtext={`實體 ${number(memory?.total_gb)} GB · 峰值 ${number(memory?.peak_gb)} GB`}
          />
        </div>
      )}

      {/* Live: what the engine is doing right now. */}
      <Card
        className="grid min-w-0 overflow-hidden p-0 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]"
        data-testid="live-panel"
      >
        <div className="flex min-w-0 flex-col justify-between gap-5 border-b border-border/60 p-5 sm:p-6 lg:border-b-0 lg:border-r">
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground">
              即時合計速度 · 近 5 分鐘走勢 · tok/s
            </p>
            <TimeSeriesChart
              {...chartLabels}
              className="mt-3"
              data={heroData}
              series={rateSeries.both}
              height={150}
              ariaLabel="近 5 分鐘解碼與預填的即時合計速度，單位 tok/s"
              formatX={clock}
              formatY={formatNumber}
              maxGap={12000}
            />
          </div>
        </div>
        <div className="flex min-w-0 flex-col p-5 sm:p-6">
          <div className="flex items-center justify-between gap-3">
            <h2 className="yunui-section-title text-base font-semibold">
              進行中請求
              <span className="ml-2 text-muted-foreground tabular-nums">
                {number(status?.requests.active, 0)}
              </span>
            </h2>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              <Slot ch={14} align="right">
                近 {number(status?.throughput.window_s ?? 60, 0)} 秒結束{" "}
                {number(status?.throughput.requests, 0)} 筆
              </Slot>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => navigate("requests")}
              >
                全部
                <ArrowRight size={13} />
              </Button>
            </div>
          </div>
          {items.length ? (
            <ul className="mt-2 divide-y divide-border/60">
              {items.slice(0, 5).map((row) => (
                <RequestLane key={row.request_id} row={row} />
              ))}
            </ul>
          ) : (
            <EmptyState
              size="inline"
              className="flex-1"
              title="目前閒置"
              description={`沒有正在處理的請求。${
                memory?.cache_gb
                  ? ` 記憶體保留池 ${number(memory.cache_gb)} GB 會在閒置後歸還系統。`
                  : ""
              }`}
            />
          )}
          <TotalsLine totals={totals} />
        </div>
      </Card>

      <SectionRow title="快速開始" />
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <QuickAction
          icon={<Play size={18} strokeWidth={1.5} />}
          title="測試一段提示"
          caption="在推理測試送出請求"
          onClick={() => navigate("playground")}
        />
        <QuickAction
          icon={<HardDrive size={18} strokeWidth={1.5} />}
          title="載入模型"
          caption="管理權重與保留時間"
          onClick={() => navigate("models")}
        />
        <QuickAction
          icon={
            copied === "url" ? (
              <Check size={18} strokeWidth={1.5} />
            ) : (
              <Link2 size={18} strokeWidth={1.5} />
            )
          }
          title={copied === "url" ? "已複製" : "複製 API 網址"}
          caption={`${baseUrl}/v1`}
          onClick={() => copy("url", `${baseUrl}/v1`)}
        />
        <QuickAction
          icon={<Stethoscope size={18} strokeWidth={1.5} />}
          title="開啟診斷"
          caption="檢查服務與環境"
          onClick={() => navigate("diagnostics")}
        />
      </div>
      <CodeBlock code={curl} language="bash" filename="curl" />

      <SectionRow
        title="效能觀測"
        action={
          <div className="flex flex-wrap items-center gap-2">
            <SegmentedSelect
              aria-label="觀測時間範圍"
              value={range}
              onChange={chooseRange}
              options={[
                { value: "5m", label: "5 分鐘" },
                { value: "15m", label: "15 分鐘" },
                { value: "1h", label: "1 小時" },
              ]}
            />
            <Button
              size="sm"
              variant="ghost"
              disabled={!points.length}
              onClick={exportData}
            >
              <Download size={13} />
              匯出觀測
            </Button>
          </div>
        }
      />
      <p className="-mt-3 text-xs text-muted-foreground">
        {engine.historyFrom != null
          ? `含引擎端歷史（自 ${clock(engine.historyFrom)}）`
          : "本頁開啟後採樣（此引擎沒有提供歷史）"}{" "}
        · {points.length} 筆 · 中斷期間不補資料
      </p>
      <div className="grid gap-5 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <ChartCard
          data-testid="throughput-panel"
          title="吞吐觀測"
          description="解碼與預填的即時合計速度 · 沒有請求時留白 · tok/s"
          action={
            <SegmentedSelect
              aria-label="吞吐指標"
              value={metric}
              onChange={setMetric}
              options={[
                { value: "decode", label: "解碼" },
                { value: "prefill", label: "預填" },
                { value: "both", label: "比較" },
              ]}
            />
          }
        >
          <TimeSeriesChart
            {...chartLabels}
            data={data}
            series={throughputSeries}
            height={220}
            ariaLabel="吞吐速度時序圖，單位 tok/s"
            formatX={clock}
            formatY={formatNumber}
            maxGap={12000}
            activeX={activeX}
            onActiveXChange={setActiveX}
          />
          <Button
            className="mt-2"
            size="sm"
            variant="ghost"
            aria-expanded={table}
            onClick={() => setTable(!table)}
          >
            {table ? "收合數值" : "檢視採樣數值"}
          </Button>
          {table && (
            <ScrollFade className="mt-3 max-h-60 overflow-auto">
              <Table scrollLabel="吞吐採樣資料">
                <Thead>
                  <Tr>
                    <Th>時間</Th>
                    <Th>解碼</Th>
                    <Th>預填</Th>
                    <Th>Metal GB</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {points.slice(-200).map((p) => (
                    <Tr key={p.at}>
                      <Td>{clock(p.at)}</Td>
                      <Td>{number(p.decode)}</Td>
                      <Td>{number(p.prefill)}</Td>
                      <Td>{number(p.memActive)}</Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </ScrollFade>
          )}
        </ChartCard>
        <ChartCard
          data-testid="memory-panel"
          title="Metal 記憶體"
          action={
            <Badge variant="outline">實體 {number(memory?.total_gb)} GB</Badge>
          }
        >
          <p className="text-2xl font-semibold tabular-nums">
            <Slot ch={5}>{fixed(memory?.active_gb)}</Slot>
            <span className="ml-1.5 text-sm font-normal text-muted-foreground">
              / {fixed(memory?.total_gb)} GB 活躍配置
            </span>
          </p>
          <SegmentedBar
            className="mt-4"
            height={10}
            total={memory?.total_gb ?? undefined}
            label={`Metal 記憶體：活躍 ${number(memory?.active_gb)} GB，快取 ${number(memory?.cache_gb)} GB`}
            segments={[
              {
                value: memory?.active_gb ?? 0,
                tone: "accent",
                label: "活躍配置",
              },
              {
                value: memory?.cache_gb ?? 0,
                tone: "neutral",
                label: "記憶體保留池",
              },
            ]}
            marks={
              memory?.peak_gb
                ? [{ value: memory.peak_gb, label: "本次峰值" }]
                : undefined
            }
            legend
            formatValue={(v) => `${number(v)} GB`}
          />
          <DetailList className="mt-4 border-t border-border/60 pt-4">
            <DetailRow
              label="本次峰值"
              value={`${number(memory?.peak_gb)} GB`}
            />
            <DetailRow
              label="已載入權重"
              value={
                loaded.length
                  ? [
                      loaded.reduce((s, m) => s + (m.size_gb ?? 0), 0) > 0
                        ? sizeGb(
                            loaded.reduce((s, m) => s + (m.size_gb ?? 0), 0),
                          )
                        : null,
                      `${loaded.length} 個`,
                    ]
                      .filter(Boolean)
                      .join(" · ")
                  : "—"
              }
            />
          </DetailList>
          <TimeSeriesChart
            {...chartLabels}
            className="mt-4"
            data={data}
            series={memorySeries}
            height={120}
            ariaLabel="Metal 記憶體時序圖，單位 GB"
            formatX={clock}
            formatY={formatNumber}
            maxGap={12000}
            activeX={activeX}
            onActiveXChange={setActiveX}
          />
        </ChartCard>
      </div>

      {activePoint && (
        <div
          className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border/60 px-4 py-2 text-xs text-muted-foreground"
          role="status"
        >
          <span>
            選取 {clock(activePoint.at)} · {number(activePoint.active, 0)}{" "}
            個活動請求 · {number(activePoint.memActive)} GB
          </span>
          <Button size="sm" variant="ghost" onClick={() => setActiveX(null)}>
            清除選取
          </Button>
        </div>
      )}
      <div className="grid min-w-0 gap-5 xl:grid-cols-2">
        <LatencyPanel records={observed} />
        <PhasePanel engine={engine} navigate={navigate} />
      </div>
      <ActivityPanel
        history={points}
        start={start}
        end={end}
        onSelectTime={setActiveX}
      />
      <ChartCard
        title="請求並行趨勢"
        description="不同處理階段，共用請求數刻度"
      >
        <TimeSeriesChart
          {...chartLabels}
          data={data}
          series={requestSeries}
          height={180}
          ariaLabel="請求階段並行數時序圖"
          formatX={clock}
          formatY={formatCount}
          maxGap={12000}
          activeX={activeX}
          onActiveXChange={setActiveX}
        />
      </ChartCard>
      <SectionRow
        title="模型"
        action={
          <Button size="sm" variant="ghost" onClick={() => navigate("models")}>
            模型庫
            <ArrowRight size={13} />
          </Button>
        }
      />
      <Card className="min-w-0 p-0">
        <ul className="divide-y divide-border/60 py-1">
          {(status?.models ?? []).slice(0, 6).map((model) => (
            <li key={model.id} className="px-3 py-0.5 sm:px-4">
              <HoverRow
                onClick={() => navigate("models")}
                aria-label={`${modelLabel(model.id)}，開啟模型庫`}
                className="flex items-center gap-3 px-2 py-2.5"
              >
                <StatusIndicator
                  status={
                    model.loading ? "away" : model.loaded ? "online" : "neutral"
                  }
                />
                <span className="min-w-0 flex-1 truncate text-sm font-medium">
                  {modelLabel(model.id)}
                </span>
                <span className="hidden text-xs text-muted-foreground sm:inline">
                  {model.type}
                </span>
                <span className="w-20 text-right text-xs tabular-nums text-muted-foreground">
                  {sizeGb(model.size_gb)}
                </span>
                <span className="w-24 text-right text-xs text-muted-foreground">
                  {model.loading
                    ? "載入中"
                    : model.loaded
                      ? model.expires_in_s != null
                        ? `${elapsed(model.expires_in_s)} 後卸載`
                        : model.pinned
                          ? "固定保留"
                          : "已載入"
                      : "未載入"}
                </span>
              </HoverRow>
            </li>
          ))}
          {!status?.models.length && (
            <li className="px-5 py-5 text-sm text-muted-foreground sm:px-6">
              {status ? "服務尚未註冊模型" : "連線後顯示可用模型"}
            </li>
          )}
        </ul>
      </Card>
    </DashboardPage>
  );
}
