import { useEffect, useState } from "react";
import {
  Button,
  Card,
  EmptyState,
  NavTabs,
  SegmentedBar,
  StatusIndicator,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import { CodeBlock } from "@yuhuanowo/yunui/content";
import {
  DashboardPage,
  DetailList,
  DetailRow,
  PageHeader,
  SectionRow,
  StatCard,
  StatGrid,
} from "@yuhuanowo/yunui/patterns";
import {
  Activity,
  Check,
  Copy,
  Cpu,
  Database,
  HeartPulse,
  Layers,
  MemoryStick,
  RefreshCw,
  Server,
  Timer,
  type LucideIcon,
} from "lucide-react";
import { ApiError, type Connection, type EngineStatus } from "./api";
import { requestServerJson } from "./management-api";
import { detailText } from "./errors.ts";
import { ErrorNote } from "./error-note";
import { SectionCard, clock, elapsed, number, type Engine } from "./ui";
const groups = {
  system: [
    { key: "system", title: "主機資源", path: "/debug/system" },
    { key: "engine", title: "引擎計數", path: "/debug/engine" },
  ],
  requests: [{ key: "requests", title: "請求與延遲", path: "/debug/requests" }],
  cache: [
    { key: "kv", title: "KV 前綴快取", path: "/debug/kv-cache" },
    { key: "ssd", title: "SSD 快取", path: "/debug/ssd-cache" },
  ],
  decode: [
    { key: "spec", title: "推測解碼", path: "/debug/spec-decode" },
    { key: "perModel", title: "逐模型運行情況", path: "/debug/per-model" },
  ],
  memory: [
    { key: "guard", title: "記憶體保護", path: "/debug/memory-guard" },
    { key: "census", title: "配置明細", path: "/debug/memory-census" },
  ],
} as const;
type Group = keyof typeof groups;
type Result = {
  key: string;
  title: string;
  path: string;
  data?: unknown;
  error?: string;
  detail?: string;
  status?: number;
};
function at(value: unknown, ...keys: string[]): unknown {
  let next = value;
  for (const key of keys) {
    if (!next || typeof next !== "object" || !Object.hasOwn(next, key))
      return undefined;
    next = (next as Record<string, unknown>)[key];
  }
  return next;
}
const metric = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;
const gb = (value: unknown) => {
  const n = metric(value);
  return n == null ? undefined : n / 1e9;
};
type Health = {
  key: string;
  name: string;
  status: "online" | "offline" | "away" | "neutral";
  value: string;
  hint: string;
  /** Real proportion behind the row (loaded / registered, active / total, CPU %), when there is one. */
  bar?: {
    value: number;
    total: number;
    tone: "accent" | "warning";
  };
};
const STATE_LABEL: Record<string, string> = {
  running: "運作中",
  ready: "就緒",
  starting: "準備中",
  loading: "載入中",
  error: "錯誤",
};
const groupIcon: Record<Group, LucideIcon> = {
  system: Server,
  requests: Timer,
  cache: Layers,
  decode: Cpu,
  memory: MemoryStick,
};
/** Health checks derived only from /yunshu/status and the /debug/system reply. */
export function healthChecks(
  status: EngineStatus | null,
  system: unknown,
  systemState: "ok" | "disabled" | "error" | "pending",
): Health[] {
  const rows: Health[] = [];
  if (!status) {
    rows.push({
      key: "engine",
      name: "引擎狀態",
      status: "offline",
      value: "無法讀取",
      hint: "/v1/yunshu/status 沒有回應，請確認引擎位址與存取權杖。",
    });
  } else {
    const running = ["running", "ready"].includes(status.state);
    rows.push({
      key: "engine",
      name: "引擎狀態",
      status: status.load_error ? "offline" : running ? "online" : "away",
      value: STATE_LABEL[status.state] ?? status.state,
      hint:
        status.load_error ??
        `版本 ${status.version}，已運行 ${elapsed(status.uptime_s)}`,
    });
    const loaded = status.models.filter((m) => m.loaded).length;
    rows.push({
      key: "models",
      name: "模型載入",
      status: loaded > 0 ? "online" : "neutral",
      value: `${loaded} / ${status.models.length}`,
      hint:
        loaded > 0
          ? "已載入的模型可直接推論。"
          : "目前沒有已載入的模型，首個請求會觸發載入。",
    });
    const active = status.memory.active_gb,
      total = status.memory.total_gb;
    const ratio = active != null && total ? active / total : undefined;
    rows.push({
      key: "memory",
      name: "Metal 記憶體",
      status: ratio == null ? "neutral" : ratio > 0.9 ? "away" : "online",
      bar:
        ratio == null || !total
          ? undefined
          : {
              value: active ?? 0,
              total,
              tone: ratio > 0.9 ? "warning" : "accent",
            },
      value:
        active != null && total
          ? `${number(active)} / ${number(total)} GB`
          : "—",
      hint:
        ratio != null && ratio > 0.9
          ? "Metal 活躍記憶體超過總量的 90%，可能觸發記憶體保護。"
          : "Metal 活躍記憶體佔總量的比例正常。",
    });
    rows.push({
      key: "queue",
      name: "請求排隊",
      status: status.requests.queued > 0 ? "away" : "online",
      value: `${number(status.requests.active, 0)} 進行 · ${number(status.requests.queued, 0)} 排隊`,
      hint:
        status.requests.queued > 0
          ? "有請求在排隊等待前一個請求完成。"
          : "沒有排隊中的請求。",
    });
    const tps =
      status.throughput.live_decode_tps ?? status.throughput.mean_decode_tps;
    rows.push({
      key: "throughput",
      name: "解碼速度",
      status: tps == null ? "neutral" : "online",
      value: tps == null ? "—" : `${number(tps)} tok/s`,
      hint: `近 ${number(status.throughput.window_s, 0)} 秒內 ${number(status.throughput.requests, 0)} 個請求。`,
    });
  }
  rows.push({
    key: "debug",
    name: "診斷介面",
    status:
      systemState === "ok"
        ? "online"
        : systemState === "pending"
          ? "neutral"
          : "offline",
    value:
      systemState === "ok"
        ? "可用"
        : systemState === "pending"
          ? "讀取中"
          : systemState === "disabled"
            ? "未啟用"
            : "讀取失敗",
    hint:
      systemState === "disabled"
        ? "引擎需設定 YUNSHU_DEBUG_ROUTES=1 並重新啟動，才會提供 /debug。"
        : systemState === "error"
          ? "/debug/system 回應錯誤，可能需要有效的存取權杖。"
          : "/debug/system 可讀取。",
  });
  const cpu = metric(at(system, "cpu", "percent"));
  if (cpu != null)
    rows.push({
      key: "cpu",
      name: "主機 CPU",
      status: cpu > 90 ? "away" : "online",
      value: `${number(cpu)}%`,
      bar: { value: cpu, total: 100, tone: cpu > 90 ? "warning" : "accent" },
      hint: `${number(metric(at(system, "cpu", "logical_cores")), 0)} 個邏輯核心。`,
    });
  return rows;
}
export function Diagnostics({
  connection,
  engine,
}: {
  connection: Connection;
  engine: Engine;
}) {
  const status = engine.status;
  const [group, setGroup] = useState<Group>("system"),
    [refresh, setRefresh] = useState(0),
    [results, setResults] = useState<Result[]>([]),
    [loading, setLoading] = useState(false),
    [updated, setUpdated] = useState<number | null>(null),
    [expanded, setExpanded] = useState<Record<string, boolean>>({}),
    [overviewSystem, setOverviewSystem] = useState<unknown>(undefined),
    [systemState, setSystemState] = useState<
      "ok" | "disabled" | "error" | "pending"
    >("pending"),
    [copied, setCopied] = useState<"idle" | "done" | "failed">("idle");
  useEffect(() => {
    const controller = new AbortController();
    setSystemState("pending");
    void requestServerJson(connection, "/debug/system", {
      signal: controller.signal,
    })
      .then((data) => {
        if (controller.signal.aborted) return;
        setOverviewSystem(data);
        setSystemState("ok");
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setOverviewSystem(undefined);
        setSystemState(
          e instanceof ApiError && e.status === 404 ? "disabled" : "error",
        );
      });
    return () => controller.abort();
  }, [connection.baseUrl, connection.token, refresh]);
  useEffect(() => {
    if (systemState === "pending") return;
    if (systemState === "disabled") {
      setLoading(false);
      setResults([]);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setResults([]);
    setExpanded({});
    setUpdated(null);
    void Promise.all(
      groups[group].map(async (endpoint) => {
        try {
          const data = await requestServerJson(connection, endpoint.path, {
            signal: controller.signal,
          });
          return {
            ...endpoint,
            data,
            error:
              typeof at(data, "error") === "string"
                ? String(at(data, "error"))
                : undefined,
          } as Result;
        } catch (e) {
          return {
            ...endpoint,
            error: e instanceof Error ? e.message : "無法取得資料",
            detail: detailText(e),
            status: e instanceof ApiError ? e.status : undefined,
          } as Result;
        }
      }),
    ).then((rows) => {
      if (!controller.signal.aborted) {
        setResults(rows);
        setUpdated(Date.now());
        setLoading(false);
      }
    });
    return () => controller.abort();
  }, [connection.baseUrl, connection.token, group, refresh, systemState]);
  const system = overviewSystem,
    engineCounters = results.find((r) => r.key === "engine")?.data,
    request = results.find((r) => r.key === "requests")?.data;
  const checks = healthChecks(status, system, systemState);
  async function copyBundle() {
    const bundle = JSON.stringify(
      {
        generated_at: new Date().toISOString(),
        service: connection.baseUrl,
        status,
        debug_system: system ?? null,
        group,
        endpoints: results.map(({ key, path, data, error, status: code }) => ({
          key,
          path,
          http_status: code ?? null,
          error: error ?? null,
          data: data ?? null,
        })),
      },
      null,
      2,
    );
    try {
      await navigator.clipboard.writeText(bundle);
      setCopied("done");
    } catch {
      setCopied("failed");
    }
    setTimeout(() => setCopied("idle"), 2500);
  }
  const debugOff = systemState === "disabled";
  const hasSystem = system !== undefined;
  return (
    <DashboardPage data-testid="diagnostics">
      <PageHeader
        title="引擎診斷"
        description="直接讀取服務的資源、請求、快取與解碼狀態。"
        actions={
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="secondary"
              disabled={loading}
              onClick={() => void copyBundle()}
            >
              {copied === "done" ? <Check size={13} /> : <Copy size={13} />}
              {copied === "done"
                ? "已複製"
                : copied === "failed"
                  ? "無法寫入剪貼簿"
                  : "複製診斷資料"}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              disabled={loading}
              onClick={() => {
                void engine.refresh();
                setRefresh((n) => n + 1);
              }}
            >
              <RefreshCw size={13} />
              {loading ? "讀取中" : "重新讀取"}
            </Button>
          </div>
        }
      />
      {/* Without /debug there is one tile at most; that figure is already a health row. */}
      {hasSystem && (
        <StatGrid data-testid="resource-readouts">
          {hasSystem && (
            <StatCard
              compact
              icon={Cpu}
              label="CPU"
              value={`${number(metric(at(system, "cpu", "percent")))}%`}
              subtext={`${number(metric(at(system, "cpu", "logical_cores")), 0)} 個邏輯核心`}
            />
          )}
          {hasSystem && (
            <StatCard
              compact
              icon={Activity}
              label="統一記憶體"
              value={`${number(metric(at(system, "memory", "percent")))}%`}
              subtext={`${number(gb(at(system, "memory", "used_bytes")))} GB 已使用`}
            />
          )}
          <StatCard
            compact
            icon={Database}
            label="Metal 記憶體（活躍）"
            value={`${number(status?.memory.active_gb ?? gb(at(system, "gpu", "active_bytes")))} GB`}
            subtext={`峰值 ${number(status?.memory.peak_gb)} GB`}
          />
          {engineCounters !== undefined && (
            <StatCard
              compact
              icon={Server}
              label="已處理請求"
              value={number(
                metric(at(engineCounters, "requests_processed")),
                0,
              )}
              subtext="引擎計數器"
            />
          )}
        </StatGrid>
      )}
      <SectionCard
        icon={HeartPulse}
        title="健康檢查"
        description="只用 /v1/yunshu/status 與 /debug/system 的實際回報判斷。"
        data-testid="health-checks"
      >
        <ul className="-my-2 divide-y divide-border">
          {checks.map((check) => (
            <li key={check.key} className="py-3">
              <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1">
                <div className="min-w-0">
                  <StatusIndicator status={check.status}>
                    <span className="text-sm font-medium text-foreground">
                      {check.name}
                    </span>
                  </StatusIndicator>
                  <p className="mt-0.5 pl-4 text-xs text-muted-foreground">
                    {check.hint}
                  </p>
                </div>
                <span className="min-w-28 text-right text-sm tabular-nums">
                  {check.value}
                </span>
              </div>
              {check.bar && (
                <SegmentedBar
                  className="mt-2 pl-4"
                  height={6}
                  total={check.bar.total}
                  segments={[{ value: check.bar.value, tone: check.bar.tone }]}
                  label={`${check.name}：${check.value}`}
                />
              )}
            </li>
          ))}
        </ul>
      </SectionCard>
      {debugOff && (
        <SectionCard
          icon={Server}
          title="/debug 診斷介面未啟用"
          description="逐項診斷（主機、請求、快取、解碼、記憶體）需要它。"
          data-testid="debug-disabled"
        >
          <p className="text-sm text-muted-foreground">
            以 <code className="font-mono">YUNSHU_DEBUG_ROUTES=1</code>{" "}
            啟動引擎後重新連線；另需存取權杖，或設定{" "}
            <code className="font-mono">YUNSHU_AUTH_DISABLED</code>
            。控制台不會自行變更引擎設定。
          </p>
        </SectionCard>
      )}
      {!debugOff && (
        <>
          <SectionRow
            title="逐項診斷"
            action={
              <p className="text-xs text-muted-foreground">
                {updated ? `讀取於 ${clock(updated)}` : "尚未取得資料"} ·
                手動更新
              </p>
            }
          />
          <div>
            <NavTabs
              ariaLabel="診斷類別"
              activeKey={group}
              onChange={(value) => setGroup(value as Group)}
              tabs={[
                {
                  key: "system",
                  label: (
                    <>
                      <Server size={14} />
                      主機與引擎
                    </>
                  ),
                },
                {
                  key: "requests",
                  label: (
                    <>
                      <Activity size={14} />
                      請求
                    </>
                  ),
                },
                {
                  key: "cache",
                  label: (
                    <>
                      <Database size={14} />
                      快取
                    </>
                  ),
                },
                {
                  key: "decode",
                  label: (
                    <>
                      <Cpu size={14} />
                      解碼
                    </>
                  ),
                },
                {
                  key: "memory",
                  label: (
                    <>
                      <Database size={14} />
                      記憶體
                    </>
                  ),
                },
              ]}
            />
          </div>
          {loading && (
            <Card className="p-6 text-sm text-muted-foreground" role="status">
              正在讀取 {groups[group].map((item) => item.title).join("、")}…
            </Card>
          )}
          {group === "requests" && request !== undefined && (
            <SectionCard
              icon={Timer}
              title="服務延遲百分位數"
              description="後端近 60 秒 HTTP 請求耗時統計，與首頁的已觀測首 token 延遲分布不同。"
            >
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                {["p50", "p90", "p95", "p99"].map((key) => (
                  <div key={key}>
                    <p className="text-xs text-muted-foreground">{key}</p>
                    <p className="mt-2 text-2xl font-semibold tabular-nums">
                      {number(metric(at(request, "latency_percentiles", key)))}{" "}
                      <span className="text-xs">ms</span>
                    </p>
                  </div>
                ))}
              </div>
            </SectionCard>
          )}
          {results.map((row) => (
            <SectionCard
              key={row.key}
              icon={groupIcon[group]}
              title={row.title}
              description={<span className="font-mono">{row.path}</span>}
              className="min-w-0"
              action={
                <StatusIndicator
                  className="gap-1.5 text-xs text-muted-foreground"
                  status={row.error ? "away" : "online"}
                >
                  {row.error
                    ? row.status === 404
                      ? "未啟用"
                      : row.status === 401 || row.status === 403
                        ? "需要授權"
                        : "讀取失敗"
                    : "已讀取"}
                </StatusIndicator>
              }
            >
              {row.error ? (
                <ErrorNote
                  tone="muted"
                  message={row.error}
                  detail={row.detail}
                  className="text-sm"
                />
              ) : (
                <div className="space-y-4">
                  <DetailList className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
                    {Object.entries(
                      row.data && typeof row.data === "object" ? row.data : {},
                    )
                      .filter(
                        ([, value]) =>
                          typeof value === "number" ||
                          typeof value === "boolean" ||
                          typeof value === "string",
                      )
                      .slice(0, 12)
                      .map(([key, value]) => (
                        <DetailRow
                          key={key}
                          label={key}
                          value={
                            typeof value === "number"
                              ? number(value, 2)
                              : String(value)
                          }
                        />
                      ))}
                  </DetailList>
                  {Array.isArray(at(row.data, "caches")) && (
                    <Table scrollLabel="模型快取診斷">
                      <Thead>
                        <Tr>
                          <Th>模型</Th>
                          <Th>前綴命中</Th>
                          <Th>常駐</Th>
                          <Th>SSD</Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        {(at(row.data, "caches") as unknown[]).map(
                          (cache, index) => (
                            <Tr key={index}>
                              <Td>
                                <span className="block max-w-44 truncate text-xs">
                                  {String(at(cache, "model_id") ?? "—")}
                                </span>
                              </Td>
                              <Td>
                                {number(
                                  metric(at(cache, "apc", "token_hit_rate")) ==
                                    null
                                    ? undefined
                                    : Number(
                                        at(cache, "apc", "token_hit_rate"),
                                      ) * 100,
                                )}
                                %
                              </Td>
                              <Td>
                                {number(
                                  metric(at(cache, "apc", "resident_bytes")) ==
                                    null
                                    ? undefined
                                    : Number(
                                        at(cache, "apc", "resident_bytes"),
                                      ) / 1e6,
                                )}{" "}
                                MB
                              </Td>
                              <Td>
                                {number(
                                  metric(at(cache, "apc", "disk_bytes")) == null
                                    ? undefined
                                    : Number(at(cache, "apc", "disk_bytes")) /
                                        1e6,
                                )}{" "}
                                MB
                              </Td>
                            </Tr>
                          ),
                        )}
                      </Tbody>
                    </Table>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    aria-expanded={!!expanded[row.key]}
                    onClick={() =>
                      setExpanded((current) => ({
                        ...current,
                        [row.key]: !current[row.key],
                      }))
                    }
                  >
                    {expanded[row.key] ? "收合原始資料" : "查看完整診斷資料"}
                  </Button>
                  {expanded[row.key] && (
                    <CodeBlock language="json">
                      {JSON.stringify(row.data, null, 2) ?? "null"}
                    </CodeBlock>
                  )}
                </div>
              )}
            </SectionCard>
          ))}
        </>
      )}
    </DashboardPage>
  );
}
