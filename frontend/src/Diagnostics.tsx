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
import { CodeBlock } from "./LazyCodeBlock";
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
  Download,
  ScrollText,
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
import { downloadBundle } from "./admin-logs-api";
import { detailText } from "./errors.ts";
import { ErrorNote } from "./error-note";
import { SupportBundlePreview } from "./SupportBundlePreview";
import { has, t, tr } from "./i18n/index.ts";
import { list } from "./i18n/format.ts";
import { SectionCard, clock, elapsed, number, type Engine } from "./ui";
const groups = {
  system: [
    { key: "system", path: "/debug/system" },
    { key: "engine", path: "/debug/engine" },
  ],
  requests: [{ key: "requests", path: "/debug/requests" }],
  cache: [
    { key: "kv", path: "/debug/kv-cache" },
    { key: "ssd", path: "/debug/ssd-cache" },
  ],
  decode: [
    { key: "spec", path: "/debug/spec-decode" },
    { key: "perModel", path: "/debug/per-model" },
  ],
  memory: [
    { key: "guard", path: "/debug/memory-guard" },
    { key: "census", path: "/debug/memory-census" },
  ],
} as const;
type Group = keyof typeof groups;
const groupTitle = (key: string) => tr(`diagnostics.group.${key}`);
type Result = {
  key: string;
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
const stateLabel = (state: string) =>
  has(`diagnostics.state.${state}`) ? tr(`diagnostics.state.${state}`) : state;
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
  systemReason?: string | null,
): Health[] {
  const rows: Health[] = [];
  if (!status) {
    rows.push({
      key: "engine",
      name: t("diagnostics.health.engine.name"),
      status: "offline",
      value: t("diagnostics.health.engine.unreadable"),
      hint: t("diagnostics.health.engine.unreachableHint"),
    });
  } else {
    const running = ["running", "ready"].includes(status.state);
    rows.push({
      key: "engine",
      name: t("diagnostics.health.engine.name"),
      status: status.load_error ? "offline" : running ? "online" : "away",
      value: stateLabel(status.state),
      hint:
        status.load_error ??
        t("diagnostics.health.engine.versionHint", {
          version: status.version,
          uptime: elapsed(status.uptime_s),
        }),
    });
    const loaded = status.models.filter((m) => m.loaded).length;
    rows.push({
      key: "models",
      name: t("diagnostics.health.models.name"),
      status: loaded > 0 ? "online" : "neutral",
      value: `${loaded} / ${status.models.length}`,
      hint:
        loaded > 0
          ? t("diagnostics.health.models.hintLoaded")
          : t("diagnostics.health.models.hintNone"),
    });
    const active = status.memory.active_gb,
      total = status.memory.total_gb;
    const ratio = active != null && total ? active / total : undefined;
    rows.push({
      key: "memory",
      name: t("diagnostics.health.memory.name"),
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
          ? t("diagnostics.health.memory.hintHigh")
          : t("diagnostics.health.memory.hintOk"),
    });
    rows.push({
      key: "queue",
      name: t("diagnostics.health.queue.name"),
      status: status.requests.queued > 0 ? "away" : "online",
      value: t("diagnostics.health.queue.value", {
        active: number(status.requests.active, 0),
        queued: number(status.requests.queued, 0),
      }),
      hint:
        status.requests.queued > 0
          ? t("diagnostics.health.queue.hintQueued")
          : t("diagnostics.health.queue.hintNone"),
    });
    const tps =
      status.throughput.live_decode_tps ?? status.throughput.mean_decode_tps;
    rows.push({
      key: "throughput",
      name: t("diagnostics.health.throughput.name"),
      status: tps == null ? "neutral" : "online",
      value: tps == null ? "—" : `${number(tps)} tok/s`,
      hint: t("diagnostics.health.throughput.hint", {
        window: number(status.throughput.window_s, 0),
        count: status.throughput.requests,
      }),
    });
  }
  // A /debug that is switched off is a setting, not a fault: its own card explains it, so no row and no verdict.
  if (systemState !== "disabled")
    rows.push({
      key: "debug",
      name: t("diagnostics.health.debug.name"),
      status: systemState === "ok" ? "online" : "neutral",
      value:
        systemState === "ok"
          ? t("diagnostics.health.debug.ok")
          : systemState === "pending"
            ? t("diagnostics.health.debug.pending")
            : t("diagnostics.health.debug.failed"),
      hint:
        systemState === "error"
          ? t("diagnostics.health.debug.hintError") +
            (systemReason ? ` (${systemReason})` : "")
          : t("diagnostics.health.debug.hintOk"),
    });
  const cpu = metric(at(system, "cpu", "percent"));
  if (cpu != null)
    rows.push({
      key: "cpu",
      name: t("diagnostics.health.cpu.name"),
      status: cpu > 90 ? "away" : "online",
      value: `${number(cpu)}%`,
      bar: { value: cpu, total: 100, tone: cpu > 90 ? "warning" : "accent" },
      hint: t("diagnostics.health.cpu.hint", {
        n: metric(at(system, "cpu", "logical_cores")) ?? 0,
      }),
    });
  return rows;
}
export type Verdict = {
  level: "ok" | "attention" | "abnormal";
  /** The checks behind a non-healthy verdict, worst first. */
  reasons: Health[];
};
/**
 * One verdict over the health rows: any offline row is 異常, any away row 注意, else 健康.
 * A disabled /debug surface is a setting, not a fault, so it never counts.
 */
export function healthVerdict(
  checks: Health[],
  _systemState?: "ok" | "disabled" | "error" | "pending",
): Verdict {
  // An unreadable /debug (a read-only proxy, a missing scope) is not evidence the engine is unhealthy.
  const counted = checks.filter((c) => c.key !== "debug");
  const bad = counted.filter((c) => c.status === "offline"),
    warn = counted.filter((c) => c.status === "away");
  return bad.length
    ? { level: "abnormal", reasons: [...bad, ...warn] }
    : warn.length
      ? { level: "attention", reasons: warn }
      : { level: "ok", reasons: [] };
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
    [systemReason, setSystemReason] = useState<string | null>(null),
    [copied, setCopied] = useState<"idle" | "done" | "failed">("idle"),
    [bundle, setBundle] = useState<{
      state: "idle" | "busy" | "done" | "missing" | "denied" | "failed";
      name?: string;
    }>({ state: "idle" });
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
        setSystemReason(e instanceof ApiError ? `HTTP ${e.status}` : null);
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
            error:
              e instanceof Error ? e.message : t("diagnostics.row.fetchFailed"),
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
  const checks = healthChecks(status, system, systemState, systemReason);
  const verdict = healthVerdict(checks, systemState);
  async function saveBundle() {
    setBundle({ state: "busy" });
    try {
      const name = await downloadBundle(connection);
      setBundle({ state: "done", name });
    } catch (e) {
      const code = e instanceof ApiError ? e.status : undefined;
      setBundle({
        state:
          code === 404 || code === 405
            ? "missing"
            : code === 401 || code === 403
              ? "denied"
              : "failed",
      });
    }
  }
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
        title={t("diagnostics.page.title")}
        description={t("diagnostics.page.description")}
        actions={
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="secondary"
              disabled={loading}
              onClick={() => void copyBundle()}
            >
              {copied === "done" ? <Check size={13} /> : <Copy size={13} />}
              {copied === "done"
                ? t("diagnostics.action.copied")
                : copied === "failed"
                  ? t("diagnostics.action.copyFailed")
                  : t("diagnostics.action.copy")}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              disabled={bundle.state === "busy"}
              onClick={() => void saveBundle()}
            >
              <Download size={13} />
              {bundle.state === "busy"
                ? t("diagnostics.action.bundling")
                : t("diagnostics.action.bundle")}
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
              {loading
                ? t("diagnostics.action.refreshing")
                : t("diagnostics.action.refresh")}
            </Button>
          </div>
        }
      />
      <SupportBundlePreview endpoints={results.map((r) => r.path)} />
      {bundle.state !== "idle" && bundle.state !== "busy" && (
        <p
          role="status"
          data-testid="bundle-note"
          className={`text-xs ${bundle.state === "done" ? "text-muted-foreground" : "text-warning"}`}
        >
          {bundle.state === "done"
            ? t("diagnostics.bundle.done", { name: bundle.name ?? "" })
            : bundle.state === "missing"
              ? t("diagnostics.bundle.missing")
              : bundle.state === "denied"
                ? t("diagnostics.bundle.denied")
                : t("diagnostics.bundle.failed")}
        </p>
      )}
      {/* Without /debug there is one tile at most; that figure is already a health row. */}
      {hasSystem && (
        <StatGrid data-stat-grid="" data-testid="resource-readouts">
          {hasSystem && (
            <StatCard
              compact
              icon={Cpu}
              label="CPU"
              value={`${number(metric(at(system, "cpu", "percent")))}%`}
              subtext={t("diagnostics.stat.cpuCores", {
                n: metric(at(system, "cpu", "logical_cores")) ?? 0,
              })}
            />
          )}
          {hasSystem && (
            <StatCard
              compact
              icon={Activity}
              label={t("diagnostics.stat.unifiedMemory")}
              value={`${number(metric(at(system, "memory", "percent")))}%`}
              subtext={t("diagnostics.stat.memoryUsed", {
                value: number(gb(at(system, "memory", "used_bytes"))),
              })}
            />
          )}
          <StatCard
            compact
            icon={Database}
            label={t("diagnostics.stat.metalActive")}
            value={`${number(status?.memory.active_gb ?? gb(at(system, "gpu", "active_bytes")))} GB`}
            subtext={t("diagnostics.stat.metalPeak", {
              value: number(status?.memory.peak_gb),
            })}
          />
          {engineCounters !== undefined && (
            <StatCard
              compact
              icon={Server}
              label={t("diagnostics.stat.requestsProcessed")}
              value={number(
                metric(at(engineCounters, "requests_processed")),
                0,
              )}
              subtext={t("diagnostics.stat.engineCounter")}
            />
          )}
        </StatGrid>
      )}
      <SectionCard
        icon={HeartPulse}
        title={t("diagnostics.health.title")}
        description={t("diagnostics.health.description")}
        data-testid="health-checks"
        action={
          <div
            className="flex items-center gap-2"
            data-testid="health-verdict"
            data-level={verdict.level}
          >
            <StatusIndicator
              status={
                verdict.level === "ok"
                  ? "online"
                  : verdict.level === "attention"
                    ? "away"
                    : "offline"
              }
            >
              <span
                className={`text-xs font-medium ${verdict.level === "abnormal" ? "text-error" : "text-foreground"}`}
              >
                {t(
                  verdict.level === "ok"
                    ? "diagnostics.verdict.ok"
                    : verdict.level === "attention"
                      ? "diagnostics.verdict.attention"
                      : "diagnostics.verdict.abnormal",
                )}
              </span>
            </StatusIndicator>
            {/* The slot keeps its box while the verdict flips, so polling never moves the header. */}
            <Button
              size="sm"
              variant="ghost"
              asChild
              className={verdict.level === "ok" ? "invisible" : undefined}
            >
              <a
                href="#/logs"
                tabIndex={verdict.level === "ok" ? -1 : undefined}
                aria-hidden={verdict.level === "ok" ? true : undefined}
              >
                <ScrollText size={14} />
                {t("diagnostics.verdict.logs")}
              </a>
            </Button>
          </div>
        }
      >
        <ul className="-my-2">
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
                  label={`${check.name}: ${check.value}`}
                />
              )}
            </li>
          ))}
        </ul>
      </SectionCard>
      {debugOff && (
        <SectionCard
          icon={Server}
          title={t("diagnostics.off.title")}
          description={t("diagnostics.off.description")}
          data-testid="debug-disabled"
        >
          <p className="text-sm text-muted-foreground">
            {t("diagnostics.off.body", {
              flag: "YUNSHU_DEBUG_ROUTES=1",
              authFlag: "YUNSHU_AUTH_DISABLED",
            })}
          </p>
        </SectionCard>
      )}
      {!debugOff && (
        <>
          <SectionRow
            title={t("diagnostics.items.title")}
            action={
              <p className="text-xs text-muted-foreground">
                {updated
                  ? t("diagnostics.items.updated", { time: clock(updated) })
                  : t("diagnostics.items.notLoaded")}
              </p>
            }
          />
          <div>
            <NavTabs
              ariaLabel={t("diagnostics.tabs.aria")}
              activeKey={group}
              onChange={(value) => setGroup(value as Group)}
              tabs={[
                {
                  key: "system",
                  label: (
                    <>
                      <Server size={14} />
                      {t("diagnostics.tabs.system")}
                    </>
                  ),
                },
                {
                  key: "requests",
                  label: (
                    <>
                      <Activity size={14} />
                      {t("diagnostics.tabs.requests")}
                    </>
                  ),
                },
                {
                  key: "cache",
                  label: (
                    <>
                      <Database size={14} />
                      {t("diagnostics.tabs.cache")}
                    </>
                  ),
                },
                {
                  key: "decode",
                  label: (
                    <>
                      <Cpu size={14} />
                      {t("diagnostics.tabs.decode")}
                    </>
                  ),
                },
                {
                  key: "memory",
                  label: (
                    <>
                      <Database size={14} />
                      {t("diagnostics.tabs.memory")}
                    </>
                  ),
                },
              ]}
            />
          </div>
          {loading && (
            <Card className="p-4 text-sm text-muted-foreground" role="status">
              {t("diagnostics.loading", {
                items: list(groups[group].map((item) => groupTitle(item.key))),
              })}
            </Card>
          )}
          {group === "requests" && request !== undefined && (
            <SectionCard
              icon={Timer}
              title={t("diagnostics.latency.title")}
              description={t("diagnostics.latency.description")}
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
              title={groupTitle(row.key)}
              description={<span className="font-mono">{row.path}</span>}
              className="min-w-0"
              action={
                <StatusIndicator
                  className="gap-1.5 text-xs text-muted-foreground"
                  status={row.error ? "away" : "online"}
                >
                  {row.error
                    ? row.status === 404
                      ? t("diagnostics.row.disabled")
                      : row.status === 401 || row.status === 403
                        ? t("diagnostics.row.unauthorized")
                        : t("diagnostics.row.failed")
                    : t("diagnostics.row.ok")}
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
                    <Table scrollLabel={t("diagnostics.cache.scroll")}>
                      <Thead>
                        <Tr>
                          <Th>{t("diagnostics.cache.model")}</Th>
                          <Th>{t("diagnostics.cache.prefixHit")}</Th>
                          <Th>{t("diagnostics.cache.resident")}</Th>
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
                    {expanded[row.key]
                      ? t("diagnostics.raw.hide")
                      : t("diagnostics.raw.show")}
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
