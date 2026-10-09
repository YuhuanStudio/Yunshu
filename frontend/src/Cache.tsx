import { useEffect, useMemo, useState } from "react";
import { t } from "./i18n/index.ts";
import {
  Button,
  Card,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  EmptyState,
  Progress,
  Table,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
} from "@yuhuanowo/yunui";
import {
  DashboardPage,
  PageHeader,
  StatCard,
  StatGrid,
} from "@yuhuanowo/yunui/patterns";
import { Database, Gauge, Layers, MemoryStick, Trash2 } from "lucide-react";
import type { Connection } from "./api";
import {
  clearableTier,
  clearCache,
  hitRate,
  type CacheEntry,
  type CacheTier,
  type ModelCache,
} from "./admin-cache-api";
import { useCache } from "./admin-hooks";
import { bytesText } from "./byte-format";
import { ByteValue } from "./ByteValue";
import { Reasoned } from "./Reasoned";
import { useMemoryLedger } from "./memory-api";
import { useRouteAction } from "./useRouteAction";
import { CacheLifecyclePanel } from "./CacheLifecyclePanel";
import { number, percent, relative } from "./i18n/format";
import { SectionCard, UnavailableNotice, modelLabel, type Engine } from "./ui";

export function tierLabel(name: string): string {
  if (name === "ram") return t("cache.tier.ram");
  if (name === "warm") return t("cache.tier.warm");
  const m = /^ssd(\d*)$/.exec(name);
  if (m) return m[1] ? t("cache.tier.ssdN", { n: m[1] }) : t("cache.tier.ssd");
  return name;
}
const tierHelp = (name: string) =>
  name === "ram"
    ? t("cache.tier.ramHelp")
    : name === "warm"
      ? t("cache.tier.warmHelp")
      : t("cache.tier.ssdHelp");

const SHOWN = 15;
type ClearTarget = { model: string; tier: CacheTier };

function EntriesTable({
  entries,
  truncated,
}: {
  entries: CacheEntry[];
  truncated: boolean;
}) {
  const [all, setAll] = useState(false);
  const sorted = useMemo(
    () => [...entries].sort((a, b) => (b.bytes ?? 0) - (a.bytes ?? 0)),
    [entries],
  );
  const shown = all ? sorted : sorted.slice(0, SHOWN);
  if (!entries.length)
    return (
      <p className="text-sm text-muted-foreground">{t("cache.entries.none")}</p>
    );
  return (
    <div className="space-y-2">
      <Table scrollLabel={t("cache.entries.title")} className="min-w-[560px]">
        <Thead>
          <Tr>
            <Th>{t("cache.entries.key")}</Th>
            <Th>{t("cache.entries.tier")}</Th>
            <Th className="text-right">{t("cache.entries.tokens")}</Th>
            <Th className="text-right">{t("cache.entries.bytes")}</Th>
            <Th className="text-right">{t("cache.entries.hits")}</Th>
            <Th className="text-right">{t("cache.entries.lastHit")}</Th>
          </Tr>
        </Thead>
        <Tbody>
          {shown.map((e) => (
            <Tr key={`${e.tier}:${e.key}`} data-testid="cache-entry">
              <Td className="font-mono text-xs">{e.key}</Td>
              <Td>{tierLabel(e.tier)}</Td>
              <Td className="text-right tabular-nums">{number(e.tokens, 0)}</Td>
              <Td className="text-right">
                <ByteValue bytes={e.bytes} />
              </Td>
              <Td className="text-right tabular-nums">{number(e.hits, 0)}</Td>
              <Td className="text-right tabular-nums text-muted-foreground">
                {e.lastHitAgeS == null ? "—" : relative(e.lastHitAgeS)}
              </Td>
            </Tr>
          ))}
        </Tbody>
      </Table>
      <p className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <span>
          {t("cache.entries.shown", {
            shown: shown.length,
            total: entries.length,
          })}
          {truncated && ` ${t("cache.entries.truncated")}`}
        </span>
        {sorted.length > SHOWN && (
          <Button size="sm" variant="ghost" onClick={() => setAll((v) => !v)}>
            {all ? t("cache.entries.less") : t("cache.entries.more")}
          </Button>
        )}
        <span>{t("cache.entries.noText")}</span>
      </p>
    </div>
  );
}

function ModelCacheCard({
  cache,
  canClear,
  clearing,
  onClear,
}: {
  cache: ModelCache;
  canClear: boolean;
  clearing: boolean;
  onClear: (target: ClearTarget) => void;
}) {
  const rate = hitRate(cache);
  return (
    <SectionCard
      icon={Database}
      title={modelLabel(cache.model)}
      description={cache.model}
      data-testid="cache-model"
    >
      {cache.error ? (
        <p className="text-sm text-muted-foreground">{t("cache.modelError")}</p>
      ) : (
        <div className="space-y-6">
          <div>
            <Table scrollLabel={t("cache.tiers.title")}>
              <Thead>
                <Tr>
                  <Th>{t("cache.tiers.tier")}</Th>
                  <Th className="md:w-56">{t("cache.tiers.usage")}</Th>
                  <Th className="hidden text-right md:table-cell">
                    {t("cache.tiers.entries")}
                  </Th>
                  <Th className="hidden text-right md:table-cell">
                    {t("cache.tiers.hits")}
                  </Th>
                  <Th className="w-24" />
                </Tr>
              </Thead>
              <Tbody>
                {cache.tiers.map((tier) => {
                  const pct =
                    tier.capBytes && tier.capBytes > 0 && tier.usedBytes != null
                      ? Math.min(100, (tier.usedBytes / tier.capBytes) * 100)
                      : null;
                  const addressable = clearableTier(tier.name);
                  const reason = !canClear
                    ? t("cache.clear.offlineReason")
                    : !addressable
                      ? t("cache.clear.notAddressable")
                      : clearing
                        ? t("cache.clear.busyReason")
                        : (tier.entries ?? 0) === 0
                          ? t("cache.clear.emptyReason")
                          : null;
                  return (
                    <Tr
                      key={tier.name}
                      data-testid="cache-tier"
                      data-tier={tier.name}
                    >
                      <Td>
                        <p className="text-sm">{tierLabel(tier.name)}</p>
                        <p className="text-xs text-muted-foreground">
                          {tierHelp(tier.name)}
                        </p>
                      </Td>
                      <Td>
                        <p className="text-sm">
                          <ByteValue bytes={tier.usedBytes} />
                          {tier.capBytes != null && tier.capBytes > 0 && (
                            <span className="text-xs text-muted-foreground">
                              {" / "}
                              <ByteValue bytes={tier.capBytes} />
                            </span>
                          )}
                        </p>
                        {pct != null && (
                          <Progress
                            className="mt-1.5 h-1"
                            value={pct}
                            label={t("cache.tiers.usageAria", {
                              tier: tierLabel(tier.name),
                            })}
                          />
                        )}
                      </Td>
                      <Td className="hidden text-right tabular-nums md:table-cell">
                        {number(tier.entries, 0)}
                      </Td>
                      <Td className="hidden text-right tabular-nums md:table-cell">
                        {number(tier.hits, 0)}
                      </Td>
                      <Td className="text-right">
                        <Reasoned reason={reason}>
                          <Button
                            size="sm"
                            variant="secondary"
                            disabled={!!reason}
                            onClick={() =>
                              onClear({ model: cache.model, tier })
                            }
                          >
                            <Trash2 size={12} />
                            {t("cache.clear.button")}
                          </Button>
                        </Reasoned>
                      </Td>
                    </Tr>
                  );
                })}
              </Tbody>
            </Table>
          </div>
          <p
            className="flex flex-wrap items-baseline gap-x-6 gap-y-1 text-sm tabular-nums"
            data-testid="cache-lookups"
          >
            <span>
              <span className="text-muted-foreground">
                {t("cache.lookups.hit")}{" "}
              </span>
              {number(cache.hit, 0)}
            </span>
            <span>
              <span className="text-muted-foreground">
                {t("cache.lookups.miss")}{" "}
              </span>
              {number(cache.miss, 0)}
            </span>
            <span>
              <span className="text-muted-foreground">
                {t("cache.lookups.rate")}{" "}
              </span>
              {rate == null ? "—" : percent(rate, 1)}
            </span>
          </p>
          <div className="space-y-2">
            <h3 className="text-sm font-semibold">
              {t("cache.entries.title")}
            </h3>
            <EntriesTable entries={cache.entries} truncated={cache.truncated} />
          </div>
        </div>
      )}
    </SectionCard>
  );
}

export default function Cache({
  connection,
  engine,
}: {
  connection: Connection;
  engine: Engine;
}) {
  const online = engine.status != null;
  const polled = useCache(connection, online);
  const ledger = useMemoryLedger(connection, online);
  const [target, setTarget] = useState<ClearTarget | null>(null),
    [clearing, setClearing] = useState(false),
    [result, setResult] = useState<{
      bytes: number | null;
      tier: string;
    } | null>(null),
    [error, setError] = useState("");
  const data = polled.data;

  // Palette verb `#/cache?action=clear[&model=id]`: open the confirm dialog for the first
  // clearable tier once the cache list has loaded (the intent itself fires only once).
  const [wantClear, setWantClear] = useState<string | null>(null);
  useRouteAction("cache", (action, q) => {
    if (action === "clear") setWantClear(q.get("model") ?? "");
  });
  useEffect(() => {
    if (wantClear == null || !online || !data) return;
    setWantClear(null);
    const pick = data.caches.find(
      (c) =>
        !c.error &&
        (wantClear === "" || c.model === wantClear) &&
        c.tiers.some((x) => clearableTier(x.name)),
    );
    const tier = pick?.tiers.find((x) => clearableTier(x.name));
    if (pick && tier) setTarget({ model: pick.model, tier });
  }, [wantClear, online, data]);

  async function confirm() {
    const tg = target;
    const tier = tg ? clearableTier(tg.tier.name) : null;
    if (!tg || !tier) return;
    setClearing(true);
    setError("");
    setResult(null);
    try {
      const r = await clearCache(connection, { tier, model: tg.model });
      setResult({ bytes: r.freedBytes, tier: tg.tier.name });
      setTarget(null);
      polled.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setTarget(null);
    } finally {
      setClearing(false);
    }
  }

  const caches = data?.caches ?? [];
  const totalBytes = caches.reduce(
    (n, c) => n + c.tiers.reduce((m, x) => m + (x.usedBytes ?? 0), 0),
    0,
  );
  const hits = caches.reduce((n, c) => n + (c.hit ?? 0), 0),
    misses = caches.reduce((n, c) => n + (c.miss ?? 0), 0);
  const apcGb = ledger.data
    ? ledger.data.owners
        .filter((o) => o.kind.startsWith("apc") && o.gb != null)
        .reduce((n, o) => n + (o.gb as number), 0)
    : null;
  const share =
    apcGb != null && ledger.data?.total_gb
      ? apcGb / ledger.data.total_gb
      : null;

  const header = (
    <PageHeader title={t("cache.title")} description={t("cache.description")} />
  );
  if (polled.unsupported)
    return (
      <DashboardPage data-testid="cache">
        {header}
        <UnavailableNotice
          data-testid="cache-unsupported"
          title={t("cache.unsupportedTitle")}
          description={t("cache.unsupportedDescription")}
        />
        <CacheLifecyclePanel connection={connection} enabled={online} />
      </DashboardPage>
    );

  return (
    <DashboardPage data-testid="cache">
      {header}
      {!data ? (
        <Card className="p-2">
          <EmptyState
            size="inline"
            icon={<Database size={22} />}
            title={t("cache.loading")}
          />
        </Card>
      ) : !data.enabled ? (
        <Card className="p-2" data-testid="cache-disabled">
          <EmptyState
            size="inline"
            icon={<Database size={22} />}
            title={t("cache.disabledTitle")}
            description={t("cache.disabledDescription")}
          />
        </Card>
      ) : (
        <>
          <StatGrid data-stat-grid="">
            <StatCard
              icon={Database}
              label={t("cache.stat.total")}
              value={<ByteValue bytes={totalBytes} />}
            />
            <StatCard
              icon={Gauge}
              label={t("cache.stat.hitRate")}
              value={
                hits + misses > 0 ? percent(hits / (hits + misses), 1) : "—"
              }
              subtext={
                hits + misses > 0
                  ? t("cache.stat.lookups", { count: number(hits + misses, 0) })
                  : t("cache.stat.noLookups")
              }
            />
            <StatCard
              icon={MemoryStick}
              label={t("cache.stat.share")}
              value={share == null ? "—" : percent(share, 1)}
              subtext={
                share == null
                  ? t("cache.stat.shareUnknown")
                  : t("cache.stat.shareOf", { gb: number(apcGb, 1) })
              }
            />
            <StatCard
              icon={Layers}
              label={t("cache.stat.models")}
              value={number(caches.length, 0)}
            />
          </StatGrid>
          {result && (
            <p role="status" className="text-sm" data-testid="cache-cleared">
              {t("cache.clear.freed", {
                tier: tierLabel(result.tier),
                size: bytesText(result.bytes),
              })}
            </p>
          )}
          {error && (
            <p role="alert" className="text-sm text-error">
              {t("cache.clear.failed")}
            </p>
          )}
          {caches.map((c) => (
            <ModelCacheCard
              key={c.model}
              cache={c}
              canClear={online}
              clearing={clearing}
              onClear={setTarget}
            />
          ))}
          <CacheLifecyclePanel connection={connection} enabled={online} />
          <p className="text-xs text-muted-foreground">
            <a className="underline underline-offset-2" href="#/models">
              {t("cache.toMemory")}
            </a>
          </p>
        </>
      )}
      <Dialog open={!!target} onOpenChange={(o) => !o && setTarget(null)}>
        <DialogContent closeLabel={t("cache.clear.close")}>
          <DialogTitle>
            {t("cache.clear.title", {
              tier: target ? tierLabel(target.tier.name) : "",
            })}
          </DialogTitle>
          <DialogDescription>
            {t("cache.clear.description", {
              model: target?.model ?? "",
              entries: number(target?.tier.entries, 0),
              size: bytesText(target?.tier.usedBytes),
            })}
          </DialogDescription>
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={() => setTarget(null)}>
              {t("cache.clear.keep")}
            </Button>
            <Button disabled={clearing} onClick={() => void confirm()}>
              {t("cache.clear.confirm")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </DashboardPage>
  );
}
