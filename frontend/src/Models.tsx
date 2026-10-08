import { t, tr } from "./i18n/index.ts";
import { useEffect, useRef, useState } from "react";
import {
  Button,
  Card,
  Dialog,
  DialogContent,
  DialogFooter,
  DialogTitle,
  DialogDescription,
  EmptyState,
  Progress,
  SearchInput,
  StatusIndicator,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
} from "@yuhuanowo/yunui";
import {
  DashboardPage,
  PageHeader,
  SectionRow,
} from "@yuhuanowo/yunui/patterns";
import { CodeBlock } from "@yuhuanowo/yunui/code";
import { IDBadge, ModelCard, isKnownCapability } from "@yuhuanowo/yunui/ai";
import { Box, Download, LayoutGrid, RefreshCw, Table2 } from "lucide-react";
import { getModel, loadModel, unloadModel, type Connection } from "./api";
import { useRouteAction } from "./useRouteAction";
import {
  elapsed,
  fixed,
  LocalModelIcon,
  isOnline,
  modelLabel,
  number,
  useStoredChoice,
  type Engine,
  type Model,
} from "./ui";
import { ModelSize } from "./ModelSize";
import { useMemoryLedger } from "./memory-api";
import { useDownloads, useLocalInventory } from "./admin-hooks";
import { isActive } from "./admin-models-api";
import { LocalInventory } from "./LocalInventory";
import { LoadingElapsed } from "./loading-clock";
import { ModelManagement } from "./ModelManagement";
import { ModelActions, type Perform } from "./ModelActions";
import { ModelDetail, modelState, retention } from "./ModelDetail";
import { SegmentedTray } from "./SegmentedTray";
export type { Perform } from "./ModelActions";
export function Models({
  engine,
  connection,
  perform,
  busy,
  test,
  selected,
  open,
}: {
  engine: Engine;
  connection: Connection;
  perform: Perform;
  busy: string | null;
  test: (id: string) => void;
  /** Model id from the route (#/models/<id>), or null for the list. */
  selected: string | null;
  open: (id: string | null) => void;
}) {
  const [query, setQuery] = useState(""),
    [filter, setFilter] = useState("all"),
    [kind, setKind] = useState("all"),
    [view, setView] = useStoredChoice(
      "yunshu.console.models.view",
      ["table", "cards"] as const,
      "table",
    ),
    [details, setDetails] = useState<Model | null>(null),
    [metadata, setMetadata] = useState<unknown>(null),
    [detailError, setDetailError] = useState(""),
    [unloading, setUnloading] = useState<Model | null>(null);
  const opener = useRef<HTMLButtonElement | null>(null),
    detailSequence = useRef(0);
  const online = isOnline(engine);
  const ledger = useMemoryLedger(connection, online);
  const downloads = useDownloads(connection, online);
  const local = useLocalInventory(connection, online);
  const running = downloads.data?.jobs.filter(isActive).length ?? 0;
  const finishedCount =
    downloads.data?.jobs.filter((j) => j.state === "done").length ?? 0;
  const lastFinished = useRef(finishedCount);
  useEffect(() => {
    // a download finished: it registered itself, so show it in the library now
    if (finishedCount > lastFinished.current) {
      void engine.refresh();
      local.refresh();
    }
    lastFinished.current = finishedCount;
  }, [finishedCount]); // eslint-disable-line react-hooks/exhaustive-deps
  const unregistered = (local.data?.models ?? []).filter(
    (m) =>
      m.registeredAs == null &&
      m.id.toLowerCase().includes(query.toLowerCase()),
  );
  const freeGb = ledger.data ? (ledger.data.free_gb ?? null) : undefined;
  const rows = (engine.status?.models ?? []).filter(
    (m) =>
      (filter === "all" || m.loaded || m.loading) &&
      (kind === "all" || m.type === kind) &&
      m.id.toLowerCase().includes(query.toLowerCase()),
  );
  const restore = (e: Event) => {
    if (opener.current?.isConnected) {
      e.preventDefault();
      opener.current.focus();
    }
  };
  async function show(model: Model, button: HTMLButtonElement) {
    opener.current = button;
    setDetails(model);
    setMetadata(null);
    setDetailError("");
    const seq = ++detailSequence.current;
    try {
      const value = await getModel(connection, model.id);
      if (seq === detailSequence.current) setMetadata(value);
    } catch (e) {
      if (seq === detailSequence.current)
        setDetailError(e instanceof Error ? e.message : String(e));
    }
  }
  const memTotal = engine.status?.memory.total_gb,
    memActive = engine.status?.memory.active_gb,
    memFree =
      memTotal != null && memActive != null ? memTotal - memActive : undefined;
  const loadedRows = rows.filter((m) => m.loaded || m.loading),
    availableRows = rows.filter((m) => !m.loaded && !m.loading);
  const requestUnload = (model: Model, button: HTMLButtonElement) => {
    opener.current = button;
    setUnloading(model);
  };
  // Palette verbs `#/models?action=load|unload[&model=id]`: with a model id they act on it
  // (load runs, unload asks to confirm); without one they narrow the list so the user picks.
  const [intent, setIntent] = useState<{
    action: string;
    model: string;
  } | null>(null);
  useRouteAction("models", (action, q) => {
    if (action === "load" || action === "unload")
      setIntent({ action, model: q.get("model") ?? "" });
  });
  const statusModels = engine.status?.models;
  useEffect(() => {
    if (!intent || !online || !statusModels) return;
    setIntent(null);
    const target = intent.model
      ? statusModels.find((m) => m.id === intent.model)
      : undefined;
    if (
      target &&
      intent.action === "load" &&
      !target.loaded &&
      !target.loading
    ) {
      void perform(`load:${target.id}`, () => loadModel(connection, target.id));
    } else if (target && intent.action === "unload" && target.loaded) {
      setUnloading(target);
    } else {
      if (intent.action === "unload") setFilter("loaded");
      setQuery(intent.model);
      requestAnimationFrame(() =>
        document.getElementById("models-search")?.focus(),
      );
    }
  }, [intent, online, statusModels]); // eslint-disable-line react-hooks/exhaustive-deps
  const kinds = [...new Set((engine.status?.models ?? []).map((m) => m.type))];
  const group = (title: string, items: Model[]) =>
    view === "cards" ? (
      <div className="space-y-3">
        <SectionRow
          title={
            <span className="flex items-center gap-2">
              {title}
              <span className="text-muted-foreground tabular-nums">
                {items.length}
              </span>
            </span>
          }
        />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {items.map((model) => {
            const state = modelState(model);
            return (
              <ModelCard
                key={model.id}
                name={modelLabel(model.id)}
                icon={<LocalModelIcon id={model.id} />}
                ids={[model.id]}
                capabilities={
                  /vlm|omni/i.test(model.type) && isKnownCapability("vision")
                    ? ["vision"]
                    : []
                }
                description={
                  <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs tabular-nums">
                    <StatusIndicator status={state.status}>
                      {state.text}
                    </StatusIndicator>
                    <span>{model.type}</span>
                    <span>
                      <ModelSize gb={model.size_gb} />
                    </span>
                    <span>{retention(model)}</span>
                  </span>
                }
                onClick={() => open(model.id)}
              />
            );
          })}
        </div>
      </div>
    ) : (
      <div className="space-y-3">
        <SectionRow
          title={
            <span className="flex items-center gap-2">
              {title}
              <span className="text-muted-foreground tabular-nums">
                {items.length}
              </span>
            </span>
          }
        />
        <div className="space-y-2 md:hidden" data-testid="model-rows-mobile">
          {items.map((model) => {
            const state = modelState(model);
            return (
              <Card key={model.id} className="space-y-3 p-4">
                <div className="flex min-w-0 items-start gap-3">
                  <LocalModelIcon id={model.id} />
                  <div className="min-w-0 flex-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-auto max-w-full whitespace-normal break-words p-0 text-left"
                      onClick={(e) => void show(model, e.currentTarget)}
                    >
                      {modelLabel(model.id)}
                    </Button>
                    <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs tabular-nums text-muted-foreground">
                      <StatusIndicator status={state.status}>
                        <span className="text-foreground">{state.text}</span>
                      </StatusIndicator>
                      {model.loading && <LoadingElapsed id={model.id} />}
                      <span>{model.type}</span>
                      <span>
                        <ModelSize gb={model.size_gb} />
                      </span>
                      <span>{retention(model)}</span>
                    </div>
                    {model.error && (
                      <p className="mt-1 break-words text-xs text-error">
                        {model.error}
                      </p>
                    )}
                  </div>
                </div>
                <ModelActions
                  model={model}
                  connection={connection}
                  online={online}
                  busy={busy}
                  perform={perform}
                  test={test}
                  requestUnload={requestUnload}
                  freeGb={freeGb}
                />
              </Card>
            );
          })}
        </div>
        <Card className="hidden overflow-hidden md:block">
          <Table scrollLabel={title} className="min-w-[640px] table-fixed">
            <Thead>
              <Tr>
                <Th>{t("models.list.col.model")}</Th>
                <Th className="w-40">{t("models.list.col.state")}</Th>
                <Th className="hidden w-24 md:table-cell">
                  {t("models.list.col.size")}
                </Th>
                <Th className="hidden w-32 xl:table-cell">
                  {t("models.list.col.keep")}
                </Th>
                <Th className="w-72">{t("models.list.col.actions")}</Th>
              </Tr>
            </Thead>
            <Tbody>
              {items.map((model) => (
                <Tr key={model.id}>
                  <Td>
                    <div className="flex min-w-0 items-start gap-3">
                      <LocalModelIcon id={model.id} />
                      <div className="min-w-0">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-auto max-w-72 whitespace-normal break-all p-0 text-left"
                          onClick={(e) => void show(model, e.currentTarget)}
                        >
                          {modelLabel(model.id)}
                        </Button>
                        <div className="mt-1 flex flex-wrap items-center gap-1.5">
                          <span className="text-xs text-muted-foreground">
                            {model.type}
                            {/vlm|omni/i.test(model.type) &&
                              ` · ${t("models.local.cap.vision")}`}
                          </span>
                        </div>
                        {model.error && (
                          <p className="mt-1 max-w-xs break-words text-xs text-error">
                            {model.error}
                          </p>
                        )}
                      </div>
                    </div>
                  </Td>
                  <Td>
                    <div className="w-28">
                      <StatusIndicator
                        status={
                          model.error
                            ? "busy"
                            : model.loading
                              ? "away"
                              : model.loaded
                                ? "online"
                                : "neutral"
                        }
                      >
                        <span className="whitespace-nowrap text-foreground">
                          {model.loading
                            ? t("models.state.loading")
                            : model.error
                              ? t("models.state.failed")
                              : model.loaded
                                ? t("models.state.loaded")
                                : t("models.state.notLoaded")}
                        </span>
                      </StatusIndicator>
                      {model.loading && (
                        <div className="mt-1 text-xs">
                          <LoadingElapsed id={model.id} />
                        </div>
                      )}
                      <div className="mt-1 h-1">
                        {model.loading && (
                          <Progress
                            indeterminate
                            className="h-1"
                            label={t("models.list.loadingAria", {
                              name: modelLabel(model.id),
                            })}
                          />
                        )}
                      </div>
                    </div>
                  </Td>
                  <Td className="hidden tabular-nums md:table-cell">
                    <ModelSize gb={model.size_gb} />
                  </Td>
                  <Td className="hidden tabular-nums text-muted-foreground xl:table-cell">
                    {model.loaded
                      ? model.pinned
                        ? t("models.retention.pinned")
                        : t("models.retention.releaseIn", {
                            time: elapsed(model.expires_in_s),
                          })
                      : "—"}
                  </Td>
                  <Td>
                    <ModelActions
                      model={model}
                      connection={connection}
                      online={online}
                      busy={busy}
                      perform={perform}
                      test={test}
                      requestUnload={requestUnload}
                      freeGb={freeGb}
                    />
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </Card>
      </div>
    );
  return (
    <DashboardPage data-testid="models">
      {selected ? (
        <ModelDetail
          id={selected}
          engine={engine}
          connection={connection}
          online={online}
          busy={busy}
          perform={perform}
          test={test}
          requestUnload={requestUnload}
          back={() => open(null)}
          ledger={ledger}
          local={
            local.data?.models.find((m) => m.registeredAs === selected) ?? null
          }
        />
      ) : (
        <>
          <PageHeader
            title={t("models.list.title")}
            description={t("models.list.description")}
            actions={
              <div className="flex flex-wrap gap-2">
                <ModelManagement
                  connection={connection}
                  disabled={!online || !!busy}
                  disabledReason={
                    !online
                      ? t("models.list.offlineReason")
                      : busy
                        ? t("models.list.busyReason")
                        : undefined
                  }
                  perform={perform}
                />
                {!downloads.unsupported && (
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => {
                      location.hash = "/downloads";
                    }}
                  >
                    <Download size={14} />
                    {t("models.list.download")}
                  </Button>
                )}
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => void engine.refresh()}
                >
                  <RefreshCw size={14} />
                  {t("models.list.refresh")}
                </Button>
              </div>
            }
          />
          <Card className="flex flex-wrap items-center gap-3 p-4">
            <div className="w-full sm:w-auto sm:max-w-xs [&>*]:w-full">
              <SearchInput
                id="models-search"
                className="w-full"
                aria-label={t("models.list.searchAria")}
                value={query}
                onChange={setQuery}
                placeholder={t("models.list.searchPlaceholder")}
              />
            </div>
            <SegmentedTray
              fillOnPhone
              value={filter}
              onChange={setFilter}
              options={[
                { value: "all", label: t("models.list.filterAll") },
                { value: "loaded", label: t("models.list.filterLoaded") },
              ]}
            />
            {kinds.length > 1 && (
              <SegmentedTray
                fillOnPhone
                value={kind}
                onChange={setKind}
                options={[
                  { value: "all", label: t("models.list.kindAll") },
                  ...kinds.map((k) => ({ value: k, label: k })),
                ]}
              />
            )}
            <SegmentedTray
              fillOnPhone
              className="sm:ml-auto"
              value={view}
              onChange={(v) => setView(v as "table" | "cards")}
              options={[
                {
                  value: "table",
                  label: t("models.list.viewTable"),
                  icon: Table2,
                },
                {
                  value: "cards",
                  label: t("models.list.viewCards"),
                  icon: LayoutGrid,
                },
              ]}
            />
          </Card>
          {running > 0 && (
            <Card
              className="flex flex-wrap items-center gap-3 px-5 py-3 text-sm"
              data-testid="downloads-running"
            >
              <Download size={14} className="text-muted-foreground" />
              <span>{t("models.list.downloading", { count: running })}</span>
              <a
                className="ml-auto text-sm underline underline-offset-2"
                href="#/downloads"
              >
                {t("models.list.viewDownloads")}
              </a>
            </Card>
          )}
          {memTotal != null && memTotal > 0 && (
            <Card
              className="flex flex-col items-stretch gap-3 px-5 py-4 sm:flex-row sm:flex-wrap sm:items-center sm:gap-x-6"
              data-testid="memory-summary"
            >
              <div className="min-w-0 text-left sm:min-w-32">
                <p className="text-xs text-muted-foreground">
                  {t("models.list.memoryUsage")}
                </p>
                <p className="mt-1 text-2xl font-semibold tabular-nums">
                  {fixed(memActive)}
                  <span className="ml-1 text-xs font-normal text-muted-foreground">
                    / {number(memTotal, 0)} GB
                  </span>
                </p>
              </div>
              <Progress
                className="h-1.5 w-full sm:min-w-40 sm:flex-1"
                value={Math.max(
                  0,
                  Math.min(100, ((memActive ?? 0) / memTotal) * 100),
                )}
                label={t("models.list.memoryUsage")}
              />
              <p className="text-left text-xs tabular-nums text-muted-foreground sm:w-44 sm:shrink-0 sm:text-right">
                {t("models.list.memorySummary", {
                  loaded: loadedRows.filter((m) => m.loaded && !m.loading)
                    .length,
                  free: fixed(memFree),
                })}
              </p>
            </Card>
          )}
          {!rows.length && !unregistered.length ? (
            <Card className="p-2">
              <EmptyState
                size="inline"
                icon={<Box size={22} />}
                title={
                  engine.status
                    ? t("models.list.emptyTitle")
                    : t("models.detail.waitingTitle")
                }
                description={
                  engine.status
                    ? t("models.list.emptyDescription")
                    : t("models.detail.waitingDescription")
                }
              />
            </Card>
          ) : (
            <>
              {loadedRows.length > 0 &&
                group(t("models.list.groupLoaded"), loadedRows)}
              {filter === "all" &&
                availableRows.length > 0 &&
                group(t("models.list.groupAvailable"), availableRows)}
            </>
          )}
          {filter === "all" && unregistered.length > 0 && (
            <LocalInventory
              items={unregistered}
              connection={connection}
              online={online}
              busy={busy}
              perform={perform}
              freeBytes={local.data?.freeBytes ?? null}
            />
          )}
          <p className="text-xs text-muted-foreground">
            {t("models.list.note")}
          </p>
        </>
      )}
      <Dialog
        open={!!unloading}
        onOpenChange={(open) => {
          if (!open) setUnloading(null);
        }}
      >
        <DialogContent
          closeLabel={t("models.unload.close")}
          onCloseAutoFocus={restore}
        >
          <DialogTitle>{t("models.unload.title")}</DialogTitle>
          <DialogDescription>
            {t("models.unload.description", { id: unloading?.id ?? "" })}
          </DialogDescription>
          <div className="flex justify-end gap-2">
            <Button variant="secondary" onClick={() => setUnloading(null)}>
              {t("models.unload.keep")}
            </Button>
            <Button
              onClick={() => {
                const m = unloading;
                setUnloading(null);
                if (m)
                  void perform(`unload:${m.id}`, () =>
                    unloadModel(connection, m.id),
                  );
              }}
            >
              {t("models.unload.confirm")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
      <Dialog
        open={!!details}
        onOpenChange={(open) => {
          if (!open) {
            setDetails(null);
            detailSequence.current++;
          }
        }}
      >
        <DialogContent
          closeLabel={t("models.info.close")}
          onCloseAutoFocus={restore}
        >
          <DialogTitle>
            {details ? modelLabel(details.id) : t("models.info.title")}
          </DialogTitle>
          <DialogDescription>{t("models.info.description")}</DialogDescription>
          {detailError ? (
            <p role="alert" className="text-sm text-error">
              {detailError}
            </p>
          ) : metadata ? (
            <CodeBlock language="json">
              {JSON.stringify(metadata, null, 2)}
            </CodeBlock>
          ) : (
            <p role="status" className="text-sm text-muted-foreground">
              {t("models.info.loading")}
            </p>
          )}
          {details && (
            <DialogFooter>
              <Button
                variant="secondary"
                onClick={() => {
                  const id = details.id;
                  setDetails(null);
                  detailSequence.current++;
                  open(id);
                }}
              >
                {t("models.info.openDetail")}
              </Button>
            </DialogFooter>
          )}
        </DialogContent>
      </Dialog>
    </DashboardPage>
  );
}
