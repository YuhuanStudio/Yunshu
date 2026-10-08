import { t } from "./i18n/index.ts";
import { useEffect, useState } from "react";
import {
  Button,
  Card,
  EmptyState,
  Gauge,
  SegmentedBar,
  StatusIndicator,
} from "@yuhuanowo/yunui";
import { IDBadge } from "@yuhuanowo/yunui/ai";
import { CodeBlock } from "@yuhuanowo/yunui/code";
import { DetailList, DetailRow, PageHeader } from "@yuhuanowo/yunui/patterns";
import {
  ArrowLeft,
  Box,
  FileJson,
  Gauge as GaugeIcon,
  MemoryStick,
  Timer,
} from "lucide-react";
import { getModel, type Connection } from "./api";
import { ModelSize } from "./ModelSize";
import { MemoryLedgerView } from "./MemoryLedger";
import type { LedgerState } from "./memory-api";
import { ModelActions, type Perform } from "./ModelActions";
import { FitPanel, useFit } from "./FitPanel";
import { LoadingElapsed } from "./loading-clock";
import { localFacts } from "./LocalInventory";
import type { LocalModel } from "./admin-models-api";
import {
  LocalModelIcon,
  SectionCard,
  elapsed,
  modelLabel,
  number,
  type Engine,
  type Model,
} from "./ui";

export const modelState = (model: Model) =>
  model.error
    ? ({ status: "busy", text: t("models.state.failed") } as const)
    : model.loading
      ? ({ status: "away", text: t("models.state.loading") } as const)
      : model.loaded
        ? ({ status: "online", text: t("models.state.loaded") } as const)
        : ({ status: "neutral", text: t("models.state.notLoaded") } as const);

export const retention = (model: Model) =>
  !model.loaded
    ? "—"
    : model.pinned
      ? t("models.retention.pinned")
      : model.expires_in_s != null
        ? t("models.retention.releaseIn", { time: elapsed(model.expires_in_s) })
        : model.keep_alive_s != null
          ? t("models.retention.idleRelease", {
              time: elapsed(model.keep_alive_s),
            })
          : t("models.retention.serverDefault");

/** Per-model page: only facts the status and model endpoints report. */
export function ModelDetail({
  id,
  engine,
  connection,
  online,
  busy,
  perform,
  test,
  requestUnload,
  back,
  ledger,
  local,
}: {
  id: string;
  engine: Engine;
  connection: Connection;
  online: boolean;
  busy: string | null;
  perform: Perform;
  test: (id: string) => void;
  requestUnload: (model: Model, button: HTMLButtonElement) => void;
  back: () => void;
  ledger?: LedgerState;
  /** Disk facts for this model from /models/local, when the server has them. */
  local?: LocalModel | null;
}) {
  const model = engine.status?.models.find((m) => m.id === id);
  const fitState = useFit(
    connection,
    id,
    online && !!model && !model.loaded && !model.loading,
  );
  const [card, setCard] = useState<unknown>(null),
    [cardError, setCardError] = useState("");
  useEffect(() => {
    if (!online) return;
    const controller = new AbortController();
    setCard(null);
    setCardError("");
    getModel(connection, id)
      .then((value) => {
        if (!controller.signal.aborted) setCard(value);
      })
      .catch((e) => {
        if (!controller.signal.aborted)
          setCardError(e instanceof Error ? e.message : String(e));
      });
    return () => controller.abort();
  }, [connection.baseUrl, connection.token, id, online]);
  const backButton = (
    <Button size="sm" variant="secondary" onClick={back}>
      <ArrowLeft size={14} />
      {t("models.detail.back")}
    </Button>
  );
  if (!model)
    return (
      <>
        <PageHeader title={modelLabel(id)} actions={backButton} />
        <Card>
          <EmptyState
            icon={<Box size={25} />}
            title={
              engine.status
                ? t("models.detail.missingTitle")
                : t("models.detail.waitingTitle")
            }
            description={
              engine.status
                ? t("models.detail.missingDescription")
                : t("models.detail.waitingDescription")
            }
          />
        </Card>
      </>
    );
  const state = modelState(model);
  const total = engine.status?.memory.total_gb,
    active = engine.status?.memory.active_gb;
  const share =
    model.size_gb && total
      ? Math.min(100, (model.size_gb / total) * 100)
      : null;
  const memorySegments =
    model.loaded && model.size_gb && total && active != null
      ? [
          {
            value: Math.min(model.size_gb, active),
            tone: "accent" as const,
            label: t("models.detail.weights"),
          },
          {
            value: Math.max(0, active - model.size_gb),
            tone: "neutral" as const,
            label: t("models.detail.otherActive"),
          },
        ]
      : null;
  return (
    <>
      <PageHeader
        title={modelLabel(model.id)}
        description={`${model.type} · ${state.text}`}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <ModelActions
              model={model}
              connection={connection}
              online={online}
              busy={busy}
              perform={perform}
              test={test}
              requestUnload={requestUnload}
              freeGb={ledger ? (ledger.data?.free_gb ?? undefined) : undefined}
            />
            {backButton}
          </div>
        }
      />
      {model.error && (
        <p role="alert" className="text-sm text-error">
          {model.error}
        </p>
      )}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-w-0 space-y-6">
          <SectionCard
            icon={MemoryStick}
            title={t("models.detail.memory.title")}
            description={
              model.loaded
                ? t("models.detail.memory.loadedDescription")
                : t("models.detail.memory.unloadedDescription")
            }
          >
            {share == null ? (
              <p className="text-sm text-muted-foreground">
                {t("models.detail.memory.unknown")}
              </p>
            ) : (
              <div className="flex flex-wrap items-center gap-6">
                <Gauge
                  value={share}
                  size={96}
                  thickness={8}
                  tone={share > 85 ? "warning" : "accent"}
                  label={
                    <span className="text-base font-semibold">{`${number(share, 0)}%`}</span>
                  }
                  ariaLabel={t("models.detail.memory.gaugeAria", {
                    percent: number(share, 0),
                  })}
                />
                <div className="min-w-0 flex-1 space-y-3">
                  <p className="text-sm tabular-nums">
                    <ModelSize gb={model.size_gb} />
                    <span className="text-muted-foreground">
                      {" "}
                      {t("models.detail.memory.ofTotal", {
                        total: number(total, 0),
                      })}
                    </span>
                  </p>
                  {memorySegments && (
                    <SegmentedBar
                      segments={memorySegments}
                      total={total}
                      legend
                      height={8}
                      formatValue={(v) => `${number(v)} GB`}
                      label={t("models.detail.memory.barLabel")}
                    />
                  )}
                </div>
              </div>
            )}
          </SectionCard>
          {!model.loaded && !model.loading && !fitState.unsupported && (
            <SectionCard
              icon={GaugeIcon}
              title={t("models.fit.title")}
              description={t("models.fit.description")}
              data-testid="fit-card"
            >
              {fitState.fit ? (
                <FitPanel fit={fitState.fit} />
              ) : (
                <p role="status" className="text-sm text-muted-foreground">
                  {fitState.error
                    ? t("models.fit.error")
                    : t("models.fit.checking")}
                </p>
              )}
            </SectionCard>
          )}
          <SectionCard
            icon={MemoryStick}
            title={t("models.detail.holders.title")}
            description={t("models.detail.holders.description")}
          >
            {ledger?.unsupported ? (
              <p
                className="text-sm text-muted-foreground"
                data-testid="memory-ledger-unsupported"
              >
                {t("models.detail.holders.unsupported")}
              </p>
            ) : ledger?.data ? (
              <MemoryLedgerView data={ledger.data} />
            ) : (
              <p role="status" className="text-sm text-muted-foreground">
                {ledger?.error
                  ? t("models.detail.holders.error", { error: ledger.error })
                  : t("models.detail.holders.loading")}
              </p>
            )}
          </SectionCard>
          <SectionCard
            icon={FileJson}
            title={t("models.detail.card.title")}
            description={t("models.detail.card.description")}
          >
            {cardError ? (
              <p role="alert" className="text-sm text-error">
                {cardError}
              </p>
            ) : card ? (
              <CodeBlock language="json">
                {JSON.stringify(card, null, 2)}
              </CodeBlock>
            ) : (
              <p role="status" className="text-sm text-muted-foreground">
                {online
                  ? t("models.detail.card.loading")
                  : t("models.detail.card.offline")}
              </p>
            )}
          </SectionCard>
        </div>
        <SectionCard
          icon={Timer}
          title={t("models.detail.facts.title")}
          className="h-fit"
          data-testid="model-facts"
        >
          <div className="mb-4 flex min-w-0 items-center gap-3">
            <LocalModelIcon id={model.id} size={32} />
            <IDBadge text={model.id} truncate />
          </div>
          <DetailList ariaLabel={t("models.detail.facts.aria")}>
            <DetailRow
              label={t("models.detail.facts.type")}
              value={model.type}
              mono={false}
            />
            <DetailRow
              label={t("models.detail.facts.state")}
              mono={false}
              value={
                <span className="inline-flex flex-wrap items-center gap-x-3">
                  <StatusIndicator status={state.status}>
                    {state.text}
                  </StatusIndicator>
                  {model.loading && <LoadingElapsed id={model.id} />}
                </span>
              }
            />
            <DetailRow
              label={t("models.detail.facts.size")}
              value={<ModelSize gb={model.size_gb} />}
            />
            <DetailRow
              label={t("models.detail.facts.retention")}
              value={retention(model)}
              mono={false}
            />
            <DetailRow
              label={t("models.detail.facts.pinned")}
              value={
                model.pinned
                  ? t("models.detail.facts.yes")
                  : t("models.detail.facts.no")
              }
              mono={false}
            />
            {local && (
              <>
                <DetailRow
                  label={t("models.detail.facts.disk")}
                  value={localFacts(local).join(" · ") || "—"}
                  mono={false}
                />
                <DetailRow
                  label={t("models.detail.facts.files")}
                  value={
                    local.complete
                      ? t("models.local.complete")
                      : t("models.local.incomplete")
                  }
                  mono={false}
                />
              </>
            )}
            <DetailRow
              label={t("models.detail.facts.idle")}
              value={elapsed(model.idle_s)}
            />
            <DetailRow
              label={t("models.detail.facts.keepAlive")}
              value={
                model.keep_alive_s != null ? elapsed(model.keep_alive_s) : "—"
              }
            />
          </DetailList>
        </SectionCard>
      </div>
    </>
  );
}
