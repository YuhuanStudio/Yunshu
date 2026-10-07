import { useCallback, useEffect, useState } from "react";
import {
  Button,
  ConfirmModal,
  EmptyState,
  Input,
  StatusIndicator,
  Switch,
  Tag,
} from "@yuhuanowo/yunui";
import { CodeBlock } from "@yuhuanowo/yunui/content";
import { Banner } from "@yuhuanowo/yunui/patterns";
import { CircleAlert, Globe, Plus, Server } from "lucide-react";
import type { Connection } from "./api";
import {
  AdminError,
  getCors,
  getService,
  manualCommand,
  normaliseOrigin,
  patchCors,
  restartService,
  type CorsInfo,
  type RestartAccepted,
  type ServiceInfo,
} from "./admin-settings-api";
import { uptimeText } from "./footer-status";
import { StackRow } from "./stack-row";
import { t } from "./i18n/index.ts";
import { CopyField, SectionCard, UnavailableNotice } from "./ui";

/** One-click restart (launchd only): confirm, then drain and kickstart. A 409 shows the manual command. */
export function RestartControl({
  connection,
  drainTimeoutS,
}: {
  connection: Connection;
  /** Shown in the confirmation when known. */
  drainTimeoutS?: number | null;
}) {
  const [open, setOpen] = useState(false),
    [state, setState] = useState<"idle" | "busy" | "done" | "failed">("idle"),
    [accepted, setAccepted] = useState<RestartAccepted | null>(null),
    [manual, setManual] = useState<string | null>(null);
  async function go() {
    setOpen(false);
    setState("busy");
    try {
      setAccepted(await restartService(connection));
      setState("done");
    } catch (e) {
      const cmd = manualCommand(e);
      if (cmd) setManual(cmd);
      setState(cmd ? "idle" : "failed");
    }
  }
  return (
    <div className="space-y-2" data-testid="restart-control">
      <Button
        size="sm"
        variant="secondary"
        disabled={state === "busy" || state === "done"}
        onClick={() => setOpen(true)}
      >
        {t("service.restart.button")}
      </Button>
      {state === "done" && (
        <p role="status" className="text-xs text-muted-foreground">
          {t("service.restart.accepted", {
            count: accepted?.activeRequests ?? 0,
            seconds: accepted?.drainTimeoutS ?? 0,
          })}
        </p>
      )}
      {state === "failed" && (
        <p role="alert" className="text-xs text-error">
          {t("service.restart.failed")}
        </p>
      )}
      {manual && (
        <div className="space-y-1">
          <p role="alert" className="text-xs text-muted-foreground">
            {t("service.restart.notLaunchd")}
          </p>
          <CopyField label={t("service.restart.manual")} value={manual} />
        </div>
      )}
      <ConfirmModal
        isOpen={open}
        variant="warning"
        title={t("service.restart.confirmTitle")}
        message={
          drainTimeoutS != null
            ? t("service.restart.confirmBody", { seconds: drainTimeoutS })
            : t("service.restart.confirmBodyUnknown")
        }
        confirmText={t("service.restart.confirm")}
        cancelText={t("settings.admin.cancel")}
        onConfirm={() => void go()}
        onClose={() => setOpen(false)}
      />
    </div>
  );
}

type Load<T> =
  | { state: "loading" }
  | { state: "ok"; data: T }
  | { state: "unsupported" | "denied" | "error" };

function useAdminLoad<T>(
  connection: Connection,
  fetcher: (c: Connection, s: AbortSignal) => Promise<T>,
) {
  const [load, setLoad] = useState<Load<T>>({ state: "loading" }),
    [tick, setTick] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    fetcher(connection, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setLoad({ state: "ok", data });
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setLoad({
          state:
            e instanceof AdminError
              ? e.unsupported
                ? "unsupported"
                : e.denied
                  ? "denied"
                  : "error"
              : "error",
        });
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connection.baseUrl, connection.token, tick]);
  const reload = useCallback(() => setTick((n) => n + 1), []);
  return [load, reload, setLoad] as const;
}

function Unavailable({ kind }: { kind: "unsupported" | "denied" | "error" }) {
  return (
    <UnavailableNotice
      title={t(`service.unavailable.${kind}.title`)}
      description={t(`service.unavailable.${kind}.description`)}
    />
  );
}

const Mono = ({ children }: { children: string }) => (
  <span className="break-all font-mono text-xs">{children}</span>
);

export function ServiceSection({ connection }: { connection: Connection }) {
  const [load, reload] = useAdminLoad<ServiceInfo>(connection, getService);
  return (
    <SectionCard
      id="settings-service"
      icon={Server}
      title={t("service.title")}
      description={t("service.description")}
      className="scroll-mt-4"
      bodyClassName="px-5 pb-5"
      data-testid="service-section"
      action={
        load.state === "ok" ? (
          <Button size="sm" variant="ghost" onClick={reload}>
            {t("service.refresh")}
          </Button>
        ) : undefined
      }
    >
      {load.state === "loading" && (
        <p role="status" className="pt-2 text-sm text-muted-foreground">
          {t("service.loading")}
        </p>
      )}
      {(load.state === "unsupported" ||
        load.state === "denied" ||
        load.state === "error") && <Unavailable kind={load.state} />}
      {load.state === "ok" && (
        <ServiceBody info={load.data} connection={connection} />
      )}
    </SectionCard>
  );
}

function ServiceBody({
  info,
  connection,
}: {
  info: ServiceInfo;
  connection: Connection;
}) {
  const status = info.underLaunchd
    ? "online"
    : info.loaded
      ? "away"
      : "neutral";
  const label = info.underLaunchd
    ? t("service.state.managed")
    : info.loaded
      ? t("service.state.other")
      : info.installed
        ? t("service.state.stopped")
        : t("service.state.notInstalled");
  return (
    <div data-testid="service-body">
      <StackRow
        title={t("service.row.status")}
        control={
          <StatusIndicator
            status={status}
            className="gap-1.5 text-sm text-muted-foreground"
          >
            {label}
          </StatusIndicator>
        }
      />
      <StackRow
        title={t("service.row.pid")}
        control={
          <span className="text-sm tabular-nums">{info.pid ?? "—"}</span>
        }
      />
      <StackRow
        title={t("service.row.uptime")}
        description={t("service.row.uptimeHelp")}
        control={
          <span className="text-sm tabular-nums">
            {info.uptimeS == null ? "—" : uptimeText(info.uptimeS)}
          </span>
        }
      />
      <StackRow
        title={t("service.row.version")}
        control={
          <span className="text-sm tabular-nums">{info.version || "—"}</span>
        }
      />
      <StackRow
        title={t("service.row.plist")}
        control={<Mono>{info.plist || "—"}</Mono>}
      />
      <StackRow
        title={t("service.row.log")}
        control={<Mono>{info.log || "—"}</Mono>}
      />
      <div className="space-y-3 pt-4">
        {info.underLaunchd ? (
          <>
            <RestartControl connection={connection} />
            <p className="text-xs text-muted-foreground">
              {t("service.restart.help")}
            </p>
          </>
        ) : (
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              {t("service.restart.notLaunchd")}
            </p>
            <CopyField
              label={t("service.restart.cli")}
              value={info.installed ? info.cli.restart : info.cli.install}
            />
          </div>
        )}
      </div>
    </div>
  );
}

// ── network (read-only) and CORS (editable) ─────────────────────────

const LOOPBACK = new Set(["127.0.0.1", "localhost", "::1", "[::1]"]);

export function bindScope(hostname: string): "loopback" | "lan" {
  return LOOPBACK.has(hostname.toLowerCase()) ? "loopback" : "lan";
}

export function NetworkSection({ connection }: { connection: Connection }) {
  let host = "—",
    scope: "loopback" | "lan" = "loopback";
  try {
    const u = new URL(connection.baseUrl);
    host = u.host;
    scope = bindScope(u.hostname);
  } catch {
    /* invalid address: the connection card reports it */
  }
  return (
    <SectionCard
      icon={Globe}
      title={t("service.network.title")}
      description={t("service.network.description")}
      className="scroll-mt-4"
      bodyClassName="px-5 pb-5"
      data-testid="network-section"
    >
      <StackRow
        title={t("service.network.address")}
        description={t("service.network.addressHelp")}
        control={<Mono>{host}</Mono>}
      />
      <StackRow
        title={t("service.network.exposure")}
        description={t("service.network.exposureHelp")}
        control={
          <StatusIndicator
            status={scope === "lan" ? "away" : "online"}
            className="gap-1.5 text-sm text-muted-foreground"
          >
            {scope === "lan"
              ? t("service.network.lan")
              : t("service.network.loopback")}
          </StatusIndicator>
        }
      />
      <div className="grid gap-4 pt-4 lg:grid-cols-2">
        {(
          [
            ["service.network.cmdLocal", "127.0.0.1"],
            ["service.network.cmdLan", "0.0.0.0"],
          ] as const
        ).map(([label, host]) => (
          <div key={host} className="min-w-0">
            <p className="mb-1.5 text-xs text-muted-foreground">{t(label)}</p>
            <CodeBlock language="bash">
              {`yunshu service install --force --host ${host} -m <model>`}
            </CodeBlock>
          </div>
        ))}
      </div>
      <p className="pt-3 text-xs text-muted-foreground">
        {t("service.network.lanWarning")}
      </p>
    </SectionCard>
  );
}

export function CorsSection({ connection }: { connection: Connection }) {
  const [load, reload] = useAdminLoad<CorsInfo>(connection, getCors),
    [saved, setSaved] = useState(false);
  return (
    <SectionCard
      id="settings-cors"
      icon={Globe}
      title={t("service.cors.title")}
      description={t("service.cors.description")}
      className="scroll-mt-4"
      bodyClassName="px-5 pb-5"
      data-testid="cors-section"
    >
      {load.state === "loading" && (
        <p role="status" className="pt-2 text-sm text-muted-foreground">
          {t("service.loading")}
        </p>
      )}
      {(load.state === "unsupported" ||
        load.state === "denied" ||
        load.state === "error") && <Unavailable kind={load.state} />}
      {load.state === "ok" && (
        <CorsEditor
          key={load.data.origins.join(",")}
          connection={connection}
          info={load.data}
          saved={saved}
          onSaved={() => {
            setSaved(true);
            reload();
          }}
        />
      )}
    </SectionCard>
  );
}

function CorsEditor({
  connection,
  info,
  saved,
  onSaved,
}: {
  connection: Connection;
  info: CorsInfo;
  saved: boolean;
  onSaved: () => void;
}) {
  const [origins, setOrigins] = useState(info.origins),
    [text, setText] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [failure, setFailure] = useState(""),
    [lastSaved, setLastSaved] = useState(""),
    [anyOk, setAnyOk] = useState(false);
  const wildcard = origins.includes("*");
  const dirty = origins.join(",") !== info.origins.join(",");
  function add() {
    const o = normaliseOrigin(text);
    if (!o) return setError(t("service.cors.invalid"));
    if ((o === "*" && origins.length) || (wildcard && o !== "*"))
      return setError(t("service.cors.mixed"));
    if (origins.includes(o)) return setError(t("service.cors.duplicate"));
    setOrigins([...origins, o]);
    setText("");
    setError("");
  }
  async function save(next: string[] | null) {
    setBusy(true);
    setFailure("");
    try {
      await patchCors(connection, next, anyOk);
      setLastSaved(origins.join(","));
      onSaved();
    } catch (e) {
      const invalid =
        e instanceof AdminError && typeof e.detail.invalid === "object"
          ? Object.keys(e.detail.invalid as object)
          : [];
      setFailure(
        invalid.length
          ? t("service.cors.rejected", { list: invalid.join(", ") })
          : e instanceof AdminError
            ? e.message
            : t("settings.admin.error.network"),
      );
    } finally {
      setBusy(false);
    }
  }
  const forced = info.source === "env" || info.source === "cli";
  return (
    <div className="space-y-4 pt-3" data-testid="cors-editor">
      {forced && (
        <Banner
          tone="warning"
          icon={<CircleAlert size={16} />}
          title={t("service.cors.forced", {
            source:
              info.source === "cli"
                ? t("service.cors.source.cli")
                : t("service.cors.source.env"),
          })}
        />
      )}
      {(wildcard || info.wildcard) && (
        <Banner
          tone="warning"
          icon={<CircleAlert size={16} />}
          title={t("service.cors.wildcardTitle")}
          description={t("service.cors.wildcardBody")}
        />
      )}
      {info.requestOrigin && info.requestOriginAllowed != null && (
        <p className="text-xs text-muted-foreground">
          {info.requestOriginAllowed
            ? t("service.cors.thisAllowed", { origin: info.requestOrigin })
            : t("service.cors.thisBlocked", { origin: info.requestOrigin })}
        </p>
      )}
      <ul
        className="flex flex-wrap gap-2"
        aria-label={t("service.cors.listAria")}
        data-testid="cors-origins"
      >
        {origins.map((o) => (
          <li key={o}>
            <Tag
              className="font-mono text-xs"
              removeLabel={t("service.cors.remove", { origin: o })}
              onRemove={() => {
                setOrigins(origins.filter((x) => x !== o));
              }}
            >
              {o}
            </Tag>
          </li>
        ))}
        {!origins.length && (
          <li className="text-xs text-muted-foreground">
            {t("service.cors.none")}
          </li>
        )}
      </ul>
      <div className="flex flex-wrap items-start gap-2">
        <div className="min-w-0 flex-1 sm:max-w-sm">
          <Input
            aria-label={t("service.cors.add")}
            className="font-mono text-sm"
            value={text}
            placeholder="https://app.example.com"
            onChange={(e) => {
              setText(e.target.value);
              setError("");
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                add();
              }
            }}
          />
          {error && (
            <p role="alert" className="mt-1 text-xs text-error">
              {error}
            </p>
          )}
        </div>
        <Button
          size="sm"
          variant="secondary"
          type="button"
          onClick={add}
          disabled={!text.trim()}
        >
          <Plus size={13} />
          {t("service.cors.addButton")}
        </Button>
      </div>
      {wildcard && (
        <span className="flex items-center gap-2 text-xs text-muted-foreground">
          <Switch
            label={t("service.cors.confirmAny")}
            checked={anyOk}
            onCheckedChange={setAnyOk}
          />
          {t("service.cors.confirmAny")}
        </span>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <Button
          size="sm"
          disabled={busy || !dirty || !origins.length || (wildcard && !anyOk)}
          title={
            wildcard && !anyOk ? t("service.cors.confirmAnyNeeded") : undefined
          }
          onClick={() => void save(origins)}
        >
          {t("service.cors.save")}
        </Button>
        <Button
          size="sm"
          variant="secondary"
          disabled={busy || info.source === "default"}
          title={
            info.source === "default"
              ? t("service.cors.alreadyDefault")
              : undefined
          }
          onClick={() => void save(null)}
        >
          {t("service.cors.reset")}
        </Button>
        {saved && (!dirty || origins.join(",") === lastSaved) && (
          <span role="status" className="text-xs text-muted-foreground">
            {t("service.cors.saved")}
          </span>
        )}
        {failure && (
          <span role="alert" className="text-xs text-error">
            {failure}
          </span>
        )}
      </div>
      <p className="text-xs text-muted-foreground">
        {t("service.cors.help", { default: info.default })}
      </p>
    </div>
  );
}
