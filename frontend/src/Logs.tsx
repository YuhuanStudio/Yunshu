import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import {
  Button,
  Card,
  CustomSelect,
  EmptyState,
  IconButton,
  Input,
} from "@yuhuanowo/yunui";
import { DashboardPage, PageHeader } from "@yuhuanowo/yunui/patterns";
import {
  ArrowDownToLine,
  Copy,
  Download,
  Pause,
  Play,
  Search,
} from "lucide-react";
import { ApiError, type Connection } from "./api";
import {
  fetchLogs,
  formatLine,
  saveBlob,
  streamLogs,
  type LogRecord,
} from "./admin-logs-api";
import { SegmentedTray } from "./SegmentedTray";
import { has, t, tr } from "./i18n/index.ts";
import { UnavailableNotice, clock, dateTime, number } from "./ui";

const CLIENT_CAP = 4000;
const RING_LIMIT = 2000;
const LEVELS = ["all", "INFO", "WARNING", "ERROR"] as const;
type LevelChoice = (typeof LEVELS)[number];

/** `#/logs/range/<from>/<to>` (epoch seconds), written by the request detail and Diagnostics links. */
export function logsRangeHash(from: number, to: number) {
  return `#/logs/range/${Math.floor(from)}/${Math.ceil(to)}`;
}
function parseRange(hash: string): { from: number; to: number } | null {
  const m = /^#\/logs\/range\/(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)$/.exec(hash);
  return m ? { from: Number(m[1]), to: Number(m[2]) } : null;
}
function useHashRange() {
  const [range, setRange] = useState(() => parseRange(location.hash));
  useEffect(() => {
    const fn = () => setRange(parseRange(location.hash));
    addEventListener("hashchange", fn);
    return () => removeEventListener("hashchange", fn);
  }, []);
  return range;
}

const levelName = (level: string) =>
  has(`logs.level.${level}`) ? tr(`logs.level.${level}`) : level;
const levelTone = (level: string) =>
  level === "ERROR" || level === "CRITICAL"
    ? "text-error"
    : level === "WARNING" || level === "WARN"
      ? "text-warning"
      : "text-muted-foreground";
const stamp = (r: LogRecord) =>
  `${clock(r.t * 1000)}.${String(Math.floor((r.t % 1) * 1000)).padStart(3, "0")}`;

type Load = "loading" | "ready" | "missing" | "denied" | "error";
type Live = "off" | "connecting" | "live" | "polling" | "reconnecting";

export default function Logs({ connection }: { connection: Connection }) {
  const range = useHashRange();
  const [level, setLevel] = useState<LevelChoice>("all"),
    [text, setText] = useState(""),
    [query, setQuery] = useState(""),
    [span, setSpan] = useState("all"),
    [paused, setPaused] = useState(false),
    [follow, setFollow] = useState(true),
    [records, setRecords] = useState<LogRecord[]>([]),
    [load, setLoad] = useState<Load>("loading"),
    [live, setLive] = useState<Live>("off"),
    [dropped, setDropped] = useState(0),
    [unseen, setUnseen] = useState(0),
    [note, setNote] = useState<string | null>(null);
  const cursor = useRef(0),
    scroller = useRef<HTMLDivElement | null>(null),
    followRef = useRef(true);
  followRef.current = follow;
  const windowed = range != null;

  useEffect(() => {
    const id = setTimeout(() => setQuery(text.trim()), 300);
    return () => clearTimeout(id);
  }, [text]);

  // The latest list lives in a ref too, so the unseen counter is bumped once per batch:
  // a side effect inside a state updater would run twice under StrictMode.
  const recordsRef = useRef<LogRecord[]>([]);
  recordsRef.current = records;
  const append = useCallback((incoming: LogRecord[]) => {
    if (!incoming.length) return;
    const prev = recordsRef.current;
    const seen = new Set(prev.slice(-incoming.length - 50).map((r) => r.id));
    const fresh = incoming.filter((r) => !seen.has(r.id));
    if (!fresh.length) return;
    if (!followRef.current) setUnseen((n) => n + fresh.length);
    const next = prev.concat(fresh);
    recordsRef.current =
      next.length > CLIENT_CAP ? next.slice(-CLIENT_CAP) : next;
    setRecords(recordsRef.current);
  }, []);

  const sinceFor = useCallback(
    () =>
      range
        ? range.from
        : span === "5m"
          ? Date.now() / 1000 - 300
          : span === "1h"
            ? Date.now() / 1000 - 3600
            : undefined,
    [range, span],
  );

  // First page (and every filter change): one query, newest records last.
  useEffect(() => {
    const controller = new AbortController();
    setLoad("loading");
    setUnseen(0);
    fetchLogs(
      connection,
      {
        level: level === "all" ? undefined : level,
        q: query || undefined,
        since: sinceFor(),
        limit: RING_LIMIT,
      },
      controller.signal,
    )
      .then((page) => {
        if (controller.signal.aborted) return;
        const end = range ? range.to + 1 : Infinity;
        setRecords(page.records.filter((r) => r.t <= end));
        cursor.current = page.next_id;
        setDropped(page.dropped);
        setLoad("ready");
        setFollow(!range);
      })
      .catch((e: unknown) => {
        if (controller.signal.aborted) return;
        const code = e instanceof ApiError ? e.status : undefined;
        setLoad(
          code === 404 || code === 405
            ? "missing"
            : code === 401 || code === 403
              ? "denied"
              : "error",
        );
      });
    return () => controller.abort();
  }, [connection.baseUrl, connection.token, level, query, sinceFor, range]);

  // Live tail: SSE with the bearer header; polls the same cursor when streaming is not offered.
  const streaming = load === "ready" && !paused && !windowed;
  useEffect(() => {
    if (!streaming) {
      setLive("off");
      return;
    }
    const controller = new AbortController();
    const filters = {
      level: level === "all" ? undefined : level,
      q: query || undefined,
    };
    const sleep = (ms: number) =>
      new Promise<void>((done) => {
        const id = setTimeout(done, ms);
        controller.signal.addEventListener("abort", () => {
          clearTimeout(id);
          done();
        });
      });
    (async () => {
      let poll = false;
      setLive("connecting");
      while (!controller.signal.aborted) {
        if (!poll) {
          let batch: LogRecord[] = [];
          const flush = setInterval(() => {
            if (!batch.length) return;
            const out = batch;
            batch = [];
            append(out);
          }, 150);
          try {
            await streamLogs(
              connection,
              { ...filters, since_id: cursor.current },
              (r) => {
                cursor.current = Math.max(cursor.current, r.id);
                batch.push(r);
              },
              controller.signal,
              () => setLive("live"),
            );
          } catch (e) {
            const code = e instanceof ApiError ? e.status : undefined;
            if (code === 404 || code === 405) poll = true;
          } finally {
            clearInterval(flush);
            append(batch);
          }
          if (controller.signal.aborted) return;
          if (!poll) setLive("reconnecting");
        } else {
          setLive("polling");
          try {
            const page = await fetchLogs(
              connection,
              { ...filters, since_id: cursor.current, limit: RING_LIMIT },
              controller.signal,
            );
            cursor.current = Math.max(cursor.current, page.next_id);
            append(page.records);
          } catch {
            /* try again on the next tick */
          }
        }
        await sleep(poll ? 2000 : 1500);
      }
    })();
    return () => controller.abort();
  }, [streaming, connection.baseUrl, connection.token, level, query, append]);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && follow) el.scrollTop = el.scrollHeight;
  }, [records, follow, load]);

  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    const atEnd = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    setFollow(atEnd);
    if (atEnd) setUnseen(0);
  };
  const jump = () => {
    setFollow(true);
    setUnseen(0);
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  };
  const flash = (message: string) => {
    setNote(message);
    setTimeout(() => setNote((n) => (n === message ? null : n)), 2200);
  };
  const copy = (value: string, message: string) => {
    void navigator.clipboard
      ?.writeText(value)
      .then(() => flash(message))
      .catch(() => flash(t("logs.copyFailed")));
  };
  const lines = () => records.map(formatLine).join("\n");
  const download = () =>
    saveBlob(
      `yunshu-logs-${new Date().toISOString().replace(/[:.]/g, "-")}.log`,
      new Blob([lines() + "\n"], { type: "text/plain;charset=utf-8" }),
    );

  const liveText =
    live === "live"
      ? t("logs.live.live")
      : live === "polling"
        ? t("logs.live.polling")
        : live === "reconnecting"
          ? t("logs.live.reconnecting")
          : live === "connecting"
            ? t("logs.live.connecting")
            : windowed
              ? t("logs.live.window")
              : paused
                ? t("logs.live.paused")
                : "";

  if (load === "missing" || load === "denied" || load === "error") {
    const kind = load === "missing" ? "missing" : load;
    return (
      <DashboardPage width="7xl" data-testid="logs">
        <PageHeader
          title={t("logs.page.title")}
          description={t("logs.page.description")}
        />
        {/* i18n-keys: logs.state. */}
        <UnavailableNotice
          data-testid="logs-unavailable"
          title={t(`logs.state.${kind}Title`)}
          description={t(`logs.state.${kind}Body`)}
        />
      </DashboardPage>
    );
  }

  return (
    <DashboardPage width="7xl" data-testid="logs">
      <PageHeader
        title={t("logs.page.title")}
        description={t("logs.page.description")}
        actions={
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="secondary"
              disabled={!records.length}
              onClick={() =>
                copy(lines(), t("logs.copiedAll", { count: records.length }))
              }
            >
              <Copy size={13} />
              {t("logs.action.copyVisible")}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              disabled={!records.length}
              onClick={download}
            >
              <Download size={13} />
              {t("logs.action.download")}
            </Button>
          </div>
        }
      />
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Input
            className="w-64 max-w-full"
            aria-label={t("logs.search.aria")}
            icon={<Search size={14} />}
            placeholder={t("logs.search.placeholder")}
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <SegmentedTray<LevelChoice>
            aria-label={t("logs.level.aria")}
            value={level}
            onChange={setLevel}
            options={LEVELS.map((v) => ({
              value: v,
              label: v === "all" ? t("logs.level.all") : levelName(v),
            }))}
          />
          <CustomSelect
            className="w-40 [&_button]:h-8 [&_button]:text-xs"
            value={windowed ? "window" : span}
            onChange={(v) => {
              if (v === "window") return;
              if (windowed) location.hash = "#/logs";
              setSpan(v);
            }}
            options={[
              { value: "all", label: t("logs.span.all") },
              { value: "5m", label: t("logs.span.5m") },
              { value: "1h", label: t("logs.span.1h") },
              ...(range
                ? [
                    {
                      value: "window",
                      label: t("logs.span.window", {
                        from: clock(range.from * 1000),
                        to: clock(range.to * 1000),
                      }),
                    },
                  ]
                : []),
            ]}
          />
        </div>
        <div className="flex items-center gap-2">
          <span
            role="status"
            data-testid="logs-live"
            className="order-last min-w-24 text-xs text-muted-foreground"
          >
            {liveText}
          </span>
          <Button
            size="sm"
            variant="secondary"
            disabled={windowed}
            title={windowed ? t("logs.pause.windowWhy") : undefined}
            onClick={() => setPaused((p) => !p)}
          >
            {paused ? <Play size={13} /> : <Pause size={13} />}
            {paused ? t("logs.action.resume") : t("logs.action.pause")}
          </Button>
        </div>
      </div>
      <Card className="relative overflow-hidden p-0">
        {load === "ready" && records.length > 0 ? (
          <div
            ref={scroller}
            onScroll={onScroll}
            data-testid="logs-scroller"
            tabIndex={0}
            aria-label={t("logs.list.aria")}
            className="h-[60dvh] min-h-80 overflow-y-auto py-1"
          >
            {records.map((r) => (
              <div
                key={r.id}
                className="group flex flex-wrap items-baseline gap-x-3 px-4 py-1 text-xs hover:bg-muted/40"
              >
                <span className="w-[6.75rem] shrink-0 tabular-nums text-muted-foreground">
                  {stamp(r)}
                </span>
                <span className={`w-14 shrink-0 ${levelTone(r.level)}`}>
                  {levelName(r.level)}
                </span>
                <span
                  title={r.logger}
                  className="hidden max-w-40 shrink-0 truncate text-muted-foreground md:inline"
                >
                  {r.logger}
                </span>
                <span className="min-w-0 basis-full break-words font-mono md:basis-0 md:flex-1">
                  {r.msg}
                </span>
                <IconButton
                  icon={<Copy size={12} />}
                  label={t("logs.action.copyLine")}
                  onClick={() => copy(formatLine(r), t("logs.copiedLine"))}
                  className="ml-auto h-5 w-5 min-h-0 min-w-0 text-muted-foreground opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-60"
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="min-h-80 py-8">
            {load === "loading" ? (
              <p className="px-4 text-center text-sm text-muted-foreground">
                {t("logs.state.loading")}
              </p>
            ) : (
              <EmptyState
                size="inline"
                title={t("logs.state.emptyTitle")}
                description={
                  query || level !== "all" || span !== "all" || windowed
                    ? t("logs.state.emptyFiltered")
                    : t("logs.state.emptyBody")
                }
              />
            )}
          </div>
        )}
        {!follow && load === "ready" && records.length > 0 && (
          <Button
            size="sm"
            className="absolute bottom-12 right-4 shadow-md"
            onClick={jump}
          >
            <ArrowDownToLine size={13} />
            {unseen > 0
              ? t("logs.action.jumpNew", { count: number(unseen, 0) })
              : t("logs.action.jump")}
          </Button>
        )}
        <div className="flex min-h-9 flex-wrap items-center justify-between gap-x-4 gap-y-1 border-t border-border/60 bg-muted/30 px-4 py-2 text-xs text-muted-foreground">
          <span className="tabular-nums">
            {t("logs.footer.counts", {
              count: number(records.length, 0),
              capacity: number(RING_LIMIT, 0),
            })}
            {dropped > 0
              ? ` · ${t("logs.footer.dropped", { count: number(dropped, 0) })}`
              : ""}
            {records.length
              ? ` · ${t("logs.footer.span", {
                  from: dateTime(records[0].t * 1000),
                })}`
              : ""}
          </span>
          <span role="status">{note ?? ""}</span>
        </div>
        <p
          className="border-t border-border/60 px-4 py-2 text-xs text-muted-foreground"
          data-testid="logs-redaction"
        >
          {t("logs.note.redaction", { capacity: number(RING_LIMIT, 0) })}
        </p>
      </Card>
    </DashboardPage>
  );
}
