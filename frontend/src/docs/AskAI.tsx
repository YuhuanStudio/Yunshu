// # Upstream (inspired): YuhuanStudio/Yunxin 6d671e87871b, Apache-2.0;
// frontend/src/components/docs/ai-search.tsx. UX reference; independent local transport.
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  IconButton,
  Spinner,
} from "@yuhuanowo/yunui";
import { ChatComposer } from "@yuhuanowo/yunui/chat";
import { Bot, Copy, Check, ThumbsUp, ThumbsDown } from "lucide-react";
import type { Connection } from "../api";
import { t, tr, useLocale } from "../i18n";
import { ChatEmpty, ChatSurface, ChatTurn } from "../chat/ChatSurface";
import { docHref } from "./links";
import {
  docsRequest,
  makeTabId,
  readAskStream,
  type AskState,
  type ChatMessage,
  type Source,
} from "./ask-api";

const Markdown = lazy(() =>
  import("@yuhuanowo/yunui/content").then((m) => ({
    default: m.MarkdownRenderer,
  })),
);
type Turn = {
  question: string;
  answer: string;
  id?: string;
  sources: Source[];
  feedback?: "good" | "bad";
  complete?: boolean;
};
const errorKey = (code: string) => {
  const known = [
    "measurement_paused",
    "no_engine",
    "no_model",
    "question_too_long",
    "history_too_long",
    "ask_in_flight",
    "context_too_small",
    "docs_index_missing",
    "history_disabled",
    "stream_incomplete",
    "auth_required",
  ];
  return known.includes(code) ? code : "engine_error";
};

export default function AskAI({
  open,
  onClose,
  connection,
  initialQuestion,
  variant = "dialog",
}: {
  open: boolean;
  onClose: () => void;
  connection: Connection;
  initialQuestion: string;
  /** `panel` renders the conversation bare, to dock it (fumadocs `aiChat.panel`); the host owns its frame. */
  variant?: "dialog" | "panel";
}) {
  const locale = useLocale();
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [state, setState] = useState<AskState | null>(null);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const [retry, setRetry] = useState(0);
  const [copied, setCopied] = useState<number | null>(null);
  const controller = useRef<AbortController | null>(null);
  const [tabId] = useState(makeTabId);
  const stop = () => controller.current?.abort();
  useEffect(
    () => () => {
      controller.current?.abort();
    },
    [],
  );
  useEffect(() => {
    stop();
    setTurns([]);
    setError("");
    setState(null);
  }, [locale, connection.baseUrl, connection.token]);
  useEffect(() => {
    if (open && initialQuestion) setInput(initialQuestion);
  }, [open, initialQuestion]);
  useEffect(() => {
    if (!open) {
      stop();
      return;
    }
    const abort = new AbortController();
    docsRequest(connection, "state", undefined, abort.signal)
      .then((r) => r.json())
      .then(setState)
      .catch((e) => {
        if (!abort.signal.aborted) setError(errorKey(e.message));
      });
    return () => abort.abort();
  }, [open, retry, locale, connection.baseUrl, connection.token]);

  async function send(question = input.trim()) {
    if (!question || controller.current || !state?.ready) return;
    const abort = new AbortController();
    controller.current = abort;
    setRunning(true);
    setError("");
    setInput("");
    const at = turns.length;
    const history: ChatMessage[] = turns
      .filter((turn) => turn.complete)
      .flatMap((turn) => [
        { role: "user" as const, content: turn.question },
        {
          role: "assistant" as const,
          content: turn.answer.slice(0, state.question),
        },
      ])
      .slice(-(state.history || 1));
    if (!state.history) history.length = 0;
    setTurns((old) => [...old, { question, answer: "", sources: [] }]);
    const update = (fn: (turn: Turn) => Turn) =>
      setTurns((old) => old.map((turn, i) => (i === at ? fn(turn) : turn)));
    try {
      const response = await docsRequest(
        connection,
        "ask",
        { question, language: locale, tab_id: tabId, history },
        abort.signal,
      );
      await readAskStream(response, (event) => {
        if (abort.signal.aborted) return;
        if (event.type === "sources")
          update((turn) => ({ ...turn, id: event.id, sources: event.sources }));
        if (event.type === "token")
          update((turn) => ({ ...turn, answer: turn.answer + event.text }));
        if (event.type === "done")
          update((turn) => ({ ...turn, complete: true }));
      });
    } catch (e) {
      if (!abort.signal.aborted)
        setError(errorKey(e instanceof Error ? e.message : "engine_error"));
    } finally {
      if (controller.current === abort) {
        controller.current = null;
        setRunning(false);
      }
    }
  }
  async function feedback(turn: Turn, opinion: "good" | "bad") {
    if (!turn.id) return;
    try {
      await docsRequest(connection, "feedback", { ask_id: turn.id, opinion });
      setTurns((old) =>
        old.map((item) =>
          item.id === turn.id ? { ...item, feedback: opinion } : item,
        ),
      );
    } catch (e) {
      setError(errorKey(e instanceof Error ? e.message : "engine_error"));
    }
  }
  const unavailable =
    state && !state.ready ? errorKey(state.code ?? "no_engine") : null;
  const limit = state?.question ?? 2000;
  const dock = (
    <>
      {unavailable && (
        <div
          role="status"
          className="mb-2 rounded-xl border border-border bg-(--bg-elevated) p-3 text-sm"
        >
          <p>{tr(`docs.ask.error.${unavailable}`)}</p>
          {["no_engine", "no_model"].includes(unavailable) && (
            <pre className="mt-2 overflow-x-auto">
              <code>
                {state?.command ?? "yunshu serve -m /path/to/mlx-model"}
              </code>
            </pre>
          )}
        </div>
      )}
      {error && (
        <p role="alert" className="mb-2 text-sm text-destructive">
          {tr(`docs.ask.error.${error}`)}
        </p>
      )}
      {(unavailable || error) && (
        <Button
          size="sm"
          variant="outline"
          className="mb-2"
          disabled={running}
          onClick={() => {
            setError("");
            setRetry((v) => v + 1);
          }}
        >
          {t("docs.ask.retry")}
        </Button>
      )}
      <ChatComposer
        value={input}
        onChange={(value) => setInput(value.slice(0, limit))}
        onSend={() => void send()}
        onStop={stop}
        loading={running}
        disabled={!state?.ready}
        sendDisabled={!state?.ready || !input.trim()}
        maxRows={5}
        placeholder={t("docs.ask.placeholder")}
        labels={{ send: t("docs.ask.send"), stop: t("docs.ask.stop") }}
      />
      <div className="mt-1.5 flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
        <p className="min-w-0 truncate">{t("docs.ask.disclaimer")}</p>
        <Button
          size="sm"
          variant="ghost"
          className="shrink-0"
          disabled={running || !turns.length}
          onClick={() => {
            setTurns([]);
            setError("");
          }}
        >
          {t("docs.ask.clear")}
        </Button>
      </div>
    </>
  );
  const panel = variant === "panel";
  const body = (
    <>
      <div className="shrink-0 px-4 pb-1 pr-12 pt-4 sm:px-6 sm:pt-5">
        {panel ? (
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <Bot size={16} />
            {t("docs.ask.title")}
          </h2>
        ) : (
          <DialogTitle className="flex items-center gap-2 text-base font-semibold">
            <Bot size={16} />
            {t("docs.ask.title")}
          </DialogTitle>
        )}
        {panel ? (
          <p className="text-xs text-muted-foreground">
            {t("docs.ask.subtitle")}
          </p>
        ) : (
          <DialogDescription className="text-xs">
            {t("docs.ask.subtitle")}
          </DialogDescription>
        )}
      </div>
      <ChatSurface
        jumpToLatestLabel={t("playground.jumpLatest")}
        empty={
          <ChatEmpty
            icon={<Bot size={22} />}
            title={t("docs.ask.welcome")}
            description={t("docs.ask.local")}
            suggestions={["start", "models", "api"].map((key) =>
              tr(`docs.ask.suggestion.${key}`),
            )}
            onPick={(text) => void send(text)}
            disabled={!state?.ready || running}
            groupLabel={t("playground.empty.examples")}
          />
        }
        dock={dock}
      >
        {turns.length ? (
          <div role="log" aria-label={t("docs.ask.answers")}>
            {turns.map((turn, i) => {
              const last = running && i === turns.length - 1;
              return (
                <section key={i} className="space-y-1 motion-enter">
                  <ChatTurn role="user">{turn.question}</ChatTurn>
                  <ChatTurn
                    role="assistant"
                    name={t("docs.ask.title")}
                    avatar={<Bot size={16} aria-hidden="true" />}
                    actions={
                      turn.answer ? (
                        <>
                          <IconButton
                            icon={
                              copied === i ? (
                                <Check size={14} />
                              ) : (
                                <Copy size={14} />
                              )
                            }
                            label={t("docs.ask.copy")}
                            onClick={async () => {
                              try {
                                await navigator.clipboard.writeText(
                                  turn.answer,
                                );
                                setCopied(i);
                              } catch {
                                setError("copy_failed");
                              }
                            }}
                          />
                          {turn.complete && (
                            <>
                              <IconButton
                                icon={<ThumbsUp size={14} />}
                                label={t("docs.ask.good")}
                                aria-pressed={turn.feedback === "good"}
                                onClick={() => void feedback(turn, "good")}
                              />
                              <IconButton
                                icon={<ThumbsDown size={14} />}
                                label={t("docs.ask.bad")}
                                aria-pressed={turn.feedback === "bad"}
                                onClick={() => void feedback(turn, "bad")}
                              />
                            </>
                          )}
                        </>
                      ) : undefined
                    }
                    footer={
                      turn.sources.length > 0 ? (
                        <nav
                          aria-label={t("docs.ask.sources")}
                          className="flex flex-wrap gap-1.5"
                        >
                          {turn.sources.map((source, n) => (
                            <Button
                              key={source.id}
                              asChild
                              size="sm"
                              variant="secondary"
                              className="h-auto max-w-full justify-start py-1 text-left"
                            >
                              <a
                                href={docHref(source.slug, source.anchor)}
                                onClick={onClose}
                              >
                                <span className="text-muted-foreground">
                                  [{n + 1}]
                                </span>
                                <span className="min-w-0 truncate">
                                  {source.title}
                                  {source.heading && (
                                    <span className="text-muted-foreground">
                                      {" · "}
                                      {source.heading}
                                    </span>
                                  )}
                                </span>
                              </a>
                            </Button>
                          ))}
                        </nav>
                      ) : undefined
                    }
                  >
                    <div
                      className="min-w-0 overflow-x-auto"
                      data-testid="docs-ask-answer"
                    >
                      <Suspense
                        fallback={
                          <p className="whitespace-pre-wrap">{turn.answer}</p>
                        }
                      >
                        <Markdown content={turn.answer} streaming={last} />
                      </Suspense>
                      {last && !turn.answer && (
                        <p className="flex items-center gap-2 text-sm text-muted-foreground">
                          <Spinner className="size-4" />
                          {t("docs.ask.waiting")}
                        </p>
                      )}
                    </div>
                  </ChatTurn>
                </section>
              );
            })}
          </div>
        ) : undefined}
      </ChatSurface>
    </>
  );
  if (panel)
    return (
      <div
        className="flex h-full min-h-0 flex-col"
        data-testid="docs-ask"
        hidden={!open}
      >
        {body}
      </div>
    );
  return (
    <Dialog
      open={open}
      onOpenChange={(value) => {
        if (!value) {
          stop();
          onClose();
        }
      }}
    >
      <DialogContent
        className="flex h-[min(44rem,92dvh)] max-w-2xl flex-col gap-0 overflow-hidden p-0 max-sm:h-[calc(100dvh-1rem)] max-sm:w-[calc(100%-1rem)]"
        data-testid="docs-ask"
      >
        {body}
      </DialogContent>
    </Dialog>
  );
}
