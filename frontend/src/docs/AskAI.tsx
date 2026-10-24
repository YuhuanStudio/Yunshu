// # Upstream (inspired): YuhuanStudio/Yunxin 6d671e87871b, Apache-2.0;
// frontend/src/components/docs/ai-search.tsx. UX reference; independent local transport.
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  Textarea,
  Spinner,
} from "@yuhuanowo/yunui";
import {
  Bot,
  Copy,
  Check,
  ThumbsUp,
  ThumbsDown,
  Send,
  Square,
} from "lucide-react";
import type { Connection } from "../api";
import { t, tr, useLocale } from "../i18n";
import { docHref } from "./links";
import {
  docsRequest,
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
}: {
  open: boolean;
  onClose: () => void;
  connection: Connection;
  initialQuestion: string;
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
  const tabId = useRef(crypto.randomUUID());
  const scroll = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
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
  useEffect(() => {
    if (pinned.current && scroll.current)
      scroll.current.scrollTop = scroll.current.scrollHeight;
  }, [turns, running]);

  async function send(question = input.trim()) {
    if (!question || controller.current || !state?.ready) return;
    const abort = new AbortController();
    controller.current = abort;
    setRunning(true);
    setError("");
    setInput("");
    pinned.current = true;
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
        { question, language: locale, tab_id: tabId.current, history },
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
        className="flex h-[min(46rem,90dvh)] max-w-2xl flex-col gap-3 overflow-hidden"
        data-testid="docs-ask"
      >
        <DialogTitle className="flex items-center gap-2">
          <Bot size={18} />
          {t("docs.ask.title")}
        </DialogTitle>
        <DialogDescription>{t("docs.ask.subtitle")}</DialogDescription>
        <div
          ref={scroll}
          onScroll={() => {
            const el = scroll.current;
            if (el)
              pinned.current =
                el.scrollHeight - el.scrollTop - el.clientHeight < 64;
          }}
          className="min-h-0 flex-1 space-y-5 overflow-y-auto overscroll-contain pr-2"
          role="log"
          aria-label={t("docs.ask.answers")}
        >
          {!turns.length && (
            <div className="space-y-3 py-6 text-center">
              <p className="font-medium">{t("docs.ask.welcome")}</p>
              <p className="text-sm text-muted-foreground">
                {t("docs.ask.local")}
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {["start", "models", "api"].map((key) => (
                  <Button
                    key={key}
                    size="sm"
                    variant="outline"
                    disabled={!state?.ready || running}
                    onClick={() => void send(tr(`docs.ask.suggestion.${key}`))}
                  >
                    {tr(`docs.ask.suggestion.${key}`)}
                  </Button>
                ))}
              </div>
            </div>
          )}
          {unavailable && (
            <div
              role="status"
              className="rounded-xl border border-border bg-(--bg-elevated) p-4 text-sm"
            >
              <p>{tr(`docs.ask.error.${unavailable}`)}</p>
              {["no_engine", "no_model"].includes(unavailable) && (
                <pre className="mt-3 overflow-x-auto">
                  <code>
                    {state?.command ?? "yunshu serve -m /path/to/mlx-model"}
                  </code>
                </pre>
              )}
            </div>
          )}
          {turns.map((turn, i) => (
            <section key={i} className="space-y-3 motion-enter">
              <p className="ml-8 whitespace-pre-wrap break-words rounded-xl bg-(--bg-elevated) p-3 text-sm">
                {turn.question}
              </p>
              <div
                className="min-w-0 overflow-x-auto"
                data-testid="docs-ask-answer"
              >
                <Suspense
                  fallback={
                    <p className="whitespace-pre-wrap">{turn.answer}</p>
                  }
                >
                  <Markdown
                    content={turn.answer}
                    streaming={running && i === turns.length - 1}
                  />
                </Suspense>
                {running && i === turns.length - 1 && !turn.answer && (
                  <p className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Spinner className="size-4" />
                    {t("docs.ask.waiting")}
                  </p>
                )}
              </div>
              {turn.sources.length > 0 && (
                <nav
                  aria-label={t("docs.ask.sources")}
                  className="grid gap-2 sm:grid-cols-2"
                >
                  {turn.sources.map((source, n) => (
                    <a
                      key={source.id}
                      href={docHref(source.slug, source.anchor)}
                      onClick={onClose}
                      className="rounded-lg border border-border px-3 py-2 text-xs transition-colors duration-(--dur-fast) hover:bg-(--bg-elevated)"
                    >
                      <span className="text-muted-foreground">[{n + 1}] </span>
                      {source.title}
                      {source.heading && (
                        <span className="mt-1 block truncate text-muted-foreground">
                          {source.heading}
                        </span>
                      )}
                    </a>
                  ))}
                </nav>
              )}
              {turn.answer && (
                <div className="flex gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    aria-label={t("docs.ask.copy")}
                    onClick={async () => {
                      try {
                        await navigator.clipboard.writeText(turn.answer);
                        setCopied(i);
                      } catch {
                        setError("copy_failed");
                      }
                    }}
                  >
                    {copied === i ? <Check size={14} /> : <Copy size={14} />}
                  </Button>
                  {turn.complete && (
                    <>
                      <Button
                        size="sm"
                        variant="ghost"
                        aria-label={t("docs.ask.good")}
                        aria-pressed={turn.feedback === "good"}
                        onClick={() => void feedback(turn, "good")}
                      >
                        <ThumbsUp size={14} />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        aria-label={t("docs.ask.bad")}
                        aria-pressed={turn.feedback === "bad"}
                        onClick={() => void feedback(turn, "bad")}
                      >
                        <ThumbsDown size={14} />
                      </Button>
                    </>
                  )}
                </div>
              )}
            </section>
          ))}
        </div>
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {tr(`docs.ask.error.${error}`)}
          </p>
        )}
        {(unavailable || error) && (
          <Button
            size="sm"
            variant="outline"
            disabled={running}
            onClick={() => {
              setError("");
              setRetry((v) => v + 1);
            }}
          >
            {t("docs.ask.retry")}
          </Button>
        )}
        <form
          className="flex items-end gap-2 border-t border-border pt-3"
          onSubmit={(e) => {
            e.preventDefault();
            void send();
          }}
        >
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={t("docs.ask.placeholder")}
            aria-label={t("docs.ask.placeholder")}
            maxLength={state?.question ?? 2000}
            rows={2}
            disabled={!state?.ready}
            onKeyDown={(e) => {
              if (
                e.key === "Enter" &&
                !e.shiftKey &&
                !e.nativeEvent.isComposing
              ) {
                e.preventDefault();
                void send();
              }
            }}
          />
          {running ? (
            <Button
              type="button"
              variant="outline"
              onClick={stop}
              aria-label={t("docs.ask.stop")}
            >
              <Square size={16} />
            </Button>
          ) : (
            <Button
              type="submit"
              disabled={!state?.ready || !input.trim()}
              aria-label={t("docs.ask.send")}
            >
              <Send size={16} />
            </Button>
          )}
        </form>
        <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
          <p>{t("docs.ask.disclaimer")}</p>
          <Button
            size="sm"
            variant="ghost"
            disabled={running || !turns.length}
            onClick={() => {
              setTurns([]);
              setError("");
            }}
          >
            {t("docs.ask.clear")}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
