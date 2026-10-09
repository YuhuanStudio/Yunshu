import { useMemo, useRef, useState } from "react";
import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  StatusIndicator,
  Textarea,
} from "@yuhuanowo/yunui";
import { ApiError, requestJson, type Connection } from "./api";
import { parseSchemaText, unsupportedKeywords } from "./json-schema-lite";
import { t, useLocale } from "./i18n/index.ts";
import { number } from "./i18n/format";
import { SegmentedTray } from "./SegmentedTray";
import {
  checkStructured,
  checkToolDefs,
  nextMessages,
  readRound,
  type ChatMessage,
  type Round,
  type ToolDef,
} from "./tools-round";

const MAX_ROUNDS = 6;
const DEFAULT_TOOLS = JSON.stringify(
  [
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "weather", // i18n-ignore
        parameters: {
          type: "object",
          required: ["city"],
          properties: {
            city: { type: "string" },
            unit: { enum: ["c", "f"] },
          },
        },
      },
    },
  ],
  null,
  2,
);
const DEFAULT_SCHEMA = JSON.stringify(
  {
    type: "object",
    required: ["city", "temperature_c"],
    additionalProperties: false,
    properties: { city: { type: "string" }, temperature_c: { type: "number" } },
  },
  null,
  2,
);

type Tab = "tools" | "schema";
type Done = { round: Round; results: Record<string, string> };

/** The text of one definitions error code ("kind:detail"). */
function defsError(code: string): string {
  const [kind, ...rest] = code.split(":");
  const detail = rest.join(":");
  switch (kind) {
    case "json":
      return t("playground.tools.defs.err.json", { detail });
    case "array":
      return t("playground.tools.defs.err.array");
    case "shape":
      return t("playground.tools.defs.err.shape", { n: detail });
    case "name":
      return t("playground.tools.defs.err.name", { n: detail });
    case "duplicate":
      return t("playground.tools.defs.err.duplicate", { name: detail });
    default:
      return t("playground.tools.defs.err.parameters", { name: detail });
  }
}

const errText = (e: unknown) =>
  e instanceof ApiError ? e.publicMessage : e instanceof Error ? e.message : "";

/**
 * Checks tool definitions and a structured-output schema before sending, then runs the rounds
 * against the engine and shows exactly what came back: each tool call, whether its arguments
 * follow the tool's schema, and the results you send back (as tool messages, never as assistant
 * text). A cancelled run keeps the rounds that finished.
 */
export function ToolsTester({
  open,
  onClose,
  connection,
  model,
}: {
  open: boolean;
  onClose: () => void;
  connection: Connection;
  model: string;
}) {
  useLocale();
  const [tab, setTab] = useState<Tab>("tools");
  const [defs, setDefs] = useState(DEFAULT_TOOLS);
  const [prompt, setPrompt] = useState(t("playground.tools.prompt.default"));
  const [schemaText, setSchemaText] = useState(DEFAULT_SCHEMA);
  const [rounds, setRounds] = useState<Done[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<{
    kind: "error" | "cancelled" | "max";
    text: string;
  } | null>(null);
  const [structured, setStructured] = useState<{ content: string } | null>(
    null,
  );
  const abort = useRef<AbortController | null>(null);
  const userCancelled = useRef(false);

  const checked = useMemo(() => checkToolDefs(defs), [defs]);
  const schema = useMemo(() => parseSchemaText(schemaText), [schemaText]);
  const unsupported = useMemo(
    () => ("schema" in schema ? unsupportedKeywords(schema.schema) : []),
    [schema],
  );
  const tools: ToolDef[] = checked.ok ? checked.tools : [];
  const last = rounds.at(-1);
  const waiting = !!last && last.round.toolCalls.length > 0 && !busy;

  async function send(body: Record<string, unknown>) {
    const controller = new AbortController();
    abort.current = controller;
    userCancelled.current = false;
    setBusy(true);
    setNote(null);
    try {
      return await requestJson<unknown>(connection, "/chat/completions", {
        method: "POST",
        body,
        signal: controller.signal,
        timeoutMs: 180_000,
      });
    } finally {
      setBusy(false);
      abort.current = null;
    }
  }

  async function runRound(history: ChatMessage[], doneBefore: Done[]) {
    try {
      const body = await send({
        model,
        messages: history,
        tools,
        temperature: 0,
        max_tokens: 512,
        stream: false,
      });
      const round = readRound(body, tools);
      if (!round) throw new Error("no choice");
      const results = Object.fromEntries(
        round.toolCalls.map((c) => [c.id, '{"ok": true}']),
      );
      setMessages(history);
      setRounds([...doneBefore, { round, results }]);
      if (round.toolCalls.length && doneBefore.length + 1 >= MAX_ROUNDS)
        setNote({
          kind: "max",
          text: t("playground.tools.maxRounds", { n: MAX_ROUNDS }),
        });
    } catch (e) {
      if (userCancelled.current) {
        setNote({
          kind: "cancelled",
          text: t("playground.tools.cancelled", { n: doneBefore.length }),
        });
      } else
        setNote({
          kind: "error",
          text: t("playground.tools.error", { error: errText(e) }),
        });
    }
  }

  const start = () => {
    setRounds([]);
    void runRound([{ role: "user", content: prompt }], []);
  };
  const resume = () => {
    if (!last) return;
    void runRound(nextMessages(messages, last.round, last.results), rounds);
  };
  const reset = () => {
    userCancelled.current = true;
    abort.current?.abort();
    setRounds([]);
    setMessages([]);
    setNote(null);
    setStructured(null);
  };

  async function runStructured() {
    setStructured(null);
    try {
      const body = (await send({
        model,
        messages: [{ role: "user", content: prompt }],
        temperature: 0,
        max_tokens: 512,
        stream: false,
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "result",
            strict: true,
            schema: "schema" in schema ? schema.schema : {},
          },
        },
      })) as { choices?: { message?: { content?: unknown } }[] };
      const content = body.choices?.[0]?.message?.content;
      setStructured({ content: typeof content === "string" ? content : "" });
    } catch (e) {
      const cancelled = userCancelled.current;
      setNote({
        kind: cancelled ? "cancelled" : "error",
        text: cancelled
          ? t("playground.tools.stopped")
          : t("playground.tools.error", { error: errText(e) }),
      });
    }
  }

  const sendBlock = !model
    ? t("playground.tools.disabled.noModel")
    : !prompt.trim()
      ? t("playground.tools.disabled.empty")
      : tab === "tools" && !checked.ok
        ? t("playground.tools.disabled.invalid")
        : tab === "schema" && "error" in schema
          ? t("playground.tools.schema.err")
          : null;
  const verdict = structured
    ? checkStructured(structured.content, schemaText)
    : null;

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent
        closeLabel={t("playground.params.close")}
        className="max-h-[90dvh] max-w-3xl overflow-y-auto"
        data-testid="tools-tester"
      >
        <DialogTitle>{t("playground.tools.title")}</DialogTitle>
        <DialogDescription>{t("playground.tools.desc")}</DialogDescription>
        <SegmentedTray
          aria-label={t("playground.tools.title")}
          value={tab}
          onChange={setTab}
          fillOnPhone
          options={[
            { value: "tools", label: t("playground.tools.tab.tools") },
            { value: "schema", label: t("playground.tools.tab.schema") },
          ]}
        />
        {tab === "tools" ? (
          <div className="space-y-1.5">
            <label
              className="text-xs text-muted-foreground"
              htmlFor="tools-defs"
            >
              {t("playground.tools.defs.label")}
            </label>
            <Textarea
              id="tools-defs"
              className="min-h-40 font-mono text-xs"
              value={defs}
              spellCheck={false}
              onChange={(e) => setDefs(e.target.value)}
            />
            <p
              className={`text-xs ${checked.ok ? "text-muted-foreground" : "text-error"}`}
              data-testid="defs-status"
              role={checked.ok ? undefined : "alert"}
            >
              {checked.ok
                ? t("playground.tools.defs.ok", { n: checked.tools.length })
                : checked.errors.map(defsError).join(" ")}
            </p>
          </div>
        ) : (
          <div className="space-y-1.5">
            <label
              className="text-xs text-muted-foreground"
              htmlFor="tools-schema"
            >
              {t("playground.tools.schema.label")}
            </label>
            <Textarea
              id="tools-schema"
              className="min-h-40 font-mono text-xs"
              value={schemaText}
              spellCheck={false}
              onChange={(e) => setSchemaText(e.target.value)}
            />
            {"error" in schema && (
              <p
                className="text-xs text-error"
                role="alert"
                data-testid="schema-status"
              >
                {t("playground.tools.schema.err")}
              </p>
            )}
            {unsupported.length > 0 && (
              <p className="text-xs text-muted-foreground">
                {t("playground.tools.schema.unsupported", {
                  list: unsupported.join(", "),
                })}
              </p>
            )}
          </div>
        )}
        <div className="space-y-1.5">
          <label
            className="text-xs text-muted-foreground"
            htmlFor="tools-prompt"
          >
            {t("playground.tools.prompt.label")}
          </label>
          <Textarea
            id="tools-prompt"
            className="min-h-16 text-sm"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span title={sendBlock ?? undefined}>
            <Button
              size="sm"
              disabled={busy || !!sendBlock}
              onClick={tab === "tools" ? start : () => void runStructured()}
            >
              {tab === "tools"
                ? t("playground.tools.run")
                : t("playground.tools.schema.run")}
            </Button>
          </span>
          {busy && (
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                userCancelled.current = true;
                abort.current?.abort();
              }}
            >
              {t("playground.tools.cancel")}
            </Button>
          )}
          {(rounds.length > 0 || structured || note) && !busy && (
            <Button size="sm" variant="ghost" onClick={reset}>
              {t("playground.tools.reset")}
            </Button>
          )}
        </div>
        {note && (
          <p
            role={note.kind === "error" ? "alert" : "status"}
            className={`text-sm ${note.kind === "error" ? "text-error" : "text-muted-foreground"}`}
            data-testid="tools-note"
          >
            {note.text}
          </p>
        )}
        {tab === "tools" && (
          <ol className="space-y-3" data-testid="tool-rounds">
            {rounds.map((d, i) => {
              const r = d.round;
              return (
                <li
                  key={i}
                  className="space-y-2 rounded-lg bg-(--bg-elevated) p-3"
                  data-testid="tool-round"
                >
                  <p className="text-sm font-semibold">
                    {t("playground.tools.round", { n: i + 1 })}
                    <span className="ml-2 text-xs font-normal text-muted-foreground">
                      {r.toolCalls.length
                        ? t("playground.tools.round.calls", {
                            n: r.toolCalls.length,
                          })
                        : t("playground.tools.round.text")}
                      {r.finishReason &&
                        ` · ${t("playground.tools.round.finish", { reason: r.finishReason })}`}
                      {r.usage.prompt != null &&
                        r.usage.completion != null &&
                        ` · ${t("playground.tools.round.usage", {
                          prompt: number(r.usage.prompt, 0),
                          completion: number(r.usage.completion, 0),
                        })}`}
                    </span>
                  </p>
                  {r.toolCalls.map((c) => (
                    <div
                      key={c.id}
                      className="space-y-1.5"
                      data-testid="tool-call"
                      data-call={c.name}
                    >
                      <p className="flex flex-wrap items-center gap-2 text-sm">
                        <span className="font-mono">{c.name || "—"}</span>
                        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                          <StatusIndicator
                            status={
                              !c.known || c.parseError || c.issues.length
                                ? "busy"
                                : "online"
                            }
                          />
                          {!c.known
                            ? t("playground.tools.call.unknown")
                            : c.parseError
                              ? t("playground.tools.call.badJson")
                              : c.issues.length
                                ? t("playground.tools.call.invalid", {
                                    n: c.issues.length,
                                  })
                                : t("playground.tools.call.valid")}
                        </span>
                      </p>
                      <pre className="overflow-x-auto rounded bg-muted/40 p-2 font-mono text-xs">
                        {c.parseError == null
                          ? JSON.stringify(c.args, null, 2)
                          : c.rawArguments}
                      </pre>
                      {c.issues.length > 0 && (
                        <ul className="space-y-0.5 font-mono text-xs text-error">
                          {c.issues.map((x, k) => (
                            <li key={k}>
                              {x.path} {x.message}
                            </li>
                          ))}
                        </ul>
                      )}
                      {i === rounds.length - 1 && waiting && (
                        <>
                          <label
                            className="text-xs text-muted-foreground"
                            htmlFor={`res-${c.id}`}
                          >
                            {t("playground.tools.call.result", {
                              name: c.name,
                            })}
                          </label>
                          <Textarea
                            id={`res-${c.id}`}
                            className="min-h-12 font-mono text-xs"
                            value={d.results[c.id] ?? ""}
                            spellCheck={false}
                            onChange={(e) =>
                              setRounds((all) =>
                                all.map((x, k) =>
                                  k === i
                                    ? {
                                        ...x,
                                        results: {
                                          ...x.results,
                                          [c.id]: e.target.value,
                                        },
                                      }
                                    : x,
                                ),
                              )
                            }
                          />
                        </>
                      )}
                    </div>
                  ))}
                  {r.content.trim() && (
                    <div data-testid="tool-final">
                      <p className="text-xs text-muted-foreground">
                        {t("playground.tools.final")}
                      </p>
                      <p className="whitespace-pre-wrap text-sm">{r.content}</p>
                    </div>
                  )}
                </li>
              );
            })}
          </ol>
        )}
        {tab === "tools" && waiting && rounds.length < MAX_ROUNDS && (
          <div>
            <Button size="sm" onClick={resume}>
              {t("playground.tools.continue")}
            </Button>
          </div>
        )}
        {tab === "schema" && structured && verdict && (
          <div className="space-y-2" data-testid="schema-result">
            <p className="flex items-center gap-2 text-sm">
              <StatusIndicator
                status={verdict.state === "ok" ? "online" : "busy"}
              />
              {verdict.state === "ok"
                ? t("playground.tools.schema.ok")
                : verdict.state === "notJson"
                  ? t("playground.tools.schema.notJson", {
                      detail: verdict.message,
                    })
                  : verdict.state === "bad"
                    ? t("playground.tools.schema.bad", {
                        n: verdict.issues.length,
                      })
                    : t("playground.tools.schema.empty")}
            </p>
            {verdict.state === "bad" && (
              <ul className="space-y-0.5 font-mono text-xs text-error">
                {verdict.issues.map((x, k) => (
                  <li key={k}>
                    {x.path} {x.message}
                  </li>
                ))}
              </ul>
            )}
            <p className="text-xs text-muted-foreground">
              {t("playground.tools.raw")}
            </p>
            <pre className="overflow-x-auto rounded bg-muted/40 p-2 font-mono text-xs">
              {structured.content}
            </pre>
            <p className="text-xs text-muted-foreground">
              {t("playground.tools.schema.note")}
            </p>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
