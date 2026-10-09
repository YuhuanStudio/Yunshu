"use client";

import { ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { REPO } from "@/lib/site";
import type { Messages } from "@/lib/messages";

/** "Was this page helpful?": no backend. A "No" opens a prefilled GitHub issue. */
export function Feedback({ messages, path }: { messages: Messages; path: string }) {
  const [opinion, setOpinion] = useState<"good" | "bad" | null>(null);
  const [text, setText] = useState("");
  const issue = `${REPO}/issues/new?title=${encodeURIComponent(`Docs: ${path}`)}&body=${encodeURIComponent(`Page: ${path}\n\n${text}`)}&labels=documentation`;
  const btn = (active: boolean) =>
    `inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm font-medium [&_svg]:size-4 ${
      active ? "bg-fd-accent text-fd-accent-foreground [&_svg]:fill-current" : "text-fd-muted-foreground hover:bg-fd-accent/50"
    }`;
  return (
    <div className="mt-8 border-y py-3">
      <div className="flex flex-row flex-wrap items-center gap-2">
        <p className="pe-2 text-sm font-medium">{messages.feedbackQ}</p>
        <button type="button" className={btn(opinion === "good")} onClick={() => setOpinion("good")}>
          <ThumbsUp aria-hidden />
          {messages.good}
        </button>
        <button type="button" className={btn(opinion === "bad")} onClick={() => setOpinion("bad")}>
          <ThumbsDown aria-hidden />
          {messages.bad}
        </button>
      </div>
      {opinion === "good" ? <p className="mt-3 text-sm text-fd-muted-foreground">{messages.feedbackThanks}</p> : null}
      {opinion === "bad" ? (
        <div className="mt-3 flex flex-col gap-3">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            placeholder={messages.feedbackPlaceholder}
            className="resize-none rounded-lg border bg-fd-secondary p-3 text-fd-secondary-foreground placeholder:text-fd-muted-foreground focus-visible:outline-none"
          />
          <a
            href={issue}
            target="_blank"
            rel="noreferrer noopener"
            className="w-fit rounded-full border px-3 py-2 text-sm font-medium hover:bg-fd-accent/50"
          >
            {messages.feedbackOpen}
          </a>
        </div>
      ) : null}
    </div>
  );
}
