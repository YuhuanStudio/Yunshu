"use client";

import { LLMCopyButton, ViewOptions } from "@yuhuanowo/yunui/patterns";
import type { Messages } from "@/lib/messages";

export function PageActions({ messages, markdownUrl, githubUrl }: { messages: Messages; markdownUrl: string; githubUrl: string }) {
  return (
    <div className="-mx-2 mb-2 flex flex-wrap items-center gap-0.5 text-fd-muted-foreground">
      <LLMCopyButton markdownUrl={markdownUrl} labels={{ copy: messages.copyMd, copied: messages.copiedMd, title: messages.copyMd }} />
      <span className="text-fd-border">·</span>
      <ViewOptions
        markdownUrl={markdownUrl}
        githubUrl={githubUrl}
        labels={{ markdown: "Markdown", markdownTitle: "Markdown", github: messages.openGithub, githubTitle: messages.openGithub }}
      />
    </div>
  );
}
