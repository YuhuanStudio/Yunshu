"use client";

import { memo } from "react";
import { MarkdownRenderer } from "yunui/content";

/**
 * Thin adapter over YunUI's `MarkdownRenderer` (`yunui/content`), which owns the
 * whole rendering stack: GFM tables/task lists, KaTeX math, Shiki-highlighted
 * code, Mermaid diagrams, GitHub callouts and lazy zoomable images. Kept as a
 * local `Markdown({ children })` wrapper so existing call sites — including
 * `ThinkingBlock`'s `renderContent` — need no changes.
 *
 * Requires `yunui/content.css` + `katex/dist/katex.min.css` (imported once in
 * globals.css).
 */
export const Markdown = memo(function Markdown({
  children,
  className,
}: {
  children: string;
  className?: string;
}) {
  // `break-words` so long unbroken strings (URLs) wrap instead of overflowing
  // on narrow screens.
  return (
    <MarkdownRenderer
      content={children}
      className={["break-words", className].filter(Boolean).join(" ")}
    />
  );
});
