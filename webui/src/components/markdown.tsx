"use client";

import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import "katex/dist/katex.min.css";
import { cn } from "yunui";

/**
 * Markdown renderer for chat / completion output. GFM tables + task lists,
 * KaTeX math, and syntax-highlighted code. Styled via the `.prose` token
 * overrides in globals.css so it follows the YunUI theme.
 */
export const Markdown = memo(function Markdown({
  children,
  className,
}: {
  children: string;
  className?: string;
}) {
  return (
    <div className={cn("prose prose-sm max-w-none break-words", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
});
