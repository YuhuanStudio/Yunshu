import { Suspense, lazy, type ComponentProps } from "react";

// The highlighter (shiki core, ~57 KB gzip) loads only when a code block is shown,
// not with every route that merely contains one.
const CodeBlockImpl = lazy(() =>
  import("@yuhuanowo/yunui/code").then((m) => ({ default: m.CodeBlock })),
);
type Props = ComponentProps<typeof CodeBlockImpl>;

export function CodeBlock(props: Props) {
  return (
    <Suspense
      fallback={
        <pre
          className={`overflow-x-auto rounded-lg border border-border bg-(--bg-elevated) p-3 font-mono text-xs ${props.className ?? ""}`}
        >
          {props.children}
        </pre>
      }
    >
      <CodeBlockImpl {...props} />
    </Suspense>
  );
}
