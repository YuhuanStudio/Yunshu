import { useEffect, useId, useState } from "react";

/** Mermaid is loaded only when a page actually contains a diagram. */
export function Mermaid({
  chart,
  children,
}: {
  chart?: string;
  children?: string;
}) {
  const id = useId().replace(/[^a-z0-9]/gi, "");
  const [dark, setDark] = useState(() =>
    document.documentElement.classList.contains("dark"),
  );
  useEffect(() => {
    const observer = new MutationObserver(() =>
      setDark(document.documentElement.classList.contains("dark")),
    );
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => observer.disconnect();
  }, []);
  const [svg, setSvg] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    import("mermaid")
      .then(async ({ default: mermaid }) => {
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: dark ? "dark" : "default",
        });
        const result = await mermaid.render(
          `diagram${id}`,
          chart ?? children ?? "",
        );
        if (live) setSvg(result.svg);
      })
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [chart, children, id, dark]);
  return error ? (
    <pre role="alert">{error}</pre>
  ) : (
    <div
      className="my-6 overflow-x-auto"
      role="img"
      aria-label={chart ?? children}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
