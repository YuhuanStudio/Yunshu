import type { ReactNode } from "react";
import "../globals.css";

// Only the bare "/" lives here: it forwards to the best-matching language.
export default function RootRedirectLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-TW">
      <body>{children}</body>
    </html>
  );
}
