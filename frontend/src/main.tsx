import { StrictMode } from "react";
import { exposePushStats, setPushEnabled } from "./push.ts";
import { createRoot } from "react-dom/client";
import App from "./App";
import "katex/dist/katex.min.css";
import "@yuhuanowo/yunui/content.css";
import "./styles.css";
import { initI18n } from "./i18n/index.ts";

exposePushStats();
// `?push=0`: poll like the previous console (A/B measurements, and a way out if a proxy mangles streams).
if (new URLSearchParams(location.search).get("push") === "0")
  setPushEnabled(false);

// The detected language loads before the first paint, so nothing flashes in another one.
void initI18n().then(() =>
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  ),
);
