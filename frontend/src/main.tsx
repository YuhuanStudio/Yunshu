import { StrictMode } from "react";
import { exposePushStats } from "./push.ts";
import { createRoot } from "react-dom/client";
import App from "./App";
import "katex/dist/katex.min.css";
import "@yuhuanowo/yunui/content.css";
import "./styles.css";
import { initI18n } from "./i18n/index.ts";

exposePushStats();

// The detected language loads before the first paint, so nothing flashes in another one.
void initI18n().then(() =>
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  ),
);
