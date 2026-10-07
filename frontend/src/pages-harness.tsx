// Dev-only host for the Downloads and Cache pages until the app shell routes them.
// Not part of the build (vite builds index.html only); the browser specs use it as a fallback.
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { YunUIProvider } from "@yuhuanowo/yunui/adapters";
import "@yuhuanowo/yunui/content.css";
import "./styles.css";
import { initI18n } from "./i18n/index.ts";
import { useEngine } from "./useEngine";
import Downloads from "./Downloads";
import Cache from "./Cache";

const adapters = { useT: () => (key: string) => key };
function Host() {
  const connection = { baseUrl: location.origin, token: "" };
  const engine = useEngine(connection);
  const [page, setPage] = useState(location.hash);
  useEffect(() => {
    const fn = () => setPage(location.hash);
    addEventListener("hashchange", fn);
    return () => removeEventListener("hashchange", fn);
  }, []);
  return (
    <YunUIProvider adapters={adapters}>
      <div className="min-h-dvh bg-(--bg-window) p-4 lg:p-6">
        <div className="mx-auto w-full max-w-7xl">
          {page.startsWith("#/cache") ? (
            <Cache connection={connection} engine={engine} />
          ) : (
            <Downloads connection={connection} engine={engine} />
          )}
        </div>
      </div>
    </YunUIProvider>
  );
}
void initI18n().then(() =>
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <Host />
    </StrictMode>,
  ),
);
