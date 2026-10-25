import { useEffect, useState } from "react";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  Spinner,
} from "@yuhuanowo/yunui";
import { createOpenAPIPageBase } from "fumadocs-openapi/ui/base";
import { createShikiFactory } from "fumadocs-core/highlight/shiki";
import { createHighlighterCore } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import { t } from "../i18n";
import { docHref, loadToc } from "./data";
import { useLocale } from "../i18n";
import type { DocsToc } from "./types";
import { routeQuery } from "../route";
import type { OpenAPIPageProps } from "fumadocs-openapi/ui";

const shiki = createShikiFactory({
  init: () =>
    createHighlighterCore({
      themes: [
        import("shiki/themes/github-light.mjs"),
        import("shiki/themes/github-dark.mjs"),
      ],
      langs: [
        import("shiki/langs/json.mjs"),
        import("shiki/langs/python.mjs"),
        import("shiki/langs/javascript.mjs"),
        import("shiki/langs/bash.mjs"),
      ],
      engine: createJavaScriptRegexEngine(),
    }),
});
const OpenAPIPage = createOpenAPIPageBase({
  shiki,
  generateTypeScriptDefinitions: false,
  generateCodeSamples: ({ path }) =>
    path === "/v1/chat/completions"
      ? [
          {
            id: "openai-sdk",
            lang: "python",
            label: t("docs.reference.openaiSdk"),
            source:
              'from openai import OpenAI\nclient = OpenAI(base_url="http://localhost:8100/v1", api_key="YOUR_KEY")\nprint(client.chat.completions.create(model="YOUR_MODEL", messages=[{"role": "user", "content": "Hello"}]))',
          },
        ]
      : path === "/v1/messages"
        ? [
            {
              id: "anthropic-sdk",
              lang: "python",
              label: t("docs.reference.anthropicSdk"),
              source:
                'from anthropic import Anthropic\nclient = Anthropic(base_url="http://localhost:8100", api_key="YOUR_KEY")\nprint(client.messages.create(model="YOUR_MODEL", max_tokens=128, messages=[{"role": "user", "content": "Hello"}]))',
            },
          ]
        : [],
});
type Document = NonNullable<
  OpenAPIPageProps extends infer _P
    ? Extract<OpenAPIPageProps, { payload: unknown }>["payload"]["bundled"]
    : never
>;
const METHODS = ["get", "post", "put", "patch", "delete", "head", "options"];
export default function ApiReference() {
  const locale = useLocale();
  const [schema, setSchema] = useState<Document>();
  const [toc, setToc] = useState<DocsToc>();
  const [error, setError] = useState(false);
  const [selected, setSelected] = useState(
    routeQuery().get("operation") ?? "get /v1/models",
  );
  useEffect(() => {
    let live = true;
    fetch("/console/docs-assets/openapi.json")
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      })
      .then((doc) => {
        // Auth middleware is not represented by FastAPI dependencies. Expose
        // optional Bearer/x-api-key inputs without changing the source schema.
        doc.components ??= {};
        doc.components.securitySchemes = {
          ...doc.components.securitySchemes,
          ConsoleBearer: { type: "http", scheme: "bearer" },
          AnthropicKey: { type: "apiKey", in: "header", name: "x-api-key" },
        };
        doc.security = [{ ConsoleBearer: [] }, { AnthropicKey: [] }, {}];
        doc.servers = [
          { url: location.origin, description: t("docs.reference.proxy") },
        ];
        if (live) setSchema(doc);
      })
      .catch(() => live && setError(true));
    return () => {
      live = false;
    };
  }, []);
  useEffect(() => {
    loadToc(locale).then(setToc);
  }, [locale]);
  const operations = Object.entries(schema?.paths ?? {}).flatMap(
    ([path, item]) =>
      METHODS.filter((m) => item && m in item).map(
        (method) => `${method} ${path}`,
      ),
  );
  const operation = operations.includes(selected) ? selected : "get /v1/models";
  const [method, path] = operation.split(" ");
  return (
    <section className="space-y-6" data-testid="api-reference">
      <h1 className="text-2xl font-semibold">{t("docs.reference")}</h1>
      <p className="text-sm text-muted-foreground">
        {t("docs.reference.description")}
      </p>
      <Select
        value={operation}
        onValueChange={(value) => {
          setSelected(value);
          location.hash =
            docHref("api/reference") +
            "?operation=" +
            encodeURIComponent(value);
        }}
      >
        <SelectTrigger aria-label={t("docs.reference.operation")}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {operations.map((op) => (
            <SelectItem key={op} value={op}>
              {op.toUpperCase().split(" ")[0]} {op.split(" ")[1]}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {error && <p role="alert">{t("docs.error.body")}</p>}
      {schema ? (
        <OpenAPIPage
          key={operation}
          payload={{ bundled: schema }}
          operations={[{ method: method as "get", path }]}
        />
      ) : (
        !error && <Spinner />
      )}
      <nav
        className="border-t border-border pt-4"
        aria-label={t("docs.reference.guides")}
      >
        <h2 className="mb-3 font-medium">{t("docs.reference.guides")}</h2>
        <div className="flex flex-wrap gap-3 text-sm">
          {Object.entries(toc?.pages ?? {})
            .filter(([slug]) => slug.startsWith("api/"))
            .map(([slug, page]) => (
              <a className="underline" href={docHref(slug)} key={slug}>
                {page.title}
              </a>
            ))}
        </div>
      </nav>
    </section>
  );
}
