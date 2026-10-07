// i18n consistency gate. Run: node scripts/check-i18n.mjs
//  1. every locale has exactly the zh-TW key set and the same {placeholders}
//  2. every key used in src exists; no zh-TW key is unused (yunui.* is requested by YunUI itself)
//  3. t("k", {vars}) passes every placeholder the message uses
//  4. no hard-coded CJK or English UI text in src outside src/i18n (// i18n-ignore on the same or previous line opts out)
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import ts from "typescript";

const root = new URL("..", import.meta.url).pathname;
const src = join(root, "src");
const errors = [];
const fail = (m) => errors.push(m);

// ---- dictionaries -------------------------------------------------------
const LOCALES = ["zh-TW", "zh-CN", "en"];
const flat = {};
for (const l of LOCALES) {
  const mod = (await import(`../src/i18n/locales/${l}/index.ts`)).default;
  flat[l] = {};
  for (const [ns, entries] of Object.entries(mod))
    for (const [k, v] of Object.entries(entries)) flat[l][`${ns}.${k}`] = v;
}
const source = flat["zh-TW"];
const placeholders = (msg) => {
  const names = new Set();
  let depth = 0;
  for (let i = 0; i < msg.length; i++) {
    if (msg[i] === "{") {
      if (depth === 0) {
        const m = /^\{\s*([A-Za-z_][\w]*)\s*[,}]/.exec(msg.slice(i));
        if (m) names.add(m[1]);
        else fail(`bad placeholder near "${msg.slice(i, i + 20)}" in: ${msg}`);
      }
      depth++;
    } else if (msg[i] === "}") depth--;
  }
  if (depth !== 0) fail(`unbalanced braces in: ${msg}`);
  if (/\bplural\b/.test(msg) && !/\bother\s*\{/.test(msg))
    fail(`plural without "other": ${msg}`);
  return names;
};
for (const l of LOCALES.slice(1)) {
  for (const k of Object.keys(source)) if (!(k in flat[l])) fail(`[${l}] missing key ${k}`);
  for (const k of Object.keys(flat[l])) if (!(k in source)) fail(`[${l}] extra key ${k}`);
  for (const [k, v] of Object.entries(flat[l])) {
    if (!(k in source)) continue;
    const a = [...placeholders(source[k])].sort().join(",");
    const b = [...placeholders(v)].sort().join(",");
    if (a !== b) fail(`[${l}] ${k}: placeholders {${b}} differ from zh-TW {${a}}`);
    if (v === "" && source[k] !== "") fail(`[${l}] ${k}: empty translation`);
  }
}

// ---- source scan --------------------------------------------------------
function walk(dir) {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    if (statSync(p).isDirectory()) return p.includes(join("src", "i18n")) ? [] : walk(p);
    return /\.(ts|tsx)$/.test(n) ? [p] : [];
  });
}
const files = walk(src);
const used = new Set();
const prefixes = new Set();
const CJK = /[㐀-鿿豈-﫿　-〿＀-￯]/;
const BRAND = new Set(
  "Yunshu YunUI Yunmo Yunxin Ollama OpenAI Anthropic Claude Codex Qwen Qwen3 MLX Metal Apple Silicon API URL JSON JSONL SSE HTTP HTTPS CORS GPU CPU RAM KV APC MTP DFlash LoRA ID IDs Token Bearer Authorization Content-Type curl Python TypeScript JavaScript Node GB MB KB ms tok/s TTFT Hugging Face Mermaid PDF CSV UTF Temperature Top Max Min Markdown LaTeX Grafana Prometheus Docker Homebrew macOS Mac".split(
    " ",
  ),
);
const ATTRS = new Set(
  "title label placeholder alt aria-label ariaLabel description subtitle tooltip hint emptyLabel helper caption closeLabel dismissLabel summary heading name text note message".split(
    " ",
  ),
);
const PROPS = new Set("label title description hint note placeholder emptyText tooltip caption message ariaLabel".split(" "));

function englishUi(s) {
  const t = s.trim();
  if (!t || CJK.test(t)) return false;
  if (!/[A-Za-z]{2}/.test(t)) return false;
  if (/^[a-z0-9_.:/#@\-\[\]{}()$%<>=+*,;'"`\\ ]+$/.test(t) && !/ [a-z]{3,} [a-z]{3,}/.test(t)) return false; // ids, paths, css
  const words = t.match(/[A-Za-z][A-Za-z'-]+/g) ?? [];
  const real = words.filter((w) => !BRAND.has(w) && w.length > 2);
  if (real.length === 0) return false;
  return /\s/.test(t) ? real.length >= 2 || /^[A-Z]/.test(t) : /^[A-Z][a-z]{2,}/.test(t);
}

for (const file of files) {
  const text = readFileSync(file, "utf8");
  const sf = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true, file.endsWith("x") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  const lines = text.split("\n");
  const rel = relative(root, file);
  const ignored = (node) => {
    const ln = sf.getLineAndCharacterOfPosition(node.getStart()).line;
    return /i18n-ignore/.test(lines[ln] ?? "") || /i18n-ignore/.test(lines[ln - 1] ?? "");
  };
  const where = (node) => `${rel}:${sf.getLineAndCharacterOfPosition(node.getStart()).line + 1}`;
  const tsx = file.endsWith(".tsx");
  const visit = (node) => {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && (node.expression.text === "t" || node.expression.text === "tr")) {
      const [k, v] = node.arguments;
      if (k && (ts.isStringLiteral(k) || ts.isNoSubstitutionTemplateLiteral(k))) {
        used.add(k.text);
        if (!(k.text in source)) fail(`${where(node)} unknown key ${k.text}`);
        else {
          const need = placeholders(source[k.text]);
          if (need.size) {
            const have = new Set();
            let opaque = !v;
            if (v && ts.isObjectLiteralExpression(v)) {
              for (const p of v.properties) {
                if (ts.isShorthandPropertyAssignment(p) || ts.isPropertyAssignment(p)) have.add(p.name.getText());
                else opaque = true;
              }
              opaque = false || v.properties.some((p) => ts.isSpreadAssignment(p));
            } else if (v) opaque = true;
            if (v == null) fail(`${where(node)} ${k.text} needs {${[...need]}} but no vars passed`);
            else if (!opaque) for (const n of need) if (!have.has(n)) fail(`${where(node)} ${k.text} missing var ${n}`);
          }
        }
      } else if (k && ts.isTemplateExpression(k)) {
        prefixes.add(k.head.text);
        if (!k.head.text) fail(`${where(node)} dynamic key with no static prefix`);
      } else if (k && ts.isIdentifier(k) && node.expression.text === "tr") {
        // tr(variable): the variable's literal keys are listed with a `// i18n-keys: prefix.` comment
        const ln = sf.getLineAndCharacterOfPosition(node.getStart()).line;
        const m = /i18n-keys:\s*(\S+)/.exec((lines[ln] ?? "") + (lines[ln - 1] ?? ""));
        if (m) prefixes.add(m[1]);
        else fail(`${where(node)} tr(${k.text}) needs a "// i18n-keys: <prefix>" comment`);
      }
    }
    // A string literal equal to a key counts as a use (keys passed around as data).
    if (ts.isStringLiteral(node) && node.text in source) used.add(node.text);
    // Hard-coded strings
    if ((ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node) || ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) && CJK.test(node.text) && !ignored(node)) {
      fail(`${where(node)} hard-coded CJK: ${JSON.stringify(node.text.slice(0, 40))}`);
    } else if (ts.isJsxText(node) && CJK.test(node.text) && !ignored(node)) {
      fail(`${where(node)} hard-coded CJK JSX text: ${JSON.stringify(node.text.trim().slice(0, 40))}`);
    }
    if (tsx) {
      if (ts.isJsxText(node) && englishUi(node.text) && !ignored(node)) fail(`${where(node)} hard-coded English JSX text: ${JSON.stringify(node.text.trim().slice(0, 50))}`);
      if (ts.isJsxAttribute(node) && node.initializer && ts.isStringLiteral(node.initializer) && ATTRS.has(node.name.getText()) && englishUi(node.initializer.text) && !ignored(node))
        fail(`${where(node)} hard-coded English ${node.name.getText()}="${node.initializer.text.slice(0, 40)}"`);
      if (ts.isPropertyAssignment(node) && PROPS.has(node.name.getText()) && ts.isStringLiteral(node.initializer) && englishUi(node.initializer.text) && !ignored(node))
        fail(`${where(node)} hard-coded English ${node.name.getText()}: ${JSON.stringify(node.initializer.text.slice(0, 40))}`);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
}

// ---- unused keys --------------------------------------------------------
for (const k of Object.keys(source)) {
  if (k.startsWith("yunui.")) continue;
  if (used.has(k)) continue;
  if ([...prefixes].some((p) => k.startsWith(p))) continue;
  fail(`unused key ${k}`);
}

if (errors.length) {
  console.error(`check-i18n: ${errors.length} problem(s)`);
  for (const e of errors.slice(0, 400)) console.error("  " + e);
  process.exit(1);
}
console.log(`check-i18n: ok (${Object.keys(source).length} keys x ${LOCALES.length} locales, ${files.length} source files)`);
