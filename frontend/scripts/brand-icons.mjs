// The Yunshu brand asset set, generated from ONE source: the mark of the console's sidebar (a pivot
// with two cloud arcs turning around it, `CloudMark` in src/App.tsx before it became an asset).
//
//   node scripts/brand-icons.mjs          write public/ (favicon.svg, favicon.ico, apple-touch-icon*.png,
//                                         web-app-manifest-{192,512}x{192,512}.png, manifest.json)
//   node scripts/brand-icons.mjs --check  verify the committed files match (CI: no drift from the source)
//
// CPU only, no new dependency: the PNGs are drawn by the Chromium that Playwright already installs,
// and the ICO container (PNG-compressed 16/32/48) is written here. The same tile is used everywhere
// (tab, sidebar, docs nav, Ask AI, chat avatar), so it must read on light and dark: a near-black tile
// with a light mark and a hairline edge.

import { chromium } from "@playwright/test";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "public");

export const TILE = "#0b0b0d";
export const INK = "#fafafa";

// The mark in its own 24-unit box (CloudMark): two arcs of a circle of radius 7.5 around (12, 12), the
// second at half opacity, and the pivot dot. The arcs reach 1 unit beyond the circle (stroke 2).
const MARK = `<path d="M4.5 12a7.5 7.5 0 0 1 12.8-5.3" stroke="${INK}" stroke-width="2" stroke-linecap="round" fill="none"/><path d="M19.5 12a7.5 7.5 0 0 1-12.8 5.3" stroke="${INK}" stroke-opacity=".5" stroke-width="2" stroke-linecap="round" fill="none"/><circle cx="12" cy="12" r="2.6" fill="${INK}"/>`;
const MARK_DIAMETER = 17; // 2 * (7.5 + 1)

/**
 * A square icon of `size` units. `radius` rounds the tile (0 = full bleed, for maskable and Apple
 * icons, which the platform masks itself). `fill` is the share of the box the mark's diameter takes:
 * a maskable icon keeps its mark inside the central 80 % circle (safe zone), so 56 %.
 */
export function iconSvg({
  size = 64,
  radius = 0.22,
  fill = 0.62,
  edge = true,
} = {}) {
  const s = (size * fill) / MARK_DIAMETER;
  const r = size * radius;
  const hairline =
    edge && radius > 0
      ? `<rect x=".5" y=".5" width="${size - 1}" height="${size - 1}" rx="${r - 0.5}" fill="none" stroke="#ffffff" stroke-opacity=".16"/>`
      : "";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}"><rect width="${size}" height="${size}" rx="${r}" fill="${TILE}"/>${hairline}<g transform="translate(${size / 2} ${size / 2}) scale(${s.toFixed(4)}) translate(-12 -12)">${MARK}</g></svg>\n`;
}

/** ICO container with PNG-compressed entries (Vista+ and every browser). */
export function ico(entries) {
  const head = Buffer.alloc(6);
  head.writeUInt16LE(1, 2);
  head.writeUInt16LE(entries.length, 4);
  let offset = 6 + 16 * entries.length;
  const dirs = [];
  for (const { size, png } of entries) {
    const d = Buffer.alloc(16);
    d.writeUInt8(size >= 256 ? 0 : size, 0);
    d.writeUInt8(size >= 256 ? 0 : size, 1);
    d.writeUInt16LE(1, 4);
    d.writeUInt16LE(32, 6);
    d.writeUInt32LE(png.length, 8);
    d.writeUInt32LE(offset, 12);
    offset += png.length;
    dirs.push(d);
  }
  return Buffer.concat([head, ...dirs, ...entries.map((e) => e.png)]);
}

export const MANIFEST = {
  name: "Yunshu",
  short_name: "Yunshu",
  description:
    "Local LLM / VLM inference engine for Apple Silicon: console, API reference and docs.",
  start_url: "/console/",
  scope: "/console/",
  icons: [
    {
      src: "/console/web-app-manifest-192x192.png",
      sizes: "192x192",
      type: "image/png",
      purpose: "maskable",
    },
    {
      src: "/console/web-app-manifest-512x512.png",
      sizes: "512x512",
      type: "image/png",
      purpose: "maskable",
    },
    {
      src: "/console/favicon.svg",
      sizes: "any",
      type: "image/svg+xml",
      purpose: "any",
    },
  ],
  theme_color: "#f7f7f8",
  background_color: "#f7f7f8",
  display: "standalone",
};

async function render(browser, svg, px) {
  const page = await browser.newPage({
    viewport: { width: px, height: px },
    deviceScaleFactor: 1,
  });
  await page.setContent(
    `<style>html,body{margin:0;background:transparent}svg{display:block;width:${px}px;height:${px}px}</style>${svg}`,
  );
  const png = await page.screenshot({
    omitBackground: true,
    clip: { x: 0, y: 0, width: px, height: px },
  });
  await page.close();
  return png;
}

export async function build() {
  const files = new Map();
  const tile = iconSvg();
  files.set("favicon.svg", Buffer.from(tile));
  files.set(
    "manifest.json",
    Buffer.from(JSON.stringify(MANIFEST, null, 2) + "\n"),
  );
  const browser = await chromium.launch();
  try {
    const entries = [];
    for (const size of [16, 32, 48])
      entries.push({
        size,
        png: await render(
          browser,
          iconSvg({ size: 64, edge: size >= 32 }),
          size,
        ),
      });
    files.set("favicon.ico", ico(entries));
    // Apple masks the touch icon itself: full bleed, no transparency, mark a little larger than maskable.
    files.set(
      "apple-touch-icon.png",
      await render(browser, iconSvg({ size: 180, radius: 0, fill: 0.6 }), 180),
    );
    files.set(
      "apple-touch-icon-precomposed.png",
      files.get("apple-touch-icon.png"),
    );
    for (const px of [192, 512])
      files.set(
        `web-app-manifest-${px}x${px}.png`,
        await render(browser, iconSvg({ size: px, radius: 0, fill: 0.56 }), px),
      );
  } finally {
    await browser.close();
  }
  return files;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const check = process.argv.includes("--check");
  const files = await build();
  mkdirSync(out, { recursive: true });
  let drift = 0;
  for (const [name, data] of files) {
    const path = join(out, name);
    if (check) {
      const same =
        existsSync(path) &&
        createHash("sha1").update(readFileSync(path)).digest("hex") ===
          createHash("sha1").update(data).digest("hex");
      if (!same) {
        drift++;
        console.error(`drift: public/${name}`);
      }
    } else {
      writeFileSync(path, data);
      console.log(`public/${name}  ${data.length} B`);
    }
  }
  if (check && drift) process.exit(1);
}
