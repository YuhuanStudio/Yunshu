// The console's brand asset set is the shared YuhuanStudio icon set, byte-identical to Yunxin, YunUI's site
// and YunNEWS (favicon.ico, apple-touch-icon*.png, web-app-manifest-{192,512}.png, yuhuanstudio-logo.png).
// The images are copied, not drawn: update them from those projects together. This script owns manifest.json.
//
//   node scripts/brand-icons.mjs          write public/manifest.json
//   node scripts/brand-icons.mjs --check  verify manifest.json and that every shared image is the pinned one

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const out = join(dirname(fileURLToPath(import.meta.url)), "..", "public");

/** sha1 of each shared YuhuanStudio image (same bytes as Yunxin/frontend/public and YunUI/site/public). */
export const SHARED = {
  "favicon.ico": "c8bd5c55a267b48913e6219fcaebb4faf820cc11",
  "apple-touch-icon.png": "3f4db84d6060bdb065694f194c4a556f09fa5541",
  "apple-touch-icon-precomposed.png": "3f4db84d6060bdb065694f194c4a556f09fa5541",
  "web-app-manifest-192x192.png": "4a42299a44fbb9d192fe49d21ea060fef5b3611d",
  "web-app-manifest-512x512.png": "23d221b5619e165425913b27697ab4073d8b2e2a",
  "yuhuanstudio-logo.png": "3f4db84d6060bdb065694f194c4a556f09fa5541",
};

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
  ],
  theme_color: "#f7f7f8",
  background_color: "#f7f7f8",
  display: "standalone",
};

const sha1 = (buf) => createHash("sha1").update(buf).digest("hex");

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const manifest = JSON.stringify(MANIFEST, null, 2) + "\n";
  const path = join(out, "manifest.json");
  if (!process.argv.includes("--check")) {
    writeFileSync(path, manifest);
    console.log(`public/manifest.json  ${manifest.length} B`);
  } else {
    let drift = 0;
    if (!existsSync(path) || readFileSync(path, "utf8") !== manifest) {
      drift++;
      console.error("drift: public/manifest.json");
    }
    for (const [name, want] of Object.entries(SHARED)) {
      const p = join(out, name);
      if (!existsSync(p) || sha1(readFileSync(p)) !== want) {
        drift++;
        console.error(`drift: public/${name} is not the shared YuhuanStudio icon`);
      }
    }
    if (drift) process.exit(1);
  }
}
