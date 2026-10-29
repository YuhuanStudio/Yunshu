import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { storm } from "./support/storm";

// A frame strip of the status band under the storm feed, for before/after pictures (not an assertion):
// one crop of the bottom band every 250 ms, stacked into one image, newest at the bottom. Rows whose
// pills move or resize from one frame to the next are the jumping.
//   BAND_STRIP_DIR=/dir BAND_STRIP_TAG=before|after pnpm exec playwright test browser/band-strip.spec.ts

const dir = process.env.BAND_STRIP_DIR;
test.skip(!dir, "set BAND_STRIP_DIR to write the strip");
test.use({ viewport: { width: 1440, height: 900 } });

test("band strip", async ({ page }) => {
  test.setTimeout(60_000);
  mkdirSync(dir!, { recursive: true });
  await storm(page, "");
  await page.goto("/console/#/overview", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(6000);
  const clip = { x: 270, y: 856, width: 720, height: 36 };
  const frames = 24;
  const shots: string[] = [];
  for (let i = 0; i < frames; i++) {
    shots.push((await page.screenshot({ clip })).toString("base64"));
    await page.waitForTimeout(250);
  }
  await page.setViewportSize({
    width: clip.width,
    height: clip.height * frames,
  });
  await page.setContent(
    `<body style="margin:0;background:#f7f7f8">${shots
      .map(
        (s) => `<img style="display:block" src="data:image/png;base64,${s}">`,
      )
      .join("")}</body>`,
  );
  const path = join(dir!, `band-${process.env.BAND_STRIP_TAG ?? "run"}.png`);
  await page.screenshot({ path });
  expect(path).toBeTruthy();
});
