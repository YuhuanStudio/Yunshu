import assert from "node:assert/strict";
import test from "node:test";
import {
  detectLocale,
  getLocale,
  matchLocale,
  setLocale,
  t,
  tr,
} from "../src/i18n/index.ts";
import { interpolate } from "../src/i18n/plural.ts";
import { elapsed, fixed, number, percent } from "../src/i18n/format.ts";

test("regional tags map onto the three locales", () => {
  assert.equal(matchLocale("zh-TW"), "zh-TW");
  assert.equal(matchLocale("zh-HK"), "zh-TW");
  assert.equal(matchLocale("zh-MO"), "zh-TW");
  assert.equal(matchLocale("zh-Hant"), "zh-TW");
  assert.equal(matchLocale("zh-CN"), "zh-CN");
  assert.equal(matchLocale("zh-SG"), "zh-CN");
  assert.equal(matchLocale("zh-Hans-CN"), "zh-CN");
  assert.equal(matchLocale("en-GB"), "en");
  assert.equal(matchLocale("fr"), null);
});

test("detection reads navigator.languages; the runtime default stays zh-TW", () => {
  // Node reports en-US; a browser without any match would get zh-TW.
  assert.equal(detectLocale(), matchLocale(navigator.languages[0]) ?? "zh-TW");
  assert.equal(getLocale(), "zh-TW");
});

test("plural and interpolation (ICU-lite)", () => {
  const msg = "{n, plural, =0 {none} one {# item} other {# items}}";
  assert.equal(interpolate(msg, { n: 0 }, "en"), "none");
  assert.equal(interpolate(msg, { n: 1 }, "en"), "1 item");
  assert.equal(interpolate(msg, { n: 5 }, "en"), "5 items");
  assert.equal(
    interpolate("{n, plural, other {# 個}}", { n: 3 }, "zh-TW"),
    "3 個",
  );
  assert.equal(interpolate("a {x} b {y}", { x: 1 }, "en"), "a 1 b ");
});

test("switching locale changes t() and formatting without a reload", async () => {
  assert.equal(getLocale(), "zh-TW");
  assert.equal(t("common.retry"), "重試");
  assert.equal(fixed(1234.5, 1), "1,234.5");
  await setLocale("en");
  assert.equal(getLocale(), "en");
  assert.equal(t("common.retry"), "Retry");
  assert.equal(elapsed(3725), "1h 2m");
  assert.equal(percent(0.42), "42%");
  await setLocale("zh-CN");
  assert.equal(t("common.retry"), "重试");
  await setLocale("zh-TW");
  assert.equal(number(null), "—");
});

test("a missing key is visible, never a blank", () => {
  assert.equal(tr("nope.missing"), "⟦nope.missing⟧");
});

test("YunUI keys exist in every locale", async () => {
  for (const l of ["en", "zh-CN", "zh-TW"] as const) {
    await setLocale(l);
    assert.notEqual(
      tr("yunui.components.confirmModal.confirm"),
      "⟦yunui.components.confirmModal.confirm⟧",
    );
    assert.match(tr("yunui.content.codeBlock.lineCount", { count: 2 }), /2/);
  }
});
