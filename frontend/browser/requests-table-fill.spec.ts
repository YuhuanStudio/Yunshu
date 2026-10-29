import { expect, test } from "@playwright/test";
import { storm } from "./support/storm";

// The requests table's cells must span its rows at every width. A fixed table layout with a column
// hidden by breakpoint left WebKit with a blank strip at the right of every row (161 px at 1728 px):
// WebKit kept the hidden column's width.
for (const width of [1440, 1512, 1728, 1920]) {
  test(`requests table cells fill each row at ${width} px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    await storm(page, "");
    await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
    await expect(page.locator("table.yunui-table tbody tr").first()).toBeVisible();
    const g = await page.evaluate(() => {
      const t = document.querySelector("table.yunui-table")!;
      const w = (e: Element) => e.getBoundingClientRect().width;
      const rows = [...t.querySelectorAll("tbody tr")].slice(0, 3);
      return {
        table: w(t),
        card: w(t.closest(".card") ?? t),
        gaps: rows.map(
          (tr) =>
            w(tr) -
            [...tr.children]
              .filter((c) => getComputedStyle(c).display !== "none")
              .reduce((a, c) => a + w(c), 0),
        ),
      };
    });
    for (const gap of g.gaps) expect(Math.abs(gap)).toBeLessThanOrEqual(1);
    expect(g.table).toBeGreaterThanOrEqual(g.card - 4);
  });
}
