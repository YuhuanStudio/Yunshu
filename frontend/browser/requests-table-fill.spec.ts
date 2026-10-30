import { expect, test } from "@playwright/test";
import { storm } from "./support/storm";

// The requests table's cells must span its rows at every width. A fixed table layout with a column
// hidden by breakpoint left WebKit with a blank strip at the right of every row (161 px at 1728 px):
// WebKit kept the hidden column's width. The engine-restart divider row is in the table too (it spans
// every column), since that is where the strip showed on the resident console.
for (const width of [1440, 1512, 1728, 1920]) {
  test(`requests table cells fill each row at ${width} px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1000 });
    await storm(page, "");
    const now = Date.now() / 1000;
    const history = {
      series: { t: [now - 30, now - 20, now - 10], decode_tps: [1, 2, 3] },
      events: [{ t: now - 4, kind: "engine_restarted" }],
    };
    await page.route(/\/v1\/yunshu\/(metrics\/)?history/, (route) =>
      route.fulfill({ json: history }),
    );
    await page.goto("/console/#/requests", { waitUntil: "domcontentloaded" });
    await expect(
      page.locator("table.yunui-table tbody tr").first(),
    ).toBeVisible();
    // The divider shows on the full list (a busy engine opens on the live lanes).
    await page
      .getByRole("tab", { name: /^(全部|All)$/ })
      .first()
      .click();
    await expect(
      page.locator("tr[data-restart-boundary]").first(),
    ).toBeVisible();
    const g = await page.evaluate(() => {
      const t = document.querySelector("table.yunui-table")!;
      const w = (e: Element) => e.getBoundingClientRect().width;
      const rows = [...t.querySelectorAll("tbody tr")].slice(0, 6);
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
