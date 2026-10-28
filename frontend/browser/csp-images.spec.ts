import { expect, test } from "@playwright/test";

// Model text is Markdown; an image it names must not be fetched from a remote host (a tracking pixel).
test("remote images are blocked by the page policy; data: images still load", async ({
  page,
}) => {
  const remote: string[] = [];
  await page.route("http://tracker.invalid/**", (route) => {
    remote.push(route.request().url());
    return route.abort();
  });
  await page.goto("/console/");
  const result = await page.evaluate(async () => {
    const violations: string[] = [];
    document.addEventListener("securitypolicyviolation", (e) =>
      violations.push(e.blockedURI),
    );
    const load = (src: string) =>
      new Promise<boolean>((done) => {
        const img = new Image();
        img.onload = () => done(true);
        img.onerror = () => done(false);
        img.src = src;
      });
    const pixel =
      "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
    return {
      remote: await load("http://tracker.invalid/p.png"),
      data: await load(pixel),
      violations,
    };
  });
  expect(result.data).toBe(true);
  expect(result.remote).toBe(false);
  expect(result.violations.join(" ")).toContain("tracker.invalid");
  expect(remote).toEqual([]);
});
