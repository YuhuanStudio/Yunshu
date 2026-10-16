// Lists the browser tests that failed first and passed on retry, as GitHub warnings and in the job
// summary, so a retry never hides an unstable test. Exits 0: the test run itself decides pass/fail.
import { appendFileSync, existsSync, readFileSync } from "node:fs";

const file = process.argv[2] ?? "playwright-report.json";
if (!existsSync(file)) {
  console.log(`report-flaky: no ${file}`);
  process.exit(0);
}
const flaky = [];
const walk = (suite, path) => {
  for (const s of suite.suites ?? []) walk(s, [...path, s.title]);
  for (const spec of suite.specs ?? [])
    for (const t of spec.tests ?? [])
      if (t.status === "flaky") {
        const first = t.results.find((r) => r.status !== "passed");
        const msg = (first?.error?.message ?? "").split("\n")[0];
        flaky.push({ name: `[${t.projectName}] ${spec.file}:${spec.line} › ${spec.title}`, msg });
      }
};
walk(JSON.parse(readFileSync(file, "utf8")), []);
for (const f of flaky) console.log(`::warning title=flaky browser test::${f.name} -- ${f.msg}`);
console.log(`report-flaky: ${flaky.length} flaky`);
if (process.env.GITHUB_STEP_SUMMARY && flaky.length)
  appendFileSync(
    process.env.GITHUB_STEP_SUMMARY,
    `### Flaky browser tests (passed only on retry)\n\n${flaky.map((f) => `- ${f.name}: ${f.msg}`).join("\n")}\n`,
  );
