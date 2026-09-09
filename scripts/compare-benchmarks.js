#!/usr/bin/env node
/**
 * Compare benchmark results against baselines.
 *
 * Usage:
 *   node scripts/compare-benchmarks.js <benchmark-results.json> [baselines.md]
 *
 * - Reads the benchmark JSON output produced by `BENCH_JSON=1 yarn bench:all`
 * - Parses baselines from benchmark/BASELINES.md (or a custom path)
 * - Fails if any benchmark suite itself failed
 * - Prints warnings for any benchmark that regressed >10%
 * - Exits with code 1 if a suite failure or regression is detected
 */

const fs = require("fs");
const path = require("path");

const REGRESSION_THRESHOLD = 0.10;

function parseBaselines(baselinePath) {
  const baselines = new Map();

  if (!fs.existsSync(baselinePath)) {
    console.log(`No baselines file found at ${baselinePath}. Skipping comparison.`);
    return baselines;
  }

  const content = fs.readFileSync(baselinePath, "utf-8");
  const lines = content.split("\n");

  for (const line of lines) {
    const match = line.match(
      /^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([\d.]+)\s*\|/,
    );
    if (!match) continue;

    const suite = match[1].trim();
    const name = match[2].trim();
    const time = parseFloat(match[3]);

    if (suite === "Suite" || suite.startsWith("-")) continue;
    if (isNaN(time)) continue;

    baselines.set(`${suite}::${name}`, { suite, name, tsTimeMs: time });
  }

  return baselines;
}

function main() {
  const args = process.argv.slice(2);

  if (args.length < 1) {
    console.error(
      "Usage: node scripts/compare-benchmarks.js <benchmark-results.json> [baselines.md]",
    );
    process.exit(2);
  }

  const resultsPath = args[0];
  const baselinePath =
    args[1] || path.join(__dirname, "..", "benchmark", "BASELINES.md");

  if (!fs.existsSync(resultsPath)) {
    console.error(`Results file not found: ${resultsPath}`);
    process.exit(2);
  }

  const results = JSON.parse(fs.readFileSync(resultsPath, "utf-8"));
  const baselines = parseBaselines(baselinePath);

  const failedSuites = results.suites.filter((suite) => suite.error);
  if (failedSuites.length > 0) {
    console.log("Benchmark suite failures detected:\n");
    for (const suite of failedSuites) {
      console.log(`  FAIL ${suite.name}: ${suite.error}`);
    }
    console.log(`\nSuite failures: ${failedSuites.length}`);
    process.exit(1);
  }

  if (baselines.size === 0) {
    console.log("No baselines to compare against. All benchmark suites completed successfully.");
    console.log("\nBenchmark results summary:");
    for (const suite of results.suites) {
      console.log(`  ${suite.name}: ${suite.benchmarks.length} benchmarks`);
      for (const b of suite.benchmarks) {
        console.log(`    ${b.name}: ${b.tsTimeMs.toFixed(2)}ms`);
      }
    }
    process.exit(0);
  }

  let regressionCount = 0;
  let comparedCount = 0;
  const regressions = [];

  console.log("Comparing benchmark results against baselines...\n");

  for (const suite of results.suites) {
    for (const bench of suite.benchmarks) {
      const key = `${suite.name}::${bench.name}`;
      const baseline = baselines.get(key);

      if (!baseline) {
        console.log(`  NEW  ${suite.name} / ${bench.name}: ${bench.tsTimeMs.toFixed(2)}ms (no baseline)`);
        continue;
      }

      comparedCount++;
      const change = (bench.tsTimeMs - baseline.tsTimeMs) / baseline.tsTimeMs;
      const changeStr = (change * 100).toFixed(1);
      const arrow = change > 0 ? "slower" : "faster";

      if (change > REGRESSION_THRESHOLD) {
        regressionCount++;
        regressions.push({
          suite: suite.name,
          name: bench.name,
          baseline: baseline.tsTimeMs,
          current: bench.tsTimeMs,
          change,
        });
        console.log(
          `  FAIL ${suite.name} / ${bench.name}: ${bench.tsTimeMs.toFixed(2)}ms vs baseline ${baseline.tsTimeMs.toFixed(2)}ms (+${changeStr}% ${arrow})`,
        );
      } else {
        console.log(
          `  PASS ${suite.name} / ${bench.name}: ${bench.tsTimeMs.toFixed(2)}ms vs baseline ${baseline.tsTimeMs.toFixed(2)}ms (${change > 0 ? "+" : ""}${changeStr}% ${arrow})`,
        );
      }
    }
  }

  console.log("\n--- Summary ---");
  console.log(`Compared: ${comparedCount} benchmarks`);
  console.log(`Regressions (>${REGRESSION_THRESHOLD * 100}%): ${regressionCount}`);

  if (regressionCount > 0) {
    console.log("\nRegressed benchmarks:");
    for (const r of regressions) {
      console.log(
        `  - ${r.suite} / ${r.name}: ${r.baseline.toFixed(2)}ms -> ${r.current.toFixed(2)}ms (+${(r.change * 100).toFixed(1)}%)`,
      );
    }
    console.log("\nPerformance regression detected! Exiting with error.");
    process.exit(1);
  }

  console.log("\nNo performance regressions detected.");
  process.exit(0);
}

main();
