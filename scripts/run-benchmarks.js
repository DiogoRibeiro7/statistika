#!/usr/bin/env node
/**
 * Runs all benchmark suites. When the BENCH_JSON=1 environment variable is set,
 * outputs results as a JSON file to benchmark/results/benchmark-results.json.
 * Otherwise, runs benchmarks normally with console output.
 */

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const suites = [
  "benchmark/special-functions.ts",
  "benchmark/linalg.ts",
  "benchmark/regression-glm.ts",
  "benchmark/mcmc-clustering.ts",
];

const jsonMode = process.env.BENCH_JSON === "1";

if (!jsonMode) {
  // Standard mode: run each suite sequentially with normal output
  for (const suite of suites) {
    execSync(`npx ts-node ${suite}`, { stdio: "inherit" });
  }
  process.exit(0);
}

// JSON mode: capture timing for each suite and write structured output
const resultsDir = path.join(__dirname, "..", "benchmark", "results");
fs.mkdirSync(resultsDir, { recursive: true });

const allResults = {
  timestamp: new Date().toISOString(),
  nodeVersion: process.version,
  platform: process.platform,
  arch: process.arch,
  suites: [],
};

for (const suite of suites) {
  const suiteName = path.basename(suite, ".ts");
  console.log(`Running benchmark suite: ${suiteName}...`);

  const startTime = performance.now();
  let output = "";
  try {
    output = execSync(`npx ts-node ${suite}`, {
      encoding: "utf-8",
      timeout: 300000,
    });
  } catch (err) {
    console.error(`Suite ${suiteName} failed: ${err.message}`);
    allResults.suites.push({
      name: suiteName,
      error: err.message,
      benchmarks: [],
    });
    continue;
  }
  const endTime = performance.now();

  // Parse benchmark output lines: look for lines with timing data
  // Format: "functionName          123.45       67.89     1.84x"
  const benchmarks = [];
  const lines = output.split("\n");
  for (const line of lines) {
    // Match lines that have a name followed by numeric values
    const match = line.match(
      /^(.+?)\s{2,}([\d.]+)\s+([\d.]+|N\/A)\s+([\d.]+x|N\/A)\s*$/,
    );
    if (match) {
      benchmarks.push({
        name: match[1].trim(),
        tsTimeMs: parseFloat(match[2]),
        nativeTimeMs: match[3] === "N/A" ? null : parseFloat(match[3]),
        speedup: match[4] === "N/A" ? null : parseFloat(match[4]),
      });
    }
  }

  allResults.suites.push({
    name: suiteName,
    totalTimeMs: endTime - startTime,
    benchmarks,
  });

  console.log(
    `  Completed ${suiteName}: ${benchmarks.length} benchmarks in ${(endTime - startTime).toFixed(0)}ms`,
  );
}

const outputPath = path.join(resultsDir, "benchmark-results.json");
fs.writeFileSync(outputPath, JSON.stringify(allResults, null, 2) + "\n");
console.log(`\nResults written to ${outputPath}`);
