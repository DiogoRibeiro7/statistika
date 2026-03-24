/**
 * Benchmark: Regression and GLM models.
 *
 * Usage:
 *   npx ts-node benchmark/regression-glm.ts
 *
 * Measures wall-clock time for simple linear regression, multiple regression,
 * logistic regression, polynomial regression, and GLM (Poisson family).
 */

import { linearRegression } from "../src/models/linear-regression";
import { multipleRegression } from "../src/models/multiple-regression";
import { polynomialRegression } from "../src/models/polynomial-regression";
import { logisticRegression } from "../src/models/logistic-regression";
import { glm, poisson } from "../src/glm";

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

interface BenchResult {
  name: string;
  timeMs: number;
  iterations: number;
}

function bench(
  name: string,
  fn: () => void,
  iterations: number,
): BenchResult {
  // Warmup
  const warmup = Math.min(Math.max(1, Math.floor(iterations / 10)), 50);
  for (let i = 0; i < warmup; i++) fn();

  // Measure
  const start = performance.now();
  for (let i = 0; i < iterations; i++) fn();
  const timeMs = performance.now() - start;

  return { name, timeMs, iterations };
}

// ---------------------------------------------------------------------------
// Simple LCG for reproducible synthetic data
// ---------------------------------------------------------------------------

function createRng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

function normalRandom(rng: () => number): number {
  // Box-Muller transform
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

function generateLinearData(n: number, seed = 42) {
  const rng = createRng(seed);
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const xi = rng() * 10;
    const noise = normalRandom(rng) * 0.5;
    x.push(xi);
    y.push(2 * xi + 1 + noise);
  }
  return { x, y };
}

function generateMultipleData(n: number, p: number, seed = 42) {
  const rng = createRng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  const trueCoeffs = Array.from({ length: p }, (_, i) => (i + 1) * 0.5);
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    let yi = 3; // intercept
    for (let j = 0; j < p; j++) {
      const xij = rng() * 10;
      row.push(xij);
      yi += trueCoeffs[j] * xij;
    }
    yi += normalRandom(rng) * 0.5;
    X.push(row);
    y.push(yi);
  }
  return { X, y };
}

function generateLogisticData(n: number, seed = 42) {
  const rng = createRng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = normalRandom(rng);
    const x2 = normalRandom(rng);
    const eta = -1 + 2 * x1 + 1.5 * x2;
    const prob = 1 / (1 + Math.exp(-eta));
    X.push([x1, x2]);
    y.push(rng() < prob ? 1 : 0);
  }
  return { X, y };
}

function generatePoissonData(n: number, seed = 42) {
  const rng = createRng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x1 = rng() * 2;
    const mu = Math.exp(0.5 + 0.8 * x1);
    // Simple Poisson draw via inverse CDF
    let count = 0;
    let p = Math.exp(-mu);
    let cumP = p;
    const u = rng();
    while (cumP < u && count < 100) {
      count++;
      p *= mu / count;
      cumP += p;
    }
    X.push([x1]);
    y.push(count);
  }
  return { X, y };
}

// ---------------------------------------------------------------------------
// Run benchmarks
// ---------------------------------------------------------------------------

const results: BenchResult[] = [];

// Simple linear regression at different sizes
for (const n of [100, 1000, 10000]) {
  const { x, y } = generateLinearData(n);
  const iters = n <= 100 ? 5000 : n <= 1000 ? 500 : 50;
  results.push(
    bench(`linearRegression (n=${n})`, () => linearRegression(x, y), iters),
  );
}

// Multiple regression (5 predictors, 1000 data points)
{
  const { X, y } = generateMultipleData(1000, 5);
  results.push(
    bench("multipleRegression (p=5, n=1000)", () => multipleRegression(X, y), 200),
  );
}

// Logistic regression (1000 data points)
{
  const { X, y } = generateLogisticData(1000);
  results.push(
    bench("logisticRegression (n=1000)", () => logisticRegression(X, y), 50),
  );
}

// GLM with Poisson family (1000 data points)
{
  const { X, y } = generatePoissonData(1000);
  results.push(
    bench("GLM Poisson (n=1000)", () => glm(X, y, poisson), 50),
  );
}

// Polynomial regression (degree 3, 1000 data points)
{
  const { x, y } = generateLinearData(1000, 99);
  results.push(
    bench("polynomialRegression (deg=3, n=1000)", () => polynomialRegression(x, y, 3), 200),
  );
}

// ---------------------------------------------------------------------------
// Output results
// ---------------------------------------------------------------------------

console.log("\n=== Regression & GLM Benchmark ===\n");

const COL = { name: 42, time: 12, iters: 10, perIter: 14 };

const header =
  "Benchmark".padEnd(COL.name) +
  "Total (ms)".padStart(COL.time) +
  "Iters".padStart(COL.iters) +
  "Per iter (ms)".padStart(COL.perIter);

console.log(header);
console.log("-".repeat(header.length));

for (const r of results) {
  const totalStr = r.timeMs.toFixed(2);
  const perIter = (r.timeMs / r.iterations).toFixed(4);
  console.log(
    r.name.padEnd(COL.name) +
      totalStr.padStart(COL.time) +
      r.iterations.toString().padStart(COL.iters) +
      perIter.padStart(COL.perIter),
  );
}

console.log("");
