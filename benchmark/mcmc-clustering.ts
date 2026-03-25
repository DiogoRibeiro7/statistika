/**
 * Benchmark: MCMC and clustering/dimensionality reduction.
 *
 * Usage:
 *   npx ts-node benchmark/mcmc-clustering.ts
 *
 * Measures wall-clock time for Metropolis-Hastings MCMC,
 * K-means clustering, and PCA.
 */

import { metropolisHastings } from "../src/mcmc";
import { kMeans, pca } from "../src/multivariate";

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
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

function generateClusterData(n: number, dims: number, k: number, seed = 42) {
  const rng = createRng(seed);
  const data: number[][] = [];
  // Generate k cluster centers spread out
  const centers: number[][] = [];
  for (let c = 0; c < k; c++) {
    centers.push(Array.from({ length: dims }, () => rng() * 20 - 10));
  }
  for (let i = 0; i < n; i++) {
    const cluster = i % k;
    const point = centers[cluster].map((c) => c + normalRandom(rng) * 0.5);
    data.push(point);
  }
  return data;
}

function generatePCAData(n: number, dims: number, seed = 42) {
  const rng = createRng(seed);
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    const base = normalRandom(rng);
    for (let j = 0; j < dims; j++) {
      // Correlated dimensions with some noise
      row.push(base * (j + 1) * 0.3 + normalRandom(rng) * 0.5);
    }
    data.push(row);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Log-density for MCMC: standard normal
// ---------------------------------------------------------------------------

const standardNormalLogDensity = (x: number) => -0.5 * x * x;

// ---------------------------------------------------------------------------
// Run benchmarks
// ---------------------------------------------------------------------------

const results: BenchResult[] = [];

// Metropolis-Hastings MCMC at different iteration counts
for (const mcmcIters of [1000, 5000, 10000]) {
  results.push(
    bench(
      `metropolisHastings (iters=${mcmcIters})`,
      () =>
        metropolisHastings(standardNormalLogDensity, {
          iterations: mcmcIters,
          proposalStd: 1.0,
          seed: 42,
        }),
      20,
    ),
  );
}

// K-means clustering (100 points, k=3, 2 dimensions)
{
  const data = generateClusterData(100, 2, 3);
  results.push(
    bench("kMeans (n=100, k=3, d=2)", () => kMeans(data, 3, { seed: 42 }), 50),
  );
}

// PCA (100 points, 10 dimensions)
{
  const data = generatePCAData(100, 10);
  results.push(
    bench("PCA (n=100, d=10)", () => pca(data, { nComponents: 3 }), 100),
  );
}

// ---------------------------------------------------------------------------
// Output results
// ---------------------------------------------------------------------------

console.log("\n=== MCMC & Clustering Benchmark ===\n");

const COL = { name: 40, time: 12, iters: 10, perIter: 14 };

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
