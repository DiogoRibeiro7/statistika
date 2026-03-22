/**
 * Benchmark: Statistical methods performance.
 *
 * Usage:
 *   npx ts-node benchmark/statistical-methods.ts
 *
 * Times kMeans clustering, PCA, gaussianMixture EM, LOESS smoothing,
 * bootstrapCI resampling, metropolisHastingsND MCMC, and kFoldCV
 * cross-validation at various problem sizes.
 */

import {
  kMeans,
  pca,
  gaussianMixture,
  loess,
  bootstrapCI,
  metropolisHastingsND,
  kFoldCV,
  mean,
} from "../src";

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

interface BenchmarkResult {
  name: string;
  timeMs: number;
  iterations: number;
  avgMs: number;
}

function bench(
  name: string,
  fn: () => void,
  iterations: number,
): BenchmarkResult {
  // Warmup
  const warmup = Math.min(3, iterations);
  for (let i = 0; i < warmup; i++) fn();

  // Timed run
  const start = performance.now();
  for (let i = 0; i < iterations; i++) fn();
  const end = performance.now();
  const timeMs = end - start;

  return {
    name,
    timeMs,
    iterations,
    avgMs: timeMs / iterations,
  };
}

// ---------------------------------------------------------------------------
// Data generators (seeded via deterministic construction)
// ---------------------------------------------------------------------------

function generateClusters(n: number, dims: number, k: number): number[][] {
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const cluster = i % k;
    const row: number[] = [];
    for (let d = 0; d < dims; d++) {
      // Deterministic pseudo-random offset per cluster
      row.push(cluster * 5 + Math.sin(i * 13 + d * 7) * 1.5);
    }
    data.push(row);
  }
  return data;
}

function generateMatrix(n: number, p: number): number[][] {
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < p; j++) {
      row.push(Math.sin(i * 7 + j * 13) + Math.cos(i * 3 + j * 11));
    }
    data.push(row);
  }
  return data;
}

function generateSeries(n: number): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const xi = i / n;
    x.push(xi);
    y.push(Math.sin(xi * 6) + Math.cos(xi * 2) + Math.sin(i * 17) * 0.3);
  }
  return { x, y };
}

function generateDataset(n: number): number[] {
  const data: number[] = [];
  for (let i = 0; i < n; i++) {
    data.push(Math.sin(i * 7) * 10 + 50 + Math.cos(i * 13) * 5);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Pre-generate test data
// ---------------------------------------------------------------------------

const clusters100 = generateClusters(100, 2, 3);
const clusters500 = generateClusters(500, 2, 3);
const clusters1000 = generateClusters(1000, 2, 5);

const matrix50x5 = generateMatrix(50, 5);
const matrix100x10 = generateMatrix(100, 10);
const matrix200x20 = generateMatrix(200, 20);

const series100 = generateSeries(100);
const series500 = generateSeries(500);

const dataset100 = generateDataset(100);
const dataset500 = generateDataset(500);

// Simple linear model for kFoldCV
const cvX = matrix50x5;
const cvY = cvX.map((row) => row[0] * 2 + row[1] * 0.5 + Math.sin(row[2]));

// ---------------------------------------------------------------------------
// Run benchmarks
// ---------------------------------------------------------------------------

const results: BenchmarkResult[] = [];

console.log("\nRunning benchmarks...\n");

// -- kMeans clustering --
results.push(
  bench(
    "kMeans  n=100, k=3, 2d",
    () => kMeans(clusters100, 3, { maxIter: 50, seed: 42 }),
    50,
  ),
);

results.push(
  bench(
    "kMeans  n=500, k=3, 2d",
    () => kMeans(clusters500, 3, { maxIter: 50, seed: 42 }),
    20,
  ),
);

results.push(
  bench(
    "kMeans  n=1000, k=5, 2d",
    () => kMeans(clusters1000, 5, { maxIter: 50, seed: 42 }),
    10,
  ),
);

// -- PCA --
results.push(
  bench(
    "PCA  50x5",
    () => pca(matrix50x5, { nComponents: 3 }),
    100,
  ),
);

results.push(
  bench(
    "PCA  100x10",
    () => pca(matrix100x10, { nComponents: 5 }),
    50,
  ),
);

results.push(
  bench(
    "PCA  200x20",
    () => pca(matrix200x20, { nComponents: 10 }),
    10,
  ),
);

// -- gaussianMixture EM --
results.push(
  bench(
    "gaussianMixture  n=100, k=2",
    () => gaussianMixture(dataset100, 2, { maxIterations: 50, seed: 42 }),
    20,
  ),
);

results.push(
  bench(
    "gaussianMixture  n=500, k=3",
    () => gaussianMixture(dataset500, 3, { maxIterations: 50, seed: 42 }),
    10,
  ),
);

// -- LOESS smoothing --
results.push(
  bench(
    "loess  n=100, span=0.3",
    () => loess(series100.x, series100.y, 0.3),
    100,
  ),
);

results.push(
  bench(
    "loess  n=500, span=0.3",
    () => loess(series500.x, series500.y, 0.3),
    10,
  ),
);

// -- bootstrapCI --
results.push(
  bench(
    "bootstrapCI  n=100, B=1000",
    () => bootstrapCI(dataset100, mean, { nReplicates: 1000, seed: 42 }),
    20,
  ),
);

results.push(
  bench(
    "bootstrapCI  n=500, B=2000",
    () => bootstrapCI(dataset500, mean, { nReplicates: 2000, seed: 42 }),
    5,
  ),
);

// -- metropolisHastingsND MCMC --
const logMvNormal = (x: number[]): number =>
  -0.5 * x.reduce((s, xi) => s + xi * xi, 0);

results.push(
  bench(
    "metropolisHastingsND  2d, 1000 iter",
    () =>
      metropolisHastingsND(logMvNormal, 2, {
        iterations: 1000,
        burnIn: 200,
        seed: 42,
      }),
    20,
  ),
);

results.push(
  bench(
    "metropolisHastingsND  5d, 2000 iter",
    () =>
      metropolisHastingsND(logMvNormal, 5, {
        iterations: 2000,
        burnIn: 400,
        seed: 42,
      }),
    10,
  ),
);

// -- kFoldCV cross-validation --
const simpleFitPredict = (
  trainX: number[][],
  trainY: number[],
  testX: number[][],
): number[] => {
  // Naive mean predictor for benchmarking purposes
  const m = trainY.reduce((a, b) => a + b, 0) / trainY.length;
  return testX.map(() => m);
};

results.push(
  bench(
    "kFoldCV  n=50, k=5",
    () => kFoldCV(cvX, cvY, simpleFitPredict, { k: 5, seed: 42 }),
    50,
  ),
);

results.push(
  bench(
    "kFoldCV  n=50, k=10",
    () => kFoldCV(cvX, cvY, simpleFitPredict, { k: 10, seed: 42 }),
    50,
  ),
);

// ---------------------------------------------------------------------------
// Output results
// ---------------------------------------------------------------------------

console.log("=== Statistical Methods Benchmark ===\n");

const colWidths = {
  name: 38,
  iters: 8,
  total: 14,
  avg: 14,
};

const header =
  "Method".padEnd(colWidths.name) +
  "Iters".padStart(colWidths.iters) +
  "Total (ms)".padStart(colWidths.total) +
  "Avg (ms)".padStart(colWidths.avg);

console.log(header);
console.log("-".repeat(header.length));

for (const r of results) {
  console.log(
    r.name.padEnd(colWidths.name) +
      String(r.iterations).padStart(colWidths.iters) +
      r.timeMs.toFixed(2).padStart(colWidths.total) +
      r.avgMs.toFixed(3).padStart(colWidths.avg),
  );
}

// Summary
console.log("");
const totalTime = results.reduce((s, r) => s + r.timeMs, 0);
console.log(`Total benchmark time: ${(totalTime / 1000).toFixed(2)}s`);
const slowest = results.reduce((a, b) => (a.avgMs > b.avgMs ? a : b));
const fastest = results.reduce((a, b) => (a.avgMs < b.avgMs ? a : b));
console.log(`Slowest (avg): ${slowest.name.trim()}  (${slowest.avgMs.toFixed(3)} ms)`);
console.log(`Fastest (avg): ${fastest.name.trim()}  (${fastest.avgMs.toFixed(3)} ms)`);
console.log("");
