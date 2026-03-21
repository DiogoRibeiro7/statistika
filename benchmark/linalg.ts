/**
 * Benchmark: TypeScript vs native LAPACK linear algebra operations.
 *
 * Usage:
 *   npx ts-node benchmark/linalg.ts
 *
 * Compares wall-clock time for matMul, solveLinearSystem, invertMatrix,
 * symmetricEigen, and normalCdf across both implementations at matrix
 * sizes 10×10, 50×50, 100×100, and 200×200.
 */

import { createRequire } from "node:module";
import { join } from "node:path";

// ── Load pure-TS implementations directly ──────────────────────────────

// We need the pure-TS fallback functions without native dispatch.
// Re-implement the core loops inline so we can benchmark them in isolation.

type Matrix = number[][];

function tsMatMul(A: Matrix, B: Matrix): Matrix {
  const m = A.length;
  const n = B[0].length;
  const k = B.length;
  const C: Matrix = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let l = 0; l < k; l++) sum += A[i][l] * B[l][j];
      C[i][j] = sum;
    }
  }
  return C;
}

function tsSolve(A: Matrix, b: number[]): number[] {
  const n = A.length;
  const aug: number[][] = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    if (Math.abs(aug[col][col]) < 1e-12) throw new Error("Singular");
    for (let row = col + 1; row < n; row++) {
      const f = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) aug[row][j] -= f * aug[col][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    x[row] = aug[row][n];
    for (let col = row + 1; col < n; col++) x[row] -= aug[row][col] * x[col];
    x[row] /= aug[row][row];
  }
  return x;
}

function tsInvert(M: Matrix): Matrix | null {
  const n = M.length;
  const aug: number[][] = M.map((row, i) => {
    const id = new Array(n).fill(0);
    id[i] = 1;
    return [...row, ...id];
  });
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-12) return null;
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = aug[row][col];
      for (let j = 0; j < 2 * n; j++) aug[row][j] -= f * aug[col][j];
    }
  }
  return aug.map((row) => row.slice(n));
}

function tsSymEigen(
  A: Matrix,
): { eigenvalues: number[]; eigenvectors: Matrix } {
  const n = A.length;
  const S: Matrix = A.map((row) => [...row]);
  const V: Matrix = Array.from({ length: n }, (_, i) => {
    const row = new Array(n).fill(0);
    row[i] = 1;
    return row;
  });
  for (let iter = 0; iter < 200; iter++) {
    let maxVal = 0;
    let p = 0;
    let q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(S[i][j]) > maxVal) {
          maxVal = Math.abs(S[i][j]);
          p = i;
          q = j;
        }
      }
    }
    if (maxVal < 1e-12) break;
    const theta =
      Math.abs(S[p][p] - S[q][q]) < 1e-15
        ? Math.PI / 4
        : 0.5 * Math.atan2(2 * S[p][q], S[p][p] - S[q][q]);
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const Spp = c * c * S[p][p] + 2 * s * c * S[p][q] + s * s * S[q][q];
    const Sqq = s * s * S[p][p] - 2 * s * c * S[p][q] + c * c * S[q][q];
    S[p][p] = Spp;
    S[q][q] = Sqq;
    S[p][q] = 0;
    S[q][p] = 0;
    for (let i = 0; i < n; i++) {
      if (i !== p && i !== q) {
        const Sip = c * S[i][p] + s * S[i][q];
        const Siq = -s * S[i][p] + c * S[i][q];
        S[i][p] = Sip;
        S[p][i] = Sip;
        S[i][q] = Siq;
        S[q][i] = Siq;
      }
    }
    for (let i = 0; i < n; i++) {
      const Vip = c * V[i][p] + s * V[i][q];
      const Viq = -s * V[i][p] + c * V[i][q];
      V[i][p] = Vip;
      V[i][q] = Viq;
    }
  }
  const eigenvalues = new Array(n);
  for (let i = 0; i < n; i++) eigenvalues[i] = S[i][i];
  return { eigenvalues, eigenvectors: V };
}

function tsNormalCdf(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1.0 / (1.0 + p * ax);
  const y =
    1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return 0.5 * (1.0 + sign * y);
}

// ── Load native addon ──────────────────────────────────────────────────

interface NativeLinalg {
  matMul(A: Matrix, B: Matrix, m: number, k: number, n: number): Matrix;
  solve(A: Matrix, b: number[], n: number): { x: number[]; info: number };
  invert(A: Matrix, n: number): { inv: Matrix; info: number };
  symEigen(
    A: Matrix,
    n: number,
  ): { eigenvalues: number[]; eigenvectors: Matrix; info: number };
  normalCdf(x: number): number;
}

let native: NativeLinalg | null = null;
try {
  const projectRoot =
    typeof __dirname !== "undefined"
      ? join(__dirname, "..")
      : process.cwd();
  const require_ = createRequire(join(projectRoot, "package.json"));
  const addon = require_(
    join(projectRoot, "build", "Release", "fortran_special.node"),
  ) as NativeLinalg;
  const probe = addon.solve(
    [
      [1, 0],
      [0, 1],
    ],
    [1, 1],
    2,
  );
  if (probe && probe.info === 0) native = addon;
} catch {
  // Native not available
}

// ── Test data generators ───────────────────────────────────────────────

/** Generate an n×n random matrix with values in [0, 1). */
function randomMatrix(n: number, seed = 42): Matrix {
  // Simple LCG for reproducibility
  let s = seed;
  const next = () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  return Array.from({ length: n }, () =>
    Array.from({ length: n }, () => next()),
  );
}

/** Generate an n×n random symmetric positive-definite matrix (for eigen). */
function randomSymmetricPD(n: number, seed = 42): Matrix {
  const A = randomMatrix(n, seed);
  // M = A^T * A + n*I  (guarantees positive definiteness)
  const M: Matrix = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += A[k][i] * A[k][j];
      M[i][j] = sum + (i === j ? n : 0);
    }
  }
  return M;
}

/** Generate a random vector of length n. */
function randomVector(n: number, seed = 99): number[] {
  let s = seed;
  const next = () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  return Array.from({ length: n }, () => next());
}

// ── Benchmark harness ──────────────────────────────────────────────────

interface BenchResult {
  operation: string;
  size: string;
  tsMs: number;
  nativeMs: number | null;
  iterations: number;
  speedup: number | null;
}

function bench(
  operation: string,
  size: string,
  tsFn: () => void,
  nativeFn: (() => void) | null,
  iterations: number,
): BenchResult {
  // Warmup
  const warmup = Math.min(Math.max(1, Math.floor(iterations / 10)), 100);
  for (let i = 0; i < warmup; i++) tsFn();
  if (nativeFn) for (let i = 0; i < warmup; i++) nativeFn();

  // Measure TS
  const tsStart = performance.now();
  for (let i = 0; i < iterations; i++) tsFn();
  const tsMs = performance.now() - tsStart;

  // Measure Native
  let nativeMs: number | null = null;
  if (nativeFn) {
    const nStart = performance.now();
    for (let i = 0; i < iterations; i++) nativeFn();
    nativeMs = performance.now() - nStart;
  }

  return {
    operation,
    size,
    tsMs,
    nativeMs,
    iterations,
    speedup: nativeMs !== null && nativeMs > 0 ? tsMs / nativeMs : null,
  };
}

// ── Run benchmarks ─────────────────────────────────────────────────────

const SIZES = [10, 50, 100, 200, 500];

// Scale iterations inversely with expected O(n^3) cost so total time is bounded.
// Ensure at least 3 iterations for statistical reliability.
function itersForSize(n: number, baseOps: number): number {
  const scale = (10 / n) ** 3;
  return Math.max(3, Math.round(baseOps * scale));
}

const results: BenchResult[] = [];

console.log("\nGenerating test matrices...");

for (const n of SIZES) {
  const A = randomMatrix(n);
  const B = randomMatrix(n, 73);
  const Asym = randomSymmetricPD(n);
  const b = randomVector(n);
  const iters = itersForSize(n, 5000);

  console.log(`  ${n}×${n}: ${iters} iterations`);

  // matMul
  results.push(
    bench(
      "matMul",
      `${n}×${n}`,
      () => tsMatMul(A, B),
      native ? () => native!.matMul(A, B, n, n, n) : null,
      iters,
    ),
  );

  // solve
  results.push(
    bench(
      "solve",
      `${n}×${n}`,
      () => tsSolve(A, b),
      native ? () => native!.solve(A, b, n) : null,
      iters,
    ),
  );

  // invert
  results.push(
    bench(
      "invert",
      `${n}×${n}`,
      () => tsInvert(A),
      native ? () => native!.invert(A, n) : null,
      iters,
    ),
  );

  // symmetricEigen
  results.push(
    bench(
      "symEigen",
      `${n}×${n}`,
      () => tsSymEigen(Asym),
      native ? () => native!.symEigen(Asym, n) : null,
      iters,
    ),
  );
}

// normalCdf (scalar — high iteration count)
const SCALAR_ITERS = 500_000;
results.push(
  bench(
    "normalCdf",
    "scalar",
    () => tsNormalCdf(1.96),
    native ? () => native!.normalCdf(1.96) : null,
    SCALAR_ITERS,
  ),
);

// ── Print results ──────────────────────────────────────────────────────

console.log("\n=== Linear Algebra Benchmark ===");
console.log(
  `Native LAPACK: ${native ? "AVAILABLE" : "NOT AVAILABLE (showing TS-only results)"}`,
);
console.log("");

const COL = { op: 14, size: 8, iters: 8, ts: 12, native: 14, speedup: 10 };

const header =
  "Operation".padEnd(COL.op) +
  "Size".padStart(COL.size) +
  "Iters".padStart(COL.iters) +
  "TS (ms)".padStart(COL.ts) +
  "LAPACK (ms)".padStart(COL.native) +
  "Speedup".padStart(COL.speedup);

console.log(header);
console.log("─".repeat(header.length));

let currentOp = "";
for (const r of results) {
  if (r.operation !== currentOp && currentOp !== "") {
    // Visual separator between different operations
    console.log("");
  }
  currentOp = r.operation;

  const tsStr = r.tsMs.toFixed(2);
  const nativeStr = r.nativeMs !== null ? r.nativeMs.toFixed(2) : "N/A";
  const speedupStr =
    r.speedup !== null ? `${r.speedup.toFixed(1)}x` : "N/A";

  console.log(
    r.operation.padEnd(COL.op) +
      r.size.padStart(COL.size) +
      r.iterations.toString().padStart(COL.iters) +
      tsStr.padStart(COL.ts) +
      nativeStr.padStart(COL.native) +
      speedupStr.padStart(COL.speedup),
  );
}

// Summary
console.log("");
if (native) {
  const matrixResults = results.filter((r) => r.speedup !== null && r.size !== "scalar");
  const scalarResults = results.filter((r) => r.speedup !== null && r.size === "scalar");

  if (matrixResults.length > 0) {
    const speedups = matrixResults.map((r) => r.speedup!);
    const avg = speedups.reduce((a, b) => a + b, 0) / speedups.length;
    const max = Math.max(...speedups);
    const min = Math.min(...speedups);
    console.log(`Matrix operations — average speedup: ${avg.toFixed(1)}x  (range: ${min.toFixed(1)}x – ${max.toFixed(1)}x)`);
  }
  if (scalarResults.length > 0) {
    console.log(`normalCdf — speedup: ${scalarResults[0].speedup!.toFixed(1)}x`);
  }
} else {
  console.log("Install LAPACK to compare: apt install liblapack-dev && npm run build:native");
}
console.log("");
