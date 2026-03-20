/**
 * Benchmark: TypeScript vs native Fortran special functions.
 *
 * Usage:
 *   npx ts-node benchmark/special-functions.ts
 *
 * Compares wall-clock time for gammaLn, gamma, betaFn, erf, erfc,
 * regularizedGammaP, and regularizedBeta across both implementations.
 */

// Force-load the pure-TS implementations by reaching into the internals.
// The public API auto-dispatches, so we need both paths explicitly.

// eslint-disable-next-line @typescript-eslint/no-require-imports
const mathModule = require("../src/utils/math");

interface NativeSpecial {
  gammaLn(x: number): number;
  gamma(x: number): number;
  logFactorial(n: number): number;
  factorial(n: number): number;
  binomialCoeff(n: number, k: number): number;
  betaFn(a: number, b: number): number;
  erf(x: number): number;
  erfc(x: number): number;
  regularizedGammaP(s: number, x: number): number;
  regularizedBeta(x: number, a: number, b: number): number;
}

let nativeMod: NativeSpecial | null = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  nativeMod = require("../build/Release/fortran_special.node") as NativeSpecial;
} catch {
  // Native not available
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

interface BenchmarkResult {
  name: string;
  tsTimeMs: number;
  nativeTimeMs: number | null;
  iterations: number;
  speedup: number | null;
}

function bench(
  name: string,
  tsFn: () => void,
  nativeFn: (() => void) | null,
  iterations: number,
): BenchmarkResult {
  // Warmup
  for (let i = 0; i < Math.min(1000, iterations); i++) tsFn();
  if (nativeFn) {
    for (let i = 0; i < Math.min(1000, iterations); i++) nativeFn();
  }

  // TypeScript
  const tsStart = performance.now();
  for (let i = 0; i < iterations; i++) tsFn();
  const tsEnd = performance.now();
  const tsTimeMs = tsEnd - tsStart;

  // Native Fortran
  let nativeTimeMs: number | null = null;
  if (nativeFn) {
    const nStart = performance.now();
    for (let i = 0; i < iterations; i++) nativeFn();
    const nEnd = performance.now();
    nativeTimeMs = nEnd - nStart;
  }

  return {
    name,
    tsTimeMs,
    nativeTimeMs,
    iterations,
    speedup: nativeTimeMs !== null ? tsTimeMs / nativeTimeMs : null,
  };
}

// ---------------------------------------------------------------------------
// Test inputs
// ---------------------------------------------------------------------------

const ITERATIONS = 100_000;

const benchmarks: BenchmarkResult[] = [];

// gammaLn
benchmarks.push(
  bench(
    "gammaLn(5.5)",
    () => mathModule.gammaLn(5.5),
    nativeMod ? () => nativeMod!.gammaLn(5.5) : null,
    ITERATIONS,
  ),
);

benchmarks.push(
  bench(
    "gammaLn(0.1)",
    () => mathModule.gammaLn(0.1),
    nativeMod ? () => nativeMod!.gammaLn(0.1) : null,
    ITERATIONS,
  ),
);

// gamma
benchmarks.push(
  bench(
    "gamma(5.5)",
    () => mathModule.gamma(5.5),
    nativeMod ? () => nativeMod!.gamma(5.5) : null,
    ITERATIONS,
  ),
);

// betaFn
benchmarks.push(
  bench(
    "betaFn(2, 5)",
    () => mathModule.betaFn(2, 5),
    nativeMod ? () => nativeMod!.betaFn(2, 5) : null,
    ITERATIONS,
  ),
);

benchmarks.push(
  bench(
    "betaFn(0.5, 0.5)",
    () => mathModule.betaFn(0.5, 0.5),
    nativeMod ? () => nativeMod!.betaFn(0.5, 0.5) : null,
    ITERATIONS,
  ),
);

// erf
benchmarks.push(
  bench(
    "erf(1.0)",
    () => mathModule.erf(1.0),
    nativeMod ? () => nativeMod!.erf(1.0) : null,
    ITERATIONS,
  ),
);

benchmarks.push(
  bench(
    "erf(2.5)",
    () => mathModule.erf(2.5),
    nativeMod ? () => nativeMod!.erf(2.5) : null,
    ITERATIONS,
  ),
);

// erfc
benchmarks.push(
  bench(
    "erfc(1.0)",
    () => mathModule.erfc(1.0),
    nativeMod ? () => nativeMod!.erfc(1.0) : null,
    ITERATIONS,
  ),
);

// regularizedGammaP
benchmarks.push(
  bench(
    "regularizedGammaP(2, 3)",
    () => mathModule.regularizedGammaP(2, 3),
    nativeMod ? () => nativeMod!.regularizedGammaP(2, 3) : null,
    ITERATIONS,
  ),
);

benchmarks.push(
  bench(
    "regularizedGammaP(10, 5)",
    () => mathModule.regularizedGammaP(10, 5),
    nativeMod ? () => nativeMod!.regularizedGammaP(10, 5) : null,
    ITERATIONS,
  ),
);

// regularizedBeta
benchmarks.push(
  bench(
    "regularizedBeta(0.5, 2, 5)",
    () => mathModule.regularizedBeta(0.5, 2, 5),
    nativeMod ? () => nativeMod!.regularizedBeta(0.5, 2, 5) : null,
    ITERATIONS,
  ),
);

benchmarks.push(
  bench(
    "regularizedBeta(0.3, 0.5, 0.5)",
    () => mathModule.regularizedBeta(0.3, 0.5, 0.5),
    nativeMod ? () => nativeMod!.regularizedBeta(0.3, 0.5, 0.5) : null,
    ITERATIONS,
  ),
);

// logFactorial
benchmarks.push(
  bench(
    "logFactorial(100)",
    () => mathModule.logFactorial(100),
    nativeMod ? () => nativeMod!.logFactorial(100) : null,
    ITERATIONS,
  ),
);

// binomialCoeff
benchmarks.push(
  bench(
    "binomialCoeff(20, 10)",
    () => mathModule.binomialCoeff(20, 10),
    nativeMod ? () => nativeMod!.binomialCoeff(20, 10) : null,
    ITERATIONS,
  ),
);

// ---------------------------------------------------------------------------
// Output results
// ---------------------------------------------------------------------------

console.log("\n=== Special Functions Benchmark ===");
console.log(`Iterations per function: ${ITERATIONS.toLocaleString()}`);
console.log(
  `Native Fortran addon: ${nativeMod ? "AVAILABLE" : "NOT AVAILABLE (showing TS-only results)"}`,
);
console.log("");

const colWidths = {
  name: 32,
  ts: 12,
  native: 12,
  speedup: 10,
};

const header =
  "Function".padEnd(colWidths.name) +
  "TS (ms)".padStart(colWidths.ts) +
  "Fortran (ms)".padStart(colWidths.native) +
  "Speedup".padStart(colWidths.speedup);

console.log(header);
console.log("-".repeat(header.length));

for (const r of benchmarks) {
  const tsStr = r.tsTimeMs.toFixed(2);
  const nativeStr = r.nativeTimeMs !== null ? r.nativeTimeMs.toFixed(2) : "N/A";
  const speedupStr =
    r.speedup !== null ? `${r.speedup.toFixed(2)}x` : "N/A";

  console.log(
    r.name.padEnd(colWidths.name) +
      tsStr.padStart(colWidths.ts) +
      nativeStr.padStart(colWidths.native) +
      speedupStr.padStart(colWidths.speedup),
  );
}

// Summary
console.log("");
if (nativeMod) {
  const speedups = benchmarks
    .map((r) => r.speedup)
    .filter((s): s is number => s !== null);
  const avgSpeedup = speedups.reduce((a, b) => a + b, 0) / speedups.length;
  const maxSpeedup = Math.max(...speedups);
  const minSpeedup = Math.min(...speedups);
  console.log(`Average speedup: ${avgSpeedup.toFixed(2)}x`);
  console.log(
    `Range: ${minSpeedup.toFixed(2)}x - ${maxSpeedup.toFixed(2)}x`,
  );
} else {
  console.log("Install the native addon to compare: npm run build:native");
}
console.log("");
