/**
 * Benchmark: TypeScript vs native Fortran special functions.
 *
 * Usage:
 *   npx ts-node benchmark/special-functions.ts
 *
 * Compares wall-clock time for gammaLn, gamma, betaFn, erf, erfc,
 * regularizedGammaP, and regularizedBeta across both implementations.
 * Each timing is the median of repeated samples to reduce hosted-runner noise.
 */

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
  // Native acceleration is optional.
}

interface BenchmarkResult {
  name: string;
  tsTimeMs: number;
  nativeTimeMs: number | null;
  iterations: number;
  speedup: number | null;
}

const ITERATIONS = 100_000;
const SAMPLES = 5;

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function measure(fn: () => void, iterations: number): number {
  for (let i = 0; i < Math.min(5_000, iterations); i++) fn();

  const samples: number[] = [];
  for (let sample = 0; sample < SAMPLES; sample++) {
    const start = performance.now();
    for (let i = 0; i < iterations; i++) fn();
    samples.push(performance.now() - start);
  }
  return median(samples);
}

function bench(
  name: string,
  tsFn: () => void,
  nativeFn: (() => void) | null,
  iterations: number,
): BenchmarkResult {
  const tsTimeMs = measure(tsFn, iterations);
  const nativeTimeMs = nativeFn ? measure(nativeFn, iterations) : null;

  return {
    name,
    tsTimeMs,
    nativeTimeMs,
    iterations,
    speedup: nativeTimeMs !== null ? tsTimeMs / nativeTimeMs : null,
  };
}

const benchmarks: BenchmarkResult[] = [
  bench("gammaLn(5.5)", () => mathModule.gammaLn(5.5), nativeMod ? () => nativeMod.gammaLn(5.5) : null, ITERATIONS),
  bench("gammaLn(0.1)", () => mathModule.gammaLn(0.1), nativeMod ? () => nativeMod.gammaLn(0.1) : null, ITERATIONS),
  bench("gamma(5.5)", () => mathModule.gamma(5.5), nativeMod ? () => nativeMod.gamma(5.5) : null, ITERATIONS),
  bench("betaFn(2, 5)", () => mathModule.betaFn(2, 5), nativeMod ? () => nativeMod.betaFn(2, 5) : null, ITERATIONS),
  bench("betaFn(0.5, 0.5)", () => mathModule.betaFn(0.5, 0.5), nativeMod ? () => nativeMod.betaFn(0.5, 0.5) : null, ITERATIONS),
  bench("erf(1.0)", () => mathModule.erf(1.0), nativeMod ? () => nativeMod.erf(1.0) : null, ITERATIONS),
  bench("erf(2.5)", () => mathModule.erf(2.5), nativeMod ? () => nativeMod.erf(2.5) : null, ITERATIONS),
  bench("erfc(1.0)", () => mathModule.erfc(1.0), nativeMod ? () => nativeMod.erfc(1.0) : null, ITERATIONS),
  bench("regularizedGammaP(2, 3)", () => mathModule.regularizedGammaP(2, 3), nativeMod ? () => nativeMod.regularizedGammaP(2, 3) : null, ITERATIONS),
  bench("regularizedGammaP(10, 5)", () => mathModule.regularizedGammaP(10, 5), nativeMod ? () => nativeMod.regularizedGammaP(10, 5) : null, ITERATIONS),
  bench("regularizedBeta(0.5, 2, 5)", () => mathModule.regularizedBeta(0.5, 2, 5), nativeMod ? () => nativeMod.regularizedBeta(0.5, 2, 5) : null, ITERATIONS),
  bench("regularizedBeta(0.3, 0.5, 0.5)", () => mathModule.regularizedBeta(0.3, 0.5, 0.5), nativeMod ? () => nativeMod.regularizedBeta(0.3, 0.5, 0.5) : null, ITERATIONS),
  bench("logFactorial(100)", () => mathModule.logFactorial(100), nativeMod ? () => nativeMod.logFactorial(100) : null, ITERATIONS),
  bench("binomialCoeff(20, 10)", () => mathModule.binomialCoeff(20, 10), nativeMod ? () => nativeMod.binomialCoeff(20, 10) : null, ITERATIONS),
];

console.log("\n=== Special Functions Benchmark ===");
console.log(`Iterations per sample: ${ITERATIONS.toLocaleString()}`);
console.log(`Samples per function: ${SAMPLES} (median reported)`);
console.log(
  `Native Fortran addon: ${nativeMod ? "AVAILABLE" : "NOT AVAILABLE (showing TS-only results)"}`,
);
console.log("");

const colWidths = { name: 32, ts: 12, native: 12, speedup: 10 };
const header =
  "Function".padEnd(colWidths.name) +
  "TS (ms)".padStart(colWidths.ts) +
  "Fortran (ms)".padStart(colWidths.native) +
  "Speedup".padStart(colWidths.speedup);

console.log(header);
console.log("-".repeat(header.length));

for (const result of benchmarks) {
  const tsStr = result.tsTimeMs.toFixed(2);
  const nativeStr = result.nativeTimeMs !== null ? result.nativeTimeMs.toFixed(2) : "N/A";
  const speedupStr = result.speedup !== null ? `${result.speedup.toFixed(2)}x` : "N/A";

  console.log(
    result.name.padEnd(colWidths.name) +
      tsStr.padStart(colWidths.ts) +
      nativeStr.padStart(colWidths.native) +
      speedupStr.padStart(colWidths.speedup),
  );
}

console.log("");
if (nativeMod) {
  const speedups = benchmarks
    .map((result) => result.speedup)
    .filter((speedup): speedup is number => speedup !== null);
  const average = speedups.reduce((sum, speedup) => sum + speedup, 0) / speedups.length;
  console.log(`Average speedup: ${average.toFixed(2)}x`);
  console.log(`Range: ${Math.min(...speedups).toFixed(2)}x - ${Math.max(...speedups).toFixed(2)}x`);
} else {
  console.log("Install the native addon to compare: npm run build:native");
}
console.log("");
