import { nativeAddon } from "./native-addon";
import { SeededRng } from "../random";
import { normalSampleBatch } from "./native-sampling";

interface NativeVariational {
  normalLogDensityBatch(n: number, mu: number, sigma: number, x: number[]): number[];
}

let native: NativeVariational | null = null;
try {
  if (nativeAddon && typeof nativeAddon.normalLogDensityBatch === "function") {
    native = nativeAddon as unknown as NativeVariational;
  }
} catch {
  // Fall back to TypeScript.
}

export const hasNativeVariational = native !== null;

function normalizeSeed(seed?: number): number {
  if (seed === undefined || seed === null) {
    return Math.floor(Math.random() * 2147483647);
  }
  if (!Number.isFinite(seed) || !Number.isInteger(seed)) {
    throw new Error(`Invalid parameter 'seed': expected an integer, received ${seed}`);
  }
  return Math.max(1, Math.min(2147483646, seed));
}

export function normalLogDensityBatch(
  n: number,
  mu: number,
  sigma: number,
  x: number[],
): number[] {
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`Invalid parameter 'n': expected a non-negative integer, received ${n}`);
  }
  if (sigma <= 0) {
    throw new Error(`Invalid parameter 'sigma': expected a positive number, received ${sigma}`);
  }
  if (x.length !== n) {
    throw new Error(`Invalid parameter 'x': expected array of length ${n}, received ${x.length}`);
  }
  if (n === 0) return [];

  if (native) {
    return native.normalLogDensityBatch(n, mu, sigma, x);
  }

  const varSigma = sigma * sigma;
  const base = -0.5 * Math.log(2 * Math.PI * varSigma);
  return x.map((value) => {
    const diff = value - mu;
    return base - 0.5 * (diff * diff) / varSigma;
  });
}
