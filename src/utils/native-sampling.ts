import { nativeAddon } from "./native-addon";
import { SeededRng } from "../random";
import { BetaDistribution } from "../distributions/continuous/beta";
import { GammaDistribution } from "../distributions/continuous/gamma";
import { Normal } from "../distributions/continuous/normal";
import { Uniform } from "../distributions/continuous/uniform";

interface NativeSampling {
  uniformSampleBatch(n: number, a: number, b: number, seed: number): number[];
  normalSampleBatch(n: number, mu: number, sigma: number, seed: number): number[];
  gammaSampleBatch(n: number, shape: number, rate: number, seed: number): number[];
  betaSampleBatch(n: number, alpha: number, beta: number, seed: number): number[];
}

let native: NativeSampling | null = null;
try {
  if (nativeAddon && typeof nativeAddon.uniformSampleBatch === "function") {
    native = nativeAddon as unknown as NativeSampling;
  }
} catch {
  // Fallback to TypeScript
}

export const hasNativeSampling = native !== null;

function normalizeSeed(seed?: number): number {
  if (seed === undefined || seed === null) {
    return Math.floor(Math.random() * 2147483647);
  }
  if (!Number.isFinite(seed) || !Number.isInteger(seed)) {
    throw new Error(`Invalid parameter 'seed': expected an integer, received ${seed}`);
  }
  return Math.max(1, Math.min(2147483646, seed));
}

function seededRng(seed: number): () => number {
  const generator = new SeededRng(seed);
  return generator.next.bind(generator);
}

export function uniformSampleBatch(
  n: number,
  a: number,
  b: number,
  seed?: number,
): number[] {
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`Invalid parameter 'n': expected a non-negative integer, received ${n}`);
  }
  if (b <= a) {
    throw new Error(`Invalid parameters 'a', 'b': expected a < b, received a=${a}, b=${b}`);
  }
  if (n === 0) return [];

  const actualSeed = normalizeSeed(seed);
  if (native) {
    return native.uniformSampleBatch(n, a, b, actualSeed);
  }

  const rng = seededRng(actualSeed);
  return new Uniform(a, b, rng).sampleN(n);
}

export function normalSampleBatch(
  n: number,
  mu: number,
  sigma: number,
  seed?: number,
): number[] {
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`Invalid parameter 'n': expected a non-negative integer, received ${n}`);
  }
  if (sigma <= 0) {
    throw new Error(`Invalid parameter 'sigma': expected a positive number, received ${sigma}`);
  }
  if (n === 0) return [];

  const actualSeed = normalizeSeed(seed);
  if (native) {
    return native.normalSampleBatch(n, mu, sigma, actualSeed);
  }

  const rng = seededRng(actualSeed);
  return new Normal(mu, sigma, rng).sampleN(n);
}

export function gammaSampleBatch(
  n: number,
  shape: number,
  rate: number,
  seed?: number,
): number[] {
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`Invalid parameter 'n': expected a non-negative integer, received ${n}`);
  }
  if (shape <= 0) {
    throw new Error(`Invalid parameter 'shape': expected a positive number, received ${shape}`);
  }
  if (rate <= 0) {
    throw new Error(`Invalid parameter 'rate': expected a positive number, received ${rate}`);
  }
  if (n === 0) return [];

  const actualSeed = normalizeSeed(seed);
  if (native) {
    return native.gammaSampleBatch(n, shape, rate, actualSeed);
  }

  const rng = seededRng(actualSeed);
  return new GammaDistribution(shape, rate, rng).sampleN(n);
}

export function betaSampleBatch(
  n: number,
  alpha: number,
  beta: number,
  seed?: number,
): number[] {
  if (!Number.isInteger(n) || n < 0) {
    throw new Error(`Invalid parameter 'n': expected a non-negative integer, received ${n}`);
  }
  if (alpha <= 0) {
    throw new Error(`Invalid parameter 'alpha': expected a positive number, received ${alpha}`);
  }
  if (beta <= 0) {
    throw new Error(`Invalid parameter 'beta': expected a positive number, received ${beta}`);
  }
  if (n === 0) return [];

  const actualSeed = normalizeSeed(seed);
  if (native) {
    return native.betaSampleBatch(n, alpha, beta, actualSeed);
  }

  const rng = seededRng(actualSeed);
  return new BetaDistribution(alpha, beta, rng).sampleN(n);
}
