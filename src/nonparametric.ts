import { Dataset } from "./types";
import { mean } from "./utils/descriptive";
import { createRng, normalCdf, normalQuantile } from "./utils/linalg";

// ── Kernel Density Estimation ───────────────────────────────────────────

/**
 * Result of a kernel density estimation.
 */
export interface KDEResult {
  /** Evaluation points on the x-axis. */
  x: number[];
  /** Estimated density at each evaluation point. */
  density: number[];
  /** Bandwidth used. */
  bandwidth: number;
  /** Kernel function name. */
  kernel: string;
}

/**
 * A kernel function K(u) used in kernel density estimation.
 * Takes a standardized distance u and returns the kernel weight.
 */
export type KernelFunction = (u: number) => number;

const kernels: Record<string, KernelFunction> = {
  gaussian: (u) => Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI),
  epanechnikov: (u) => (Math.abs(u) <= 1 ? 0.75 * (1 - u * u) : 0),
  uniform: (u) => (Math.abs(u) <= 1 ? 0.5 : 0),
  triangular: (u) => (Math.abs(u) <= 1 ? 1 - Math.abs(u) : 0),
};

/**
 * Silverman's rule-of-thumb bandwidth selector.
 * h = 0.9 * min(stdDev, IQR/1.34) * n^(-1/5)
 */
function silvermanBandwidth(data: Dataset): number {
  const n = data.length;
  const m = mean(data);
  const s = Math.sqrt(data.reduce((acc, v) => acc + (v - m) ** 2, 0) / (n - 1));
  const sorted = [...data].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(n * 0.25)];
  const q3 = sorted[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  const spread = Math.min(s, iqr / 1.34);
  return 0.9 * spread * Math.pow(n, -0.2);
}

/**
 * Kernel density estimation.
 *
 * Estimates the probability density function of a continuous random variable
 * using a kernel smoothing approach. For each evaluation point x, the density
 * is estimated as: f(x) = (1 / (n * h)) * sum(K((x - x_i) / h)) where K is
 * the kernel function and h is the bandwidth.
 *
 * @param data - Input sample data
 * @param options - Configuration options
 * @param options.bandwidth - Smoothing bandwidth (default: Silverman's rule)
 * @param options.kernel - Kernel function: "gaussian", "epanechnikov", "uniform", "triangular" (default: "gaussian")
 * @param options.nPoints - Number of evaluation points (default: 512)
 * @param options.from - Lower bound of evaluation range (default: min - 3*bandwidth)
 * @param options.to - Upper bound of evaluation range (default: max + 3*bandwidth)
 * @returns KDEResult with evaluation points, density values, bandwidth, and kernel name
 * @throws Error if dataset has fewer than 2 elements
 * @throws Error if an unknown kernel name is specified
 * @throws Error if bandwidth is not positive
 *
 * @example
 * ```ts
 * const data = [1, 1.5, 2, 2.5, 3, 3.5, 4];
 * const result = kernelDensity(data, { kernel: "gaussian", nPoints: 100 });
 * // result.x — 100 evaluation points
 * // result.density — estimated density at each point
 * ```
 */
export function kernelDensity(
  data: Dataset,
  options: {
    bandwidth?: number;
    kernel?: string;
    nPoints?: number;
    from?: number;
    to?: number;
  } = {},
): KDEResult {
  if (data.length < 2) throw new Error(`Invalid parameter 'data': expected at least 2 elements, received ${data.length}`);

  const kernelName = options.kernel ?? "gaussian";
  const K = kernels[kernelName];
  if (!K) {
    throw new Error(`Unknown kernel: ${kernelName}`);
  }

  const h = options.bandwidth ?? silvermanBandwidth(data);
  if (h <= 0) throw new Error(`Invalid parameter 'bandwidth': expected a positive number, received ${h}`);

  const n = data.length;
  const nPoints = options.nPoints ?? 512;
  const dataMin = Math.min(...data);
  const dataMax = Math.max(...data);
  const from = options.from ?? dataMin - 3 * h;
  const to = options.to ?? dataMax + 3 * h;

  const x = new Array<number>(nPoints);
  const density = new Array<number>(nPoints);
  const step = (to - from) / (nPoints - 1);

  for (let i = 0; i < nPoints; i++) {
    x[i] = from + i * step;
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += K((x[i] - data[j]) / h);
    }
    density[i] = sum / (n * h);
  }

  return { x, density, bandwidth: h, kernel: kernelName };
}

// ── Bootstrap Confidence Intervals ──────────────────────────────────────

/**
 * Result of a bootstrap confidence interval estimation.
 */
export interface BootstrapCIResult {
  /** The statistic computed on the original sample. */
  estimate: number;
  /** Lower bound of the confidence interval. */
  lower: number;
  /** Upper bound of the confidence interval. */
  upper: number;
  /** Confidence level. */
  confidenceLevel: number;
  /** Bootstrap standard error. */
  standardError: number;
  /** Method used. */
  method: "percentile" | "bca";
  /** Number of bootstrap replicates. */
  nReplicates: number;
}

/**
 * Bootstrap confidence interval.
 *
 * Estimates a confidence interval for any statistic by resampling with
 * replacement. Supports percentile and BCa (bias-corrected and accelerated)
 * methods. The BCa method adjusts for bias and skewness in the bootstrap
 * distribution using jackknife acceleration.
 *
 * @param data - Input sample data
 * @param statistic - Function that computes the statistic of interest from a sample
 * @param options - Configuration options
 * @param options.confidence - Confidence level (default: 0.95)
 * @param options.nReplicates - Number of bootstrap resamples (default: 10000)
 * @param options.method - "percentile" or "bca" (default: "percentile")
 * @param options.seed - Random seed for reproducibility
 * @returns BootstrapCIResult with the estimate, confidence interval bounds, and standard error
 * @throws Error if dataset has fewer than 2 elements
 *
 * @example
 * ```ts
 * const data = [2, 4, 6, 8, 10, 12];
 * const result = bootstrapCI(data, (s) => s.reduce((a, b) => a + b) / s.length, {
 *   confidence: 0.95,
 *   nReplicates: 5000,
 *   seed: 42,
 * });
 * // result.estimate — sample mean
 * // result.lower, result.upper — 95% CI bounds
 * ```
 */
export function bootstrapCI(
  data: Dataset,
  statistic: (sample: Dataset) => number,
  options: {
    confidence?: number;
    nReplicates?: number;
    method?: "percentile" | "bca";
    seed?: number;
  } = {},
): BootstrapCIResult {
  if (data.length < 2) throw new Error(`Invalid parameter 'data': expected at least 2 elements, received ${data.length}`);

  const confidence = options.confidence ?? 0.95;
  const nReplicates = options.nReplicates ?? 10000;
  const method = options.method ?? "percentile";
  const rng = options.seed != null ? createRng(options.seed) : Math.random;
  const n = data.length;

  const estimate = statistic(data);

  // Generate bootstrap replicates
  const replicates = new Array<number>(nReplicates);
  const resample = new Array<number>(n);
  for (let b = 0; b < nReplicates; b++) {
    for (let i = 0; i < n; i++) {
      resample[i] = data[Math.floor(rng() * n)];
    }
    replicates[b] = statistic([...resample]);
  }

  replicates.sort((a, b) => a - b);

  // Standard error
  const bootMean = replicates.reduce((s, v) => s + v, 0) / nReplicates;
  const standardError = Math.sqrt(
    replicates.reduce((s, v) => s + (v - bootMean) ** 2, 0) / (nReplicates - 1),
  );

  const alpha = 1 - confidence;

  if (method === "percentile") {
    const lowerIdx = Math.max(0, Math.floor((alpha / 2) * nReplicates) - 1);
    const upperIdx = Math.min(nReplicates - 1, Math.ceil((1 - alpha / 2) * nReplicates) - 1);
    return {
      estimate,
      lower: replicates[lowerIdx],
      upper: replicates[upperIdx],
      confidenceLevel: confidence,
      standardError,
      method,
      nReplicates,
    };
  }

  // BCa method
  // Bias correction factor z0
  const countBelow = replicates.filter((v) => v < estimate).length;
  const z0 = normalQuantile(countBelow / nReplicates);

  // Acceleration factor a (jackknife)
  const jackknife = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const jackSample = [...data.slice(0, i), ...data.slice(i + 1)];
    jackknife[i] = statistic(jackSample);
  }
  const jackMean = jackknife.reduce((s, v) => s + v, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const diff = jackMean - jackknife[i];
    num += diff ** 3;
    den += diff ** 2;
  }
  const a = den === 0 ? 0 : num / (6 * Math.pow(den, 1.5));

  // Adjusted percentiles
  const zAlphaLower = normalQuantile(alpha / 2);
  const zAlphaUpper = normalQuantile(1 - alpha / 2);

  const adjLower = normalCdf(z0 + (z0 + zAlphaLower) / (1 - a * (z0 + zAlphaLower)));
  const adjUpper = normalCdf(z0 + (z0 + zAlphaUpper) / (1 - a * (z0 + zAlphaUpper)));

  const lowerIdx = Math.max(0, Math.floor(adjLower * nReplicates) - 1);
  const upperIdx = Math.min(nReplicates - 1, Math.ceil(adjUpper * nReplicates) - 1);

  return {
    estimate,
    lower: replicates[lowerIdx],
    upper: replicates[upperIdx],
    confidenceLevel: confidence,
    standardError,
    method,
    nReplicates,
  };
}

// ── Permutation Test ────────────────────────────────────────────────────

/**
 * Result of a permutation test.
 */
export interface PermutationTestResult {
  /** Observed test statistic. */
  observedStatistic: number;
  /** Estimated p-value. */
  pValue: number;
  /** Number of permutations performed. */
  nPermutations: number;
  /** Whether to reject H0 at the given alpha. */
  rejected: boolean;
}

/**
 * Two-sample permutation test.
 *
 * Tests whether two independent samples come from the same distribution
 * by randomly permuting group assignments and comparing the test statistic.
 * The p-value is computed as (count_extreme + 1) / (nPermutations + 1).
 *
 * @param data1 - First sample
 * @param data2 - Second sample
 * @param options - Configuration options
 * @param options.statistic - Function to compute the test statistic from two samples (default: difference of means)
 * @param options.nPermutations - Number of random permutations (default: 10000)
 * @param options.alternative - "two-sided", "greater", or "less" (default: "two-sided")
 * @param options.alpha - Significance level (default: 0.05)
 * @param options.seed - Random seed for reproducibility
 * @returns PermutationTestResult with observed statistic, p-value, and rejection decision
 * @throws Error if either sample is empty
 *
 * @example
 * ```ts
 * const group1 = [1, 2, 3, 4, 5];
 * const group2 = [6, 7, 8, 9, 10];
 * const result = permutationTest(group1, group2, { seed: 42 });
 * // result.pValue — estimated p-value
 * // result.rejected — whether to reject H0 at alpha = 0.05
 * ```
 */
export function permutationTest(
  data1: Dataset,
  data2: Dataset,
  options: {
    statistic?: (a: Dataset, b: Dataset) => number;
    nPermutations?: number;
    alternative?: "two-sided" | "greater" | "less";
    alpha?: number;
    seed?: number;
  } = {},
): PermutationTestResult {
  if (data1.length < 1 || data2.length < 1) {
    throw new Error(`Invalid parameters 'data1', 'data2': expected at least 1 observation each, received data1.length=${data1.length}, data2.length=${data2.length}`);
  }

  const stat = options.statistic ?? ((a, b) => mean(a) - mean(b));
  const nPermutations = options.nPermutations ?? 10000;
  const alternative = options.alternative ?? "two-sided";
  const alpha = options.alpha ?? 0.05;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  const n1 = data1.length;
  const combined = [...data1, ...data2];
  const N = combined.length;

  const observedStatistic = stat(data1, data2);

  // Fisher-Yates partial shuffle to draw n1 elements
  let extremeCount = 0;
  for (let p = 0; p < nPermutations; p++) {
    // Shuffle combined array
    const perm = [...combined];
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = perm[i];
      perm[i] = perm[j];
      perm[j] = tmp;
    }
    const permStat = stat(perm.slice(0, n1), perm.slice(n1));

    if (alternative === "two-sided") {
      if (Math.abs(permStat) >= Math.abs(observedStatistic)) extremeCount++;
    } else if (alternative === "greater") {
      if (permStat >= observedStatistic) extremeCount++;
    } else {
      if (permStat <= observedStatistic) extremeCount++;
    }
  }

  const pValue = (extremeCount + 1) / (nPermutations + 1);

  return {
    observedStatistic,
    pValue,
    nPermutations,
    rejected: pValue < alpha,
  };
}

/**
 * One-sample permutation test (sign test variant).
 *
 * Tests whether the population center equals a hypothesized value by
 * randomly flipping signs of deviations from the center.
 *
 * @param data - Sample data
 * @param center - Hypothesized center value (default: 0)
 * @param options - Configuration options
 * @param options.statistic - Function computing the test statistic (default: mean)
 * @param options.nPermutations - Number of random permutations (default: 10000)
 * @param options.alternative - "two-sided", "greater", or "less" (default: "two-sided")
 * @param options.alpha - Significance level (default: 0.05)
 * @param options.seed - Random seed for reproducibility
 * @returns PermutationTestResult with observed statistic, p-value, and rejection decision
 * @throws Error if dataset has fewer than 1 element
 *
 * @example
 * ```ts
 * const data = [1.2, 2.3, 0.8, 1.9, 2.1];
 * const result = oneSamplePermutationTest(data, 0, { seed: 42 });
 * // Tests whether the center of `data` differs from 0
 * ```
 */
export function oneSamplePermutationTest(
  data: Dataset,
  center: number = 0,
  options: {
    statistic?: (sample: Dataset) => number;
    nPermutations?: number;
    alternative?: "two-sided" | "greater" | "less";
    alpha?: number;
    seed?: number;
  } = {},
): PermutationTestResult {
  if (data.length < 1) throw new Error(`Invalid parameter 'data': expected at least 1 element, received ${data.length}`);

  const stat = options.statistic ?? mean;
  const nPermutations = options.nPermutations ?? 10000;
  const alternative = options.alternative ?? "two-sided";
  const alpha = options.alpha ?? 0.05;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  const deviations = data.map((v) => v - center);
  const observedStatistic = stat(data);

  let extremeCount = 0;
  for (let p = 0; p < nPermutations; p++) {
    // Randomly flip signs of deviations
    const flipped = deviations.map((d) => (rng() < 0.5 ? d : -d));
    const permSample = flipped.map((d) => d + center);
    const permStat = stat(permSample);

    if (alternative === "two-sided") {
      if (Math.abs(permStat - center) >= Math.abs(observedStatistic - center)) extremeCount++;
    } else if (alternative === "greater") {
      if (permStat >= observedStatistic) extremeCount++;
    } else {
      if (permStat <= observedStatistic) extremeCount++;
    }
  }

  const pValue = (extremeCount + 1) / (nPermutations + 1);

  return {
    observedStatistic,
    pValue,
    nPermutations,
    rejected: pValue < alpha,
  };
}

