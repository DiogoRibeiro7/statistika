import { Dataset } from "./types";
import { mean } from "./utils/descriptive";

// ── Kernel Density Estimation ───────────────────────────────────────────

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
 * using a kernel smoothing approach.
 *
 * @param data - Input sample data
 * @param options - Configuration options
 * @param options.bandwidth - Smoothing bandwidth (default: Silverman's rule)
 * @param options.kernel - Kernel function: "gaussian", "epanechnikov", "uniform", "triangular" (default: "gaussian")
 * @param options.nPoints - Number of evaluation points (default: 512)
 * @param options.from - Lower bound of evaluation range (default: min - 3*bandwidth)
 * @param options.to - Upper bound of evaluation range (default: max + 3*bandwidth)
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
  if (data.length < 2) throw new Error("Dataset must have at least 2 elements");

  const kernelName = options.kernel ?? "gaussian";
  const K = kernels[kernelName];
  if (!K) {
    throw new Error(`Unknown kernel "${kernelName}". Available: ${Object.keys(kernels).join(", ")}`);
  }

  const h = options.bandwidth ?? silvermanBandwidth(data);
  if (h <= 0) throw new Error("Bandwidth must be positive");

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
 * Seeded pseudo-random number generator (xorshift128+).
 */
function createRng(seed: number): () => number {
  let s0 = seed | 0 || 1;
  let s1 = (seed * 2654435761) | 0 || 2;
  return () => {
    let a = s0;
    const b = s1;
    s0 = b;
    a ^= a << 23;
    a ^= a >> 17;
    a ^= b;
    a ^= b >> 26;
    s1 = a;
    return ((s0 + s1) >>> 0) / 4294967296;
  };
}

/**
 * Bootstrap confidence interval.
 *
 * Estimates a confidence interval for any statistic by resampling with
 * replacement. Supports percentile and BCa (bias-corrected and accelerated)
 * methods.
 *
 * @param data - Input sample data
 * @param statistic - Function that computes the statistic of interest from a sample
 * @param options - Configuration options
 * @param options.confidence - Confidence level (default: 0.95)
 * @param options.nReplicates - Number of bootstrap resamples (default: 10000)
 * @param options.method - "percentile" or "bca" (default: "percentile")
 * @param options.seed - Random seed for reproducibility
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
  if (data.length < 2) throw new Error("Dataset must have at least 2 elements");

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
  const z0 = normalQuantileApprox(countBelow / nReplicates);

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
  const zAlphaLower = normalQuantileApprox(alpha / 2);
  const zAlphaUpper = normalQuantileApprox(1 - alpha / 2);

  const adjLower = normalCdfApprox(z0 + (z0 + zAlphaLower) / (1 - a * (z0 + zAlphaLower)));
  const adjUpper = normalCdfApprox(z0 + (z0 + zAlphaUpper) / (1 - a * (z0 + zAlphaUpper)));

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
 *
 * @param data1 - First sample
 * @param data2 - Second sample
 * @param options - Configuration options
 * @param options.statistic - Function to compute the test statistic from two samples (default: difference of means)
 * @param options.nPermutations - Number of random permutations (default: 10000)
 * @param options.alternative - "two-sided", "greater", or "less" (default: "two-sided")
 * @param options.alpha - Significance level (default: 0.05)
 * @param options.seed - Random seed for reproducibility
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
    throw new Error("Both samples must have at least 1 observation");
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
  if (data.length < 1) throw new Error("Dataset must have at least 1 element");

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

// ── Internal helpers ────────────────────────────────────────────────────

/** Standard normal CDF approximation. */
function normalCdfApprox(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1.0 / (1.0 + p * ax);
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return 0.5 * (1.0 + sign * y);
}

/** Rational approximation of the standard normal quantile function. */
function normalQuantileApprox(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;

  // Rational approximation (Peter Acklam)
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
    -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
    3.754408661907416e0,
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  } else if (p <= pHigh) {
    q = p - 0.5;
    const r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
}
