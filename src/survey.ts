/**
 * Survey statistics for complex survey designs.
 *
 * - **Weighted descriptive statistics** — mean, variance, quantile.
 * - **Horvitz-Thompson estimator** — population total and mean.
 * - **Design effect (DEFF)** — ratio of design-based to SRS variance.
 * - **Ratio estimator** — with linearised variance.
 * - **Post-stratification** — weight adjustment to known population margins.
 */

import { mean } from "./utils/descriptive";
import { normalCdf } from "./utils/linalg";

// ── Weighted Descriptive Statistics ───────────────────────────────────────

/**
 * Result of weighted descriptive statistics computation.
 */
export interface WeightedStatsResult {
  /** Weighted mean. */
  mean: number;
  /** Weighted variance (frequency weights interpretation). */
  variance: number;
  /** Weighted standard deviation. */
  stdDev: number;
  /** Effective sample size: (Σwᵢ)² / Σwᵢ². */
  effectiveSampleSize: number;
  /** Sum of weights. */
  sumWeights: number;
}

/**
 * Compute weighted descriptive statistics.
 *
 * The weighted mean is sum(w_i * x_i) / sum(w_i). The variance uses
 * Bessel-corrected reliability weights: Var = sum(w_i * (x_i - x_bar)^2) /
 * (sum(w_i) - sum(w_i^2) / sum(w_i)).
 *
 * @param data - Observations (length n)
 * @param weights - Non-negative weights (length n)
 * @returns WeightedStatsResult with mean, variance, stdDev, effective sample size, and sum of weights
 * @throws Error if data and weights have different lengths
 * @throws Error if no observations provided
 * @throws Error if any weight is negative
 * @throws Error if sum of weights is zero
 *
 * @example
 * ```ts
 * const result = weightedStats([1, 2, 3], [1, 2, 1]);
 * // result.mean — weighted mean = (1+4+3)/4 = 2
 * ```
 */
export function weightedStats(
  data: number[],
  weights: number[],
): WeightedStatsResult {
  const n = data.length;
  if (n !== weights.length) {
    throw new Error(`Invalid parameters 'data', 'weights': expected same length, received data.length=${data.length}, weights.length=${weights.length}`);
  }
  if (n === 0) throw new Error(`Invalid parameter 'data': expected at least 1 observation, received 0`);

  let sumW = 0;
  let sumW2 = 0;
  let sumWX = 0;
  for (let i = 0; i < n; i++) {
    if (weights[i] < 0) throw new Error(`Invalid parameter 'weights[${i}]': expected a non-negative number, received ${weights[i]}`);
    sumW += weights[i];
    sumW2 += weights[i] * weights[i];
    sumWX += weights[i] * data[i];
  }

  if (sumW === 0) throw new Error("Sum of weights must be positive");

  const wMean = sumWX / sumW;

  // Reliability weights (Bessel-corrected weighted variance)
  let sumWD2 = 0;
  for (let i = 0; i < n; i++) {
    sumWD2 += weights[i] * (data[i] - wMean) ** 2;
  }

  const variance = sumW > sumW2 / sumW
    ? sumWD2 / (sumW - sumW2 / sumW)
    : 0;

  return {
    mean: wMean,
    variance,
    stdDev: Math.sqrt(variance),
    effectiveSampleSize: sumW2 > 0 ? (sumW * sumW) / sumW2 : 0,
    sumWeights: sumW,
  };
}

/**
 * Weighted quantile using linear interpolation.
 *
 * Sorts the data by value and walks through accumulated weights until
 * the target cumulative weight p * sum(w) is reached.
 *
 * @param data - Observations (length n)
 * @param weights - Non-negative weights (length n)
 * @param p - Quantile level in [0, 1]
 * @returns The weighted quantile value
 * @throws Error if p is not in [0, 1]
 * @throws Error if data and weights have different lengths
 * @throws Error if no observations provided
 * @throws Error if sum of weights is zero
 *
 * @example
 * ```ts
 * weightedQuantile([10, 20, 30], [1, 2, 1], 0.5); // weighted median
 * ```
 */
export function weightedQuantile(
  data: number[],
  weights: number[],
  p: number,
): number {
  if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
  const n = data.length;
  if (n !== weights.length) {
    throw new Error(`Invalid parameters 'data', 'weights': expected same length, received data.length=${data.length}, weights.length=${weights.length}`);
  }
  if (n === 0) throw new Error(`Invalid parameter 'data': expected at least 1 observation, received 0`);

  // Create sorted index-weight pairs
  const indices = Array.from({ length: n }, (_, i) => i);
  indices.sort((a, b) => data[a] - data[b]);

  let sumW = 0;
  for (let i = 0; i < n; i++) sumW += weights[i];
  if (sumW === 0) throw new Error("Sum of weights must be positive");

  const target = p * sumW;
  let cumW = 0;
  for (let k = 0; k < n; k++) {
    const idx = indices[k];
    cumW += weights[idx];
    if (cumW >= target) return data[idx];
  }
  return data[indices[n - 1]];
}

// ── Horvitz-Thompson Estimator ────────────────────────────────────────────

/**
 * Result of a Horvitz-Thompson estimation.
 */
export interface HorvitzThompsonResult {
  /** Estimated population total: Σ yᵢ / πᵢ. */
  total: number;
  /** Estimated population mean: total / N. */
  mean: number;
  /** Estimated variance of the total (Sen-Yates-Grundy approximation). */
  varianceTotal: number;
  /** Standard error of the total. */
  seTotal: number;
  /** Approximate 95% confidence interval for the total. */
  ciTotal: [number, number];
}

/**
 * Horvitz-Thompson estimator for population total and mean.
 *
 * Estimates the population total as T_hat = sum(y_i / pi_i) where pi_i
 * is the first-order inclusion probability. Variance is estimated using
 * the with-replacement approximation: V_hat = sum((1-pi_i)/pi_i^2 * y_i^2).
 *
 * @param y - Observed values (length n, the sample)
 * @param inclusionProbs - First-order inclusion probabilities pi_i (length n, each in (0, 1])
 * @param populationSize - Known population size N
 * @param alpha - Significance level for CI (default 0.05)
 * @returns HorvitzThompsonResult with total, mean, variance, SE, and confidence interval
 * @throws Error if y and inclusionProbs have different lengths
 * @throws Error if populationSize < sample size
 * @throws Error if any inclusion probability is not in (0, 1]
 *
 * @example
 * ```ts
 * const result = horvitzThompson([10, 20, 30], [0.1, 0.2, 0.3], 100);
 * // result.total — estimated population total
 * // result.mean — estimated population mean (total / N)
 * ```
 */
export function horvitzThompson(
  y: number[],
  inclusionProbs: number[],
  populationSize: number,
  alpha = 0.05,
): HorvitzThompsonResult {
  const n = y.length;
  if (n !== inclusionProbs.length) {
    throw new Error("y and inclusionProbs must have the same length");
  }
  if (populationSize < n) {
    throw new Error("Population size must be at least as large as sample size");
  }

  for (let i = 0; i < n; i++) {
    if (inclusionProbs[i] <= 0 || inclusionProbs[i] > 1) {
      throw new Error("Inclusion probabilities must be in (0, 1]");
    }
  }

  // Total estimator
  let total = 0;
  for (let i = 0; i < n; i++) {
    total += y[i] / inclusionProbs[i];
  }

  // Variance estimator (assuming with-replacement approximation)
  // V̂ = Σ ((1 - πᵢ) / πᵢ²) yᵢ²
  let varTotal = 0;
  for (let i = 0; i < n; i++) {
    const pi = inclusionProbs[i];
    varTotal += ((1 - pi) / (pi * pi)) * y[i] * y[i];
  }

  const seTotal = Math.sqrt(Math.max(0, varTotal));

  // Normal approximation CI
  const z = normalQuantileApprox(1 - alpha / 2);
  const ciTotal: [number, number] = [total - z * seTotal, total + z * seTotal];

  return {
    total,
    mean: total / populationSize,
    varianceTotal: varTotal,
    seTotal,
    ciTotal,
  };
}

// ── Design Effect ─────────────────────────────────────────────────────────

/**
 * Result of a design effect (DEFF) computation.
 */
export interface DesignEffectResult {
  /** Design effect: Var_design / Var_SRS. */
  deff: number;
  /** Effective sample size: n / DEFF. */
  effectiveSampleSize: number;
  /** Variance under the complex design. */
  varianceDesign: number;
  /** Variance under SRS of same size. */
  varianceSRS: number;
}

/**
 * Compute the design effect (DEFF) comparing complex survey variance
 * to simple random sampling variance.
 *
 * Uses Kish's approximation: DEFF = n * sum(w_i^2) / (sum(w_i))^2.
 * A DEFF > 1 means the complex design is less efficient than SRS.
 *
 * @param data - Observations (length n)
 * @param weights - Survey weights (length n)
 * @returns DesignEffectResult with DEFF, effective sample size, and variance estimates
 * @throws Error if data and weights have different lengths
 * @throws Error if fewer than 2 observations
 *
 * @example
 * ```ts
 * const result = designEffect([1, 2, 3, 4], [1, 1, 2, 2]);
 * console.log(result.deff); // design effect (1.0 for equal weights)
 * ```
 */
export function designEffect(
  data: number[],
  weights: number[],
): DesignEffectResult {
  const n = data.length;
  if (n !== weights.length) {
    throw new Error(`Invalid parameters 'data', 'weights': expected same length, received data.length=${data.length}, weights.length=${weights.length}`);
  }
  if (n < 2) throw new Error("Need at least 2 observations");

  const ws = weightedStats(data, weights);

  // SRS variance of the mean
  const srsVar = sampleVariance(data) / n;

  // Design-based variance (using weighted variance / effective n)
  const designVar = ws.variance / ws.effectiveSampleSize;

  // DEFF = design variance / SRS variance
  // Also: Kish's DEFF ≈ n * Σwᵢ² / (Σwᵢ)²
  let sumW = 0;
  let sumW2 = 0;
  for (let i = 0; i < n; i++) {
    sumW += weights[i];
    sumW2 += weights[i] * weights[i];
  }
  const kishDeff = (n * sumW2) / (sumW * sumW);

  return {
    deff: kishDeff,
    effectiveSampleSize: n / kishDeff,
    varianceDesign: designVar,
    varianceSRS: srsVar,
  };
}

// ── Ratio Estimator ───────────────────────────────────────────────────────

/**
 * Result of a ratio estimation.
 */
export interface RatioEstimatorResult {
  /** Ratio estimate R̂ = Σwᵢyᵢ / Σwᵢxᵢ. */
  ratio: number;
  /** Estimated population total of y: R̂ × known total of x. */
  total: number;
  /** Linearised variance of the ratio. */
  variance: number;
  /** Standard error. */
  standardError: number;
}

/**
 * Ratio estimator for survey data.
 *
 * R_hat = (sum w_i * y_i) / (sum w_i * x_i). The population total of y
 * is estimated as R_hat * X_total. Variance uses the linearized Taylor
 * series approximation on residuals e_i = y_i - R_hat * x_i.
 *
 * @param y - Study variable (length n)
 * @param x - Auxiliary variable (length n)
 * @param weights - Survey weights (length n)
 * @param xTotal - Known population total of x (for estimating Y total)
 * @returns RatioEstimatorResult with ratio, estimated total, variance, and standard error
 * @throws Error if y, x, and weights have different lengths
 * @throws Error if fewer than 2 observations
 * @throws Error if weighted sum of x is zero
 *
 * @example
 * ```ts
 * const result = ratioEstimator([10, 20], [5, 10], [1, 1], 1000);
 * console.log(result.ratio); // R_hat = sum(w*y) / sum(w*x)
 * console.log(result.total); // estimated population total of y
 * ```
 */
export function ratioEstimator(
  y: number[],
  x: number[],
  weights: number[],
  xTotal: number,
): RatioEstimatorResult {
  const n = y.length;
  if (n !== x.length || n !== weights.length) {
    throw new Error("y, x, and weights must have the same length");
  }
  if (n < 2) throw new Error("Need at least 2 observations");

  let sumWY = 0;
  let sumWX = 0;
  let sumW = 0;
  for (let i = 0; i < n; i++) {
    sumWY += weights[i] * y[i];
    sumWX += weights[i] * x[i];
    sumW += weights[i];
  }

  if (sumWX === 0) throw new Error("Weighted sum of x must be non-zero");

  const ratio = sumWY / sumWX;
  const total = ratio * xTotal;

  // Linearised variance: Var(R̂) ≈ (1/X̄²) Var(ȳ - R̂ x̄)
  let sumE2 = 0;
  const meanE = 0; // residuals should be mean-zero by construction
  for (let i = 0; i < n; i++) {
    const e = weights[i] * (y[i] - ratio * x[i]);
    sumE2 += e * e;
  }
  const varianceRatio = sumE2 / (sumWX * sumWX * (n - 1) / n);

  return {
    ratio,
    total,
    variance: varianceRatio,
    standardError: Math.sqrt(Math.max(0, varianceRatio)),
  };
}

// ── Post-stratification ──────────────────────────────────────────────────

/**
 * Adjust survey weights to match known population margins
 * (post-stratification).
 *
 * For each stratum s, the adjustment factor is:
 * f_s = N_s / sum(w_i for i in stratum s),
 * and adjusted weights are w_i * f_s.
 *
 * @param weights - Original survey weights (length n)
 * @param strata - Stratum assignment for each observation (length n, integer-coded)
 * @param populationCounts - Known population count for each stratum (indexed by stratum code)
 * @returns Adjusted weights (length n) that sum to known population totals within each stratum
 * @throws Error if weights and strata have different lengths
 * @throws Error if any stratum in the data lacks a corresponding population count
 *
 * @example
 * ```ts
 * const adj = postStratify([1, 1, 1, 1], [0, 0, 1, 1], { 0: 100, 1: 200 });
 * // Adjusted weights so stratum sums match population counts
 * ```
 */
export function postStratify(
  weights: number[],
  strata: number[],
  populationCounts: Record<number, number>,
): number[] {
  const n = weights.length;
  if (n !== strata.length) {
    throw new Error("weights and strata must have the same length");
  }

  // Compute weighted sum per stratum
  const stratumWeightSum: Record<number, number> = {};
  for (let i = 0; i < n; i++) {
    const s = strata[i];
    stratumWeightSum[s] = (stratumWeightSum[s] || 0) + weights[i];
  }

  // Compute adjustment factors
  const adjustedWeights = new Array(n);
  for (let i = 0; i < n; i++) {
    const s = strata[i];
    const popCount = populationCounts[s];
    if (popCount === undefined) {
      throw new Error(`No population count for stratum ${s}`);
    }
    const factor = popCount / stratumWeightSum[s];
    adjustedWeights[i] = weights[i] * factor;
  }

  return adjustedWeights;
}

// ── Helpers ───────────────────────────────────────────────────────────────

function sampleVariance(data: number[]): number {
  const n = data.length;
  if (n < 2) return 0;
  const m = mean(data);
  let s = 0;
  for (const x of data) s += (x - m) ** 2;
  return s / (n - 1);
}

function normalQuantileApprox(p: number): number {
  if (p < 0.5) return -normalQuantileApprox(1 - p);
  const t = Math.sqrt(-2 * Math.log(1 - p));
  return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
    (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
}
