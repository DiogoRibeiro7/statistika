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
 * @param data  Observations (length n).
 * @param weights  Non-negative weights (length n).
 */
export function weightedStats(
  data: number[],
  weights: number[],
): WeightedStatsResult {
  const n = data.length;
  if (n !== weights.length) {
    throw new Error("data and weights must have the same length");
  }
  if (n === 0) throw new Error("Need at least 1 observation");

  let sumW = 0;
  let sumW2 = 0;
  let sumWX = 0;
  for (let i = 0; i < n; i++) {
    if (weights[i] < 0) throw new Error("Weights must be non-negative");
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
 * @param data  Observations (length n).
 * @param weights  Non-negative weights (length n).
 * @param p  Quantile level in [0, 1].
 */
export function weightedQuantile(
  data: number[],
  weights: number[],
  p: number,
): number {
  if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
  const n = data.length;
  if (n !== weights.length) {
    throw new Error("data and weights must have the same length");
  }
  if (n === 0) throw new Error("Need at least 1 observation");

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
 * @param y  Observed values (length n, the sample).
 * @param inclusionProbs  First-order inclusion probabilities πᵢ (length n).
 * @param populationSize  Known population size N.
 * @param alpha  Significance level for CI (default 0.05).
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
 * Uses Kish's approximation: DEFF ≈ 1 + cv²(w), where cv(w) is the
 * coefficient of variation of the weights.
 *
 * @param data  Observations (length n).
 * @param weights  Survey weights (length n).
 */
export function designEffect(
  data: number[],
  weights: number[],
): DesignEffectResult {
  const n = data.length;
  if (n !== weights.length) {
    throw new Error("data and weights must have the same length");
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
 * R̂ = (Σ wᵢ yᵢ) / (Σ wᵢ xᵢ)
 *
 * @param y  Study variable (length n).
 * @param x  Auxiliary variable (length n).
 * @param weights  Survey weights (length n).
 * @param xTotal  Known population total of x (for estimating Y total).
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
 * @param weights  Original survey weights (length n).
 * @param strata  Stratum assignment for each observation (length n, integer-coded).
 * @param populationCounts  Known population count for each stratum (indexed by stratum code).
 * @returns Adjusted weights (length n).
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
