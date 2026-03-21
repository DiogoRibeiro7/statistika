import { Dataset } from "./types";
import { mean, median } from "./utils/descriptive";

/**
 * Median Absolute Deviation (MAD).
 *
 * A robust measure of statistical dispersion. MAD = median(|Xi - median(X)|).
 * The consistency constant (default 1.4826) makes it a consistent estimator
 * of the standard deviation for normally distributed data.
 *
 * @param data - Input dataset
 * @param constant - Consistency constant (default: 1.4826 for normal distribution)
 * @returns The MAD value scaled by the consistency constant
 * @throws Error if dataset is empty
 * @throws Error if dataset contains NaN values
 *
 * @example
 * ```ts
 * const data = [1, 2, 3, 4, 5, 100];
 * const result = mad(data);
 * // result ≈ 1.4826 * median(|xi - 3.5|) — robust unlike std deviation
 * ```
 */
export function mad(data: Dataset, constant = 1.4826): number {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  validateNoNaN(data);
  const med = median(data);
  const absDeviations = data.map((v) => Math.abs(v - med));
  return median(absDeviations) * constant;
}

/**
 * Trimmed mean.
 *
 * Computes the mean after discarding a proportion of observations from
 * each tail of the distribution. More robust to outliers than the
 * arithmetic mean.
 *
 * @param data - Input dataset
 * @param proportion - Proportion to trim from each tail (0 to 0.5, default: 0.1)
 * @returns The mean of the remaining observations after trimming
 * @throws Error if dataset is empty
 * @throws Error if dataset contains NaN values
 * @throws Error if proportion is not in [0, 0.5)
 * @throws Error if too few observations remain after trimming
 *
 * @example
 * ```ts
 * const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100];
 * trimmedMean(data, 0.1); // trims lowest and highest 10%, returns mean of [2..9]
 * ```
 */
export function trimmedMean(data: Dataset, proportion = 0.1): number {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  validateNoNaN(data);
  if (proportion < 0 || proportion >= 0.5) {
    throw new Error("Trim proportion must be between 0 (inclusive) and 0.5 (exclusive)");
  }

  const sorted = [...data].sort((a, b) => a - b);
  const trimCount = Math.floor(sorted.length * proportion);
  const trimmed = sorted.slice(trimCount, sorted.length - trimCount);

  if (trimmed.length === 0) {
    throw new Error("Too few observations after trimming");
  }

  return mean(trimmed);
}

/**
 * Winsorized mean.
 *
 * Replaces extreme values with the nearest non-extreme values rather
 * than removing them. Retains the original sample size.
 *
 * @param data - Input dataset
 * @param proportion - Proportion to winsorize from each tail (0 to 0.5, default: 0.1)
 * @returns The mean after replacing extreme values with boundary values
 * @throws Error if dataset is empty
 * @throws Error if dataset contains NaN values
 * @throws Error if proportion is not in [0, 0.5)
 */
export function winsorizedMean(data: Dataset, proportion = 0.1): number {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  validateNoNaN(data);
  if (proportion < 0 || proportion >= 0.5) {
    throw new Error("Winsorize proportion must be between 0 (inclusive) and 0.5 (exclusive)");
  }

  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;
  const k = Math.floor(n * proportion);

  if (k === 0) return mean(sorted);

  const lower = sorted[k];
  const upper = sorted[n - 1 - k];

  const winsorized = sorted.map((v) => {
    if (v < lower) return lower;
    if (v > upper) return upper;
    return v;
  });

  return mean(winsorized);
}

/**
 * Interquartile Range (IQR).
 *
 * The range between the 25th and 75th percentiles — a robust measure
 * of spread.
 *
 * @param data - Input dataset
 * @returns The difference Q3 - Q1
 * @throws Error if dataset has fewer than 4 elements
 * @throws Error if dataset contains NaN values
 */
export function iqr(data: Dataset): number {
  if (data.length < 4) throw new Error("Dataset must have at least 4 elements");
  validateNoNaN(data);
  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;
  const q1 = median(sorted.slice(0, Math.floor(n / 2)));
  const q3 = median(sorted.slice(Math.ceil(n / 2)));
  return q3 - q1;
}

/**
 * Detect outliers using the IQR method.
 *
 * Points below Q1 - k*IQR or above Q3 + k*IQR are flagged as outliers.
 *
 * @param data - Input dataset
 * @param k - IQR multiplier (default: 1.5 for mild outliers, use 3 for extreme)
 * @returns Object with outlier values, their indices, and the lower/upper fences
 * @throws Error if dataset has fewer than 4 elements
 * @throws Error if dataset contains NaN values
 */
export function detectOutliers(
  data: Dataset,
  k = 1.5,
): { outliers: number[]; indices: number[]; lower: number; upper: number } {
  if (data.length < 4) throw new Error("Dataset must have at least 4 elements");
  validateNoNaN(data);

  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;
  const q1 = median(sorted.slice(0, Math.floor(n / 2)));
  const q3 = median(sorted.slice(Math.ceil(n / 2)));
  const range = q3 - q1;
  const lower = q1 - k * range;
  const upper = q3 + k * range;

  const outliers: number[] = [];
  const indices: number[] = [];

  for (let i = 0; i < data.length; i++) {
    if (data[i] < lower || data[i] > upper) {
      outliers.push(data[i]);
      indices.push(i);
    }
  }

  return { outliers, indices, lower, upper };
}

/**
 * Huber M-estimate of location.
 *
 * Iteratively re-weighted mean that downweights observations far from
 * the center. Robust to outliers while being more efficient than the
 * median for normally distributed data.
 *
 * @param data - Input dataset
 * @param k - Huber tuning constant (default: 1.345 for 95% efficiency at normal)
 * @param maxIterations - Maximum iterations (default: 50)
 * @param tol - Convergence tolerance (default: 1e-6)
 * @returns The Huber M-estimate of location
 * @throws Error if dataset is empty
 * @throws Error if dataset contains NaN values
 */
export function huberMean(
  data: Dataset,
  k = 1.345,
  maxIterations = 50,
  tol = 1e-6,
): number {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  validateNoNaN(data);

  let mu = median(data);
  const s = mad(data);

  if (s === 0) return mu;

  for (let iter = 0; iter < maxIterations; iter++) {
    let sumW = 0;
    let sumWx = 0;

    for (const x of data) {
      const r = (x - mu) / s;
      const w = Math.abs(r) <= k ? 1 : k / Math.abs(r);
      sumW += w;
      sumWx += w * x;
    }

    const muNew = sumWx / sumW;
    if (Math.abs(muNew - mu) < tol) {
      return muNew;
    }
    mu = muNew;
  }

  return mu;
}

/**
 * Biweight (Tukey's biweight) midvariance.
 *
 * A robust estimator of scale that is highly resistant to outliers.
 *
 * @param data - Input dataset
 * @param c - Tuning constant (default: 9.0)
 * @returns The biweight midvariance estimate
 * @throws Error if dataset has fewer than 2 elements
 * @throws Error if dataset contains NaN values
 */
export function biweightMidvariance(data: Dataset, c = 9.0): number {
  if (data.length < 2) throw new Error("Dataset must have at least 2 elements");
  validateNoNaN(data);

  const med = median(data);
  const madVal = mad(data, 1); // raw MAD without consistency constant

  if (madVal === 0) return 0;

  const n = data.length;
  const u = data.map((x) => (x - med) / (c * madVal));

  let num = 0;
  let den = 0;

  for (let i = 0; i < n; i++) {
    const ui2 = u[i] * u[i];
    if (ui2 < 1) {
      const diff = data[i] - med;
      num += diff * diff * (1 - ui2) ** 4;
      den += (1 - ui2) * (1 - 5 * ui2);
    }
  }

  return (n * num) / (den * den);
}

// ── Helpers ─────────────────────────────────────────────────────────────

function validateNoNaN(data: Dataset): void {
  for (const v of data) {
    if (Number.isNaN(v)) {
      throw new Error("Dataset must not contain NaN values");
    }
  }
}
