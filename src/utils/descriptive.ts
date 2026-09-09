import { Dataset, DescriptiveStats } from "../types";

/**
 * Computes the arithmetic mean (average) of a dataset.
 *
 * @param data - Array of numeric values.
 * @returns The arithmetic mean of the values.
 * @throws {Error} If the dataset is empty.
 * @example
 * mean([1, 2, 3, 4, 5]); // 3
 */
export function mean(data: Dataset): number {
  if (data.length === 0) throw new Error("Invalid parameter 'data': expected non-empty array, received length 0");
  return data.reduce((sum, v) => sum + v, 0) / data.length;
}

/**
 * Computes the median of a dataset. For even-length datasets, returns the
 * average of the two middle values.
 *
 * @param data - Array of numeric values.
 * @returns The median value.
 * @throws {Error} If the dataset is empty.
 * @example
 * median([3, 1, 2]); // 2
 */
export function median(data: Dataset): number {
  if (data.length === 0) throw new Error("Invalid parameter 'data': expected non-empty array, received length 0");
  const sorted = [...data].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Computes the variance of a dataset. By default uses Bessel's correction
 * (sample variance, dividing by n-1). Set `sample` to false for population variance.
 *
 * @param data - Array of numeric values.
 * @param sample - If true (default), computes sample variance s²; if false, population variance σ².
 * @returns The variance.
 * @throws {Error} If the dataset has fewer than 2 elements.
 * @example
 * variance([2, 4, 4, 4, 5, 5, 7, 9]); // ~4.571 (sample)
 */
export function variance(data: Dataset, sample = true): number {
  if (data.length < 2) throw new Error("Invalid parameter 'data': expected at least 2 elements, received length " + data.length);
  const m = mean(data);
  const sumSq = data.reduce((acc, v) => acc + (v - m) ** 2, 0);
  return sumSq / (sample ? data.length - 1 : data.length);
}

/**
 * Computes the standard deviation of a dataset (square root of the variance).
 *
 * @param data - Array of numeric values.
 * @param sample - If true (default), computes sample standard deviation s; if false, population σ.
 * @returns The standard deviation.
 * @throws {Error} If the dataset has fewer than 2 elements.
 */
export function stdDev(data: Dataset, sample = true): number {
  return Math.sqrt(variance(data, sample));
}

/**
 * Alias for {@link stdDev}. Computes the standard deviation of a dataset.
 *
 * @see stdDev
 */
export const standardDeviation = stdDev;

/**
 * Computes the adjusted Fisher-Pearson sample skewness of a dataset.
 * Uses the bias-corrected formula: [n/((n-1)(n-2))] * Σ[((xᵢ - x̄)/s)³].
 *
 * @param data - Array of numeric values.
 * @returns The sample skewness (0 for symmetric distributions).
 * @throws {Error} If the dataset has fewer than 3 elements.
 * @example
 * skewness([2, 8, 0, 4, 1, 9, 9, 0]); // positive skew
 */
export function skewness(data: Dataset): number {
  const n = data.length;
  if (n < 3) throw new Error("Invalid parameter 'data': expected at least 3 elements, received length " + n);
  const m = mean(data);
  const s = stdDev(data, true);
  if (s === 0) return 0;
  const m3 = data.reduce((acc, v) => acc + ((v - m) / s) ** 3, 0);
  return (n / ((n - 1) * (n - 2))) * m3;
}

/**
 * Computes the excess kurtosis of a dataset using the bias-corrected formula.
 * Applies the -3(n-1)²/((n-2)(n-3)) adjustment so that a normal distribution has kurtosis 0.
 *
 * @param data - Array of numeric values.
 * @returns The excess kurtosis (0 for normal distributions, positive for heavy-tailed).
 * @throws {Error} If the dataset has fewer than 4 elements.
 */
export function kurtosis(data: Dataset): number {
  const n = data.length;
  if (n < 4) throw new Error("Invalid parameter 'data': expected at least 4 elements, received length " + n);
  const m = mean(data);
  const s = stdDev(data, true);
  if (s === 0) return 0;
  const m4 = data.reduce((acc, v) => acc + ((v - m) / s) ** 4, 0);
  const num = n * (n + 1) * m4;
  const den = (n - 1) * (n - 2) * (n - 3);
  const correction = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3));
  return num / den - correction;
}

/**
 * Computes the p-th percentile of a dataset using linear interpolation
 * between adjacent ranks.
 *
 * @param data - Array of numeric values.
 * @param p - Percentile value between 0 and 100 (inclusive).
 * @returns The interpolated percentile value.
 * @throws {Error} If the dataset is empty.
 * @throws {Error} If p is not in [0, 100].
 * @example
 * percentile([15, 20, 35, 40, 50], 50); // 35 (median)
 */
export function percentile(data: Dataset, p: number): number {
  if (data.length === 0) throw new Error("Invalid parameter 'data': expected non-empty array, received length 0");
  if (p < 0 || p > 100) throw new Error(`Invalid parameter 'p': expected value in [0, 100], received ${p}`);
  const sorted = [...data].sort((a, b) => a - b);
  if (p === 0) return sorted[0];
  if (p === 100) return sorted[sorted.length - 1];
  const rank = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(rank);
  const frac = rank - lower;
  return sorted[lower] + frac * (sorted[lower + 1] - sorted[lower]);
}

/**
 * Computes a summary of descriptive statistics for a dataset, including
 * count, mean, median, variance, standard deviation, min, and max.
 *
 * @param data - Array of numeric values.
 * @returns A {@link DescriptiveStats} object with all summary statistics.
 * @throws {Error} If the dataset is empty or has fewer than 2 elements.
 * @example
 * describe([1, 2, 3, 4, 5]);
 * // { count: 5, mean: 3, median: 3, variance: 2.5, stdDev: ~1.58, min: 1, max: 5 }
 */
export function describe(data: Dataset): DescriptiveStats {
  return {
    count: data.length,
    mean: mean(data),
    median: median(data),
    variance: variance(data),
    stdDev: stdDev(data),
    min: Math.min(...data),
    max: Math.max(...data),
  };
}
