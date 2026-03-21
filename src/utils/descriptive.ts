import { Dataset, DescriptiveStats } from "../types";

export function mean(data: Dataset): number {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  return data.reduce((sum, v) => sum + v, 0) / data.length;
}

export function median(data: Dataset): number {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  const sorted = [...data].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function variance(data: Dataset, sample = true): number {
  if (data.length < 2) throw new Error("Dataset must have at least 2 elements");
  const m = mean(data);
  const sumSq = data.reduce((acc, v) => acc + (v - m) ** 2, 0);
  return sumSq / (sample ? data.length - 1 : data.length);
}

export function stdDev(data: Dataset, sample = true): number {
  return Math.sqrt(variance(data, sample));
}

/**
 * Compute the sample skewness (adjusted Fisher-Pearson) of a dataset.
 * Uses the bias-corrected formula: [n/((n-1)(n-2))] * sum[((xi - mean)/s)^3]
 */
export function skewness(data: Dataset): number {
  const n = data.length;
  if (n < 3) throw new Error("Dataset must have at least 3 elements");
  const m = mean(data);
  const s = stdDev(data, true);
  if (s === 0) return 0;
  const m3 = data.reduce((acc, v) => acc + ((v - m) / s) ** 3, 0);
  return (n / ((n - 1) * (n - 2))) * m3;
}

/**
 * Compute the excess kurtosis of a dataset.
 * Uses the bias-corrected formula with the -3(n-1)^2/((n-2)(n-3)) adjustment.
 */
export function kurtosis(data: Dataset): number {
  const n = data.length;
  if (n < 4) throw new Error("Dataset must have at least 4 elements");
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
 * Compute the p-th percentile of a dataset using linear interpolation.
 * @param p Percentile value between 0 and 100.
 */
export function percentile(data: Dataset, p: number): number {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  if (p < 0 || p > 100) throw new Error("Percentile must be between 0 and 100");
  const sorted = [...data].sort((a, b) => a - b);
  if (p === 0) return sorted[0];
  if (p === 100) return sorted[sorted.length - 1];
  const rank = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(rank);
  const frac = rank - lower;
  return sorted[lower] + frac * (sorted[lower + 1] - sorted[lower]);
}

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
