/**
 * Visualization data generators — plot-ready data structures.
 *
 * - **Histogram bins** — configurable bin count, density normalisation.
 * - **Q-Q plot coordinates** — theoretical vs sample quantiles.
 * - **Box plot statistics** — five-number summary + outliers.
 * - **KDE (kernel density)** — Gaussian kernel with Silverman bandwidth.
 * - **ECDF** — empirical cumulative distribution function coordinates.
 * - **Scatter matrix data** — pairwise variable combinations.
 */

import { ContinuousDistribution } from "./types";

// ── Types ─────────────────────────────────────────────────────────────────

export interface HistogramBin {
  /** Left edge of the bin (inclusive). */
  lo: number;
  /** Right edge of the bin (exclusive, except last bin). */
  hi: number;
  /** Mid-point of the bin. */
  mid: number;
  /** Number of observations in the bin. */
  count: number;
  /** Relative frequency (count / n). */
  frequency: number;
  /** Density (frequency / bin width). */
  density: number;
}

export interface QQPoint {
  /** Theoretical quantile. */
  theoretical: number;
  /** Sample quantile. */
  sample: number;
}

export interface BoxPlotStats {
  /** Minimum (excluding outliers). */
  min: number;
  /** First quartile (Q1). */
  q1: number;
  /** Median (Q2). */
  median: number;
  /** Third quartile (Q3). */
  q3: number;
  /** Maximum (excluding outliers). */
  max: number;
  /** Interquartile range. */
  iqr: number;
  /** Lower fence: Q1 − 1.5 × IQR. */
  lowerFence: number;
  /** Upper fence: Q3 + 1.5 × IQR. */
  upperFence: number;
  /** Values below the lower fence. */
  lowerOutliers: number[];
  /** Values above the upper fence. */
  upperOutliers: number[];
  /** Mean of the data. */
  mean: number;
}

export interface KDEPoint {
  /** Evaluation point. */
  x: number;
  /** Density estimate. */
  density: number;
}

export interface ECDFPoint {
  /** Data value. */
  x: number;
  /** Cumulative probability F(x). */
  probability: number;
}

// ── Histogram ─────────────────────────────────────────────────────────────

/**
 * Generate histogram bins from data.
 *
 * @param data  Numeric observations.
 * @param options.bins  Number of bins (default: Sturges' rule).
 * @param options.range  [min, max] range to bin over.
 */
export function histogramBins(
  data: number[],
  options: { bins?: number; range?: [number, number] } = {},
): HistogramBin[] {
  if (data.length === 0) return [];

  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;

  const lo = options.range?.[0] ?? sorted[0];
  const hi = options.range?.[1] ?? sorted[n - 1];
  const nBins = options.bins ?? Math.max(1, Math.ceil(Math.log2(n) + 1)); // Sturges

  if (lo === hi) {
    return [{
      lo, hi: lo + 1, mid: lo + 0.5,
      count: n, frequency: 1, density: n,
    }];
  }

  const binWidth = (hi - lo) / nBins;
  const result: HistogramBin[] = [];

  for (let b = 0; b < nBins; b++) {
    const binLo = lo + b * binWidth;
    const binHi = lo + (b + 1) * binWidth;
    let count = 0;
    for (const x of sorted) {
      if (b === nBins - 1 ? (x >= binLo && x <= binHi) : (x >= binLo && x < binHi)) {
        count++;
      }
    }
    result.push({
      lo: binLo,
      hi: binHi,
      mid: (binLo + binHi) / 2,
      count,
      frequency: count / n,
      density: count / (n * binWidth),
    });
  }

  return result;
}

// ── Q-Q Plot ──────────────────────────────────────────────────────────────

/**
 * Generate Q-Q plot coordinates.
 *
 * @param data  Sample data.
 * @param distribution  Theoretical distribution to compare against.
 */
export function qqPlot(
  data: number[],
  distribution: ContinuousDistribution,
): QQPoint[] {
  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;
  const points: QQPoint[] = [];

  for (let i = 0; i < n; i++) {
    // Filliben's estimate: (i − 0.3175) / (n + 0.365)
    const p = (i + 0.6825) / (n + 0.365);
    points.push({
      theoretical: distribution.quantile(p),
      sample: sorted[i],
    });
  }

  return points;
}

/**
 * Generate Q-Q plot coordinates against a standard normal distribution.
 */
export function normalQQPlot(data: number[]): QQPoint[] {
  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;
  const points: QQPoint[] = [];

  for (let i = 0; i < n; i++) {
    const p = (i + 0.6825) / (n + 0.365);
    points.push({
      theoretical: standardNormalQuantile(p),
      sample: sorted[i],
    });
  }

  return points;
}

// ── Box Plot ──────────────────────────────────────────────────────────────

/**
 * Compute box plot statistics (five-number summary + outliers).
 *
 * @param data  Numeric observations.
 */
export function boxPlotStats(data: number[]): BoxPlotStats {
  if (data.length === 0) throw new Error("Need at least 1 observation");

  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;

  const q1 = quantile(sorted, 0.25);
  const median = quantile(sorted, 0.5);
  const q3 = quantile(sorted, 0.75);
  const iqr = q3 - q1;

  const lowerFence = q1 - 1.5 * iqr;
  const upperFence = q3 + 1.5 * iqr;

  const lowerOutliers: number[] = [];
  const upperOutliers: number[] = [];
  let min = sorted[n - 1];
  let max = sorted[0];

  for (const x of sorted) {
    if (x < lowerFence) {
      lowerOutliers.push(x);
    } else if (x > upperFence) {
      upperOutliers.push(x);
    } else {
      if (x < min) min = x;
      if (x > max) max = x;
    }
  }

  // If all values are outliers
  if (min > max) {
    min = sorted[0];
    max = sorted[n - 1];
  }

  let sum = 0;
  for (const x of sorted) sum += x;
  const mean = sum / n;

  return { min, q1, median, q3, max, iqr, lowerFence, upperFence, lowerOutliers, upperOutliers, mean };
}

// ── KDE ───────────────────────────────────────────────────────────────────

/**
 * Kernel Density Estimate using a Gaussian kernel.
 *
 * @param data  Sample data.
 * @param options.nPoints  Number of evaluation points (default 100).
 * @param options.bandwidth  Bandwidth h (default: Silverman's rule).
 * @param options.range  [min, max] evaluation range.
 */
export function kde(
  data: number[],
  options: { nPoints?: number; bandwidth?: number; range?: [number, number] } = {},
): KDEPoint[] {
  if (data.length === 0) return [];

  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;

  // Silverman's rule of thumb
  const sd = stdDev(sorted);
  const iqr = quantile(sorted, 0.75) - quantile(sorted, 0.25);
  const h = options.bandwidth ?? 0.9 * Math.min(sd, iqr / 1.34) * Math.pow(n, -0.2);

  const nPoints = options.nPoints ?? 100;
  const lo = options.range?.[0] ?? sorted[0] - 3 * h;
  const hi = options.range?.[1] ?? sorted[n - 1] + 3 * h;
  const step = (hi - lo) / (nPoints - 1);

  const result: KDEPoint[] = [];
  const norm = 1 / (n * h * Math.sqrt(2 * Math.PI));

  for (let i = 0; i < nPoints; i++) {
    const x = lo + i * step;
    let density = 0;
    for (const xi of sorted) {
      const z = (x - xi) / h;
      density += Math.exp(-0.5 * z * z);
    }
    result.push({ x, density: density * norm });
  }

  return result;
}

// ── ECDF ──────────────────────────────────────────────────────────────────

/**
 * Empirical CDF coordinates.
 *
 * Returns sorted (x, F(x)) pairs suitable for a step plot.
 *
 * @param data  Sample data.
 */
export function ecdf(data: number[]): ECDFPoint[] {
  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;
  const points: ECDFPoint[] = [];

  for (let i = 0; i < n; i++) {
    points.push({ x: sorted[i], probability: (i + 1) / n });
  }

  return points;
}

// ── Scatter matrix data ───────────────────────────────────────────────────

/**
 * Generate pairwise scatter data for multiple variables.
 *
 * @param variables  Named numeric arrays, all same length.
 * @returns Array of { xName, yName, x, y } for each pair.
 */
export function scatterMatrixData(
  variables: Record<string, number[]>,
): { xName: string; yName: string; x: number[]; y: number[] }[] {
  const names = Object.keys(variables);
  const result: { xName: string; yName: string; x: number[]; y: number[] }[] = [];

  for (let i = 0; i < names.length; i++) {
    for (let j = 0; j < names.length; j++) {
      if (i === j) continue;
      result.push({
        xName: names[i],
        yName: names[j],
        x: [...variables[names[i]]],
        y: [...variables[names[j]]],
      });
    }
  }

  return result;
}

// ── Helpers ───────────────────────────────────────────────────────────────

function quantile(sorted: number[], p: number): number {
  const n = sorted.length;
  const idx = p * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (hi - idx) + sorted[hi] * (idx - lo);
}

function stdDev(data: number[]): number {
  const n = data.length;
  if (n < 2) return 0;
  let sum = 0;
  let sumSq = 0;
  for (const x of data) {
    sum += x;
    sumSq += x * x;
  }
  const m = sum / n;
  return Math.sqrt((sumSq / n - m * m) * n / (n - 1));
}

function standardNormalQuantile(p: number): number {
  if (p < 0.5) return -standardNormalQuantile(1 - p);
  const t = Math.sqrt(-2 * Math.log(1 - p));
  return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
    (1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
}
