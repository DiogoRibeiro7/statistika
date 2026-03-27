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

/**
 * A single bin of a histogram.
 */
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

/**
 * A single point on a Q-Q plot.
 */
export interface QQPoint {
  /** Theoretical quantile. */
  theoretical: number;
  /** Sample quantile. */
  sample: number;
}

/**
 * Box plot statistics (five-number summary plus outliers and mean).
 */
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

/**
 * A single point of a kernel density estimate.
 */
export interface KDEPoint {
  /** Evaluation point. */
  x: number;
  /** Density estimate. */
  density: number;
}

/**
 * A single point of the empirical CDF.
 */
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
 * Uses Sturges' rule by default: nBins = ceil(log2(n) + 1).
 *
 * @param data - Numeric observations
 * @param options - Configuration options
 * @param options.bins - Number of bins (default: Sturges' rule)
 * @param options.range - [min, max] range to bin over (default: data range)
 * @returns Array of HistogramBin objects with lo, hi, mid, count, frequency, and density
 *
 * @example
 * ```ts
 * const bins = histogramBins([1, 2, 2, 3, 3, 3, 4], { bins: 4 });
 * // bins[i].count — number of observations in bin i
 * // bins[i].density — density = frequency / bin_width
 * ```
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
 * Generate Q-Q plot coordinates against a theoretical distribution.
 *
 * Uses Filliben's plotting position: p_i = (i + 0.6825) / (n + 0.365).
 *
 * @param data - Sample data
 * @param distribution - Theoretical continuous distribution to compare against
 * @returns Array of QQPoint objects with theoretical and sample quantiles
 *
 * @example
 * ```ts
 * const points = qqPlot([1, 2, 3, 4, 5], new Normal(3, 1.5));
 * // points[i].theoretical — expected quantile from the distribution
 * // points[i].sample — corresponding sample quantile
 * ```
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
 * Generate Q-Q plot coordinates against a standard normal distribution (N(0,1)).
 *
 * Convenience wrapper that uses an approximation of the standard normal quantile.
 *
 * @param data - Sample data
 * @returns Array of QQPoint objects with theoretical (normal) and sample quantiles
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
 * Computes Q1, median, Q3, IQR, and identifies outliers using Tukey's fences
 * (Q1 - 1.5*IQR, Q3 + 1.5*IQR). The min/max are the non-outlier extremes.
 *
 * @param data - Numeric observations
 * @returns BoxPlotStats with five-number summary, fences, outliers, and mean
 * @throws Error if data is empty
 *
 * @example
 * ```ts
 * const stats = boxPlotStats([1, 2, 3, 4, 5, 6, 7, 8, 100]);
 * // stats.median — 5
 * // stats.upperOutliers — [100]
 * ```
 */
export function boxPlotStats(data: number[]): BoxPlotStats {
  if (data.length === 0) throw new Error(`Invalid parameter 'data': expected at least 1 observation, received ${data.length}`);

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
 * Estimates f(x) = (1 / (n * h * sqrt(2*pi))) * sum(exp(-0.5 * ((x - x_i) / h)^2)).
 * Uses Silverman's rule of thumb for bandwidth selection by default:
 * h = 0.9 * min(sd, IQR/1.34) * n^(-1/5).
 *
 * @param data - Sample data
 * @param options - Configuration options
 * @param options.nPoints - Number of evaluation points (default 100)
 * @param options.bandwidth - Bandwidth h (default: Silverman's rule)
 * @param options.range - [min, max] evaluation range (default: data range +/- 3h)
 * @returns Array of KDEPoint objects with x and density values
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
 * F(x_i) = (i + 1) / n for the i-th order statistic.
 *
 * @param data - Sample data
 * @returns Array of ECDFPoint objects with x and cumulative probability
 *
 * @example
 * ```ts
 * const points = ecdf([3, 1, 2]);
 * // [{x: 1, probability: 1/3}, {x: 2, probability: 2/3}, {x: 3, probability: 1}]
 * ```
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
 * Produces all off-diagonal pairs (i, j) where i != j, creating
 * data suitable for a scatter plot matrix.
 *
 * @param variables - Named numeric arrays, all same length
 * @returns Array of { xName, yName, x, y } for each pair of variables
 *
 * @example
 * ```ts
 * const pairs = scatterMatrixData({ height: [170, 180], weight: [60, 80] });
 * // [{xName: "height", yName: "weight", x: [170, 180], y: [60, 80]},
 * //  {xName: "weight", yName: "height", x: [60, 80], y: [170, 180]}]
 * ```
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
