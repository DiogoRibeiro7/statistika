/**
 * Native statistics acceleration layer.
 *
 * Exposes Fortran-accelerated routines for computationally intensive
 * operations (distance matrices, KDE, GLM cross-products, streaming stats).
 * Falls back to pure TypeScript when the native addon is unavailable.
 */

import { nativeAddon } from "./native-addon";

// ── Native interface ────────────────────────────────────────────────────

interface NativeStats {
  pairwiseEuclidean(data: number[][], n: number, p: number): number[][];
  gaussianPdfBatch(x: number[], mu: number, sigma2: number): number[];
  kdeGaussian(data: number[], evalPoints: number[], bandwidth: number): number[];
  weightedCrossProducts(
    X: number[][],
    W: number[],
    z: number[],
    n: number,
    cols: number,
  ): { XtWX: number[][]; XtWz: number[] };
  welfordBatch(
    values: number[],
    count: number,
    mean: number,
    m2: number,
    min: number,
    max: number,
  ): { count: number; mean: number; m2: number; min: number; max: number };
}

// Probe for native stats availability
let native: NativeStats | null = null;
try {
  if (nativeAddon && typeof nativeAddon.pairwiseEuclidean === "function") {
    native = nativeAddon as unknown as NativeStats;
  }
} catch {
  // Fallback to TypeScript
}

/** Whether native statistics acceleration is available. */
export const hasNativeStats = native !== null;

// ── Pairwise Euclidean Distance Matrix ──────────────────────────────────

/**
 * Compute pairwise Euclidean distance matrix.
 * Uses Fortran when available for O(n²p) acceleration.
 *
 * @param data - Array of n observation vectors, each of length p
 * @returns n×n symmetric distance matrix
 */
export function pairwiseEuclidean(data: number[][]): number[][] {
  const n = data.length;
  if (n === 0) return [];
  const p = data[0].length;

  if (native) {
    return native.pairwiseEuclidean(data, n, p);
  }

  return tsPairwiseEuclidean(data, n, p);
}

function tsPairwiseEuclidean(data: number[][], n: number, p: number): number[][] {
  const matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < p; k++) {
        const diff = data[i][k] - data[j][k];
        sum += diff * diff;
      }
      const d = Math.sqrt(sum);
      matrix[i][j] = d;
      matrix[j][i] = d;
    }
  }
  return matrix;
}

// ── Gaussian PDF Batch ──────────────────────────────────────────────────

/**
 * Evaluate Gaussian PDF at multiple points.
 * Uses Fortran when available for vectorized exp() acceleration.
 *
 * @param x - Array of evaluation points
 * @param mu - Mean
 * @param sigma2 - Variance (must be > 0)
 * @returns Array of PDF values
 */
export function gaussianPdfBatch(x: number[], mu: number, sigma2: number): number[] {
  if (native) {
    return native.gaussianPdfBatch(x, mu, sigma2);
  }

  const n = x.length;
  const result = new Array<number>(n);
  const normFactor = 1 / Math.sqrt(2 * Math.PI * sigma2);
  const inv2s = -0.5 / sigma2;
  for (let i = 0; i < n; i++) {
    result[i] = normFactor * Math.exp(inv2s * (x[i] - mu) ** 2);
  }
  return result;
}

// ── KDE (Gaussian kernel) ───────────────────────────────────────────────

/**
 * Kernel density estimation with Gaussian kernel.
 * Uses Fortran when available for O(n×m) acceleration.
 *
 * @param data - Input observations (n points)
 * @param evalPoints - Points at which to evaluate density (m points)
 * @param bandwidth - Smoothing bandwidth
 * @returns Array of density values at each evaluation point
 */
export function kdeGaussian(data: number[], evalPoints: number[], bandwidth: number): number[] {
  if (native) {
    return native.kdeGaussian(data, evalPoints, bandwidth);
  }

  return tsKdeGaussian(data, evalPoints, bandwidth);
}

function tsKdeGaussian(data: number[], evalPoints: number[], bandwidth: number): number[] {
  const n = data.length;
  const m = evalPoints.length;
  const result = new Array<number>(m);
  const invH = 1 / bandwidth;
  const normFactor = invH / (n * Math.sqrt(2 * Math.PI));

  for (let i = 0; i < m; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      const u = (evalPoints[i] - data[j]) * invH;
      sum += Math.exp(-0.5 * u * u);
    }
    result[i] = normFactor * sum;
  }
  return result;
}

// ── Weighted Cross-Products (for GLM IRLS) ──────────────────────────────

/**
 * Compute X^T W X and X^T W z for weighted least squares.
 * Uses Fortran when available for O(n×cols²) acceleration.
 *
 * @param X - Design matrix (n×cols array of arrays)
 * @param W - Weight vector (length n)
 * @param z - Adjusted response vector (length n)
 * @returns Object with XtWX (cols×cols) and XtWz (length cols)
 */
export function weightedCrossProducts(
  X: number[][],
  W: number[],
  z: number[],
): { XtWX: number[][]; XtWz: number[] } {
  const n = X.length;
  const cols = X[0].length;

  if (native) {
    return native.weightedCrossProducts(X, W, z, n, cols);
  }

  return tsWeightedCrossProducts(X, W, z, n, cols);
}

function tsWeightedCrossProducts(
  X: number[][],
  W: number[],
  z: number[],
  n: number,
  cols: number,
): { XtWX: number[][]; XtWz: number[] } {
  const XtWX = Array.from({ length: cols }, () => new Array<number>(cols).fill(0));
  const XtWz = new Array<number>(cols).fill(0);

  for (let i = 0; i < n; i++) {
    const wi = W[i];
    for (let j = 0; j < cols; j++) {
      XtWz[j] += X[i][j] * wi * z[i];
      for (let k = 0; k < cols; k++) {
        XtWX[j][k] += X[i][j] * wi * X[i][k];
      }
    }
  }

  return { XtWX, XtWz };
}

// ── Welford Batch Update ────────────────────────────────────────────────

/**
 * Batch update Welford's online statistics.
 * Uses Fortran when available for tight loop acceleration.
 *
 * @param values - New values to incorporate
 * @param state - Current state {count, mean, m2, min, max}
 * @returns Updated state
 */
export function welfordBatch(
  values: number[],
  state: { count: number; mean: number; m2: number; min: number; max: number },
): { count: number; mean: number; m2: number; min: number; max: number } {
  if (native) {
    return native.welfordBatch(
      values,
      state.count,
      state.mean,
      state.m2,
      state.min,
      state.max,
    );
  }

  return tsWelfordBatch(values, state);
}

function tsWelfordBatch(
  values: number[],
  state: { count: number; mean: number; m2: number; min: number; max: number },
): { count: number; mean: number; m2: number; min: number; max: number } {
  let { count, mean, m2, min, max } = state;

  for (const v of values) {
    count++;
    const delta = v - mean;
    mean += delta / count;
    const delta2 = v - mean;
    m2 += delta * delta2;
    if (v < min) min = v;
    if (v > max) max = v;
  }

  return { count, mean, m2, min, max };
}
