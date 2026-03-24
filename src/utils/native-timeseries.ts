/**
 * Native time series acceleration layer.
 *
 * Exposes Fortran-accelerated routines for ACF, PACF, exponential smoothing,
 * and differencing. Falls back to pure TypeScript when the native addon
 * is unavailable.
 */

import { nativeAddon } from "./native-addon";

// ── Native interface ────────────────────────────────────────────────────

interface NativeTimeSeries {
  acf(series: number[], maxLag: number): number[];
  pacfDurbinLevinson(acf: number[], maxLag: number): number[];
  exponentialSmoothing(series: number[], alpha: number): number[];
  holtWinters(
    series: number[],
    alpha: number,
    beta: number,
  ): { level: number[]; trend: number[]; fitted: number[] };
  difference(series: number[], d: number): number[];
}

// Probe for native time series availability
let native: NativeTimeSeries | null = null;
try {
  if (nativeAddon && typeof nativeAddon.acf === "function") {
    native = nativeAddon as unknown as NativeTimeSeries;
  }
} catch {
  // Fallback to TypeScript
}

/** Whether native time series acceleration is available. */
export const hasNativeTimeSeries = native !== null;

// ── ACF ─────────────────────────────────────────────────────────────────

/**
 * Compute the autocorrelation function (ACF).
 * Uses Fortran when available for O(n * maxLag) acceleration.
 *
 * @param series - Time series data
 * @param maxLag - Maximum lag to compute
 * @returns ACF values for lags 0..maxLag
 */
export function acf(series: number[], maxLag: number): number[] {
  if (native) {
    return native.acf(series, maxLag);
  }
  return tsAcf(series, maxLag);
}

function tsAcf(series: number[], maxLag: number): number[] {
  const n = series.length;
  let mu = 0;
  for (let i = 0; i < n; i++) mu += series[i];
  mu /= n;

  let gamma0 = 0;
  for (let i = 0; i < n; i++) gamma0 += (series[i] - mu) ** 2;
  gamma0 /= n;

  if (gamma0 === 0) return new Array(maxLag + 1).fill(0);

  const result = new Array(maxLag + 1);
  for (let k = 0; k <= maxLag; k++) {
    let s = 0;
    for (let t = 0; t < n - k; t++) {
      s += (series[t] - mu) * (series[t + k] - mu);
    }
    result[k] = (s / n) / gamma0;
  }
  return result;
}

// ── PACF via Durbin-Levinson ────────────────────────────────────────────

/**
 * Compute PACF from ACF using Durbin-Levinson recursion.
 * Uses Fortran when available.
 *
 * @param acfValues - ACF values for lags 0..maxLag
 * @param maxLag - Maximum lag
 * @returns PACF values for lags 0..maxLag
 */
export function pacfDurbinLevinson(acfValues: number[], maxLag: number): number[] {
  if (native) {
    return native.pacfDurbinLevinson(acfValues, maxLag);
  }
  return tsPacfDurbinLevinson(acfValues, maxLag);
}

function tsPacfDurbinLevinson(acfValues: number[], maxLag: number): number[] {
  const pacf = new Array(maxLag + 1).fill(0);
  pacf[0] = 1;
  if (maxLag < 1) return pacf;

  const phi: number[][] = [];
  for (let k = 0; k <= maxLag; k++) {
    phi[k] = new Array(maxLag + 1).fill(0);
  }

  phi[1][1] = acfValues[1];
  pacf[1] = acfValues[1];

  for (let k = 2; k <= maxLag; k++) {
    let num = acfValues[k];
    let den = 1;
    for (let j = 1; j < k; j++) {
      num -= phi[k - 1][j] * acfValues[k - j];
      den -= phi[k - 1][j] * acfValues[j];
    }
    phi[k][k] = den === 0 ? 0 : num / den;
    for (let j = 1; j < k; j++) {
      phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
    }
    pacf[k] = phi[k][k];
  }
  return pacf;
}

// ── Exponential Smoothing ───────────────────────────────────────────────

/**
 * Simple exponential smoothing.
 * Uses Fortran when available.
 *
 * @param series - Time series data
 * @param alpha - Smoothing parameter (0 < alpha <= 1)
 * @returns Smoothed series
 */
export function exponentialSmoothing(series: number[], alpha: number): number[] {
  if (native) {
    return native.exponentialSmoothing(series, alpha);
  }
  const n = series.length;
  const result = new Array(n);
  result[0] = series[0];
  for (let t = 1; t < n; t++) {
    result[t] = alpha * series[t] + (1 - alpha) * result[t - 1];
  }
  return result;
}

// ── Holt-Winters ────────────────────────────────────────────────────────

/**
 * Holt-Winters double exponential smoothing.
 * Uses Fortran when available.
 *
 * @param series - Time series data
 * @param alpha - Level smoothing parameter
 * @param beta - Trend smoothing parameter
 * @returns Object with level, trend, and fitted arrays
 */
export function holtWinters(
  series: number[],
  alpha: number,
  beta: number,
): { level: number[]; trend: number[]; fitted: number[] } {
  if (native) {
    return native.holtWinters(series, alpha, beta);
  }

  const n = series.length;
  const level = new Array(n);
  const trend = new Array(n);
  const fitted = new Array(n);

  level[0] = series[0];
  trend[0] = n >= 2 ? series[1] - series[0] : 0;
  fitted[0] = series[0];

  for (let t = 1; t < n; t++) {
    fitted[t] = level[t - 1] + trend[t - 1];
    level[t] = alpha * series[t] + (1 - alpha) * fitted[t];
    trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1];
  }

  return { level, trend, fitted };
}

// ── Differencing ────────────────────────────────────────────────────────

/**
 * Compute d-th order differences.
 * Uses Fortran when available.
 *
 * @param series - Time series data
 * @param d - Order of differencing
 * @returns Differenced series of length n - d
 */
export function difference(series: number[], d: number): number[] {
  if (native) {
    return native.difference(series, d);
  }

  let current = series.slice();
  for (let i = 0; i < d; i++) {
    const next = new Array(current.length - 1);
    for (let t = 0; t < next.length; t++) {
      next[t] = current[t + 1] - current[t];
    }
    current = next;
  }
  return current;
}
