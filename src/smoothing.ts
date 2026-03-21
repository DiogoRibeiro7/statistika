import { Dataset } from "./types";

/**
 * Simple Moving Average (SMA).
 *
 * @param data - Input time series
 * @param window - Window size
 */
export function simpleMovingAverage(data: Dataset, window: number): number[] {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  if (window < 1 || window > data.length) throw new Error("Window must be between 1 and data length");
  if (!Number.isInteger(window)) throw new Error("Window must be an integer");

  const result: number[] = [];
  let sum = 0;

  for (let i = 0; i < data.length; i++) {
    sum += data[i];
    if (i >= window) sum -= data[i - window];
    if (i >= window - 1) result.push(sum / window);
  }

  return result;
}

/**
 * Exponential Moving Average (EMA).
 *
 * @param data - Input time series
 * @param alpha - Smoothing factor (0 < alpha <= 1). Higher = more weight on recent values.
 */
export function exponentialMovingAverage(data: Dataset, alpha: number): number[] {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  if (alpha <= 0 || alpha > 1) throw new Error("Alpha must be between 0 (exclusive) and 1 (inclusive)");

  const result = new Array<number>(data.length);
  result[0] = data[0];

  for (let i = 1; i < data.length; i++) {
    result[i] = alpha * data[i] + (1 - alpha) * result[i - 1];
  }

  return result;
}

/**
 * Weighted Moving Average (WMA).
 *
 * Applies linearly increasing weights (most recent gets highest weight).
 *
 * @param data - Input time series
 * @param window - Window size
 */
export function weightedMovingAverage(data: Dataset, window: number): number[] {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  if (window < 1 || window > data.length) throw new Error("Window must be between 1 and data length");

  const weightSum = (window * (window + 1)) / 2;
  const result: number[] = [];

  for (let i = window - 1; i < data.length; i++) {
    let sum = 0;
    for (let j = 0; j < window; j++) {
      sum += data[i - window + 1 + j] * (j + 1);
    }
    result.push(sum / weightSum);
  }

  return result;
}

/**
 * LOESS (LOcally Estimated Scatterplot Smoothing).
 *
 * Fits weighted linear regressions at each point using nearby observations.
 *
 * @param x - Independent variable
 * @param y - Dependent variable
 * @param span - Proportion of data to use for each local fit (default: 0.3)
 */
export function loess(x: Dataset, y: Dataset, span = 0.3): Dataset {
  if (x.length !== y.length) throw new Error("x and y must have the same length");
  const n = x.length;
  if (n < 3) throw new Error("Need at least 3 observations");
  if (span <= 0 || span > 1) throw new Error("Span must be between 0 (exclusive) and 1 (inclusive)");

  const k = Math.max(2, Math.floor(span * n));
  const result = new Array<number>(n);

  for (let i = 0; i < n; i++) {
    // Find k nearest neighbors by x-distance
    const distances = x.map((xj, j) => ({ dist: Math.abs(xj - x[i]), j }));
    distances.sort((a, b) => a.dist - b.dist);
    const neighbors = distances.slice(0, k);
    const maxDist = neighbors[neighbors.length - 1].dist || 1;

    // Tricube weights
    const weights = neighbors.map((nb) => {
      const u = nb.dist / (maxDist * 1.001); // slight padding to avoid 0 denominator
      return (1 - u ** 3) ** 3;
    });

    // Weighted linear regression
    let swx = 0, swy = 0, swxx = 0, swxy = 0, sw = 0;
    for (let m = 0; m < neighbors.length; m++) {
      const j = neighbors[m].j;
      const w = weights[m];
      sw += w;
      swx += w * x[j];
      swy += w * y[j];
      swxx += w * x[j] * x[j];
      swxy += w * x[j] * y[j];
    }

    const det = sw * swxx - swx * swx;
    if (Math.abs(det) < 1e-12) {
      result[i] = swy / sw; // fall back to weighted mean
    } else {
      const slope = (sw * swxy - swx * swy) / det;
      const intercept = (swy - slope * swx) / sw;
      result[i] = intercept + slope * x[i];
    }
  }

  return result;
}

/**
 * Natural cubic spline interpolation.
 *
 * Fits a cubic spline through the data points with natural boundary
 * conditions (second derivative = 0 at endpoints).
 *
 * @param xs - Knot x-coordinates (must be sorted and unique)
 * @param ys - Knot y-coordinates
 * @returns A function that evaluates the spline at any x
 */
export function cubicSpline(
  xs: Dataset,
  ys: Dataset,
): (x: number) => number {
  if (xs.length !== ys.length) throw new Error("xs and ys must have the same length");
  const n = xs.length;
  if (n < 3) throw new Error("Need at least 3 knots");

  // Check sorted
  for (let i = 1; i < n; i++) {
    if (xs[i] <= xs[i - 1]) throw new Error("xs must be strictly increasing");
  }

  // Compute spline coefficients using tridiagonal solver
  const h = new Array<number>(n - 1);
  for (let i = 0; i < n - 1; i++) h[i] = xs[i + 1] - xs[i];

  // Set up tridiagonal system for second derivatives (c)
  const alpha = new Array<number>(n - 2);
  for (let i = 0; i < n - 2; i++) {
    alpha[i] =
      (3 / h[i + 1]) * (ys[i + 2] - ys[i + 1]) -
      (3 / h[i]) * (ys[i + 1] - ys[i]);
  }

  // Natural spline: c[0] = c[n-1] = 0
  const c = new Array<number>(n).fill(0);
  const l = new Array<number>(n - 2);
  const mu = new Array<number>(n - 2);
  const z = new Array<number>(n - 2);

  l[0] = 2 * (h[0] + h[1]);
  mu[0] = h[1] / l[0];
  z[0] = alpha[0] / l[0];

  for (let i = 1; i < n - 2; i++) {
    l[i] = 2 * (h[i] + h[i + 1]) - h[i] * mu[i - 1];
    mu[i] = i < n - 3 ? h[i + 1] / l[i] : 0;
    z[i] = (alpha[i] - h[i] * z[i - 1]) / l[i];
  }

  // Back substitution
  for (let i = n - 3; i >= 0; i--) {
    c[i + 1] = z[i] - mu[i] * c[i + 2];
  }

  // Compute b and d coefficients
  const b = new Array<number>(n - 1);
  const d = new Array<number>(n - 1);

  for (let i = 0; i < n - 1; i++) {
    d[i] = (c[i + 1] - c[i]) / (3 * h[i]);
    b[i] = (ys[i + 1] - ys[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3;
  }

  return (x: number): number => {
    // Find the appropriate interval
    let i: number;
    if (x <= xs[0]) {
      i = 0;
    } else if (x >= xs[n - 1]) {
      i = n - 2;
    } else {
      // Binary search
      let lo = 0, hi = n - 1;
      while (lo < hi - 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (xs[mid] <= x) lo = mid;
        else hi = mid;
      }
      i = lo;
    }

    const dx = x - xs[i];
    return ys[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
  };
}
