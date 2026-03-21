import { Dataset } from "./types";
import { mean, variance } from "./utils/descriptive";
import { autocorrelation, arima, difference, ARIMAResult } from "./time-series";
import { Normal } from "./distributions/continuous/normal";

/**
 * Augmented Dickey-Fuller test for stationarity.
 *
 * Tests the null hypothesis that the series has a unit root (non-stationary).
 * A low p-value suggests the series is stationary.
 *
 * @param series - Time series data
 * @param maxLags - Maximum number of lags (default: floor(cbrt(n)))
 */
export function adfTest(
  series: Dataset,
  maxLags?: number,
): { statistic: number; pValue: number; lags: number; isStationary: boolean } {
  const n = series.length;
  if (n < 10) throw new Error("Need at least 10 observations for ADF test");

  const lags = maxLags ?? Math.floor(Math.cbrt(n));
  const diffed = difference(series, 1);

  // OLS regression: Δy_t = α + β*y_{t-1} + Σ γ_i*Δy_{t-i} + ε_t
  const start = lags;
  const nObs = diffed.length - start;

  if (nObs < lags + 3) throw new Error("Not enough observations for the specified lag");

  // Build design matrix
  const y: number[] = [];
  const X: number[][] = [];

  for (let t = start; t < diffed.length; t++) {
    y.push(diffed[t]);
    const row: number[] = [1, series[t]]; // intercept + lagged level
    for (let j = 1; j <= lags; j++) {
      row.push(diffed[t - j]);
    }
    X.push(row);
  }

  // OLS: β = (X'X)^{-1} X'y
  const cols = X[0].length;
  const XtX = Array.from({ length: cols }, () => new Array<number>(cols).fill(0));
  const Xty = new Array<number>(cols).fill(0);

  for (let i = 0; i < nObs; i++) {
    for (let j = 0; j < cols; j++) {
      Xty[j] += X[i][j] * y[i];
      for (let k = 0; k < cols; k++) {
        XtX[j][k] += X[i][j] * X[i][k];
      }
    }
  }

  // Solve
  const beta = solveSystem(XtX, Xty);

  // Compute residuals and SE
  let sse = 0;
  for (let i = 0; i < nObs; i++) {
    let pred = 0;
    for (let j = 0; j < cols; j++) pred += X[i][j] * beta[j];
    sse += (y[i] - pred) ** 2;
  }

  const s2 = sse / (nObs - cols);
  const XtXinv = invertMatrix(XtX);
  const seBeta = Math.sqrt(Math.max(0, s2 * XtXinv[1][1]));

  const statistic = seBeta > 0 ? beta[1] / seBeta : 0;

  // Approximate p-value using MacKinnon critical values (rough approximation)
  // Critical values for ADF with intercept: -3.43 (1%), -2.86 (5%), -2.57 (10%)
  let pValue: number;
  if (statistic < -3.43) pValue = 0.01;
  else if (statistic < -2.86) pValue = 0.05;
  else if (statistic < -2.57) pValue = 0.10;
  else if (statistic < -1.94) pValue = 0.30;
  else pValue = 0.50 + 0.5 * new Normal().cdf(statistic);

  return {
    statistic,
    pValue,
    lags,
    isStationary: pValue < 0.05,
  };
}

/**
 * Auto-ARIMA: automatically select the best ARIMA(p,d,q) model.
 *
 * Searches over combinations of p and q (with d determined by ADF tests)
 * and selects the model with the lowest AIC.
 *
 * @param series - Time series data
 * @param options - Configuration
 */
export function autoArima(
  series: Dataset,
  options: {
    maxP?: number;
    maxD?: number;
    maxQ?: number;
  } = {},
): ARIMAResult & { selectedOrder: { p: number; d: number; q: number } } {
  const maxP = options.maxP ?? 3;
  const maxD = options.maxD ?? 2;
  const maxQ = options.maxQ ?? 3;

  // Determine d by testing stationarity
  let d = 0;
  let current = series;
  for (d = 0; d < maxD; d++) {
    try {
      const test = adfTest(current);
      if (test.isStationary) break;
      current = difference(current, 1);
    } catch {
      break;
    }
  }

  // Grid search over p and q
  let bestAIC = Infinity;
  let bestResult: ARIMAResult | null = null;
  let bestP = 0;
  let bestQ = 0;

  for (let p = 0; p <= maxP; p++) {
    for (let q = 0; q <= maxQ; q++) {
      if (p === 0 && q === 0) continue;
      try {
        if (series.length < p + d + q + 2) continue;
        const result = arima(series, p, d, q);
        if (result.aic < bestAIC) {
          bestAIC = result.aic;
          bestResult = result;
          bestP = p;
          bestQ = q;
        }
      } catch {
        // Skip invalid combinations
      }
    }
  }

  if (!bestResult) {
    // Fallback to ARIMA(1,d,0)
    bestResult = arima(series, 1, d, 0);
    bestP = 1;
    bestQ = 0;
  }

  return {
    ...bestResult,
    selectedOrder: { p: bestP, d, q: bestQ },
  };
}

/**
 * Forecast with prediction intervals.
 *
 * @param model - Fitted ARIMA result
 * @param steps - Number of steps ahead
 * @param confidence - Confidence level (default: 0.95)
 */
export function forecastWithIntervals(
  model: ARIMAResult,
  steps: number,
  confidence = 0.95,
): {
  point: number[];
  lower: number[];
  upper: number[];
  confidence: number;
} {
  const point = model.forecast(steps);
  const normal = new Normal();
  const z = normal.quantile(1 - (1 - confidence) / 2);
  const sigma = Math.sqrt(model.sigma2);

  const lower = new Array<number>(steps);
  const upper = new Array<number>(steps);

  for (let h = 0; h < steps; h++) {
    // Prediction interval widens with horizon
    // For AR models, error accumulates roughly as sigma * sqrt(h+1)
    const se = sigma * Math.sqrt(h + 1);
    lower[h] = point[h] - z * se;
    upper[h] = point[h] + z * se;
  }

  return { point, lower, upper, confidence };
}

/**
 * Seasonal decomposition (additive).
 *
 * Decomposes a time series into trend, seasonal, and residual components
 * using moving averages.
 *
 * @param series - Time series data
 * @param period - Seasonal period (e.g., 12 for monthly, 4 for quarterly)
 */
export function seasonalDecompose(
  series: Dataset,
  period: number,
): { trend: (number | null)[]; seasonal: number[]; residual: (number | null)[] } {
  const n = series.length;
  if (period < 2) throw new Error("Period must be at least 2");
  if (n < 2 * period) throw new Error("Need at least 2 full periods of data");

  // Step 1: Compute trend using centered moving average
  const trend = new Array<number | null>(n).fill(null);
  const half = Math.floor(period / 2);

  for (let i = half; i < n - half; i++) {
    let sum = 0;
    let count = 0;
    for (let j = -half; j <= half; j++) {
      const idx = i + j;
      if (idx >= 0 && idx < n) {
        // For even periods, average first and last weights
        const w = (j === -half || j === half) && period % 2 === 0 ? 0.5 : 1;
        sum += series[idx] * w;
        count += w;
      }
    }
    trend[i] = sum / count;
  }

  // Step 2: Detrend
  const detrended = series.map((v, i) =>
    trend[i] !== null ? v - (trend[i] as number) : 0,
  );

  // Step 3: Compute seasonal component (average detrended by position)
  const seasonal = new Array<number>(n);
  const seasonalAvg = new Array<number>(period).fill(0);
  const seasonalCount = new Array<number>(period).fill(0);

  for (let i = 0; i < n; i++) {
    if (trend[i] !== null) {
      seasonalAvg[i % period] += detrended[i];
      seasonalCount[i % period]++;
    }
  }

  for (let s = 0; s < period; s++) {
    if (seasonalCount[s] > 0) seasonalAvg[s] /= seasonalCount[s];
  }

  // Center seasonal component (subtract its mean)
  const seasonalMean = seasonalAvg.reduce((a, b) => a + b, 0) / period;
  for (let s = 0; s < period; s++) seasonalAvg[s] -= seasonalMean;

  for (let i = 0; i < n; i++) {
    seasonal[i] = seasonalAvg[i % period];
  }

  // Step 4: Residual
  const residual = series.map((v, i) =>
    trend[i] !== null ? v - (trend[i] as number) - seasonal[i] : null,
  );

  return { trend, seasonal, residual };
}

// ── Helpers ─────────────────────────────────────────────────────────────

function solveSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-12) continue;

    for (let j = col; j <= n; j++) aug[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = col; j <= n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }

  return aug.map((row) => row[n]);
}

function invertMatrix(matrix: number[][]): number[][] {
  const n = matrix.length;
  const aug = matrix.map((row, i) => {
    const r = [...row];
    for (let j = 0; j < n; j++) r.push(i === j ? 1 : 0);
    return r;
  });

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-12) continue;

    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }

  return aug.map((row) => row.slice(n));
}
