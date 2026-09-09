import { Dataset } from "./types";
import { mean, variance } from "./utils/descriptive";
import { autocorrelation, arima, difference, ARIMAResult } from "./time-series";
import { normalCdf, normalQuantile } from "./utils/linalg";

/**
 * Augmented Dickey-Fuller test for stationarity.
 *
 * Tests the null hypothesis that the series has a unit root (non-stationary).
 * A low p-value suggests the series is stationary.
 *
 * @param series - Time series data
 * @param maxLags - Maximum number of lags (default: floor(cbrt(n)))
 * @returns An object with the test statistic, p-value, lag count, and whether the series is stationary at the 0.05 level
 * @throws If the series has fewer than 10 observations or not enough observations for the specified lag
 *
 * @example
 * ```ts
 * const result = adfTest([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
 * console.log(result.isStationary); // whether the series is stationary at 0.05
 * ```
 */
export function adfTest(
  series: Dataset,
  maxLags?: number,
): { statistic: number; pValue: number; lags: number; isStationary: boolean } {
  const n = series.length;
  if (n < 10) throw new Error(`Invalid parameter 'series': expected at least 10 observations, received ${n}`);

  // NaN / Infinity guard
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(series[i])) {
      throw new Error(`Invalid parameter 'series': expected finite number at index ${i}, received ${series[i]}`);
    }
  }

  const lags = maxLags ?? Math.floor(Math.cbrt(n));
  const diffed = difference(series, 1);

  // OLS regression: delta_y_t = alpha + beta*y_{t-1} + sum gamma_i*delta_y_{t-i} + eps_t
  const start = lags;
  const nObs = diffed.length - start;

  if (nObs < lags + 3) throw new Error(`Invalid parameter 'maxLags': expected enough observations for the specified lag, received nObs=${nObs}, lags=${lags}`);

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

  // OLS: beta = (X'X)^{-1} X'y
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

  // Solve using Gaussian elimination (tolerant of near-singular systems)
  const beta = solveSystemTolerant(XtX, Xty);

  // Compute residuals and SE
  let sse = 0;
  for (let i = 0; i < nObs; i++) {
    let pred = 0;
    for (let j = 0; j < cols; j++) pred += X[i][j] * beta[j];
    sse += (y[i] - pred) ** 2;
  }

  const s2 = sse / (nObs - cols);
  const XtXinv = invertMatrixTolerant(XtX);
  const seBeta = Math.sqrt(Math.max(0, s2 * XtXinv[1][1]));

  const statistic = seBeta > 0 ? beta[1] / seBeta : 0;

  // Approximate p-value using MacKinnon critical values (rough approximation)
  // Critical values for ADF with intercept: -3.43 (1%), -2.86 (5%), -2.57 (10%)
  let pValue: number;
  if (statistic < -3.43) pValue = 0.01;
  else if (statistic < -2.86) pValue = 0.05;
  else if (statistic < -2.57) pValue = 0.10;
  else if (statistic < -1.94) pValue = 0.30;
  else pValue = 0.50 + 0.5 * normalCdf(statistic);

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
 * @returns The best-fit {@link ARIMAResult} augmented with the selected (p, d, q) order
 * @throws If the series is too short for any valid ARIMA model or all candidate models fail
 *
 * @example
 * ```ts
 * const sales = [120, 135, 148, 160, 172, 185, 190, 205, 218, 230, 245, 260];
 * const result = autoArima(sales, { maxP: 2, maxQ: 2 });
 * console.log(result.selectedOrder); // e.g. { p: 1, d: 1, q: 1 }
 * console.log(result.forecast(6));   // 6-step-ahead forecast
 * ```
 */
export function autoArima(
  series: Dataset,
  options: {
    maxP?: number;
    maxD?: number;
    maxQ?: number;
  } = {},
): ARIMAResult & { selectedOrder: { p: number; d: number; q: number } } {
  if (series.length < 10) throw new Error(`Invalid parameter 'series': expected at least 10 observations, received ${series.length}`);

  // NaN / Infinity guard
  for (let i = 0; i < series.length; i++) {
    if (!Number.isFinite(series[i])) {
      throw new Error(`Invalid parameter 'series': expected finite number at index ${i}, received ${series[i]}`);
    }
  }

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
 * @returns An object containing point forecasts and lower/upper prediction interval bounds
 * @throws If steps is not a positive integer or confidence is not in (0, 1)
 *
 * @example
 * ```ts
 * const model = arima(series, 1, 1, 0);
 * const fc = forecastWithIntervals(model, 5, 0.95);
 * console.log(fc.point);  // 5-step point forecasts
 * console.log(fc.lower);  // lower 95% bounds
 * ```
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
  if (!Number.isFinite(steps) || steps < 1) {
    throw new Error(`Invalid parameter 'steps': expected a positive integer, received ${steps}`);
  }
  if (!Number.isFinite(confidence) || confidence <= 0 || confidence >= 1) {
    throw new Error(`Invalid parameter 'confidence': confidence must be in (0, 1), received ${confidence}`);
  }

  const point = model.forecast(steps);
  const z = normalQuantile(1 - (1 - confidence) / 2);
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
 * @returns An object with trend (nullable where edges are undefined), seasonal, and residual arrays
 * @throws If period is less than 2 or the series has fewer than 2 full periods
 *
 * @example
 * ```ts
 * const result = seasonalDecompose(monthlyData, 12);
 * console.log(result.trend);    // trend component (null at edges)
 * console.log(result.seasonal); // repeating seasonal pattern
 * ```
 */
export function seasonalDecompose(
  series: Dataset,
  period: number,
): { trend: (number | null)[]; seasonal: number[]; residual: (number | null)[] } {
  const n = series.length;
  if (period < 2) throw new Error(`Invalid parameter 'period': expected at least 2, received ${period}`);
  if (n < 2 * period) throw new Error(`Invalid parameter 'series': expected at least 2 full periods (${2 * period}), received ${n}`);

  // NaN / Infinity guard
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(series[i])) {
      throw new Error(`Invalid parameter 'series[${i}]': expected a finite number, received ${series[i]}`);
    }
  }

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

// ── SARIMA ──────────────────────────────────────────────────────────────

/**
 * Result of fitting a SARIMA(p,d,q)(P,D,Q)[m] model.
 *
 * Contains both non-seasonal and seasonal AR/MA coefficients,
 * differencing orders, residual variance, AIC, and a forecast function.
 */
export interface SARIMAResult {
  /** Non-seasonal AR coefficients */
  arCoefficients: number[];
  /** Non-seasonal MA coefficients */
  maCoefficients: number[];
  /** Seasonal AR coefficients */
  seasonalArCoefficients: number[];
  /** Seasonal MA coefficients */
  seasonalMaCoefficients: number[];
  /** Intercept / constant term */
  intercept: number;
  /** Non-seasonal order (p, d, q) */
  order: { p: number; d: number; q: number };
  /** Seasonal order (P, D, Q, m) */
  seasonalOrder: { P: number; D: number; Q: number; m: number };
  /** Residual variance */
  sigma2: number;
  /** Akaike Information Criterion */
  aic: number;
  /** Forecast future values */
  forecast: (steps: number) => number[];
  /** Fitted residuals */
  residuals: number[];
}

/**
 * Seasonal differencing: removes seasonal pattern by subtracting the value
 * from `m` periods ago.
 */
function seasonalDifference(series: number[], m: number, D: number): number[] {
  let result = series;
  for (let iter = 0; iter < D; iter++) {
    if (result.length <= m) {
      throw new Error(`Invalid parameter 'series': expected length > ${m} for seasonal differencing, received ${result.length}`);
    }
    const diff = new Array(result.length - m);
    for (let i = m; i < result.length; i++) {
      diff[i - m] = result[i] - result[i - m];
    }
    result = diff;
  }
  return result;
}

/**
 * Integrate seasonal differenced forecasts back to original scale.
 */
function integrateSeasonalForecasts(
  original: number[],
  forecasts: number[],
  m: number,
  D: number,
): number[] {
  let result = [...forecasts];

  for (let iter = 0; iter < D; iter++) {
    // Get the series differenced (D - 1 - iter) times seasonally
    let ref = original;
    for (let d = 0; d < D - 1 - iter; d++) {
      const tmp = new Array(ref.length - m);
      for (let i = m; i < ref.length; i++) tmp[i - m] = ref[i] - ref[i - m];
      ref = tmp;
    }

    const integrated: number[] = [];
    for (let s = 0; s < result.length; s++) {
      // Need value from m steps back
      const idx = ref.length - m + s;
      const prev =
        idx >= 0 && idx < ref.length
          ? ref[idx]
          : idx >= ref.length
            ? integrated[idx - ref.length]
            : 0;
      integrated.push(result[s] + prev);
    }
    result = integrated;
  }

  return result;
}

/**
 * Fit a SARIMA(p,d,q)(P,D,Q)[m] model.
 *
 * Extends ARIMA to handle seasonal patterns by applying both non-seasonal
 * differencing (order d) and seasonal differencing (order D, at lag m).
 * AR and MA coefficients are estimated via Yule-Walker equations for the AR
 * part and iterative residual autocovariance matching for the MA part.
 * Forecasts are integrated back through both differencing steps.
 *
 * @param series - Time series data (must not contain NaN or Infinity)
 * @param p - Non-seasonal autoregressive order (>= 0)
 * @param d - Non-seasonal differencing order (>= 0)
 * @param q - Non-seasonal moving average order (>= 0)
 * @param P - Seasonal autoregressive order (>= 0)
 * @param D - Seasonal differencing order (>= 0)
 * @param Q - Seasonal moving average order (>= 0)
 * @param m - Seasonal period (e.g., 12 for monthly data, 4 for quarterly)
 * @returns A {@link SARIMAResult} with coefficients, diagnostics, and forecast function
 * @throws {Error} If m < 2, any order is negative, or the series is too short
 *
 * @example
 * ```ts
 * // Fit SARIMA(1,1,1)(1,1,0)[12] to monthly data
 * const result = sarima(monthlyData, 1, 1, 1, 1, 1, 0, 12);
 * const forecast = result.forecast(12); // 12-step-ahead forecast
 * ```
 */
export function sarima(
  series: Dataset,
  p: number,
  d: number,
  q: number,
  P: number,
  D: number,
  Q: number,
  m: number,
): SARIMAResult {
  if (m < 2) throw new Error(`Invalid parameter 'm': expected at least 2, received ${m}`);
  if (p < 0 || d < 0 || q < 0 || P < 0 || D < 0 || Q < 0) {
    throw new Error(`Invalid parameters: expected non-negative orders, received p=${p}, d=${d}, q=${q}, P=${P}, D=${D}, Q=${Q}`);
  }

  const minLen = p + d + q + (P + D + Q) * m + 2;
  if (series.length < minLen) {
    throw new Error(`Invalid parameter 'series': expected at least ${minLen} observations for the specified SARIMA order, received ${series.length}`);
  }

  for (let i = 0; i < series.length; i++) {
    if (!Number.isFinite(series[i])) {
      throw new Error(`Invalid parameter 'series[${i}]': expected a finite number, received ${series[i]}`);
    }
  }

  // Step 1: Apply seasonal differencing, then non-seasonal differencing
  const seasonallyDiffed = seasonalDifference(series, m, D);
  const diffed = difference(seasonallyDiffed, d);
  const n = diffed.length;
  const mu = mean(diffed);
  const centered = diffed.map((v) => v - mu);

  // Step 2: Estimate AR coefficients (non-seasonal + seasonal)
  // Total effective AR order includes seasonal lags
  const maxArLag = Math.max(p, P > 0 ? P * m : 0);
  const maxMaLag = Math.max(q, Q > 0 ? Q * m : 0);
  const maxStart = Math.max(maxArLag, maxMaLag);

  if (n <= maxStart + 1) {
    throw new Error(`Invalid parameter 'series': expected more than ${maxStart + 1} observations after differencing, received ${n}`);
  }

  // Compute autocovariances up to needed lag
  const maxCovLag = Math.max(maxArLag, maxMaLag) + 1;
  const gamma = new Array(maxCovLag + 1);
  for (let k = 0; k <= maxCovLag; k++) {
    let sum = 0;
    for (let t = 0; t < n - k; t++) sum += centered[t] * centered[t + k];
    gamma[k] = sum / n;
  }

  // Non-seasonal AR coefficients via Yule-Walker
  const arCoeffs = new Array(p).fill(0);
  if (p > 0 && gamma[0] !== 0) {
    if (p === 1) {
      arCoeffs[0] = gamma[1] / gamma[0];
    } else {
      const a: number[][] = [];
      for (let i = 0; i <= p; i++) a[i] = new Array(p + 1).fill(0);
      a[1][1] = gamma[1] / gamma[0];
      let e = gamma[0] * (1 - a[1][1] * a[1][1]);
      for (let k = 2; k <= p; k++) {
        let lambda = gamma[k];
        for (let j = 1; j < k; j++) lambda -= a[k - 1][j] * gamma[k - j];
        a[k][k] = e === 0 ? 0 : lambda / e;
        for (let j = 1; j < k; j++) a[k][j] = a[k - 1][j] - a[k][k] * a[k - 1][k - j];
        e *= 1 - a[k][k] * a[k][k];
      }
      for (let j = 1; j <= p; j++) arCoeffs[j - 1] = a[p][j];
    }
  }

  // Seasonal AR coefficients: use autocovariance at seasonal lags
  const sarCoeffs = new Array(P).fill(0);
  if (P > 0 && gamma[0] !== 0) {
    for (let s = 0; s < P; s++) {
      const lag = (s + 1) * m;
      if (lag <= maxCovLag) {
        sarCoeffs[s] = gamma[lag] / gamma[0];
      }
    }
  }

  // Step 3: Compute residuals and estimate MA coefficients
  const residuals = new Array(n).fill(0);

  // Compute initial residuals
  for (let t = maxStart; t < n; t++) {
    let pred = 0;
    for (let j = 0; j < p; j++) {
      if (t - 1 - j >= 0) pred += arCoeffs[j] * centered[t - 1 - j];
    }
    for (let s = 0; s < P; s++) {
      const lag = (s + 1) * m;
      if (t - lag >= 0) pred += sarCoeffs[s] * centered[t - lag];
    }
    residuals[t] = centered[t] - pred;
  }

  // Non-seasonal MA coefficients
  let maCoeffs = new Array(q).fill(0);
  // Seasonal MA coefficients
  let smaCoeffs = new Array(Q).fill(0);

  if (q > 0 || Q > 0) {
    for (let iter = 0; iter < 20; iter++) {
      const resGamma = new Array(Math.max(q, Q > 0 ? Q * m : 0) + 1).fill(0);
      for (let k = 0; k < resGamma.length; k++) {
        let sum = 0;
        let count = 0;
        for (let t = maxStart + k; t < n; t++) {
          sum += residuals[t] * residuals[t - k];
          count++;
        }
        resGamma[k] = count > 0 ? sum / count : 0;
      }

      if (resGamma[0] !== 0) {
        for (let j = 0; j < q; j++) {
          maCoeffs[j] = resGamma[j + 1] / resGamma[0];
        }
        for (let s = 0; s < Q; s++) {
          const lag = (s + 1) * m;
          if (lag < resGamma.length) {
            smaCoeffs[s] = resGamma[lag] / resGamma[0];
          }
        }
      }

      // Recompute residuals
      for (let t = maxStart; t < n; t++) {
        let pred = 0;
        for (let j = 0; j < p; j++) {
          if (t - 1 - j >= 0) pred += arCoeffs[j] * centered[t - 1 - j];
        }
        for (let s = 0; s < P; s++) {
          const lag = (s + 1) * m;
          if (t - lag >= 0) pred += sarCoeffs[s] * centered[t - lag];
        }
        for (let j = 0; j < q; j++) {
          if (t - 1 - j >= 0) pred += maCoeffs[j] * residuals[t - 1 - j];
        }
        for (let s = 0; s < Q; s++) {
          const lag = (s + 1) * m;
          if (t - lag >= 0) pred += smaCoeffs[s] * residuals[t - lag];
        }
        residuals[t] = centered[t] - pred;
      }
    }
  }

  // Residual variance
  let ssRes = 0;
  let resCount = 0;
  for (let t = maxStart; t < n; t++) {
    ssRes += residuals[t] * residuals[t];
    resCount++;
  }
  const sigma2 = resCount > 0 ? ssRes / resCount : 0;

  // AIC
  const nParams = p + q + P + Q + 1;
  const aic =
    resCount > 0 && sigma2 > 0
      ? resCount * Math.log(sigma2) + 2 * nParams
      : Infinity;

  // Forecast function
  const forecast = (steps: number): number[] => {
    const predictions: number[] = [];
    const extCentered = [...centered];
    const extResiduals = [...residuals];

    for (let s = 0; s < steps; s++) {
      const t = n + s;
      let pred = 0;

      for (let j = 0; j < p; j++) {
        const idx = t - 1 - j;
        pred += arCoeffs[j] * (idx < extCentered.length ? extCentered[idx] : 0);
      }
      for (let si = 0; si < P; si++) {
        const lag = (si + 1) * m;
        const idx = t - lag;
        pred += sarCoeffs[si] * (idx >= 0 && idx < extCentered.length ? extCentered[idx] : 0);
      }
      for (let j = 0; j < q; j++) {
        const idx = t - 1 - j;
        pred += maCoeffs[j] * (idx < extResiduals.length ? extResiduals[idx] : 0);
      }
      for (let si = 0; si < Q; si++) {
        const lag = (si + 1) * m;
        const idx = t - lag;
        pred += smaCoeffs[si] * (idx >= 0 && idx < extResiduals.length ? extResiduals[idx] : 0);
      }

      predictions.push(pred + mu);
      extCentered.push(pred);
      extResiduals.push(0);
    }

    // Integrate back: first non-seasonal, then seasonal
    let result = predictions;
    if (d > 0) {
      result = integrateNonSeasonalForecasts(seasonallyDiffed, result, d);
    }
    if (D > 0) {
      result = integrateSeasonalForecasts(series, result, m, D);
    }

    return result;
  };

  return {
    arCoefficients: arCoeffs,
    maCoefficients: maCoeffs,
    seasonalArCoefficients: sarCoeffs,
    seasonalMaCoefficients: smaCoeffs,
    intercept: mu,
    order: { p, d, q },
    seasonalOrder: { P, D, Q, m },
    sigma2,
    aic,
    forecast,
    residuals: residuals.slice(maxStart),
  };
}

/**
 * Integrate non-seasonal differenced forecasts back.
 */
function integrateNonSeasonalForecasts(
  original: number[],
  forecasts: number[],
  d: number,
): number[] {
  let result = [...forecasts];
  for (let iter = 0; iter < d; iter++) {
    const diffedOriginal = difference(original, d - 1 - iter);
    const lastVal = diffedOriginal[diffedOriginal.length - 1];
    const integrated: number[] = [];
    let prev = lastVal;
    for (const val of result) {
      prev = prev + val;
      integrated.push(prev);
    }
    result = integrated;
  }
  return result;
}

// ── Prophet-Style Decomposition ─────────────────────────────────────────

/**
 * Result of a Prophet-style time series decomposition.
 *
 * Contains piecewise linear trend, Fourier-based seasonal component,
 * residuals, detected changepoints, and segment slopes.
 */
export interface ProphetDecomposition {
  /** Piecewise linear trend */
  trend: number[];
  /** Seasonal component (Fourier-based) */
  seasonal: number[];
  /** Residual (observed - trend - seasonal) */
  residual: number[];
  /** Detected changepoints (indices) */
  changepoints: number[];
  /** Trend slopes for each segment */
  slopes: number[];
}

/**
 * Prophet-style time series decomposition.
 *
 * Decomposes a time series into trend + seasonality + residual using:
 * - **Trend**: Piecewise linear trend with automatic changepoint detection
 *   via second-difference magnitudes
 * - **Seasonality**: Fourier series approximation fitted by least squares
 *
 * Inspired by Facebook Prophet's approach but uses a simpler implementation.
 *
 * @param series - Time series data (must not contain NaN or Infinity)
 * @param period - Seasonal period (e.g., 12 for monthly, 7 for daily-weekly; must be >= 2)
 * @param options - Configuration for changepoint detection and Fourier order
 * @returns A {@link ProphetDecomposition} with trend, seasonal, residual,
 *   changepoint indices, and segment slopes
 * @throws {Error} If period < 2 or the series has fewer than 2 full periods
 * @throws {Error} If series contains non-finite values
 *
 * @example
 * ```ts
 * const result = prophetDecompose(monthlySales, 12, { nChangepoints: 5 });
 * console.log(result.changepoints); // detected trend change indices
 * console.log(result.seasonal);     // repeating seasonal pattern
 * ```
 */
export function prophetDecompose(
  series: Dataset,
  period: number,
  options: {
    /** Number of changepoints to detect (default: 10) */
    nChangepoints?: number;
    /** Number of Fourier terms for seasonality (default: 3) */
    fourierOrder?: number;
    /** Fraction of series to place changepoints in (default: 0.8) */
    changepointRange?: number;
  } = {},
): ProphetDecomposition {
  const n = series.length;
  if (n < 2 * period) throw new Error(`Invalid parameter 'series': expected at least 2 full periods (${2 * period}), received ${n}`);
  if (period < 2) throw new Error(`Invalid parameter 'period': expected at least 2, received ${period}`);

  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(series[i])) {
      throw new Error(`Invalid parameter 'series[${i}]': expected a finite number, received ${series[i]}`);
    }
  }

  const nChangepoints = Math.min(options.nChangepoints ?? 10, Math.floor(n / 2));
  const fourierOrder = options.fourierOrder ?? 3;
  const changepointRange = options.changepointRange ?? 0.8;

  // Step 1: Detect changepoints using total variation
  const changepoints = detectChangepoints(series, nChangepoints, changepointRange);

  // Step 2: Fit piecewise linear trend
  const { trend, slopes } = fitPiecewiseLinearTrend(series, changepoints);

  // Step 3: Detrend
  const detrended = series.map((v, i) => v - trend[i]);

  // Step 4: Fit Fourier seasonality to detrended series
  const seasonal = fitFourierSeasonality(detrended, period, fourierOrder);

  // Step 5: Residual
  const residual = series.map((v, i) => v - trend[i] - seasonal[i]);

  return { trend, seasonal, residual, changepoints, slopes };
}

/**
 * Detect changepoints by finding positions with largest slope changes.
 */
function detectChangepoints(
  series: number[],
  nChangepoints: number,
  range: number,
): number[] {
  const n = series.length;
  const maxIdx = Math.floor(n * range);

  if (maxIdx < 3 || nChangepoints < 1) return [];

  // Compute second differences (proxy for slope change)
  const secondDiff: { idx: number; value: number }[] = [];
  for (let i = 1; i < maxIdx - 1; i++) {
    const d2 = Math.abs(series[i + 1] - 2 * series[i] + series[i - 1]);
    secondDiff.push({ idx: i, value: d2 });
  }

  // Sort by magnitude and take top nChangepoints
  secondDiff.sort((a, b) => b.value - a.value);
  const points = secondDiff
    .slice(0, nChangepoints)
    .map((d) => d.idx)
    .sort((a, b) => a - b);

  // Remove points that are too close together (within period/2 of each other)
  const filtered: number[] = [];
  for (const cp of points) {
    if (filtered.length === 0 || cp - filtered[filtered.length - 1] > 2) {
      filtered.push(cp);
    }
  }

  return filtered;
}

/**
 * Fit a piecewise linear trend through the series at the given changepoints.
 */
function fitPiecewiseLinearTrend(
  series: number[],
  changepoints: number[],
): { trend: number[]; slopes: number[] } {
  const n = series.length;
  const cps = [0, ...changepoints, n - 1];
  const slopes: number[] = [];
  const trend = new Array(n);

  for (let s = 0; s < cps.length - 1; s++) {
    const start = cps[s];
    const end = cps[s + 1];
    const startVal = series[start];
    const endVal = series[end];
    const segLen = end - start;
    const slope = segLen > 0 ? (endVal - startVal) / segLen : 0;
    slopes.push(slope);

    for (let i = start; i <= end; i++) {
      trend[i] = startVal + slope * (i - start);
    }
  }

  return { trend, slopes };
}

/**
 * Fit Fourier seasonality to a series.
 * Uses least-squares to fit sin/cos terms at the specified Fourier order.
 */
function fitFourierSeasonality(
  series: number[],
  period: number,
  order: number,
): number[] {
  const n = series.length;
  const nCols = 2 * order;

  // Build Fourier design matrix
  const X: number[][] = new Array(n);
  for (let t = 0; t < n; t++) {
    X[t] = new Array(nCols);
    for (let k = 1; k <= order; k++) {
      const freq = (2 * Math.PI * k * t) / period;
      X[t][2 * (k - 1)] = Math.sin(freq);
      X[t][2 * (k - 1) + 1] = Math.cos(freq);
    }
  }

  // Normal equations: (X^T X) β = X^T y
  const XtX: number[][] = Array.from({ length: nCols }, () =>
    new Array(nCols).fill(0),
  );
  const Xty = new Array(nCols).fill(0);

  for (let t = 0; t < n; t++) {
    for (let i = 0; i < nCols; i++) {
      Xty[i] += X[t][i] * series[t];
      for (let j = 0; j < nCols; j++) {
        XtX[i][j] += X[t][i] * X[t][j];
      }
    }
  }

  // Solve via Gauss-Jordan (small system)
  const beta = solveSmallSystem(XtX, Xty);

  // Compute seasonal component
  const seasonal = new Array(n);
  for (let t = 0; t < n; t++) {
    let val = 0;
    for (let i = 0; i < nCols; i++) val += X[t][i] * beta[i];
    seasonal[t] = val;
  }

  return seasonal;
}

/**
 * Solve a small linear system using Gauss-Jordan elimination.
 */
function solveSmallSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-30) continue;

    for (let j = col; j <= n; j++) aug[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = col; j <= n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }

  return aug.map((row) => row[n]);
}

// ── Local tolerant linear algebra (for ADF where near-singularity is common) ──

function solveSystemTolerant(A: number[][], b: number[]): number[] {
  const n = A.length;
  const aug = A.map((row, i) => [...row.map(v => v), b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-30) continue; // skip truly zero pivots

    for (let j = col; j <= n; j++) aug[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = col; j <= n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }

  return aug.map((row) => row[n]);
}

function invertMatrixTolerant(matrix: number[][]): number[][] {
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
    if (Math.abs(pivot) < 1e-15) continue;

    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }

  return aug.map((row) => row.slice(n));
}
