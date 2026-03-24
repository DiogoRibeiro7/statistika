/**
 * Time series analysis: autocorrelation, moving averages, differencing, and ARIMA.
 */

import { mean } from "./utils/descriptive";
import {
  acf as nativeAcf,
  pacfDurbinLevinson as nativePacf,
  hasNativeTimeSeries,
} from "./utils/native-timeseries";

// ---- Autocorrelation ----

/**
 * Result of an autocorrelation computation (ACF and PACF).
 */
export interface AutocorrelationResult {
  /** ACF values for lags 0..maxLag */
  acf: number[];
  /** Partial ACF values for lags 0..maxLag */
  pacf: number[];
  /** Maximum lag computed */
  maxLag: number;
  /** Approximate 95% confidence bound (1.96 / sqrt(n)) */
  confidenceBound: number;
}

/**
 * Compute the autocorrelation function (ACF) and partial autocorrelation
 * function (PACF) for a time series.
 *
 * ACF at lag k is: r(k) = gamma(k) / gamma(0) where gamma(k) is the
 * autocovariance at lag k, computed with the biased estimator (dividing by n).
 * PACF is computed via the Durbin-Levinson recursion.
 *
 * @param series - The time series data
 * @param maxLag - Maximum lag to compute (default: min(n-1, floor(10*log10(n))))
 * @returns AutocorrelationResult with ACF, PACF arrays, max lag, and 95% confidence bound
 * @throws Error if series has fewer than 2 observations
 * @throws Error if maxLag < 1
 *
 * @example
 * ```ts
 * const series = [1, 2, 3, 2, 1, 2, 3, 2, 1];
 * const result = autocorrelation(series);
 * // result.acf — autocorrelation at each lag
 * // result.pacf — partial autocorrelation at each lag
 * // result.confidenceBound — 1.96 / sqrt(n), for significance testing
 * ```
 */
export function autocorrelation(
  series: number[],
  maxLag?: number,
): AutocorrelationResult {
  const n = series.length;
  if (n < 2) throw new Error("Series must have at least 2 observations");

  const defaultMaxLag = Math.min(n - 1, Math.floor(10 * Math.log10(n)));
  const lag = maxLag ?? defaultMaxLag;
  if (lag < 1) throw new Error("maxLag must be at least 1");
  const effectiveLag = Math.min(lag, n - 1);

  // Use Fortran-accelerated ACF/PACF when available
  if (hasNativeTimeSeries) {
    const acf = nativeAcf(series, effectiveLag);
    const pacf = nativePacf(acf, effectiveLag);
    return {
      acf,
      pacf,
      maxLag: effectiveLag,
      confidenceBound: 1.96 / Math.sqrt(n),
    };
  }

  const mu = mean(series);

  // Autocovariance at each lag (biased estimator)
  const gamma = new Array(effectiveLag + 1);
  for (let k = 0; k <= effectiveLag; k++) {
    let sum = 0;
    for (let t = 0; t < n - k; t++) {
      sum += (series[t] - mu) * (series[t + k] - mu);
    }
    gamma[k] = sum / n;
  }

  // ACF = gamma[k] / gamma[0]
  const acf = new Array(effectiveLag + 1);
  for (let k = 0; k <= effectiveLag; k++) {
    acf[k] = gamma[0] === 0 ? 0 : gamma[k] / gamma[0];
  }

  // PACF via Durbin-Levinson recursion
  const pacf = new Array(effectiveLag + 1).fill(0);
  pacf[0] = 1;

  if (effectiveLag >= 1) {
    // phi[k][j] = partial autocorrelation coefficients
    const phi: number[][] = [];
    for (let k = 0; k <= effectiveLag; k++) {
      phi[k] = new Array(effectiveLag + 1).fill(0);
    }

    phi[1][1] = acf[1];
    pacf[1] = acf[1];

    for (let k = 2; k <= effectiveLag; k++) {
      let num = acf[k];
      for (let j = 1; j < k; j++) {
        num -= phi[k - 1][j] * acf[k - j];
      }
      let den = 1;
      for (let j = 1; j < k; j++) {
        den -= phi[k - 1][j] * acf[j];
      }

      phi[k][k] = den === 0 ? 0 : num / den;
      pacf[k] = phi[k][k];

      for (let j = 1; j < k; j++) {
        phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
      }
    }
  }

  return {
    acf,
    pacf,
    maxLag: effectiveLag,
    confidenceBound: 1.96 / Math.sqrt(n),
  };
}

// ---- Moving Averages ----

/**
 * Result of a moving average computation.
 */
export interface MovingAverageResult {
  /** Smoothed values (shorter than original by window-1, or same length if centered) */
  values: number[];
  /** Window size used */
  window: number;
}

/**
 * Simple moving average (SMA).
 *
 * Computes the unweighted mean of the previous `window` values at each position.
 *
 * @param series - The time series data
 * @param window - Window size (must be >= 1 and <= series.length)
 * @returns MovingAverageResult with smoothed values (length = series.length - window + 1)
 * @throws Error if window is not a positive integer
 * @throws Error if window exceeds series length
 */
export function simpleMovingAverage(
  series: number[],
  window: number,
): MovingAverageResult {
  if (!Number.isInteger(window) || window < 1) {
    throw new Error("Window must be a positive integer");
  }
  if (window > series.length) {
    throw new Error("Window must not exceed series length");
  }

  const result: number[] = [];
  let sum = 0;

  // Initialize first window
  for (let i = 0; i < window; i++) {
    sum += series[i];
  }
  result.push(sum / window);

  // Slide window
  for (let i = window; i < series.length; i++) {
    sum += series[i] - series[i - window];
    result.push(sum / window);
  }

  return { values: result, window };
}

/**
 * Exponential moving average (EMA).
 *
 * EMA(t) = alpha * x_t + (1 - alpha) * EMA(t-1). The equivalent window
 * size is approximately 2/alpha - 1.
 *
 * @param series - The time series data
 * @param alpha - Smoothing factor in (0, 1]. Higher = more weight on recent values.
 * @returns MovingAverageResult with smoothed values (same length as input)
 * @throws Error if alpha is not in (0, 1]
 * @throws Error if series is empty
 */
export function exponentialMovingAverage(
  series: number[],
  alpha: number,
): MovingAverageResult {
  if (alpha <= 0 || alpha > 1) {
    throw new Error("Alpha must be in (0, 1]");
  }
  if (series.length === 0) {
    throw new Error("Series must not be empty");
  }

  const result: number[] = new Array(series.length);
  result[0] = series[0];

  for (let i = 1; i < series.length; i++) {
    result[i] = alpha * series[i] + (1 - alpha) * result[i - 1];
  }

  return { values: result, window: Math.round(2 / alpha - 1) };
}

/**
 * Weighted moving average (WMA) with linearly increasing weights.
 *
 * The most recent observation gets the highest weight. Weight for position
 * j in the window is (j + 1), normalized by the sum w(w+1)/2.
 *
 * @param series - The time series data
 * @param window - Window size
 * @returns MovingAverageResult with smoothed values (length = series.length - window + 1)
 * @throws Error if window is not a positive integer
 * @throws Error if window exceeds series length
 */
export function weightedMovingAverage(
  series: number[],
  window: number,
): MovingAverageResult {
  if (!Number.isInteger(window) || window < 1) {
    throw new Error("Window must be a positive integer");
  }
  if (window > series.length) {
    throw new Error("Window must not exceed series length");
  }

  const weights: number[] = [];
  let weightSum = 0;
  for (let i = 1; i <= window; i++) {
    weights.push(i);
    weightSum += i;
  }

  const result: number[] = [];
  for (let i = window - 1; i < series.length; i++) {
    let sum = 0;
    for (let j = 0; j < window; j++) {
      sum += series[i - window + 1 + j] * weights[j];
    }
    result.push(sum / weightSum);
  }

  return { values: result, window };
}

// ---- Differencing ----

/**
 * Difference the series d times (for stationarity).
 *
 * Each differencing reduces the series length by 1. Applied d times,
 * the output has length n - d.
 *
 * @param series - The time series data
 * @param d - Number of differences (default 1)
 * @returns Differenced series
 * @throws Error if d is not a non-negative integer
 * @throws Error if series becomes too short for further differencing
 */
export function difference(series: number[], d = 1): number[] {
  if (d < 0 || !Number.isInteger(d)) {
    throw new Error("d must be a non-negative integer");
  }

  let result = series;
  for (let iter = 0; iter < d; iter++) {
    if (result.length < 2) {
      throw new Error("Series too short for further differencing");
    }
    const diff: number[] = new Array(result.length - 1);
    for (let i = 1; i < result.length; i++) {
      diff[i - 1] = result[i] - result[i - 1];
    }
    result = diff;
  }
  return result;
}

// ---- ARIMA ----

/**
 * Result of an ARIMA model fit.
 */
export interface ARIMAResult {
  /** AR coefficients (length p) */
  arCoefficients: number[];
  /** MA coefficients (length q) */
  maCoefficients: number[];
  /** Intercept / constant term */
  intercept: number;
  /** Order (p, d, q) */
  order: { p: number; d: number; q: number };
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
 * Fit an ARIMA(p, d, q) model to a time series.
 *
 * The model is: (1 - phi_1 B - ... - phi_p B^p)(1-B)^d X_t =
 * (1 + theta_1 B + ... + theta_q B^q) epsilon_t.
 *
 * Uses conditional least squares for parameter estimation:
 * - AR coefficients are estimated via Yule-Walker equations (Levinson-Durbin recursion).
 * - MA coefficients are estimated via iterative residual fitting.
 * - AIC = n_eff * ln(sigma^2) + 2(p + q + 1).
 *
 * @param series - The time series data
 * @param p - AR order (autoregressive)
 * @param d - Differencing order
 * @param q - MA order (moving average)
 * @returns ARIMAResult with coefficients, residuals, AIC, and a forecast function
 * @throws Error if series is too short for the specified order
 * @throws Error if p, d, or q are negative or non-integer
 *
 * @example
 * ```ts
 * const series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
 * const model = arima(series, 1, 1, 0); // AR(1) with first differencing
 * const forecast = model.forecast(3); // predict next 3 values
 * ```
 */
export function arima(
  series: number[],
  p: number,
  d: number,
  q: number,
): ARIMAResult {
  if (series.length < p + d + q + 2) {
    throw new Error("Series too short for the specified ARIMA order");
  }
  if (p < 0 || d < 0 || q < 0) {
    throw new Error("p, d, q must be non-negative");
  }
  if (!Number.isInteger(p) || !Number.isInteger(d) || !Number.isInteger(q)) {
    throw new Error("p, d, q must be integers");
  }

  // Step 1: Difference
  const diffed = difference(series, d);
  const n = diffed.length;
  const mu = mean(diffed);

  // Center the series
  const centered = diffed.map((v) => v - mu);

  // Step 2: Estimate AR coefficients via Yule-Walker
  const arCoeffs = new Array(p).fill(0);

  if (p > 0) {
    // Compute autocovariances
    const gamma = new Array(p + 1);
    for (let k = 0; k <= p; k++) {
      let sum = 0;
      for (let t = 0; t < n - k; t++) {
        sum += centered[t] * centered[t + k];
      }
      gamma[k] = sum / n;
    }

    // Solve Yule-Walker: R * phi = r
    // R is the Toeplitz matrix of gamma[0..p-1]
    // r is gamma[1..p]
    if (p === 1) {
      arCoeffs[0] = gamma[0] === 0 ? 0 : gamma[1] / gamma[0];
    } else {
      // Levinson-Durbin recursion
      const a: number[][] = [];
      for (let i = 0; i <= p; i++) a[i] = new Array(p + 1).fill(0);

      a[1][1] = gamma[0] === 0 ? 0 : gamma[1] / gamma[0];
      let e = gamma[0] * (1 - a[1][1] * a[1][1]);

      for (let k = 2; k <= p; k++) {
        let lambda = gamma[k];
        for (let j = 1; j < k; j++) {
          lambda -= a[k - 1][j] * gamma[k - j];
        }
        a[k][k] = e === 0 ? 0 : lambda / e;
        for (let j = 1; j < k; j++) {
          a[k][j] = a[k - 1][j] - a[k][k] * a[k - 1][k - j];
        }
        e *= 1 - a[k][k] * a[k][k];
      }

      for (let j = 1; j <= p; j++) {
        arCoeffs[j - 1] = a[p][j];
      }
    }
  }

  // Step 3: Compute AR residuals and estimate MA coefficients
  const residuals = new Array(n).fill(0);
  const maxStart = Math.max(p, q);

  // Initialize residuals from AR model
  for (let t = maxStart; t < n; t++) {
    let predicted = 0;
    for (let j = 0; j < p; j++) {
      predicted += arCoeffs[j] * centered[t - 1 - j];
    }
    residuals[t] = centered[t] - predicted;
  }

  let maCoeffs = new Array(q).fill(0);

  if (q > 0) {
    // Iterative estimation of MA coefficients
    for (let iter = 0; iter < 20; iter++) {
      // Estimate MA coefficients from residual autocorrelations
      const resGamma = new Array(q + 1);
      for (let k = 0; k <= q; k++) {
        let sum = 0;
        for (let t = maxStart + k; t < n; t++) {
          sum += residuals[t] * residuals[t - k];
        }
        resGamma[k] = sum / (n - maxStart);
      }

      // Simple innovation estimation for MA coefficients
      const newMaCoeffs = new Array(q).fill(0);
      for (let j = 0; j < q; j++) {
        newMaCoeffs[j] =
          resGamma[0] === 0 ? 0 : resGamma[j + 1] / resGamma[0];
      }
      maCoeffs = newMaCoeffs;

      // Re-compute residuals with AR + MA
      for (let t = maxStart; t < n; t++) {
        let predicted = 0;
        for (let j = 0; j < p; j++) {
          predicted += arCoeffs[j] * centered[t - 1 - j];
        }
        for (let j = 0; j < q; j++) {
          if (t - 1 - j >= 0) {
            predicted += maCoeffs[j] * residuals[t - 1 - j];
          }
        }
        residuals[t] = centered[t] - predicted;
      }
    }
  }

  // Compute residual variance
  let ssRes = 0;
  let resCount = 0;
  for (let t = maxStart; t < n; t++) {
    ssRes += residuals[t] * residuals[t];
    resCount++;
  }
  const sigma2 = resCount > 0 ? ssRes / resCount : 0;

  // AIC = n * ln(sigma2) + 2 * (p + q + 1)
  const nEff = n - maxStart;
  const aic =
    nEff > 0 && sigma2 > 0
      ? nEff * Math.log(sigma2) + 2 * (p + q + 1)
      : Infinity;

  // Forecast function
  const forecast = (steps: number): number[] => {
    const predictions: number[] = [];
    // Extend the centered series and residuals
    const extCentered = [...centered];
    const extResiduals = [...residuals];

    for (let s = 0; s < steps; s++) {
      let pred = 0;
      const t = n + s;

      for (let j = 0; j < p; j++) {
        const idx = t - 1 - j;
        pred += arCoeffs[j] * (idx < extCentered.length ? extCentered[idx] : 0);
      }
      for (let j = 0; j < q; j++) {
        const idx = t - 1 - j;
        pred +=
          maCoeffs[j] * (idx < extResiduals.length ? extResiduals[idx] : 0);
      }

      predictions.push(pred + mu);
      extCentered.push(pred);
      extResiduals.push(0); // Future residuals are 0
    }

    // If differenced, integrate back
    if (d > 0) {
      return integrateForecasts(series, predictions, d);
    }

    return predictions;
  };

  return {
    arCoefficients: arCoeffs,
    maCoefficients: maCoeffs,
    intercept: mu,
    order: { p, d, q },
    sigma2,
    aic,
    forecast,
    residuals: residuals.slice(maxStart),
  };
}

/**
 * Integrate differenced forecasts back to original scale.
 */
function integrateForecasts(
  original: number[],
  forecasts: number[],
  d: number,
): number[] {
  // Get the last d values needed to integrate
  let result = [...forecasts];

  for (let iter = 0; iter < d; iter++) {
    // For each level of integration, we need the last value from the
    // (d - 1 - iter)-differenced original series
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
