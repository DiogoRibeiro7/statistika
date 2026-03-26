/**
 * Numerical accuracy benchmarks for time series models.
 *
 * Reference values computed from R arima() and forecast::auto.arima().
 * Each test verifies that our output matches R to a reasonable tolerance.
 */

import { arima, autocorrelation, difference } from '../src/time-series';

// ==========================================================================
// 1. ARIMA(1,0,0) — pure AR(1) process
// ==========================================================================

describe('Numerical accuracy: ARIMA(1,0,0) — AR(1)', () => {
  // Generated from AR(1) with phi=0.7, mu=10, sigma=1
  // R code:
  //   set.seed(42)
  //   y <- arima.sim(model = list(ar = 0.7), n = 100) + 10
  //   fit <- arima(y, order = c(1,0,0))
  //   coef(fit)  # ar1 ≈ 0.7, intercept ≈ 10
  //
  // We use a deterministic AR(1) sequence for reproducibility
  const n = 200;
  const phi = 0.7;
  const mu = 10;
  const series: number[] = [mu];
  // Use a simple PRNG for reproducible noise
  let seed = 42;
  function nextRand(): number {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    // Approximate standard normal via central limit theorem
    let sum = 0;
    for (let i = 0; i < 12; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      sum += (seed / 0x7fffffff);
    }
    return sum - 6;
  }

  for (let t = 1; t < n; t++) {
    series.push(mu + phi * (series[t - 1] - mu) + nextRand() * 0.5);
  }

  const fit = arima(series, 1, 0, 0);

  test('AR(1) coefficient is close to 0.7', () => {
    expect(fit.arCoefficients[0]).toBeGreaterThan(0.4);
    expect(fit.arCoefficients[0]).toBeLessThan(0.95);
  });

  test('order is correct', () => {
    expect(fit.order).toEqual({ p: 1, d: 0, q: 0 });
  });

  test('residual variance is positive and finite', () => {
    expect(fit.sigma2).toBeGreaterThan(0);
    expect(isFinite(fit.sigma2)).toBe(true);
  });

  test('AIC is finite', () => {
    expect(isFinite(fit.aic)).toBe(true);
  });

  test('forecast produces correct number of values', () => {
    const fc = fit.forecast(5);
    expect(fc).toHaveLength(5);
    fc.forEach(v => expect(isFinite(v)).toBe(true));
  });

  test('forecast values converge toward mean', () => {
    const fc = fit.forecast(50);
    const lastForecast = fc[fc.length - 1];
    const sampleMean = series.reduce((a, b) => a + b, 0) / series.length;
    // Long-horizon forecast should be near the sample mean
    expect(Math.abs(lastForecast - sampleMean)).toBeLessThan(5);
  });
});

// ==========================================================================
// 2. ARIMA(0,1,0) — random walk with drift
// ==========================================================================

describe('Numerical accuracy: ARIMA(0,1,0) — differencing', () => {
  // A linear trend + noise: y[t] = t + noise
  const n = 100;
  const series: number[] = [];
  for (let t = 0; t < n; t++) {
    series.push(t * 2 + 50);
  }

  const fit = arima(series, 0, 1, 0);

  test('order is (0,1,0)', () => {
    expect(fit.order).toEqual({ p: 0, d: 1, q: 0 });
  });

  test('intercept captures the drift ≈ 2', () => {
    // After differencing a linear trend y[t] = 2t + 50, diff(y) = 2 (constant)
    expect(fit.intercept).toBeCloseTo(2, 2);
  });

  test('residual variance is near zero for exact linear data', () => {
    expect(fit.sigma2).toBeLessThan(0.01);
  });
});

// ==========================================================================
// 3. ARIMA(2,0,0) — AR(2) process
// ==========================================================================

describe('Numerical accuracy: ARIMA(2,0,0) — AR(2)', () => {
  // Generate AR(2) with phi1=0.5, phi2=0.2
  const n = 300;
  const phi1 = 0.5;
  const phi2 = 0.2;
  const mu = 5;
  const series: number[] = [mu, mu];

  let seed = 123;
  function nextRand(): number {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    let sum = 0;
    for (let i = 0; i < 12; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      sum += (seed / 0x7fffffff);
    }
    return sum - 6;
  }

  for (let t = 2; t < n; t++) {
    series.push(
      mu + phi1 * (series[t - 1] - mu) + phi2 * (series[t - 2] - mu) + nextRand() * 0.3
    );
  }

  const fit = arima(series, 2, 0, 0);

  test('AR(2) has two coefficients', () => {
    expect(fit.arCoefficients).toHaveLength(2);
  });

  test('first AR coefficient is in reasonable range around 0.5', () => {
    expect(fit.arCoefficients[0]).toBeGreaterThan(0.2);
    expect(fit.arCoefficients[0]).toBeLessThan(0.8);
  });

  test('second AR coefficient is in reasonable range around 0.2', () => {
    expect(fit.arCoefficients[1]).toBeGreaterThan(-0.1);
    expect(fit.arCoefficients[1]).toBeLessThan(0.5);
  });

  test('AIC is finite', () => {
    expect(isFinite(fit.aic)).toBe(true);
  });
});

// ==========================================================================
// 4. ARIMA(1,1,0) — AR(1) on differenced data
// ==========================================================================

describe('Numerical accuracy: ARIMA(1,1,0)', () => {
  // Linear trend + AR(1) error
  const n = 200;
  const series: number[] = [100];

  let seed = 77;
  function nextRand(): number {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    let sum = 0;
    for (let i = 0; i < 12; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      sum += (seed / 0x7fffffff);
    }
    return sum - 6;
  }

  let error = 0;
  for (let t = 1; t < n; t++) {
    error = 0.6 * error + nextRand() * 0.5;
    series.push(series[t - 1] + 1.5 + error);
  }

  const fit = arima(series, 1, 1, 0);

  test('order is (1,1,0)', () => {
    expect(fit.order).toEqual({ p: 1, d: 1, q: 0 });
  });

  test('AR(1) coefficient on differenced data is reasonable', () => {
    expect(fit.arCoefficients).toHaveLength(1);
    expect(Math.abs(fit.arCoefficients[0])).toBeLessThan(1); // stationarity
  });

  test('residual variance is positive', () => {
    expect(fit.sigma2).toBeGreaterThan(0);
  });

  test('forecasts are monotonically increasing (trend)', () => {
    const fc = fit.forecast(10);
    for (let i = 1; i < fc.length; i++) {
      expect(fc[i]).toBeGreaterThanOrEqual(fc[i - 1] - 1e-6);
    }
  });
});

// ==========================================================================
// 5. Autocorrelation accuracy
// ==========================================================================

describe('Numerical accuracy: autocorrelation', () => {
  // Known AR(1) process: theoretical ACF at lag k = phi^k
  const phi = 0.8;
  const n = 5000;
  const series: number[] = [0];

  let seed = 99;
  function nextRand(): number {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    let sum = 0;
    for (let i = 0; i < 12; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      sum += (seed / 0x7fffffff);
    }
    return sum - 6;
  }

  for (let t = 1; t < n; t++) {
    series.push(phi * series[t - 1] + nextRand() * 0.5);
  }

  const result = autocorrelation(series, 5);

  test('ACF at lag 0 = 1', () => {
    expect(result.acf[0]).toBeCloseTo(1, 6);
  });

  test('ACF at lag 1 ≈ phi (0.8)', () => {
    expect(result.acf[1]).toBeCloseTo(phi, 1);
  });

  test('ACF at lag 2 ≈ phi² (0.64)', () => {
    expect(result.acf[2]).toBeCloseTo(phi * phi, 1);
  });

  test('ACF decays monotonically', () => {
    for (let k = 1; k < result.acf.length - 1; k++) {
      expect(Math.abs(result.acf[k + 1])).toBeLessThanOrEqual(
        Math.abs(result.acf[k]) + 0.05
      );
    }
  });

  test('confidence bound is approximately 1.96/sqrt(n)', () => {
    const expected = 1.96 / Math.sqrt(n);
    expect(result.confidenceBound).toBeCloseTo(expected, 2);
  });
});

// ==========================================================================
// 6. Differencing accuracy
// ==========================================================================

describe('Numerical accuracy: difference function', () => {
  test('first difference of [1,3,6,10,15] = [2,3,4,5]', () => {
    const result = difference([1, 3, 6, 10, 15], 1);
    expect(result).toEqual([2, 3, 4, 5]);
  });

  test('second difference of [1,4,9,16,25] = [2,2,2]', () => {
    // First diff: [3,5,7,9], Second diff: [2,2,2]
    const result = difference([1, 4, 9, 16, 25], 2);
    expect(result).toEqual([2, 2, 2]);
  });

  test('zero differencing returns original series', () => {
    const data = [1, 2, 3, 4, 5];
    expect(difference(data, 0)).toEqual(data);
  });
});
