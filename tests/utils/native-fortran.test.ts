/**
 * Tests for Fortran-accelerated native modules.
 *
 * These tests verify the TypeScript fallback implementations (which are
 * always available) and will also exercise the Fortran backends when
 * the native addon is built.
 */

import {
  acf,
  pacfDurbinLevinson,
  exponentialSmoothing,
  holtWinters,
  difference,
} from "../../src/utils/native-timeseries";

import {
  kalmanFilterUnivariate,
} from "../../src/utils/native-kalman";

import {
  chi2Cdf,
  chi2Pdf,
  tCdf,
  tPdf,
  fCdf,
  fPdf,
  normalPdf,
  gammaCdf,
  betaCdf,
  chi2CdfBatch,
  tCdfBatch,
  normalCdfBatch,
} from "../../src/utils/native-distributions";

import {
  garch11Loglik,
  garchPqLoglik,
  gjrGarch11Loglik,
  egarch11Loglik,
  garch11Forecast,
} from "../../src/utils/native-garch";

// ── Time Series Tests ───────────────────────────────────────────────────

describe("native-timeseries", () => {
  describe("acf", () => {
    it("should compute ACF for a simple series", () => {
      const series = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2];
      const result = acf(series, 5);

      expect(result).toHaveLength(6); // lags 0..5
      expect(result[0]).toBeCloseTo(1.0, 10); // lag 0 is always 1
      expect(Math.abs(result[1])).toBeLessThan(1); // all ACF values in [-1, 1]
    });

    it("should return all zeros for constant series", () => {
      const series = [5, 5, 5, 5, 5, 5, 5, 5];
      const result = acf(series, 3);

      expect(result).toHaveLength(4);
      result.forEach((v) => expect(v).toBeCloseTo(0, 10));
    });

    it("should handle sinusoidal series", () => {
      const n = 100;
      const series = Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * i / 10));
      const result = acf(series, 20);

      // For sinusoidal signal, ACF should oscillate
      expect(result[0]).toBeCloseTo(1, 10);
      expect(result[10]).toBeGreaterThan(0.8); // Period 10 -> ACF at lag 10 should be high
    });
  });

  describe("pacfDurbinLevinson", () => {
    it("should compute PACF from ACF values", () => {
      const series = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2];
      const acfValues = acf(series, 5);
      const result = pacfDurbinLevinson(acfValues, 5);

      expect(result).toHaveLength(6);
      expect(result[0]).toBeCloseTo(1, 10); // PACF at lag 0 is always 1
      expect(result[1]).toBeCloseTo(acfValues[1], 10); // PACF at lag 1 = ACF at lag 1
    });
  });

  describe("exponentialSmoothing", () => {
    it("should smooth with alpha=1 giving original series", () => {
      const series = [1, 3, 2, 5, 4];
      const result = exponentialSmoothing(series, 1.0);

      expect(result).toHaveLength(5);
      for (let i = 0; i < 5; i++) {
        expect(result[i]).toBeCloseTo(series[i], 10);
      }
    });

    it("should smooth with small alpha", () => {
      const series = [1, 10, 1, 10, 1, 10];
      const result = exponentialSmoothing(series, 0.1);

      // With small alpha, smoothed values should change slowly
      expect(result[0]).toBeCloseTo(1, 10);
      expect(result[1]).toBeCloseTo(1.9, 5);
    });
  });

  describe("holtWinters", () => {
    it("should track a linear trend", () => {
      const series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const result = holtWinters(series, 0.5, 0.5);

      expect(result.level).toHaveLength(10);
      expect(result.trend).toHaveLength(10);
      expect(result.fitted).toHaveLength(10);
      // The level should roughly follow the series
      expect(result.level[9]).toBeGreaterThan(8);
    });
  });

  describe("difference", () => {
    it("should compute first differences", () => {
      const series = [1, 3, 6, 10, 15];
      const result = difference(series, 1);

      expect(result).toHaveLength(4);
      expect(result[0]).toBeCloseTo(2, 10);
      expect(result[1]).toBeCloseTo(3, 10);
      expect(result[2]).toBeCloseTo(4, 10);
      expect(result[3]).toBeCloseTo(5, 10);
    });

    it("should compute second differences", () => {
      const series = [1, 3, 6, 10, 15];
      const result = difference(series, 2);

      expect(result).toHaveLength(3);
      expect(result[0]).toBeCloseTo(1, 10); // 3-2 = 1
      expect(result[1]).toBeCloseTo(1, 10); // 4-3 = 1
      expect(result[2]).toBeCloseTo(1, 10); // 5-4 = 1
    });
  });
});

// ── Kalman Filter Tests ─────────────────────────────────────────────────

describe("native-kalman", () => {
  describe("kalmanFilterUnivariate", () => {
    it("should filter a local level model", () => {
      // Local level model: x_t = x_{t-1} + w_t, y_t = x_t + v_t
      const F = [[1]]; // m=1
      const H = [1];
      const Q = [[0.1]];
      const R = 1.0;
      const x0 = [0];
      const P0 = [[1]];
      const y = [1.2, 0.8, 1.5, 2.1, 1.8, 2.5, 3.0, 2.7];

      const result = kalmanFilterUnivariate(F, H, Q, R, y, x0, P0);

      expect(result.states).toHaveLength(8);
      expect(result.logLikelihood).toBeLessThan(0); // Log-likelihood is negative
      expect(isFinite(result.logLikelihood)).toBe(true);

      // Filtered states should be smoothed version of observations
      for (let t = 0; t < 8; t++) {
        expect(result.states[t]).toHaveLength(1);
        expect(isFinite(result.states[t][0])).toBe(true);
      }
    });

    it("should filter a local linear trend model", () => {
      // x = [level, trend], F = [[1,1],[0,1]], H = [1, 0]
      const F = [[1, 1], [0, 1]];
      const H = [1, 0];
      const Q = [[0.1, 0], [0, 0.01]];
      const R = 1.0;
      const x0 = [0, 0];
      const P0 = [[1, 0], [0, 1]];
      const y = [1, 2, 3, 5, 6, 8, 9, 11];

      const result = kalmanFilterUnivariate(F, H, Q, R, y, x0, P0);

      expect(result.states).toHaveLength(8);
      expect(isFinite(result.logLikelihood)).toBe(true);
      // States should have 2 elements each (level, trend)
      expect(result.states[7]).toHaveLength(2);
    });
  });
});

// ── Distribution Tests ──────────────────────────────────────────────────

describe("native-distributions", () => {
  describe("chi2Cdf", () => {
    it("should compute chi-squared CDF", () => {
      // chi2(df=1) at x=3.841 should be ~0.95
      expect(chi2Cdf(3.841, 1)).toBeCloseTo(0.95, 2);
      expect(chi2Cdf(0, 5)).toBeCloseTo(0, 10);
      expect(chi2Cdf(100, 5)).toBeCloseTo(1, 5);
    });
  });

  describe("chi2Pdf", () => {
    it("should compute chi-squared PDF", () => {
      const result = chi2Pdf(2, 2);
      // chi2(df=2) PDF at x=2 = 0.5 * exp(-1) ≈ 0.1839
      expect(result).toBeCloseTo(0.5 * Math.exp(-1), 3);
      expect(chi2Pdf(-1, 5)).toBeCloseTo(0, 10);
    });
  });

  describe("tCdf", () => {
    it("should compute Student's t CDF", () => {
      // t(df=inf) is normal, so tCdf(0, 1000) ≈ 0.5
      expect(tCdf(0, 1000)).toBeCloseTo(0.5, 3);
      // t(df=1) at x=0 should be 0.5 (symmetric)
      expect(tCdf(0, 1)).toBeCloseTo(0.5, 5);
      // Large positive value should be close to 1
      expect(tCdf(10, 5)).toBeGreaterThan(0.999);
    });
  });

  describe("tPdf", () => {
    it("should compute Student's t PDF at 0", () => {
      // t(df=1) at x=0 is 1/pi ≈ 0.3183 (Cauchy distribution)
      expect(tPdf(0, 1)).toBeCloseTo(1 / Math.PI, 3);
    });
  });

  describe("fCdf", () => {
    it("should compute F-distribution CDF", () => {
      expect(fCdf(0, 5, 10)).toBeCloseTo(0, 10);
      expect(fCdf(100, 5, 10)).toBeCloseTo(1, 3);
      // F(d1=1, d2=1) at x=1 should be ~0.5
      expect(fCdf(1, 1, 1)).toBeCloseTo(0.5, 2);
    });
  });

  describe("fPdf", () => {
    it("should return 0 for negative x", () => {
      expect(fPdf(-1, 5, 10)).toBeCloseTo(0, 10);
    });

    it("should return positive value for positive x", () => {
      expect(fPdf(1, 5, 10)).toBeGreaterThan(0);
    });
  });

  describe("normalPdf", () => {
    it("should compute standard normal PDF", () => {
      expect(normalPdf(0)).toBeCloseTo(1 / Math.sqrt(2 * Math.PI), 10);
      expect(normalPdf(1)).toBeCloseTo(Math.exp(-0.5) / Math.sqrt(2 * Math.PI), 10);
    });
  });

  describe("gammaCdf", () => {
    it("should compute gamma CDF", () => {
      // Gamma(shape=1, scale=1) is exponential, CDF at x=1 = 1-e^{-1} ≈ 0.6321
      expect(gammaCdf(1, 1, 1)).toBeCloseTo(1 - Math.exp(-1), 3);
      expect(gammaCdf(0, 2, 1)).toBeCloseTo(0, 10);
    });
  });

  describe("betaCdf", () => {
    it("should compute beta CDF", () => {
      // Beta(1,1) = uniform, CDF at x=0.5 = 0.5
      expect(betaCdf(0.5, 1, 1)).toBeCloseTo(0.5, 5);
      expect(betaCdf(0, 2, 3)).toBeCloseTo(0, 10);
      expect(betaCdf(1, 2, 3)).toBeCloseTo(1, 10);
    });
  });

  describe("batch operations", () => {
    it("should compute chi2CdfBatch", () => {
      const result = chi2CdfBatch([1, 2, 3, 4, 5], 3);
      expect(result).toHaveLength(5);
      for (let i = 0; i < 4; i++) {
        expect(result[i]).toBeLessThan(result[i + 1]); // monotonically increasing
      }
    });

    it("should compute tCdfBatch", () => {
      const result = tCdfBatch([-2, -1, 0, 1, 2], 10);
      expect(result).toHaveLength(5);
      expect(result[2]).toBeCloseTo(0.5, 3);
      for (let i = 0; i < 4; i++) {
        expect(result[i]).toBeLessThan(result[i + 1]);
      }
    });

    it("should compute normalCdfBatch", () => {
      const result = normalCdfBatch([-3, -1, 0, 1, 3]);
      expect(result).toHaveLength(5);
      expect(result[2]).toBeCloseTo(0.5, 5);
      expect(result[0]).toBeLessThan(0.01);
      expect(result[4]).toBeGreaterThan(0.99);
    });
  });
});

// ── GARCH Tests ─────────────────────────────────────────────────────────

describe("native-garch", () => {
  // Generate synthetic GARCH(1,1) data
  const T = 200;
  const trueOmega = 0.01;
  const trueAlpha = 0.1;
  const trueBeta = 0.85;

  const eps: number[] = [];
  const sigma2: number[] = [];
  let s2 = trueOmega / (1 - trueAlpha - trueBeta);
  sigma2.push(s2);
  eps.push(Math.sqrt(s2) * 0.5);

  for (let t = 1; t < T; t++) {
    s2 = trueOmega + trueAlpha * eps[t - 1] * eps[t - 1] + trueBeta * sigma2[t - 1];
    sigma2.push(s2);
    // Deterministic "noise" for reproducibility
    eps.push(Math.sqrt(s2) * Math.sin(t * 0.1));
  }

  describe("garch11Loglik", () => {
    it("should compute GARCH(1,1) log-likelihood", () => {
      const result = garch11Loglik(eps, trueOmega, trueAlpha, trueBeta);

      expect(result.sigma2).toHaveLength(T);
      expect(isFinite(result.logLikelihood)).toBe(true);
      expect(result.logLikelihood).toBeLessThan(0);

      // Conditional variances should be positive
      result.sigma2.forEach((v) => expect(v).toBeGreaterThan(0));
    });

    it("should give higher likelihood near true parameters", () => {
      const trueLL = garch11Loglik(eps, trueOmega, trueAlpha, trueBeta).logLikelihood;
      const badLL = garch11Loglik(eps, 0.5, 0.4, 0.4).logLikelihood;

      expect(trueLL).toBeGreaterThan(badLL);
    });
  });

  describe("garchPqLoglik", () => {
    it("should match GARCH(1,1) when p=1, q=1", () => {
      const result11 = garch11Loglik(eps, trueOmega, trueAlpha, trueBeta);
      const resultPq = garchPqLoglik(eps, trueOmega, [trueAlpha], [trueBeta]);

      expect(resultPq.logLikelihood).toBeCloseTo(result11.logLikelihood, 3);
    });
  });

  describe("gjrGarch11Loglik", () => {
    it("should compute GJR-GARCH(1,1) log-likelihood", () => {
      const result = gjrGarch11Loglik(eps, trueOmega, trueAlpha, trueBeta, 0.05);

      expect(result.sigma2).toHaveLength(T);
      expect(isFinite(result.logLikelihood)).toBe(true);
      result.sigma2.forEach((v) => expect(v).toBeGreaterThan(0));
    });

    it("should reduce to GARCH(1,1) when gamma=0", () => {
      const result = gjrGarch11Loglik(eps, trueOmega, trueAlpha, trueBeta, 0);
      const garchResult = garch11Loglik(eps, trueOmega, trueAlpha, trueBeta);

      expect(result.logLikelihood).toBeCloseTo(garchResult.logLikelihood, 3);
    });
  });

  describe("egarch11Loglik", () => {
    it("should compute EGARCH(1,1) log-likelihood", () => {
      const result = egarch11Loglik(eps, -0.1, 0.1, 0.95, -0.05);

      expect(result.sigma2).toHaveLength(T);
      expect(isFinite(result.logLikelihood)).toBe(true);
      result.sigma2.forEach((v) => expect(v).toBeGreaterThan(0));
    });
  });

  describe("garch11Forecast", () => {
    it("should forecast conditional variances", () => {
      const lastEps2 = eps[T - 1] * eps[T - 1];
      const lastSigma2 = sigma2[T - 1];
      const h = 10;

      const forecast = garch11Forecast(lastEps2, lastSigma2, trueOmega, trueAlpha, trueBeta, h);

      expect(forecast).toHaveLength(h);
      forecast.forEach((v) => expect(v).toBeGreaterThan(0));

      // Forecasts should move toward unconditional variance
      const uncondVar = trueOmega / (1 - trueAlpha - trueBeta);
      // With alpha+beta=0.95, convergence is slow; use h=50 for closer match
      const longForecast = garch11Forecast(lastEps2, lastSigma2, trueOmega, trueAlpha, trueBeta, 50);
      // At h=50 the forecast should be closer to uncondVar than at h=1
      expect(Math.abs(longForecast[49] - uncondVar)).toBeLessThan(Math.abs(longForecast[0] - uncondVar) + 0.01);
    });
  });
});
