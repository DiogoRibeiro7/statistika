import {
  autocorrelation,
  simpleMovingAverage,
  exponentialMovingAverage,
  weightedMovingAverage,
  difference,
  arima,
} from "../src/time-series";

describe("Time Series Analysis", () => {
  // ---- Autocorrelation ----
  describe("autocorrelation", () => {
    it("ACF at lag 0 is always 1", () => {
      const series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const result = autocorrelation(series);
      expect(result.acf[0]).toBe(1);
    });

    it("PACF at lag 0 is 1", () => {
      const series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const result = autocorrelation(series);
      expect(result.pacf[0]).toBe(1);
    });

    it("ACF decays for random noise", () => {
      // Seeded pseudo-random for reproducibility
      const series: number[] = [];
      let seed = 42;
      for (let i = 0; i < 100; i++) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        series.push(seed / 0x7fffffff);
      }
      const result = autocorrelation(series, 10);
      // ACF at lag > 0 should be small for white noise
      for (let k = 1; k <= 10; k++) {
        expect(Math.abs(result.acf[k])).toBeLessThan(0.3);
      }
    });

    it("ACF detects strong positive autocorrelation", () => {
      // AR(1) like series: x_t = 0.9 * x_{t-1} + noise
      const series: number[] = [0];
      let seed = 1;
      for (let i = 1; i < 200; i++) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        const noise = (seed / 0x7fffffff - 0.5) * 0.2;
        series.push(0.9 * series[i - 1] + noise);
      }
      const result = autocorrelation(series, 5);
      expect(result.acf[1]).toBeGreaterThan(0.5);
    });

    it("respects maxLag parameter", () => {
      const series = Array.from({ length: 50 }, (_, i) => Math.sin(i * 0.5));
      const result = autocorrelation(series, 5);
      expect(result.maxLag).toBe(5);
      expect(result.acf.length).toBe(6); // 0..5
    });

    it("provides confidence bound", () => {
      const series = Array.from({ length: 100 }, (_, i) => i);
      const result = autocorrelation(series);
      expect(result.confidenceBound).toBeCloseTo(1.96 / 10, 4);
    });

    it("throws on too-short series", () => {
      expect(() => autocorrelation([1])).toThrow();
    });
  });

  // ---- Simple Moving Average ----
  describe("simpleMovingAverage", () => {
    it("computes correct SMA", () => {
      const series = [1, 2, 3, 4, 5];
      const result = simpleMovingAverage(series, 3);
      expect(result.values).toEqual([2, 3, 4]);
    });

    it("window of 1 returns the original series", () => {
      const series = [10, 20, 30];
      const result = simpleMovingAverage(series, 1);
      expect(result.values).toEqual([10, 20, 30]);
    });

    it("window of n returns the mean", () => {
      const series = [2, 4, 6, 8, 10];
      const result = simpleMovingAverage(series, 5);
      expect(result.values.length).toBe(1);
      expect(result.values[0]).toBeCloseTo(6, 8);
    });

    it("throws on invalid window", () => {
      expect(() => simpleMovingAverage([1, 2, 3], 0)).toThrow();
      expect(() => simpleMovingAverage([1, 2, 3], 4)).toThrow();
      expect(() => simpleMovingAverage([1, 2, 3], 1.5)).toThrow();
    });
  });

  // ---- Exponential Moving Average ----
  describe("exponentialMovingAverage", () => {
    it("first value equals the first data point", () => {
      const result = exponentialMovingAverage([10, 20, 30], 0.5);
      expect(result.values[0]).toBe(10);
    });

    it("alpha=1 returns the original series", () => {
      const series = [1, 5, 3, 7];
      const result = exponentialMovingAverage(series, 1);
      expect(result.values).toEqual(series);
    });

    it("smooths the series", () => {
      const series = [10, 20, 10, 20, 10, 20];
      const result = exponentialMovingAverage(series, 0.3);
      // EMA should be smoother (less variation) than original
      const emaRange =
        Math.max(...result.values) - Math.min(...result.values);
      expect(emaRange).toBeLessThan(10);
    });

    it("throws on invalid alpha", () => {
      expect(() => exponentialMovingAverage([1, 2], 0)).toThrow();
      expect(() => exponentialMovingAverage([1, 2], 1.5)).toThrow();
    });

    it("throws on empty series", () => {
      expect(() => exponentialMovingAverage([], 0.5)).toThrow();
    });
  });

  // ---- Weighted Moving Average ----
  describe("weightedMovingAverage", () => {
    it("weights recent values more", () => {
      const series = [10, 10, 10, 100];
      const sma = simpleMovingAverage(series, 3);
      const wma = weightedMovingAverage(series, 3);
      // WMA gives more weight to 100, so last value should be higher
      expect(wma.values[wma.values.length - 1]).toBeGreaterThan(
        sma.values[sma.values.length - 1],
      );
    });

    it("window of 1 returns the original series", () => {
      const series = [5, 10, 15];
      const result = weightedMovingAverage(series, 1);
      expect(result.values).toEqual([5, 10, 15]);
    });

    it("throws on invalid window", () => {
      expect(() => weightedMovingAverage([1, 2], 0)).toThrow();
      expect(() => weightedMovingAverage([1, 2], 3)).toThrow();
    });
  });

  // ---- Differencing ----
  describe("difference", () => {
    it("first difference of linear series is constant", () => {
      const series = [2, 4, 6, 8, 10];
      const result = difference(series, 1);
      expect(result).toEqual([2, 2, 2, 2]);
    });

    it("second difference of quadratic series is constant", () => {
      const series = [1, 4, 9, 16, 25]; // i^2
      const result = difference(series, 2);
      expect(result).toEqual([2, 2, 2]);
    });

    it("d=0 returns original series", () => {
      const series = [1, 2, 3];
      expect(difference(series, 0)).toEqual([1, 2, 3]);
    });

    it("throws on negative d", () => {
      expect(() => difference([1, 2, 3], -1)).toThrow();
    });

    it("throws when series becomes too short", () => {
      expect(() => difference([1, 2], 2)).toThrow();
    });
  });

  // ---- ARIMA ----
  describe("arima", () => {
    it("AR(1) model estimates coefficient close to true value", () => {
      // Generate AR(1) series: x_t = 0.7 * x_{t-1} + noise
      const n = 500;
      const series: number[] = [0];
      let seed = 123;
      for (let i = 1; i < n; i++) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        const noise = (seed / 0x7fffffff - 0.5) * 0.5;
        series.push(0.7 * series[i - 1] + noise);
      }

      const result = arima(series, 1, 0, 0);
      expect(result.order).toEqual({ p: 1, d: 0, q: 0 });
      expect(result.arCoefficients.length).toBe(1);
      expect(result.arCoefficients[0]).toBeCloseTo(0.7, 0);
    });

    it("ARIMA(0,1,0) is a random walk model", () => {
      const series = [100, 101, 103, 102, 105, 107, 106, 108];
      const result = arima(series, 0, 1, 0);
      expect(result.order.d).toBe(1);
      expect(result.arCoefficients.length).toBe(0);
      expect(result.maCoefficients.length).toBe(0);
    });

    it("forecast returns correct number of steps", () => {
      const series = Array.from({ length: 50 }, (_, i) => i + Math.sin(i));
      const result = arima(series, 2, 0, 0);
      const fc = result.forecast(5);
      expect(fc.length).toBe(5);
    });

    it("forecast with differencing integrates back", () => {
      const series = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28];
      const result = arima(series, 1, 1, 0);
      const fc = result.forecast(3);
      // Linear trend should continue approximately
      expect(fc[0]).toBeGreaterThan(28);
    });

    it("computes AIC", () => {
      const series = Array.from({ length: 50 }, (_, i) => i + Math.sin(i));
      const result = arima(series, 1, 0, 0);
      expect(isFinite(result.aic)).toBe(true);
    });

    it("returns residuals", () => {
      const series = Array.from({ length: 50 }, (_, i) => i);
      const result = arima(series, 1, 0, 0);
      expect(result.residuals.length).toBeGreaterThan(0);
    });

    it("throws on too-short series", () => {
      expect(() => arima([1, 2], 1, 0, 0)).toThrow();
    });

    it("throws on negative parameters", () => {
      expect(() => arima([1, 2, 3, 4, 5], -1, 0, 0)).toThrow();
    });
  });
});
