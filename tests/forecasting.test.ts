import {
  adfTest,
  autoArima,
  forecastWithIntervals,
  seasonalDecompose,
} from "../src/forecasting";
import { arima } from "../src/time-series";

describe("Forecasting", () => {
  // Trending series (non-stationary)
  const trending = Array.from({ length: 100 }, (_, i) => i * 0.5 + Math.sin(i * 0.3) * 2);

  // Stationary series (AR(1) with phi=0.5, clearly mean-reverting)
  const stationary: number[] = [0];
  for (let i = 1; i < 200; i++) stationary.push(0.5 * stationary[i - 1] + Math.sin(i) * 2);

  describe("adfTest", () => {
    it("detects non-stationarity in trending series", () => {
      const result = adfTest(trending);
      expect(result.isStationary).toBe(false);
      expect(result.lags).toBeGreaterThan(0);
    });

    it("detects stationarity in stationary series", () => {
      const result = adfTest(stationary, 1);
      expect(result.isStationary).toBe(true);
    });

    it("throws on too few observations", () => {
      expect(() => adfTest([1, 2, 3])).toThrow("at least 10");
    });
  });

  describe("autoArima", () => {
    it("selects a model", () => {
      const result = autoArima(stationary, { maxP: 2, maxQ: 2 });
      expect(result.selectedOrder).toBeDefined();
      expect(result.selectedOrder.p).toBeGreaterThanOrEqual(0);
      expect(result.selectedOrder.q).toBeGreaterThanOrEqual(0);
      expect(result.aic).not.toBe(Infinity);
    });

    it("handles trending data with differencing", () => {
      const result = autoArima(trending, { maxP: 2, maxQ: 1 });
      expect(result.selectedOrder.d).toBeGreaterThanOrEqual(0);
    });
  });

  describe("forecastWithIntervals", () => {
    it("produces prediction intervals", () => {
      const model = arima(stationary, 2, 0, 0);
      const fc = forecastWithIntervals(model, 5);

      expect(fc.point).toHaveLength(5);
      expect(fc.lower).toHaveLength(5);
      expect(fc.upper).toHaveLength(5);
      expect(fc.confidence).toBe(0.95);

      for (let i = 0; i < 5; i++) {
        expect(fc.lower[i]).toBeLessThan(fc.point[i]);
        expect(fc.upper[i]).toBeGreaterThan(fc.point[i]);
      }
    });

    it("intervals widen with horizon", () => {
      const model = arima(stationary, 2, 0, 0);
      const fc = forecastWithIntervals(model, 10);
      const width0 = fc.upper[0] - fc.lower[0];
      const width9 = fc.upper[9] - fc.lower[9];
      expect(width9).toBeGreaterThan(width0);
    });
  });

  describe("seasonalDecompose", () => {
    it("decomposes seasonal data", () => {
      // Create data with trend + seasonal + noise
      const period = 12;
      const n = 48;
      const series = Array.from({ length: n }, (_, i) => {
        const trend = i * 0.1;
        const seasonal = 5 * Math.sin((2 * Math.PI * (i % period)) / period);
        return trend + seasonal;
      });

      const result = seasonalDecompose(series, period);
      expect(result.trend).toHaveLength(n);
      expect(result.seasonal).toHaveLength(n);
      expect(result.residual).toHaveLength(n);

      // Seasonal should repeat with the period
      for (let i = period; i < n; i++) {
        expect(result.seasonal[i]).toBeCloseTo(result.seasonal[i % period], 5);
      }
    });

    it("throws on short series", () => {
      expect(() => seasonalDecompose([1, 2, 3], 4)).toThrow("2 full periods");
    });

    it("throws on invalid period", () => {
      expect(() => seasonalDecompose([1, 2, 3, 4], 1)).toThrow("at least 2");
    });
  });
});
