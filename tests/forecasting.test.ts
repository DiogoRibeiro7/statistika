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

    it("trend is null at edges", () => {
      const period = 4;
      const n = 16;
      const series = Array.from({ length: n }, (_, i) => i + Math.sin(i));
      const result = seasonalDecompose(series, period);
      // Edges should be null
      expect(result.trend[0]).toBeNull();
      expect(result.trend[n - 1]).toBeNull();
    });

    it("residual is null where trend is null", () => {
      const period = 4;
      const n = 16;
      const series = Array.from({ length: n }, (_, i) => i * 0.5);
      const result = seasonalDecompose(series, period);
      for (let i = 0; i < n; i++) {
        if (result.trend[i] === null) {
          expect(result.residual[i]).toBeNull();
        }
      }
    });

    it("throws on NaN in series", () => {
      expect(() => seasonalDecompose([1, NaN, 3, 4, 5, 6, 7, 8], 4)).toThrow("finite");
    });
  });

  describe("adfTest edge cases", () => {
    it("accepts custom maxLags parameter", () => {
      const result = adfTest(stationary, 3);
      expect(result.lags).toBe(3);
    });

    it("throws on non-finite values", () => {
      const bad = Array.from({ length: 20 }, (_, i) => i);
      bad[5] = NaN;
      expect(() => adfTest(bad)).toThrow("finite");
    });

    it("returns expected fields", () => {
      const result = adfTest(stationary, 1);
      expect(typeof result.statistic).toBe("number");
      expect(typeof result.pValue).toBe("number");
      expect(typeof result.lags).toBe("number");
      expect(typeof result.isStationary).toBe("boolean");
    });
  });

  describe("autoArima edge cases", () => {
    it("throws on too few observations", () => {
      expect(() => autoArima([1, 2, 3])).toThrow("at least 10");
    });

    it("throws on non-finite values", () => {
      const bad = Array.from({ length: 20 }, (_, i) => i);
      bad[5] = Infinity;
      expect(() => autoArima(bad)).toThrow("finite");
    });

    it("returns selectedOrder with p, d, q", () => {
      const result = autoArima(stationary, { maxP: 1, maxQ: 1 });
      expect(result.selectedOrder).toHaveProperty("p");
      expect(result.selectedOrder).toHaveProperty("d");
      expect(result.selectedOrder).toHaveProperty("q");
    });

    it("respects maxD option", () => {
      const result = autoArima(trending, { maxP: 1, maxQ: 0, maxD: 1 });
      expect(result.selectedOrder.d).toBeLessThanOrEqual(1);
    });
  });

  describe("forecastWithIntervals edge cases", () => {
    it("throws on non-positive steps", () => {
      const model = arima(stationary, 1, 0, 0);
      expect(() => forecastWithIntervals(model, 0)).toThrow("positive integer");
      expect(() => forecastWithIntervals(model, -1)).toThrow("positive integer");
    });

    it("throws on invalid confidence", () => {
      const model = arima(stationary, 1, 0, 0);
      expect(() => forecastWithIntervals(model, 5, 0)).toThrow("confidence must be in (0, 1)");
      expect(() => forecastWithIntervals(model, 5, 1)).toThrow("confidence must be in (0, 1)");
      expect(() => forecastWithIntervals(model, 5, 1.5)).toThrow("confidence must be in (0, 1)");
    });

    it("respects custom confidence level", () => {
      const model = arima(stationary, 2, 0, 0);
      const fc90 = forecastWithIntervals(model, 5, 0.90);
      const fc99 = forecastWithIntervals(model, 5, 0.99);
      expect(fc90.confidence).toBe(0.90);
      expect(fc99.confidence).toBe(0.99);
      // 99% interval should be wider than 90%
      const width90 = fc90.upper[0] - fc90.lower[0];
      const width99 = fc99.upper[0] - fc99.lower[0];
      expect(width99).toBeGreaterThan(width90);
    });

    it("throws on NaN steps", () => {
      const model = arima(stationary, 1, 0, 0);
      expect(() => forecastWithIntervals(model, NaN)).toThrow("positive integer");
    });

    it("throws on Infinity steps", () => {
      const model = arima(stationary, 1, 0, 0);
      expect(() => forecastWithIntervals(model, Infinity)).toThrow("positive integer");
    });

    it("throws on NaN confidence", () => {
      const model = arima(stationary, 1, 0, 0);
      expect(() => forecastWithIntervals(model, 5, NaN)).toThrow("confidence must be in (0, 1)");
    });

    it("single step forecast has non-zero width interval", () => {
      const model = arima(stationary, 2, 0, 0);
      const fc = forecastWithIntervals(model, 1);
      expect(fc.point).toHaveLength(1);
      expect(fc.upper[0] - fc.lower[0]).toBeGreaterThan(0);
    });
  });

  describe("adfTest additional edge cases", () => {
    it("throws on Infinity in series", () => {
      const bad = Array.from({ length: 20 }, (_, i) => i);
      bad[5] = Infinity;
      expect(() => adfTest(bad)).toThrow("finite");
    });

    it("pValue is between 0 and 1", () => {
      const result = adfTest(stationary, 1);
      expect(result.pValue).toBeGreaterThanOrEqual(0);
      expect(result.pValue).toBeLessThanOrEqual(1);
    });

    it("default lags use cube root rule", () => {
      const data = Array.from({ length: 64 }, (_, i) => Math.sin(i));
      const result = adfTest(data);
      // floor(cbrt(64)) = 4
      expect(result.lags).toBe(4);
    });
  });

  describe("autoArima additional edge cases", () => {
    it("respects maxP and maxQ options", () => {
      const result = autoArima(stationary, { maxP: 1, maxQ: 0 });
      expect(result.selectedOrder.p).toBeLessThanOrEqual(1);
      expect(result.selectedOrder.q).toBe(0);
    });

    it("forecast method is callable on autoArima result", () => {
      const result = autoArima(stationary, { maxP: 2, maxQ: 1 });
      const forecast = result.forecast(5);
      expect(forecast).toHaveLength(5);
      for (const v of forecast) {
        expect(Number.isFinite(v)).toBe(true);
      }
    });

    it("aic is finite", () => {
      const result = autoArima(stationary, { maxP: 1, maxQ: 1 });
      expect(Number.isFinite(result.aic)).toBe(true);
    });
  });

  describe("seasonalDecompose additional edge cases", () => {
    it("works with odd period", () => {
      const period = 5;
      const n = 20;
      const series = Array.from({ length: n }, (_, i) => i * 0.1 + Math.sin(2 * Math.PI * i / period));
      const result = seasonalDecompose(series, period);
      expect(result.trend).toHaveLength(n);
      expect(result.seasonal).toHaveLength(n);
      expect(result.residual).toHaveLength(n);
    });

    it("throws on Infinity in series", () => {
      expect(() => seasonalDecompose([1, 2, Infinity, 4, 5, 6, 7, 8], 4)).toThrow("finite");
    });

    it("seasonal component sums close to zero across one period", () => {
      const period = 4;
      const n = 16;
      const series = Array.from({ length: n }, (_, i) => 10 + 3 * Math.sin(2 * Math.PI * i / period));
      const result = seasonalDecompose(series, period);
      const seasonalSum = result.seasonal.slice(0, period).reduce((a, b) => a + b, 0);
      expect(Math.abs(seasonalSum)).toBeLessThan(1);
    });
  });
});
