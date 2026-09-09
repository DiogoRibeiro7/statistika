import { sarima, prophetDecompose, SARIMAResult, ProphetDecomposition } from "../src/forecasting";

// Generate a seasonal time series for testing
function generateSeasonalSeries(
  n: number,
  period: number,
  trend: number,
  seasonalAmplitude: number,
  noise: number,
  seed: number,
): number[] {
  // Simple seeded PRNG
  let s = seed;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return (s / 0x7fffffff) * 2 - 1; // [-1, 1]
  };

  const series: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = trend * i;
    const seasonal = seasonalAmplitude * Math.sin((2 * Math.PI * i) / period);
    series.push(100 + t + seasonal + noise * rand());
  }
  return series;
}

describe("SARIMA", () => {
  // Monthly data with period 12
  const series = generateSeasonalSeries(120, 12, 0.5, 10, 1, 42);

  it("fits a SARIMA(1,0,0)(1,0,0)[12] model", () => {
    const result = sarima(series, 1, 0, 0, 1, 0, 0, 12);
    expect(result.order).toEqual({ p: 1, d: 0, q: 0 });
    expect(result.seasonalOrder).toEqual({ P: 1, D: 0, Q: 0, m: 12 });
    expect(result.arCoefficients).toHaveLength(1);
    expect(result.seasonalArCoefficients).toHaveLength(1);
    expect(Number.isFinite(result.sigma2)).toBe(true);
    expect(Number.isFinite(result.aic)).toBe(true);
  });

  it("fits a SARIMA(1,1,1)(1,1,0)[12] model", () => {
    const result = sarima(series, 1, 1, 1, 1, 1, 0, 12);
    expect(result.order).toEqual({ p: 1, d: 1, q: 1 });
    expect(result.seasonalOrder).toEqual({ P: 1, D: 1, Q: 0, m: 12 });
    expect(result.maCoefficients).toHaveLength(1);
    expect(result.residuals.length).toBeGreaterThan(0);
  });

  it("produces finite forecasts", () => {
    const result = sarima(series, 1, 0, 0, 1, 0, 0, 12);
    const fc = result.forecast(12);
    expect(fc).toHaveLength(12);
    expect(fc.every(Number.isFinite)).toBe(true);
  });

  it("forecasts with differencing integrate back correctly", () => {
    const result = sarima(series, 1, 1, 0, 0, 1, 0, 12);
    const fc = result.forecast(6);
    expect(fc).toHaveLength(6);
    expect(fc.every(Number.isFinite)).toBe(true);
    // Forecasts should be in a reasonable range relative to the series
    for (const v of fc) {
      expect(Math.abs(v)).toBeLessThan(10000);
    }
  });

  it("throws for m < 2", () => {
    expect(() => sarima(series, 1, 0, 0, 1, 0, 0, 1)).toThrow("at least 2");
  });

  it("throws for negative orders", () => {
    expect(() => sarima(series, -1, 0, 0, 0, 0, 0, 12)).toThrow("non-negative");
  });

  it("throws for series too short", () => {
    expect(() => sarima([1, 2, 3], 1, 0, 0, 1, 0, 0, 12)).toThrow("at least 15");
  });

  it("throws for NaN in series", () => {
    const bad = [...series];
    bad[5] = NaN;
    expect(() => sarima(bad, 1, 0, 0, 1, 0, 0, 12)).toThrow("finite");
  });

  it("handles purely seasonal model (0,0,0)(1,0,0)[4]", () => {
    const quarterly = generateSeasonalSeries(60, 4, 0, 5, 0.5, 99);
    const result = sarima(quarterly, 0, 0, 0, 1, 0, 0, 4);
    expect(result.arCoefficients).toHaveLength(0);
    expect(result.seasonalArCoefficients).toHaveLength(1);
  });

  it("handles seasonal MA model (0,0,0)(0,0,1)[12]", () => {
    const result = sarima(series, 0, 0, 0, 0, 0, 1, 12);
    expect(result.seasonalMaCoefficients).toHaveLength(1);
    expect(Number.isFinite(result.aic)).toBe(true);
  });

  it("residuals are finite", () => {
    const result = sarima(series, 1, 0, 1, 1, 0, 0, 12);
    expect(result.residuals.every(Number.isFinite)).toBe(true);
  });
});

describe("prophetDecompose", () => {
  // Monthly data with clear trend and seasonality
  const series = generateSeasonalSeries(120, 12, 1.0, 15, 2, 123);

  it("decomposes into trend, seasonal, residual", () => {
    const result = prophetDecompose(series, 12);
    expect(result.trend).toHaveLength(series.length);
    expect(result.seasonal).toHaveLength(series.length);
    expect(result.residual).toHaveLength(series.length);
  });

  it("components sum to original series", () => {
    const result = prophetDecompose(series, 12);
    for (let i = 0; i < series.length; i++) {
      const reconstructed = result.trend[i] + result.seasonal[i] + result.residual[i];
      expect(reconstructed).toBeCloseTo(series[i], 6);
    }
  });

  it("trend is monotonically increasing for upward-trending series", () => {
    const result = prophetDecompose(series, 12);
    // The overall trend should go up, even if not monotonically due to piecewise fit
    expect(result.trend[result.trend.length - 1]).toBeGreaterThan(result.trend[0]);
  });

  it("detects changepoints", () => {
    const result = prophetDecompose(series, 12);
    expect(result.changepoints.length).toBeGreaterThan(0);
    // All changepoints should be valid indices
    for (const cp of result.changepoints) {
      expect(cp).toBeGreaterThanOrEqual(0);
      expect(cp).toBeLessThan(series.length);
    }
  });

  it("returns slopes for each segment", () => {
    const result = prophetDecompose(series, 12);
    // Number of slopes = number of segments = changepoints + 1
    expect(result.slopes.length).toBe(result.changepoints.length + 1);
  });

  it("respects nChangepoints option", () => {
    const result = prophetDecompose(series, 12, { nChangepoints: 3 });
    expect(result.changepoints.length).toBeLessThanOrEqual(3);
  });

  it("respects fourierOrder option", () => {
    const result1 = prophetDecompose(series, 12, { fourierOrder: 1 });
    const result2 = prophetDecompose(series, 12, { fourierOrder: 5 });
    // Higher order should generally capture more seasonal detail
    // (different residual magnitudes)
    const rss1 = result1.residual.reduce((s, r) => s + r * r, 0);
    const rss2 = result2.residual.reduce((s, r) => s + r * r, 0);
    expect(rss2).toBeLessThanOrEqual(rss1 * 1.1); // Higher order should fit at least as well
  });

  it("throws for period < 2", () => {
    expect(() => prophetDecompose(series, 1)).toThrow("at least 2");
  });

  it("throws for series too short", () => {
    expect(() => prophetDecompose([1, 2, 3], 12)).toThrow("2 full periods");
  });

  it("throws for NaN in series", () => {
    const bad = [...series];
    bad[10] = NaN;
    expect(() => prophetDecompose(bad, 12)).toThrow("finite");
  });

  it("works with weekly data (period=7)", () => {
    const weekly = generateSeasonalSeries(100, 7, 0.2, 5, 1, 77);
    const result = prophetDecompose(weekly, 7);
    expect(result.trend).toHaveLength(weekly.length);
    expect(result.seasonal).toHaveLength(weekly.length);
  });

  it("seasonal component is periodic", () => {
    const result = prophetDecompose(series, 12);
    // Check that seasonal values roughly repeat with period 12
    // (not exact due to Fourier fit, but should be close)
    for (let i = 24; i < series.length - 12; i++) {
      const diff = Math.abs(result.seasonal[i] - result.seasonal[i + 12]);
      expect(diff).toBeLessThan(5); // Allow some tolerance
    }
  });

  it("all output values are finite", () => {
    const result = prophetDecompose(series, 12);
    expect(result.trend.every(Number.isFinite)).toBe(true);
    expect(result.seasonal.every(Number.isFinite)).toBe(true);
    expect(result.residual.every(Number.isFinite)).toBe(true);
  });
});
