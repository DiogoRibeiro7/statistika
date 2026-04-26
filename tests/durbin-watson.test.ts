import { durbinWatsonTest } from "../src/diagnostics";

describe("durbinWatsonTest", () => {
  it("returns statistic near 2 for uncorrelated residuals", () => {
    // Alternating signs with smaller variation suggest no autocorrelation
    const residuals = [0.1, -0.05, 0.08, -0.03, 0.06, -0.02, 0.04, -0.01, 0.03, -0.015];
    const result = durbinWatsonTest(residuals);
    expect(result.statistic).toBeGreaterThan(1.0);
    expect(result.statistic).toBeLessThan(3.0);
    expect(result.interpretation).toBe("none");
  });

  it("detects positive autocorrelation", () => {
    // Residuals that drift slowly (positive autocorrelation)
    const residuals = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                       2.0, 2.1, 2.2, 2.3, 2.4, 2.5];
    const result = durbinWatsonTest(residuals);
    expect(result.statistic).toBeLessThan(1.0);
    expect(result.interpretation).toBe("positive");
  });

  it("detects negative autocorrelation", () => {
    // Residuals alternating with large magnitude (negative autocorrelation)
    const residuals = [5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5];
    const result = durbinWatsonTest(residuals);
    expect(result.statistic).toBeGreaterThan(3.0);
    expect(result.interpretation).toBe("negative");
  });

  it("statistic is bounded between 0 and 4", () => {
    const residuals = [0.5, -0.3, 0.8, -0.1, 0.4, -0.6, 0.2, -0.4, 0.7, -0.2];
    const result = durbinWatsonTest(residuals);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.statistic).toBeLessThanOrEqual(4);
  });

  it("returns 2 for all-zero residuals", () => {
    const residuals = [0, 0, 0, 0, 0];
    const result = durbinWatsonTest(residuals);
    expect(result.statistic).toBe(2);
    expect(result.interpretation).toBe("none");
  });

  it("handles the minimum of 3 residuals", () => {
    const result = durbinWatsonTest([0.1, -0.2, 0.3]);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.statistic).toBeLessThanOrEqual(4);
    expect(["positive", "negative", "none"]).toContain(result.interpretation);
  });

  it("computes correct DW statistic for a known example", () => {
    // Manual calculation:
    // residuals = [1, -1, 1, -1]
    // numerator = (-1-1)^2 + (1-(-1))^2 + (-1-1)^2 = 4+4+4 = 12
    // denominator = 1+1+1+1 = 4
    // DW = 12/4 = 3.0
    const result = durbinWatsonTest([1, -1, 1, -1]);
    expect(result.statistic).toBeCloseTo(3.0, 5);
  });

  it("computes DW=0 for perfectly correlated residuals", () => {
    // All identical non-zero residuals: numerator = 0, denominator > 0
    const result = durbinWatsonTest([1, 1, 1, 1, 1]);
    expect(result.statistic).toBeCloseTo(0, 5);
    expect(result.interpretation).toBe("positive");
  });

  it("throws for fewer than 3 residuals", () => {
    expect(() => durbinWatsonTest([1, 2])).toThrow("at least 3");
    expect(() => durbinWatsonTest([1])).toThrow("at least 3");
    expect(() => durbinWatsonTest([])).toThrow("at least 3");
  });

  it("returns correct result structure", () => {
    const result = durbinWatsonTest([0.1, -0.2, 0.3, -0.1, 0.05]);
    expect(result).toHaveProperty("statistic");
    expect(result).toHaveProperty("interpretation");
    expect(typeof result.statistic).toBe("number");
    expect(typeof result.interpretation).toBe("string");
  });

  it("handles large sample sizes", () => {
    // Generate alternating residuals
    const residuals = Array.from({ length: 200 }, (_, i) =>
      Math.sin(i * 0.5) * 0.1,
    );
    const result = durbinWatsonTest(residuals);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.statistic).toBeLessThanOrEqual(4);
  });
});
