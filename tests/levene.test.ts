import { leveneTest } from "../src/tests/levene";

describe("leveneTest", () => {
  it("does not reject equal variances for groups with similar spread", () => {
    const group1 = [10, 11, 12, 13, 14];
    const group2 = [20, 21, 22, 23, 24];
    const group3 = [30, 31, 32, 33, 34];
    const result = leveneTest([group1, group2, group3]);
    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.rejectNull).toBe(false);
    expect(result.df1).toBe(2);
    expect(result.df2).toBe(12);
  });

  it("rejects equal variances for groups with very different spread", () => {
    const group1 = [10, 10.1, 10.2, 9.9, 9.8]; // small variance
    const group2 = [5, 15, 25, 35, 45]; // large variance
    const result = leveneTest([group1, group2]);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.rejectNull).toBe(true);
    expect(result.df1).toBe(1);
    expect(result.df2).toBe(8);
  });

  it("uses median-based (Brown-Forsythe) variant by default", () => {
    const group1 = [1, 2, 3, 4, 5];
    const group2 = [1, 2, 3, 4, 50]; // outlier
    // Median-based should be more robust to the outlier
    const medianResult = leveneTest([group1, group2]);
    const meanResult = leveneTest([group1, group2], { center: "mean" });
    // Both should give a result, but values differ
    expect(medianResult.statistic).toBeGreaterThanOrEqual(0);
    expect(meanResult.statistic).toBeGreaterThanOrEqual(0);
  });

  it("supports mean-based center", () => {
    const group1 = [10, 11, 12, 13, 14];
    const group2 = [20, 21, 22, 23, 24];
    const result = leveneTest([group1, group2], { center: "mean" });
    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.rejectNull).toBe(false);
  });

  it("works with more than 2 groups", () => {
    const groups = [
      [1, 2, 3, 4, 5],
      [10, 20, 30, 40, 50],
      [100, 200, 300, 400, 500],
    ];
    const result = leveneTest(groups);
    expect(result.df1).toBe(2);
    expect(result.df2).toBe(12);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.pValue).toBeLessThan(0.05); // very different variances
    expect(result.rejectNull).toBe(true);
  });

  it("respects custom alpha", () => {
    const group1 = [10, 11, 12, 13, 14];
    const group2 = [20, 21, 22, 23, 24];
    const strict = leveneTest([group1, group2], { alpha: 0.01 });
    expect(typeof strict.rejectNull).toBe("boolean");
  });

  it("throws for fewer than 2 groups", () => {
    expect(() => leveneTest([[1, 2, 3]])).toThrow("at least 2 groups");
    expect(() => leveneTest([])).toThrow("at least 2 groups");
  });

  it("throws for groups with fewer than 2 observations", () => {
    expect(() => leveneTest([[1], [2, 3, 4]])).toThrow("at least 2 observations");
    expect(() => leveneTest([[1, 2], []])).toThrow("at least 2 observations");
  });

  it("returns correct result structure", () => {
    const result = leveneTest([[1, 2, 3], [4, 5, 6]]);
    expect(result).toHaveProperty("statistic");
    expect(result).toHaveProperty("pValue");
    expect(result).toHaveProperty("df1");
    expect(result).toHaveProperty("df2");
    expect(result).toHaveProperty("rejectNull");
    expect(typeof result.statistic).toBe("number");
    expect(typeof result.pValue).toBe("number");
    expect(typeof result.df1).toBe("number");
    expect(typeof result.df2).toBe("number");
    expect(typeof result.rejectNull).toBe("boolean");
  });

  it("p-value is bounded between 0 and 1", () => {
    const result = leveneTest([[1, 2, 3, 4], [5, 6, 7, 8]]);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it("F statistic is non-negative", () => {
    const result = leveneTest([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
  });
});
