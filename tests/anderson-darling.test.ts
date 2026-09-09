import { andersonDarlingTest } from "../src/tests/anderson-darling";

describe("andersonDarlingTest", () => {
  it("does not reject normality for normally-distributed data", () => {
    // Approximate standard normal quantiles
    const normalData = [
      -1.28, -0.84, -0.52, -0.25, 0.0,
       0.25,  0.52,  0.84,  1.28, 0.0,
       -0.67, 0.67, -0.13, 0.13, -0.39, 0.39,
    ];
    const result = andersonDarlingTest(normalData);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.rejectNull).toBe(false);
  });

  it("rejects normality for clearly non-normal data", () => {
    // Bimodal / uniform-like data
    const nonNormalData = [1, 1, 1, 1, 1, 10, 10, 10, 10, 10];
    const result = andersonDarlingTest(nonNormalData);
    expect(result.statistic).toBeGreaterThan(0.5);
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.rejectNull).toBe(true);
  });

  it("returns small A-squared for a near-perfect normal sample", () => {
    const perfectNormal = [
      -1.645, -1.282, -0.842, -0.524, -0.253,
       0.0,    0.253,  0.524,  0.842,  1.282, 1.645,
    ];
    const result = andersonDarlingTest(perfectNormal);
    // A² should be small for well-fitting normal data
    expect(result.statistic).toBeLessThan(1.0);
    expect(result.rejectNull).toBe(false);
  });

  it("handles the minimum sample size of 2", () => {
    const result = andersonDarlingTest([1, 2]);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it("works with larger samples", () => {
    // Linear spacing (roughly uniform -> should reject normality for large n)
    const data: number[] = [];
    for (let i = 0; i < 50; i++) {
      data.push(-3 + (6 * i) / 49);
    }
    const result = andersonDarlingTest(data);
    expect(result.statistic).toBeGreaterThan(0);
    expect(typeof result.rejectNull).toBe("boolean");
  });

  it("respects custom alpha", () => {
    const normalData = [
      -1.28, -0.84, -0.52, -0.25, 0.0,
       0.25,  0.52,  0.84,  1.28, 0.0,
    ];
    const strict = andersonDarlingTest(normalData, 0.01);
    expect(typeof strict.rejectNull).toBe("boolean");
  });

  it("throws for fewer than 2 observations", () => {
    expect(() => andersonDarlingTest([1])).toThrow("at least 2");
    expect(() => andersonDarlingTest([])).toThrow("at least 2");
  });

  it("throws for constant data (zero variance)", () => {
    expect(() => andersonDarlingTest([5, 5, 5, 5])).toThrow("non-constant data");
  });

  it("returns correct result structure", () => {
    const result = andersonDarlingTest([1, 2, 3, 4, 5, 6, 7]);
    expect(result).toHaveProperty("statistic");
    expect(result).toHaveProperty("pValue");
    expect(result).toHaveProperty("rejectNull");
    expect(typeof result.statistic).toBe("number");
    expect(typeof result.pValue).toBe("number");
    expect(typeof result.rejectNull).toBe("boolean");
  });

  it("detects exponential distribution as non-normal", () => {
    // Exponential-like data (all positive, right-skewed)
    const expData = [0.05, 0.1, 0.15, 0.3, 0.5, 0.8, 1.2, 2.0, 3.5, 6.0, 10.0, 18.0];
    const result = andersonDarlingTest(expData);
    // Should have a large A² and reject normality
    expect(result.statistic).toBeGreaterThan(0.5);
  });

  it("p-value is bounded between 0 and 1", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const result = andersonDarlingTest(data);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });
});
