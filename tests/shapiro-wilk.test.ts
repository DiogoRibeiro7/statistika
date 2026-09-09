import { shapiroWilkTest } from "../src/tests/shapiro-wilk";

describe("shapiroWilkTest", () => {
  it("does not reject normality for normally-distributed data", () => {
    // Data drawn from a standard normal (known normal sample)
    const normalData = [
      -0.93, -0.67, -0.44, -0.27, -0.12,
       0.03,  0.18,  0.35,  0.56,  0.88,
    ];
    const result = shapiroWilkTest(normalData);
    expect(result.statistic).toBeGreaterThan(0.85);
    expect(result.statistic).toBeLessThanOrEqual(1);
    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.isNormal).toBe(true);
  });

  it("rejects normality for clearly non-normal data", () => {
    // Uniform-like / heavy bimodal data
    const nonNormalData = [1, 1, 1, 1, 1, 10, 10, 10, 10, 10];
    const result = shapiroWilkTest(nonNormalData);
    expect(result.statistic).toBeLessThan(0.85);
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.isNormal).toBe(false);
  });

  it("returns W close to 1 for a near-perfect normal sample", () => {
    // Sorted standard normal quantiles (theoretical)
    const perfectNormal = [
      -1.645, -1.282, -0.842, -0.524, -0.253,
       0.0,    0.253,  0.524,  0.842,  1.282, 1.645,
    ];
    const result = shapiroWilkTest(perfectNormal);
    expect(result.statistic).toBeGreaterThan(0.95);
    expect(result.isNormal).toBe(true);
  });

  it("handles the minimum sample size of 3", () => {
    const result = shapiroWilkTest([1, 2, 3]);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.statistic).toBeLessThanOrEqual(1);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it("handles larger sample sizes", () => {
    // Generate quasi-normal data using CLT
    const data: number[] = [];
    for (let i = 0; i < 100; i++) {
      data.push(-2 + (4 * i) / 99); // linearly spaced from -2 to 2
    }
    const result = shapiroWilkTest(data);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.statistic).toBeLessThanOrEqual(1);
  });

  it("respects custom alpha", () => {
    const normalData = [
      -0.93, -0.67, -0.44, -0.27, -0.12,
       0.03,  0.18,  0.35,  0.56,  0.88,
    ];
    const strict = shapiroWilkTest(normalData, 0.01);
    expect(typeof strict.isNormal).toBe("boolean");
  });

  it("throws for fewer than 3 observations", () => {
    expect(() => shapiroWilkTest([1, 2])).toThrow("at least 3");
    expect(() => shapiroWilkTest([1])).toThrow("at least 3");
    expect(() => shapiroWilkTest([])).toThrow("at least 3");
  });

  it("throws for constant data (zero variance)", () => {
    expect(() => shapiroWilkTest([5, 5, 5, 5])).toThrow("identical");
  });

  it("returns correct result structure", () => {
    const result = shapiroWilkTest([1, 2, 3, 4, 5]);
    expect(result).toHaveProperty("statistic");
    expect(result).toHaveProperty("pValue");
    expect(result).toHaveProperty("isNormal");
    expect(typeof result.statistic).toBe("number");
    expect(typeof result.pValue).toBe("number");
    expect(typeof result.isNormal).toBe("boolean");
  });

  it("detects skewed distribution as non-normal", () => {
    // Heavily right-skewed data (exponential-like)
    const skewed = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 3.0, 8.0, 15.0, 25.0];
    const result = shapiroWilkTest(skewed);
    expect(result.statistic).toBeLessThan(0.95);
  });
});
