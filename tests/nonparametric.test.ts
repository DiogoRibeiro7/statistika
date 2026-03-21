import {
  kernelDensity,
  bootstrapCI,
  permutationTest,
  oneSamplePermutationTest,
} from "../src/nonparametric";
import { mean, median } from "../src/utils/descriptive";

// ── Kernel Density Estimation ───────────────────────────────────────────

describe("kernelDensity", () => {
  const data = [1, 2, 2, 3, 3, 3, 4, 4, 5];

  it("returns density estimates with correct structure", () => {
    const result = kernelDensity(data);
    expect(result.x).toHaveLength(512);
    expect(result.density).toHaveLength(512);
    expect(result.bandwidth).toBeGreaterThan(0);
    expect(result.kernel).toBe("gaussian");
  });

  it("densities are non-negative", () => {
    const result = kernelDensity(data);
    for (const d of result.density) {
      expect(d).toBeGreaterThanOrEqual(0);
    }
  });

  it("density integrates approximately to 1", () => {
    const result = kernelDensity(data, { nPoints: 1000 });
    const step = (result.x[result.x.length - 1] - result.x[0]) / (result.x.length - 1);
    const integral = result.density.reduce((s, d) => s + d * step, 0);
    expect(integral).toBeCloseTo(1, 1);
  });

  it("peak is near the mode of the data", () => {
    const result = kernelDensity(data, { nPoints: 200 });
    const maxIdx = result.density.indexOf(Math.max(...result.density));
    expect(result.x[maxIdx]).toBeCloseTo(3, 0);
  });

  it("respects custom bandwidth", () => {
    const result = kernelDensity(data, { bandwidth: 0.5 });
    expect(result.bandwidth).toBe(0.5);
  });

  it("supports epanechnikov kernel", () => {
    const result = kernelDensity(data, { kernel: "epanechnikov" });
    expect(result.kernel).toBe("epanechnikov");
    expect(result.density.some((d) => d > 0)).toBe(true);
  });

  it("supports uniform kernel", () => {
    const result = kernelDensity(data, { kernel: "uniform" });
    expect(result.kernel).toBe("uniform");
  });

  it("supports triangular kernel", () => {
    const result = kernelDensity(data, { kernel: "triangular" });
    expect(result.kernel).toBe("triangular");
  });

  it("throws for unknown kernel", () => {
    expect(() => kernelDensity(data, { kernel: "quartic" })).toThrow("Unknown kernel");
  });

  it("throws for fewer than 2 elements", () => {
    expect(() => kernelDensity([1])).toThrow();
  });

  it("respects custom from/to range", () => {
    const result = kernelDensity(data, { from: 0, to: 6, nPoints: 100 });
    expect(result.x[0]).toBeCloseTo(0, 5);
    expect(result.x[result.x.length - 1]).toBeCloseTo(6, 5);
  });
});

// ── Bootstrap Confidence Intervals ──────────────────────────────────────

describe("bootstrapCI", () => {
  const data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20];

  it("returns a confidence interval with correct structure", () => {
    const result = bootstrapCI(data, mean, { seed: 42 });
    expect(result.estimate).toBe(11);
    expect(result.lower).toBeLessThan(result.estimate);
    expect(result.upper).toBeGreaterThan(result.estimate);
    expect(result.confidenceLevel).toBe(0.95);
    expect(result.standardError).toBeGreaterThan(0);
    expect(result.method).toBe("percentile");
    expect(result.nReplicates).toBe(10000);
  });

  it("CI contains the true population parameter for the mean", () => {
    const result = bootstrapCI(data, mean, { seed: 123 });
    expect(result.lower).toBeLessThan(11);
    expect(result.upper).toBeGreaterThan(11);
  });

  it("narrower CI with more data", () => {
    const smallData = [5, 10, 15];
    const largeData = Array.from({ length: 100 }, (_, i) => 5 + (i / 99) * 10);
    const ciSmall = bootstrapCI(smallData, mean, { seed: 42 });
    const ciLarge = bootstrapCI(largeData, mean, { seed: 42 });
    const widthSmall = ciSmall.upper - ciSmall.lower;
    const widthLarge = ciLarge.upper - ciLarge.lower;
    expect(widthLarge).toBeLessThan(widthSmall);
  });

  it("works with median as the statistic", () => {
    const result = bootstrapCI(data, median, { seed: 42 });
    expect(result.estimate).toBe(11);
    expect(result.lower).toBeLessThan(result.upper);
  });

  it("supports BCa method", () => {
    const result = bootstrapCI(data, mean, { method: "bca", seed: 42 });
    expect(result.method).toBe("bca");
    expect(result.lower).toBeLessThan(result.upper);
  });

  it("wider CI at higher confidence level", () => {
    const ci90 = bootstrapCI(data, mean, { confidence: 0.90, seed: 42 });
    const ci99 = bootstrapCI(data, mean, { confidence: 0.99, seed: 42 });
    expect(ci99.upper - ci99.lower).toBeGreaterThan(ci90.upper - ci90.lower);
  });

  it("seed produces reproducible results", () => {
    const r1 = bootstrapCI(data, mean, { seed: 7 });
    const r2 = bootstrapCI(data, mean, { seed: 7 });
    expect(r1.lower).toBe(r2.lower);
    expect(r1.upper).toBe(r2.upper);
  });

  it("throws for fewer than 2 elements", () => {
    expect(() => bootstrapCI([1], mean)).toThrow();
  });
});

// ── Permutation Test ────────────────────────────────────────────────────

describe("permutationTest", () => {
  it("detects a significant difference between two groups", () => {
    const group1 = [10, 12, 14, 16, 18];
    const group2 = [2, 4, 6, 8, 10];
    const result = permutationTest(group1, group2, { seed: 42 });
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.rejected).toBe(true);
    expect(result.observedStatistic).toBeCloseTo(8, 5);
  });

  it("fails to reject when groups are similar", () => {
    const group1 = [10, 11, 12, 13, 14];
    const group2 = [10, 11, 12, 13, 14];
    const result = permutationTest(group1, group2, { seed: 42 });
    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.rejected).toBe(false);
  });

  it("supports one-sided 'greater' alternative", () => {
    const group1 = [10, 12, 14, 16, 18];
    const group2 = [2, 4, 6, 8, 10];
    const result = permutationTest(group1, group2, {
      alternative: "greater",
      seed: 42,
    });
    expect(result.pValue).toBeLessThan(0.05);
  });

  it("supports one-sided 'less' alternative", () => {
    const group1 = [2, 4, 6, 8, 10];
    const group2 = [10, 12, 14, 16, 18];
    const result = permutationTest(group1, group2, {
      alternative: "less",
      seed: 42,
    });
    expect(result.pValue).toBeLessThan(0.05);
  });

  it("supports custom test statistic", () => {
    const group1 = [1, 2, 3, 10, 100];
    const group2 = [1, 2, 3, 4, 5];
    const medianDiff = (a: number[], b: number[]) => median(a) - median(b);
    const result = permutationTest(group1, group2, {
      statistic: medianDiff,
      seed: 42,
    });
    expect(result.observedStatistic).toBe(0);
    expect(result.pValue).toBeGreaterThan(0.05);
  });

  it("seed produces reproducible results", () => {
    const g1 = [10, 12, 14, 16, 18];
    const g2 = [2, 4, 6, 8, 10];
    const r1 = permutationTest(g1, g2, { seed: 7 });
    const r2 = permutationTest(g1, g2, { seed: 7 });
    expect(r1.pValue).toBe(r2.pValue);
  });

  it("throws when either sample is empty", () => {
    expect(() => permutationTest([], [1, 2])).toThrow();
    expect(() => permutationTest([1, 2], [])).toThrow();
  });
});

describe("oneSamplePermutationTest", () => {
  it("detects data significantly different from hypothesized center", () => {
    const data = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
    const result = oneSamplePermutationTest(data, 0, { seed: 42 });
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.rejected).toBe(true);
  });

  it("fails to reject when data is centered at the hypothesized value", () => {
    const data = [-2, -1, 0, 1, 2, -1.5, 1.5, -0.5, 0.5, 0];
    const result = oneSamplePermutationTest(data, 0, { seed: 42 });
    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.rejected).toBe(false);
  });

  it("throws for empty data", () => {
    expect(() => oneSamplePermutationTest([])).toThrow();
  });
});
