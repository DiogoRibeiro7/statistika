import { ksTest, ksTwoSampleTest } from "../../src/tests/ks-test";

describe("Kolmogorov-Smirnov tests", () => {
  describe("ksTest (one-sample)", () => {
    it("does not reject for data matching the CDF", () => {
      // Uniform(0,1) data tested against uniform CDF
      const data = Array.from({ length: 100 }, (_, i) => (i + 0.5) / 100);
      const uniformCdf = (x: number) => Math.max(0, Math.min(1, x));
      const result = ksTest(data, uniformCdf);
      expect(result.rejected).toBe(false);
      expect(result.statistic).toBeLessThan(0.1);
    });

    it("rejects for data not matching the CDF", () => {
      // Data concentrated near 0, tested against uniform CDF
      const data = Array.from({ length: 50 }, (_, i) => i / 500);
      const uniformCdf = (x: number) => Math.max(0, Math.min(1, x));
      const result = ksTest(data, uniformCdf);
      expect(result.rejected).toBe(true);
      expect(result.statistic).toBeGreaterThan(0.5);
    });

    it("statistic is between 0 and 1", () => {
      const data = [0.1, 0.3, 0.5, 0.7, 0.9];
      const cdf = (x: number) => Math.max(0, Math.min(1, x));
      const result = ksTest(data, cdf);
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.statistic).toBeLessThanOrEqual(1);
    });

    it("standard normal test", () => {
      // Generate data from approx standard normal
      const normalCdf = (x: number) =>
        0.5 * (1 + erf(x / Math.SQRT2));
      // Evenly spaced quantiles of normal (should pass)
      const data = [-1.645, -0.674, 0, 0.674, 1.645];
      const result = ksTest(data, normalCdf);
      // Small sample, so D might be biggish but shouldn't reject
      expect(result.pValue).toBeGreaterThan(0.01);
    });
  });

  describe("ksTwoSampleTest", () => {
    it("does not reject for samples from the same distribution", () => {
      const a = Array.from({ length: 50 }, (_, i) => (i + 0.5) / 50);
      const b = Array.from({ length: 50 }, (_, i) => (i + 0.3) / 50);
      const result = ksTwoSampleTest(a, b);
      expect(result.rejected).toBe(false);
    });

    it("rejects for samples from very different distributions", () => {
      const a = Array.from({ length: 50 }, (_, i) => i / 50);
      const b = Array.from({ length: 50 }, (_, i) => 10 + i / 50);
      const result = ksTwoSampleTest(a, b);
      expect(result.rejected).toBe(true);
      expect(result.statistic).toBeCloseTo(1, 1);
    });

    it("statistic is symmetric", () => {
      const a = [1, 2, 3, 4, 5];
      const b = [3, 4, 5, 6, 7];
      const r1 = ksTwoSampleTest(a, b);
      const r2 = ksTwoSampleTest(b, a);
      expect(r1.statistic).toBeCloseTo(r2.statistic, 10);
    });
  });
});

// Simple erf for test use
function erf(x: number): number {
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const poly =
    t * (0.254829592 + t * (-0.284496736 +
      t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1 - poly * Math.exp(-x * x));
}
