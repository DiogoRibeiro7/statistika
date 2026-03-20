import {
  oneSampleTTest,
  twoSampleTTest,
  welchTTest,
  pairedTTest,
} from "../../src/tests/t-test";

describe("t-tests", () => {
  describe("oneSampleTTest", () => {
    it("does not reject when mean equals mu0", () => {
      // Data with mean ~5
      const data = [4.8, 5.1, 5.0, 4.9, 5.2, 5.0, 4.8, 5.1];
      const result = oneSampleTTest(data, 5.0);
      expect(result.rejected).toBe(false);
      expect(result.pValue).toBeGreaterThan(0.05);
      expect(result.degreesOfFreedom).toBe(7);
    });

    it("rejects when mean differs significantly from mu0", () => {
      const data = [10, 11, 12, 10, 11, 12, 10, 11, 12, 11];
      const result = oneSampleTTest(data, 0);
      expect(result.rejected).toBe(true);
      expect(result.pValue).toBeLessThan(0.001);
    });

    it("statistic is positive when mean > mu0", () => {
      const data = [10, 12, 14, 11, 13];
      const result = oneSampleTTest(data, 5);
      expect(result.statistic).toBeGreaterThan(0);
    });

    it("throws for insufficient data", () => {
      expect(() => oneSampleTTest([5])).toThrow();
    });
  });

  describe("twoSampleTTest", () => {
    it("does not reject when samples come from same distribution", () => {
      const a = [5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.9];
      const b = [5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1];
      const result = twoSampleTTest(a, b);
      expect(result.rejected).toBe(false);
      expect(result.degreesOfFreedom).toBe(14);
    });

    it("rejects when means differ significantly", () => {
      const a = [10, 11, 12, 10, 11, 12];
      const b = [1, 2, 3, 1, 2, 3];
      const result = twoSampleTTest(a, b);
      expect(result.rejected).toBe(true);
      expect(result.pValue).toBeLessThan(0.001);
    });
  });

  describe("welchTTest", () => {
    it("handles unequal variances", () => {
      const a = [10, 11, 12, 10, 11, 12, 11, 10];
      const b = [1, 2, 3, 1, 2, 3, 2, 1];
      const result = welchTTest(a, b);
      expect(result.rejected).toBe(true);
      expect(result.pValue).toBeLessThan(0.001);
    });

    it("Welch df differs from pooled df", () => {
      const a = [10, 11, 12, 10, 11]; // smaller variance
      const b = [1, 5, 9, 2, 8, 4, 6]; // larger variance
      const welch = welchTTest(a, b);
      const pooled = twoSampleTTest(a, b);
      // Welch df is typically not an integer and differs from n1+n2-2
      expect(welch.degreesOfFreedom).not.toBeCloseTo(pooled.degreesOfFreedom, 0);
    });
  });

  describe("pairedTTest", () => {
    it("does not reject when differences are ~0", () => {
      const before = [5, 6, 7, 8, 9];
      const after = [5.1, 5.9, 7.1, 7.9, 9.1];
      const result = pairedTTest(before, after);
      expect(result.rejected).toBe(false);
    });

    it("rejects when there is a consistent shift", () => {
      const before = [5, 6, 7, 8, 9, 10, 11, 12];
      const after = [15, 16, 17, 18, 19, 20, 21, 22];
      const result = pairedTTest(before, after);
      expect(result.rejected).toBe(true);
      expect(result.pValue).toBeLessThan(0.001);
    });

    it("throws for unequal length arrays", () => {
      expect(() => pairedTTest([1, 2], [1, 2, 3])).toThrow();
    });
  });
});
