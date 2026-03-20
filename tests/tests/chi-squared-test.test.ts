import {
  chiSquaredGoodnessOfFit,
  chiSquaredIndependence,
} from "../../src/tests/chi-squared-test";

describe("chi-squared tests", () => {
  describe("chiSquaredGoodnessOfFit", () => {
    it("does not reject for matching frequencies", () => {
      const observed = [50, 50, 50, 50];
      const expected = [50, 50, 50, 50];
      const result = chiSquaredGoodnessOfFit(observed, expected);
      expect(result.statistic).toBe(0);
      expect(result.pValue).toBeCloseTo(1, 5);
      expect(result.rejected).toBe(false);
      expect(result.degreesOfFreedom).toBe(3);
    });

    it("rejects for very different frequencies", () => {
      const observed = [90, 10, 10, 10];
      const expected = [30, 30, 30, 30];
      const result = chiSquaredGoodnessOfFit(observed, expected);
      expect(result.rejected).toBe(true);
      expect(result.pValue).toBeLessThan(0.001);
    });

    it("fair die roll test", () => {
      // Reasonably fair die
      const observed = [18, 16, 15, 17, 14, 20];
      const expected = [100 / 6, 100 / 6, 100 / 6, 100 / 6, 100 / 6, 100 / 6];
      const result = chiSquaredGoodnessOfFit(observed, expected);
      expect(result.rejected).toBe(false);
      expect(result.degreesOfFreedom).toBe(5);
    });

    it("throws for mismatched lengths", () => {
      expect(() => chiSquaredGoodnessOfFit([1, 2], [1])).toThrow();
    });

    it("throws for non-positive expected", () => {
      expect(() => chiSquaredGoodnessOfFit([1, 2], [0, 2])).toThrow();
    });
  });

  describe("chiSquaredIndependence", () => {
    it("detects independence (no association)", () => {
      // Perfectly proportional table
      const table = [
        [10, 20],
        [10, 20],
      ];
      const result = chiSquaredIndependence(table);
      expect(result.statistic).toBeCloseTo(0, 8);
      expect(result.rejected).toBe(false);
      expect(result.degreesOfFreedom).toBe(1);
    });

    it("detects strong association", () => {
      const table = [
        [50, 5],
        [5, 50],
      ];
      const result = chiSquaredIndependence(table);
      expect(result.rejected).toBe(true);
      expect(result.pValue).toBeLessThan(0.001);
    });

    it("handles 3x3 table", () => {
      const table = [
        [10, 20, 30],
        [20, 15, 25],
        [30, 25, 15],
      ];
      const result = chiSquaredIndependence(table);
      expect(result.degreesOfFreedom).toBe(4); // (3-1)*(3-1)
    });

    it("throws for invalid table dimensions", () => {
      expect(() => chiSquaredIndependence([[1, 2]])).toThrow();
    });
  });
});
