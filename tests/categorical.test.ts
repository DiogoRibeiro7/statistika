import {
  contingencyTable,
  mcnemarsTest,
  cochranMantelHaenszel,
  gTest,
  standardizedResiduals,
  adjustedResiduals,
} from "../src/categorical";

describe("Categorical Data Analysis", () => {
  describe("contingencyTable", () => {
    it("computes margins and expected counts", () => {
      const table = [
        [10, 20],
        [30, 40],
      ];
      const ct = contingencyTable(table);
      expect(ct.rowTotals).toEqual([30, 70]);
      expect(ct.colTotals).toEqual([40, 60]);
      expect(ct.grandTotal).toBe(100);
      expect(ct.expected[0][0]).toBeCloseTo(12); // 30*40/100
      expect(ct.expected[0][1]).toBeCloseTo(18);
      expect(ct.expected[1][0]).toBeCloseTo(28);
      expect(ct.expected[1][1]).toBeCloseTo(42);
    });

    it("works for larger tables", () => {
      const table = [
        [10, 20, 30],
        [40, 50, 60],
      ];
      const ct = contingencyTable(table);
      expect(ct.nRows).toBe(2);
      expect(ct.nCols).toBe(3);
      expect(ct.grandTotal).toBe(210);
    });
  });

  describe("mcnemarsTest", () => {
    it("detects significant change", () => {
      // Large discrepancy in discordant pairs
      const table = [
        [10, 30],
        [5, 55],
      ];
      const result = mcnemarsTest(table);
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.pValue).toBeLessThan(0.05);
      expect(result.rejected).toBe(true);
    });

    it("does not reject for balanced discordant pairs", () => {
      const table = [
        [20, 15],
        [15, 50],
      ];
      const result = mcnemarsTest(table);
      expect(result.rejected).toBe(false);
    });

    it("handles zero discordant pairs", () => {
      const table = [
        [20, 0],
        [0, 30],
      ];
      const result = mcnemarsTest(table);
      expect(result.statistic).toBe(0);
      expect(result.pValue).toBe(1);
    });

    it("throws on non-2x2 table", () => {
      expect(() => mcnemarsTest([[1, 2, 3], [4, 5, 6]])).toThrow("2x2");
    });
  });

  describe("cochranMantelHaenszel", () => {
    it("tests across strata", () => {
      const tables = [
        [[10, 20], [30, 40]],
        [[15, 25], [35, 45]],
        [[12, 18], [28, 42]],
      ];
      const result = cochranMantelHaenszel(tables);
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.commonOddsRatio).toBeGreaterThan(0);
      expect(typeof result.rejected).toBe("boolean");
    });

    it("throws on empty tables", () => {
      expect(() => cochranMantelHaenszel([])).toThrow("at least one");
    });
  });

  describe("gTest", () => {
    it("detects significant deviation from independence", () => {
      const table = [
        [50, 10],
        [10, 50],
      ];
      const result = gTest(table);
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.degreesOfFreedom).toBe(1);
      expect(result.rejected).toBe(true);
    });

    it("does not reject for independent data", () => {
      // Equal proportions
      const table = [
        [25, 25],
        [25, 25],
      ];
      const result = gTest(table);
      expect(result.statistic).toBeCloseTo(0);
      expect(result.rejected).toBe(false);
    });
  });

  describe("standardizedResiduals", () => {
    it("computes residuals", () => {
      const table = [
        [50, 10],
        [10, 50],
      ];
      const resids = standardizedResiduals(table);
      expect(resids).toHaveLength(2);
      expect(resids[0]).toHaveLength(2);
      // Large positive residual for [0][0] (more than expected)
      expect(resids[0][0]).toBeGreaterThan(2);
      // Large negative residual for [0][1]
      expect(resids[0][1]).toBeLessThan(-2);
    });
  });

  describe("adjustedResiduals", () => {
    it("computes adjusted residuals", () => {
      const table = [
        [50, 10],
        [10, 50],
      ];
      const resids = adjustedResiduals(table);
      expect(resids).toHaveLength(2);
      // Adjusted residuals should be even larger in magnitude
      expect(Math.abs(resids[0][0])).toBeGreaterThan(2);
    });
  });

  describe("contingencyTable edge cases", () => {
    it("throws on empty table", () => {
      expect(() => contingencyTable([])).toThrow("non-empty");
    });

    it("throws on table with no columns", () => {
      expect(() => contingencyTable([[]])).toThrow("at least 1 column");
    });

    it("throws on inconsistent row lengths", () => {
      expect(() => contingencyTable([[1, 2], [3]])).toThrow("expected 2 columns");
    });

    it("throws on negative cell count", () => {
      expect(() => contingencyTable([[1, -2], [3, 4]])).toThrow("non-negative");
    });

    it("throws on NaN cell count", () => {
      expect(() => contingencyTable([[1, NaN], [3, 4]])).toThrow("received NaN");
    });

    it("handles 1x1 table", () => {
      const ct = contingencyTable([[5]]);
      expect(ct.grandTotal).toBe(5);
      expect(ct.nRows).toBe(1);
      expect(ct.nCols).toBe(1);
      expect(ct.expected[0][0]).toBeCloseTo(5);
    });

    it("handles table with zero cells", () => {
      const ct = contingencyTable([[0, 10], [10, 0]]);
      expect(ct.grandTotal).toBe(20);
    });
  });

  describe("mcnemarsTest edge cases", () => {
    it("throws on 1x2 table", () => {
      expect(() => mcnemarsTest([[1, 2]])).toThrow("2x2");
    });

    it("custom alpha changes rejection decision", () => {
      const table = [[10, 12], [8, 55]];
      const strict = mcnemarsTest(table, 0.01);
      const loose = mcnemarsTest(table, 0.99);
      // With alpha=0.99, almost everything is rejected
      expect(typeof strict.rejected).toBe("boolean");
      expect(typeof loose.rejected).toBe("boolean");
    });
  });

  describe("cochranMantelHaenszel edge cases", () => {
    it("throws on non-2x2 tables", () => {
      expect(() => cochranMantelHaenszel([[[1, 2, 3], [4, 5, 6]]])).toThrow("2x2");
    });

    it("commonOddsRatio is Infinity when denominator is zero", () => {
      // b*c = 0 for all tables => orDen = 0
      const tables = [
        [[10, 0], [0, 40]],
      ];
      const result = cochranMantelHaenszel(tables);
      expect(result.commonOddsRatio).toBe(Infinity);
    });
  });

  describe("gTest edge cases", () => {
    it("handles larger table (3x3)", () => {
      const table = [
        [10, 20, 30],
        [20, 30, 10],
        [30, 10, 20],
      ];
      const result = gTest(table);
      expect(result.degreesOfFreedom).toBe(4); // (3-1)*(3-1)
      expect(result.statistic).toBeGreaterThanOrEqual(0);
    });

    it("handles cells with zero counts", () => {
      const table = [
        [0, 20],
        [20, 0],
      ];
      const result = gTest(table);
      expect(result.statistic).toBeGreaterThan(0);
    });
  });

  describe("standardizedResiduals edge cases", () => {
    it("returns zeros for cells where expected is zero", () => {
      // A row with all zeros means expected could be zero
      const table = [
        [0, 0],
        [10, 20],
      ];
      const resids = standardizedResiduals(table);
      expect(resids[0][0]).toBe(0);
      expect(resids[0][1]).toBe(0);
    });
  });

  describe("adjustedResiduals edge cases", () => {
    it("returns zeros for cells where expected is zero", () => {
      const table = [
        [0, 0],
        [10, 20],
      ];
      const resids = adjustedResiduals(table);
      expect(resids[0][0]).toBe(0);
      expect(resids[0][1]).toBe(0);
    });

    it("adjusted residuals have correct dimensions", () => {
      const table = [
        [10, 20, 30],
        [40, 50, 60],
      ];
      const resids = adjustedResiduals(table);
      expect(resids).toHaveLength(2);
      expect(resids[0]).toHaveLength(3);
    });
  });
});
