import {
  bonferroni,
  sidak,
  holm,
  hochberg,
  benjaminiHochberg,
  benjaminiYekutieli,
} from "../src/multiple-testing";

describe("Multiple Testing Corrections", () => {
  const pValues = [0.01, 0.04, 0.03, 0.005, 0.5];

  describe("bonferroni", () => {
    it("multiplies p-values by number of tests", () => {
      const result = bonferroni(pValues);
      expect(result.adjustedPValues[0]).toBeCloseTo(0.05); // 0.01 * 5
      expect(result.adjustedPValues[1]).toBeCloseTo(0.2); // 0.04 * 5
      expect(result.adjustedPValues[3]).toBeCloseTo(0.025); // 0.005 * 5
      expect(result.method).toBe("Bonferroni");
    });

    it("caps adjusted p-values at 1", () => {
      const result = bonferroni(pValues);
      expect(result.adjustedPValues[4]).toBe(1); // 0.5 * 5 = 2.5 → 1
    });

    it("rejects correctly at alpha=0.05", () => {
      const result = bonferroni(pValues, 0.05);
      expect(result.rejected).toEqual([false, false, false, true, false]);
    });
  });

  describe("sidak", () => {
    it("adjusts using 1-(1-p)^m", () => {
      const result = sidak([0.01, 0.04], 0.05);
      expect(result.adjustedPValues[0]).toBeCloseTo(1 - Math.pow(0.99, 2));
      expect(result.adjustedPValues[1]).toBeCloseTo(1 - Math.pow(0.96, 2));
      expect(result.method).toBe("Šidák");
    });
  });

  describe("holm", () => {
    it("applies step-down correction", () => {
      const result = holm(pValues);
      // Sorted: 0.005(idx3), 0.01(idx0), 0.03(idx2), 0.04(idx1), 0.5(idx4)
      // Adjusted: 0.005*5=0.025, max(0.025, 0.01*4=0.04), max(0.04, 0.03*3=0.09), ...
      expect(result.adjustedPValues[3]).toBeCloseTo(0.025);
      expect(result.adjustedPValues[0]).toBeCloseTo(0.04);
      expect(result.method).toBe("Holm-Bonferroni");
    });

    it("enforces monotonicity", () => {
      const result = holm(pValues);
      // Adjusted p-values when sorted by original rank should be non-decreasing
      const indexed = pValues.map((p, i) => ({ p, i }));
      indexed.sort((a, b) => a.p - b.p);
      const sortedAdj = indexed.map((item) => result.adjustedPValues[item.i]);
      for (let i = 1; i < sortedAdj.length; i++) {
        expect(sortedAdj[i]).toBeGreaterThanOrEqual(sortedAdj[i - 1]);
      }
    });
  });

  describe("hochberg", () => {
    it("applies step-up correction", () => {
      const result = hochberg(pValues);
      expect(result.method).toBe("Hochberg");
      // Should be less conservative or equal to Holm
      const holmResult = holm(pValues);
      for (let i = 0; i < pValues.length; i++) {
        expect(result.adjustedPValues[i]).toBeLessThanOrEqual(holmResult.adjustedPValues[i] + 1e-10);
      }
    });
  });

  describe("benjaminiHochberg", () => {
    it("controls FDR", () => {
      const result = benjaminiHochberg(pValues);
      expect(result.method).toBe("Benjamini-Hochberg");
      // BH is generally less conservative than Bonferroni
      const bonfResult = bonferroni(pValues);
      for (let i = 0; i < pValues.length; i++) {
        expect(result.adjustedPValues[i]).toBeLessThanOrEqual(bonfResult.adjustedPValues[i] + 1e-10);
      }
    });

    it("produces correct adjusted p-values", () => {
      const result = benjaminiHochberg([0.01, 0.04, 0.03, 0.005]);
      // Sorted: 0.005(r1), 0.01(r2), 0.03(r3), 0.04(r4)
      // Raw adj: 0.005*4/1=0.02, 0.01*4/2=0.02, 0.03*4/3=0.04, 0.04*4/4=0.04
      // Enforced: from right: min(0.04, 0.04)=0.04, min(0.04, 0.04)=0.04, min(0.04, 0.02)=0.02, min(0.02, 0.02)=0.02
      expect(result.adjustedPValues[3]).toBeCloseTo(0.02); // idx3 = original 0.005
      expect(result.adjustedPValues[0]).toBeCloseTo(0.02); // idx0 = original 0.01
    });
  });

  describe("benjaminiYekutieli", () => {
    it("is more conservative than BH", () => {
      const bh = benjaminiHochberg(pValues);
      const by = benjaminiYekutieli(pValues);
      expect(by.method).toBe("Benjamini-Yekutieli");
      for (let i = 0; i < pValues.length; i++) {
        expect(by.adjustedPValues[i]).toBeGreaterThanOrEqual(bh.adjustedPValues[i] - 1e-10);
      }
    });
  });

  describe("validation", () => {
    it("throws on empty array", () => {
      expect(() => bonferroni([])).toThrow("must not be empty");
    });

    it("throws on invalid p-values", () => {
      expect(() => bonferroni([-0.1, 0.5])).toThrow("between 0 and 1");
      expect(() => bonferroni([0.5, 1.1])).toThrow("between 0 and 1");
    });
  });
});
