import {
  analyzeMissing,
  listwiseDeletion,
  pairwiseDeletion,
  meanImputation,
  medianImputation,
  modeImputation,
  linearInterpolation,
  forwardFill,
} from "../src/missing-data";

describe("Missing Data Handling", () => {
  describe("analyzeMissing", () => {
    it("detects null and undefined values", () => {
      const result = analyzeMissing([1, null, 3, undefined, 5]);
      expect(result.missingCount).toBe(2);
      expect(result.completeCount).toBe(3);
      expect(result.missingIndices).toEqual([1, 3]);
      expect(result.missingProportion).toBeCloseTo(0.4);
    });

    it("detects NaN values", () => {
      const result = analyzeMissing([1, NaN, 3]);
      expect(result.missingCount).toBe(1);
    });

    it("handles no missing data", () => {
      const result = analyzeMissing([1, 2, 3]);
      expect(result.missingCount).toBe(0);
    });
  });

  describe("listwiseDeletion", () => {
    it("removes rows with any missing values", () => {
      const a = [1, null, 3, 4];
      const b = [10, 20, null, 40];
      const [ra, rb] = listwiseDeletion(a, b);
      expect(ra).toEqual([1, 4]);
      expect(rb).toEqual([10, 40]);
    });

    it("keeps all rows when no missing", () => {
      const [ra] = listwiseDeletion([1, 2, 3]);
      expect(ra).toEqual([1, 2, 3]);
    });
  });

  describe("pairwiseDeletion", () => {
    it("keeps only complete pairs", () => {
      const result = pairwiseDeletion([1, null, 3, 4], [10, 20, null, 40]);
      expect(result.a).toEqual([1, 4]);
      expect(result.b).toEqual([10, 40]);
      expect(result.indices).toEqual([0, 3]);
    });
  });

  describe("meanImputation", () => {
    it("fills missing with mean", () => {
      const result = meanImputation([1, null, 3, null, 5]);
      // mean of [1,3,5] = 3
      expect(result).toEqual([1, 3, 3, 3, 5]);
    });

    it("throws when all missing", () => {
      expect(() => meanImputation([null, null])).toThrow("No observed");
    });
  });

  describe("medianImputation", () => {
    it("fills missing with median", () => {
      const result = medianImputation([1, null, 3, null, 10]);
      // median of [1,3,10] = 3
      expect(result).toEqual([1, 3, 3, 3, 10]);
    });
  });

  describe("modeImputation", () => {
    it("fills missing with mode", () => {
      const result = modeImputation([1, 1, null, 3, null]);
      // mode = 1 (most frequent)
      expect(result).toEqual([1, 1, 1, 3, 1]);
    });
  });

  describe("linearInterpolation", () => {
    it("interpolates interior missing values", () => {
      const result = linearInterpolation([0, null, 4, null, null, 10]);
      expect(result[0]).toBeCloseTo(0);
      expect(result[1]).toBeCloseTo(2); // midpoint of 0 and 4
      expect(result[2]).toBeCloseTo(4);
      expect(result[3]).toBeCloseTo(6); // 4 + 1/3 * 6
      expect(result[4]).toBeCloseTo(8); // 4 + 2/3 * 6
      expect(result[5]).toBeCloseTo(10);
    });

    it("forward-fills leading missing values", () => {
      const result = linearInterpolation([null, null, 5, 10]);
      expect(result[0]).toBe(5);
      expect(result[1]).toBe(5);
    });

    it("backward-fills trailing missing values", () => {
      const result = linearInterpolation([5, 10, null, null]);
      expect(result[2]).toBe(10);
      expect(result[3]).toBe(10);
    });
  });

  describe("forwardFill", () => {
    it("carries last observation forward", () => {
      const result = forwardFill([1, null, null, 4, null]);
      expect(result).toEqual([1, 1, 1, 4, 4]);
    });

    it("fills leading missing with first observed", () => {
      const result = forwardFill([null, null, 3, null, 5]);
      expect(result).toEqual([3, 3, 3, 3, 5]);
    });

    it("throws when all values are missing", () => {
      expect(() => forwardFill([null, undefined, null])).toThrow("all values are missing");
    });

    it("returns empty array for empty input", () => {
      expect(forwardFill([])).toEqual([]);
    });

    it("handles NaN as missing", () => {
      const result = forwardFill([1, NaN, 3]);
      expect(result).toEqual([1, 1, 3]);
    });

    it("single observed value fills everything", () => {
      const result = forwardFill([null, null, 5, null, null]);
      expect(result).toEqual([5, 5, 5, 5, 5]);
    });
  });

  describe("analyzeMissing edge cases", () => {
    it("handles empty array", () => {
      const result = analyzeMissing([]);
      expect(result.totalValues).toBe(0);
      expect(result.missingCount).toBe(0);
      expect(result.missingProportion).toBe(0);
    });

    it("handles all missing", () => {
      const result = analyzeMissing([null, undefined, NaN]);
      expect(result.missingCount).toBe(3);
      expect(result.completeCount).toBe(0);
      expect(result.missingProportion).toBeCloseTo(1);
    });

    it("totalValues matches input length", () => {
      const result = analyzeMissing([1, null, 3, undefined, 5]);
      expect(result.totalValues).toBe(5);
    });
  });

  describe("listwiseDeletion edge cases", () => {
    it("throws on no columns", () => {
      expect(() => listwiseDeletion()).toThrow("at least one");
    });

    it("throws on mismatched column lengths", () => {
      expect(() => listwiseDeletion([1, 2], [1])).toThrow("same length");
    });

    it("returns empty arrays when all rows have missing values", () => {
      const [ra, rb] = listwiseDeletion([null, 1], [1, null]);
      expect(ra).toEqual([]);
      expect(rb).toEqual([]);
    });

    it("handles NaN as missing", () => {
      const [ra] = listwiseDeletion([1, NaN, 3]);
      expect(ra).toEqual([1, 3]);
    });

    it("handles single column", () => {
      const [ra] = listwiseDeletion([1, null, 3, 4]);
      expect(ra).toEqual([1, 3, 4]);
    });
  });

  describe("pairwiseDeletion edge cases", () => {
    it("throws on mismatched lengths", () => {
      expect(() => pairwiseDeletion([1, 2], [1])).toThrow("same length");
    });

    it("returns empty arrays when no complete pairs", () => {
      const result = pairwiseDeletion([null, 1], [1, null]);
      expect(result.a).toEqual([]);
      expect(result.b).toEqual([]);
      expect(result.indices).toEqual([]);
    });

    it("handles NaN as missing", () => {
      const result = pairwiseDeletion([1, NaN, 3], [4, 5, NaN]);
      expect(result.a).toEqual([1]);
      expect(result.b).toEqual([4]);
      expect(result.indices).toEqual([0]);
    });

    it("keeps all when no missing", () => {
      const result = pairwiseDeletion([1, 2, 3], [4, 5, 6]);
      expect(result.a).toEqual([1, 2, 3]);
      expect(result.b).toEqual([4, 5, 6]);
    });
  });

  describe("meanImputation edge cases", () => {
    it("handles NaN as missing", () => {
      const result = meanImputation([2, NaN, 4]);
      expect(result).toEqual([2, 3, 4]);
    });

    it("handles undefined as missing", () => {
      const result = meanImputation([2, undefined, 4]);
      expect(result).toEqual([2, 3, 4]);
    });

    it("no missing values returns copy of original", () => {
      const result = meanImputation([1, 2, 3]);
      expect(result).toEqual([1, 2, 3]);
    });
  });

  describe("medianImputation edge cases", () => {
    it("throws when all missing", () => {
      expect(() => medianImputation([null, null])).toThrow("No observed");
    });

    it("handles even number of observed values", () => {
      const result = medianImputation([1, null, 2, null]);
      // median of [1,2] = 1.5
      expect(result).toEqual([1, 1.5, 2, 1.5]);
    });

    it("no missing values returns copy of original", () => {
      const result = medianImputation([5, 10, 15]);
      expect(result).toEqual([5, 10, 15]);
    });
  });

  describe("modeImputation edge cases", () => {
    it("throws when all missing", () => {
      expect(() => modeImputation([null, undefined])).toThrow("No observed");
    });

    it("uses first most frequent if tied", () => {
      const result = modeImputation([1, 2, null]);
      // Both 1 and 2 appear once; mode takes whichever has higher count first
      expect([1, 2]).toContain(result[2]);
    });

    it("no missing returns copy of original", () => {
      const result = modeImputation([3, 3, 5]);
      expect(result).toEqual([3, 3, 5]);
    });
  });

  describe("linearInterpolation edge cases", () => {
    it("throws when all missing", () => {
      expect(() => linearInterpolation([null, null, null])).toThrow("No observed");
    });

    it("returns empty array for empty input", () => {
      expect(linearInterpolation([])).toEqual([]);
    });

    it("single observed value fills everything", () => {
      const result = linearInterpolation([null, 5, null]);
      expect(result).toEqual([5, 5, 5]);
    });

    it("handles no missing values", () => {
      const result = linearInterpolation([1, 2, 3]);
      expect(result).toEqual([1, 2, 3]);
    });

    it("handles undefined as missing", () => {
      const result = linearInterpolation([1, undefined, 3]);
      expect(result[1]).toBeCloseTo(2);
    });

    it("handles NaN as missing", () => {
      const result = linearInterpolation([0, NaN, 10]);
      expect(result[1]).toBeCloseTo(5);
    });
  });
});
