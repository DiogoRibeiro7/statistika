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
  });
});
