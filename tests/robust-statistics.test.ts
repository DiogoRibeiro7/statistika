import {
  mad,
  trimmedMean,
  winsorizedMean,
  iqr,
  detectOutliers,
  huberMean,
  biweightMidvariance,
} from "../src/robust-statistics";

describe("Robust Statistics", () => {
  const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const withOutliers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100];

  describe("mad", () => {
    it("computes MAD with consistency constant", () => {
      const result = mad(data);
      // median = 5.5, deviations: |x - 5.5|, median of deviations = 2.5
      expect(result).toBeCloseTo(2.5 * 1.4826, 4);
    });

    it("computes raw MAD without constant", () => {
      expect(mad(data, 1)).toBeCloseTo(2.5, 4);
    });

    it("throws on empty dataset", () => {
      expect(() => mad([])).toThrow("must not be empty");
    });
  });

  describe("trimmedMean", () => {
    it("removes extreme observations", () => {
      const result = trimmedMean(withOutliers, 0.1);
      // Remove 1 from each side: [2,3,4,5,6,7,8,9] → mean = 5.5
      expect(result).toBeCloseTo(5.5, 4);
    });

    it("returns regular mean with 0 proportion", () => {
      expect(trimmedMean(data, 0)).toBeCloseTo(5.5, 4);
    });

    it("throws on invalid proportion", () => {
      expect(() => trimmedMean(data, 0.5)).toThrow("between 0");
      expect(() => trimmedMean(data, -0.1)).toThrow("between 0");
    });
  });

  describe("winsorizedMean", () => {
    it("replaces extremes with boundary values", () => {
      const result = winsorizedMean(withOutliers, 0.1);
      // Replace 1→2 and 100→9: [2,2,3,4,5,6,7,8,9,9] → mean = 5.5
      expect(result).toBeCloseTo(5.5, 4);
    });

    it("handles no winsorization", () => {
      expect(winsorizedMean(data, 0)).toBeCloseTo(5.5, 4);
    });
  });

  describe("iqr", () => {
    it("computes interquartile range", () => {
      const result = iqr(data);
      expect(result).toBeGreaterThan(0);
    });

    it("throws on small dataset", () => {
      expect(() => iqr([1, 2, 3])).toThrow("at least 4");
    });
  });

  describe("detectOutliers", () => {
    it("detects outliers with IQR method", () => {
      const result = detectOutliers(withOutliers);
      expect(result.outliers).toContain(100);
      expect(result.indices.length).toBeGreaterThan(0);
    });

    it("returns no outliers for well-behaved data", () => {
      const result = detectOutliers(data);
      expect(result.outliers).toHaveLength(0);
    });
  });

  describe("huberMean", () => {
    it("is more robust than regular mean", () => {
      const regularMean = withOutliers.reduce((a, b) => a + b, 0) / withOutliers.length;
      const robust = huberMean(withOutliers);
      // Huber mean should be closer to the median (5.5) than the arithmetic mean (14.5)
      const med = 5.5;
      expect(Math.abs(robust - med)).toBeLessThan(Math.abs(regularMean - med));
    });

    it("equals mean for symmetric data without outliers", () => {
      const result = huberMean(data);
      expect(result).toBeCloseTo(5.5, 1);
    });
  });

  describe("biweightMidvariance", () => {
    it("returns a positive value for non-constant data", () => {
      const result = biweightMidvariance(data);
      expect(result).toBeGreaterThan(0);
    });

    it("returns 0 for constant data", () => {
      expect(biweightMidvariance([5, 5, 5, 5, 5])).toBe(0);
    });

    it("throws on small dataset", () => {
      expect(() => biweightMidvariance([1])).toThrow("at least 2");
    });
  });
});
