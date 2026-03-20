import {
  huberRegression,
  ransacRegression,
} from "../../src/models/robust-regression";

describe("Robust Regression", () => {
  // ---- Huber Regression ----
  describe("huberRegression", () => {
    it("recovers slope and intercept for clean data", () => {
      // y = 2x + 3
      const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const y = x.map((xi) => 2 * xi + 3);

      const result = huberRegression(x, y);
      expect(result.slopes[0]).toBeCloseTo(2, 2);
      expect(result.intercept).toBeCloseTo(3, 2);
    });

    it("is robust to outliers", () => {
      // y = x + noise, with one extreme outlier
      const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const y = [1.1, 2.0, 3.1, 3.9, 5.1, 6.0, 7.2, 7.9, 9.0, 100]; // last is outlier

      const result = huberRegression(x, y);
      // Should still get roughly slope ~1, not pulled to outlier
      expect(result.slopes[0]).toBeCloseTo(1, 0);
    });

    it("predict works for simple regression", () => {
      const x = [1, 2, 3, 4, 5];
      const y = [3, 5, 7, 9, 11]; // y = 2x + 1
      const result = huberRegression(x, y);
      expect(result.predict(6)).toBeCloseTo(13, 0);
    });

    it("works with multiple predictors", () => {
      const X = [
        [1, 0],
        [2, 1],
        [3, 0],
        [4, 1],
        [5, 0],
        [6, 1],
      ];
      const y = [2, 4, 5, 7, 8, 10];

      const result = huberRegression(X, y);
      expect(result.slopes.length).toBe(2);
      expect(typeof result.predict([3, 0])).toBe("number");
    });

    it("returns coefficients array with intercept first", () => {
      const x = [1, 2, 3, 4, 5];
      const y = [2, 4, 6, 8, 10];
      const result = huberRegression(x, y);
      expect(result.coefficients.length).toBe(2);
      expect(result.coefficients[0]).toBeCloseTo(result.intercept, 8);
    });

    it("returns positive scale estimate", () => {
      const x = [1, 2, 3, 4, 5, 6, 7, 8];
      const y = x.map((xi) => xi + (Math.sin(xi) * 0.5));
      const result = huberRegression(x, y);
      expect(result.scale).toBeGreaterThan(0);
    });

    it("throws on mismatched lengths", () => {
      expect(() => huberRegression([1, 2, 3], [1, 2])).toThrow();
    });

    it("throws on too few observations", () => {
      expect(() => huberRegression([1], [1])).toThrow();
    });

    it("throws on non-positive delta", () => {
      expect(() => huberRegression([1, 2, 3], [1, 2, 3], 0)).toThrow();
    });
  });

  // ---- RANSAC Regression ----
  describe("ransacRegression", () => {
    it("recovers slope and intercept with outliers", () => {
      // y = 2x + 1, with 3 extreme outliers
      const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
      const y = x.map((xi) => 2 * xi + 1);
      // Add outliers
      y[5] = 500;
      y[10] = -500;
      y[13] = 300;

      // Use seeded random for reproducibility
      let seed = 42;
      const random = () => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
      };

      const result = ransacRegression(x, y, 5, 200, undefined, random);
      expect(result.slopes[0]).toBeCloseTo(2, 0);
      expect(result.intercept).toBeCloseTo(1, 0);
      // Most points should be inliers
      expect(result.nInliers).toBeGreaterThan(10);
    });

    it("identifies inlier indices", () => {
      const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 100]; // outlier at 10

      let seed = 7;
      const random = () => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
      };

      const result = ransacRegression(x, y, 3, 100, undefined, random);
      // Index 9 (the outlier) should not be an inlier
      expect(result.inlierIndices).not.toContain(9);
    });

    it("predict works", () => {
      const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const y = x.map((xi) => 3 * xi + 2);

      let seed = 42;
      const random = () => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
      };

      const result = ransacRegression(x, y, undefined, 50, undefined, random);
      expect(result.predict(11)).toBeCloseTo(35, 0);
    });

    it("works with multiple predictors", () => {
      const X = [
        [1, 0],
        [2, 1],
        [3, 0],
        [4, 1],
        [5, 0],
        [6, 1],
        [7, 0],
        [8, 1],
      ];
      const y = [1, 3, 4, 6, 7, 9, 10, 12];

      let seed = 42;
      const random = () => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
      };

      const result = ransacRegression(X, y, undefined, 50, undefined, random);
      expect(result.slopes.length).toBe(2);
    });

    it("throws on mismatched lengths", () => {
      expect(() => ransacRegression([1, 2, 3], [1, 2])).toThrow();
    });

    it("throws on too few observations", () => {
      expect(() =>
        ransacRegression([1], [1], 1, 10, 2),
      ).toThrow();
    });
  });
});
