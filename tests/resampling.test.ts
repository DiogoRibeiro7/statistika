import {
  kFoldCV,
  loocv,
  jackknife,
  stratifiedSample,
  mse,
  rmse,
  mae,
  r2Score,
} from "../src/resampling";
import { mean } from "../src/utils/descriptive";

describe("Resampling & Cross-Validation", () => {
  // Simple linear model: y = 2*x + 1
  const X = Array.from({ length: 20 }, (_, i) => [i]);
  const y = X.map(([x]) => 2 * x + 1 + (x % 3) * 0.1);

  // Simple fit-predict using mean (baseline)
  const fitPredictMean = (trainX: number[][], trainY: number[], testX: number[][]) => {
    const m = mean(trainY);
    return testX.map(() => m);
  };

  describe("kFoldCV", () => {
    it("performs k-fold cross-validation", () => {
      const result = kFoldCV(X, y, fitPredictMean, { k: 5, seed: 42 });
      expect(result.nFolds).toBe(5);
      expect(result.foldScores).toHaveLength(5);
      expect(result.meanScore).toBeGreaterThan(0);
      expect(result.stdScore).toBeGreaterThanOrEqual(0);
    });

    it("throws on invalid k", () => {
      expect(() => kFoldCV(X, y, fitPredictMean, { k: 1 })).toThrow("at least 2");
      expect(() => kFoldCV(X, y, fitPredictMean, { k: 100 })).toThrow("cannot exceed");
    });
  });

  describe("loocv", () => {
    it("uses n folds", () => {
      const smallX = [[1], [2], [3], [4], [5]];
      const smallY = [3, 5, 7, 9, 11];
      const result = loocv(smallX, smallY, fitPredictMean);
      expect(result.nFolds).toBe(5);
    });
  });

  describe("jackknife", () => {
    it("estimates bias and standard error", () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const result = jackknife(data, mean);
      expect(result.estimate).toBeCloseTo(5.5);
      expect(result.bias).toBeCloseTo(0, 5); // Mean is unbiased
      expect(result.standardError).toBeGreaterThan(0);
      expect(result.pseudoValues).toHaveLength(10);
    });

    it("throws on single observation", () => {
      expect(() => jackknife([1], mean)).toThrow("at least 2");
    });
  });

  describe("stratifiedSample", () => {
    it("maintains stratum proportions", () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const strata = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]; // 50/50 split
      const result = stratifiedSample(data, strata, 6, 42);
      expect(result.sample.length).toBeGreaterThan(0);
      expect(result.sample.length).toBeLessThanOrEqual(6);
    });

    it("throws on mismatched lengths", () => {
      expect(() => stratifiedSample([1, 2], [0], 1)).toThrow("same length");
    });
  });

  describe("scoring functions", () => {
    const actual = [1, 2, 3, 4, 5];
    const predicted = [1.1, 2.2, 2.8, 4.1, 4.9];

    it("computes MSE", () => {
      const result = mse(actual, predicted);
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThan(1);
    });

    it("computes RMSE", () => {
      expect(rmse(actual, predicted)).toBeCloseTo(Math.sqrt(mse(actual, predicted)));
    });

    it("computes MAE", () => {
      const result = mae(actual, predicted);
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThan(1);
    });

    it("computes R²", () => {
      const result = r2Score(actual, predicted);
      expect(result).toBeGreaterThan(0.9);
      expect(result).toBeLessThanOrEqual(1);
    });

    it("R² = 1 for perfect predictions", () => {
      expect(r2Score(actual, actual)).toBeCloseTo(1);
    });
  });
});
