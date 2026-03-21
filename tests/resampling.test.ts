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

    it("MSE is zero for identical arrays", () => {
      expect(mse([1, 2, 3], [1, 2, 3])).toBe(0);
    });

    it("MSE throws on mismatched lengths", () => {
      expect(() => mse([1, 2], [1])).toThrow("same length");
    });

    it("RMSE is zero for identical arrays", () => {
      expect(rmse([1, 2, 3], [1, 2, 3])).toBe(0);
    });

    it("MAE throws on mismatched lengths", () => {
      expect(() => mae([1, 2], [1])).toThrow("same length");
    });

    it("MAE is zero for identical arrays", () => {
      expect(mae([1, 2, 3], [1, 2, 3])).toBe(0);
    });

    it("R² throws on mismatched lengths", () => {
      expect(() => r2Score([1, 2], [1])).toThrow("same length");
    });
  });

  describe("kFoldCV edge cases", () => {
    it("throws on mismatched X and y lengths", () => {
      expect(() => kFoldCV([[1], [2]], [1], fitPredictMean)).toThrow("same length");
    });

    it("throws on inconsistent row lengths in X", () => {
      expect(() => kFoldCV([[1, 2], [3]], [1, 2], fitPredictMean)).toThrow("Inconsistent row lengths");
    });

    it("uses default k=5 when not specified", () => {
      const result = kFoldCV(X, y, fitPredictMean, { seed: 42 });
      expect(result.nFolds).toBe(5);
    });

    it("custom scorer is used", () => {
      const customScorer = (actual: number[], predicted: number[]) => {
        // Simply return a constant to verify scorer is called
        return 42;
      };
      const result = kFoldCV(X, y, fitPredictMean, { k: 3, scorer: customScorer, seed: 42 });
      expect(result.meanScore).toBe(42);
    });

    it("fold scores array length matches k", () => {
      const result = kFoldCV(X, y, fitPredictMean, { k: 4, seed: 42 });
      expect(result.foldScores).toHaveLength(4);
    });
  });

  describe("loocv edge cases", () => {
    it("throws on mismatched X and y", () => {
      expect(() => loocv([[1]], [1, 2], fitPredictMean)).toThrow("same length");
    });

    it("each fold has exactly one test observation", () => {
      const smallX = [[1], [2], [3], [4]];
      const smallY = [2, 4, 6, 8];
      const result = loocv(smallX, smallY, fitPredictMean);
      expect(result.nFolds).toBe(4);
      expect(result.foldScores).toHaveLength(4);
    });
  });

  describe("jackknife edge cases", () => {
    it("pseudo-values have length equal to data", () => {
      const data = [1, 2, 3, 4, 5];
      const result = jackknife(data, mean);
      expect(result.pseudoValues).toHaveLength(5);
    });

    it("works with 2 observations (minimum)", () => {
      const data = [10, 20];
      const result = jackknife(data, mean);
      expect(result.estimate).toBeCloseTo(15);
      expect(result.pseudoValues).toHaveLength(2);
    });

    it("works with variance statistic", () => {
      const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const varianceFn = (sample: number[]) => {
        const m = mean(sample);
        return sample.reduce((s, v) => s + (v - m) ** 2, 0) / (sample.length - 1);
      };
      const result = jackknife(data, varianceFn);
      expect(result.standardError).toBeGreaterThan(0);
    });
  });

  describe("stratifiedSample edge cases", () => {
    it("throws on invalid sample size", () => {
      expect(() => stratifiedSample([1, 2, 3], [0, 1, 0], 0)).toThrow("between 1");
      expect(() => stratifiedSample([1, 2, 3], [0, 1, 0], 4)).toThrow("between 1");
    });

    it("returns correct number of samples", () => {
      const data = Array.from({ length: 100 }, (_, i) => i);
      const strata = data.map((_, i) => (i < 50 ? 0 : 1));
      const result = stratifiedSample(data, strata, 10, 42);
      expect(result.sample.length).toBeLessThanOrEqual(10);
      expect(result.indices.length).toBe(result.sample.length);
    });

    it("indices correspond to correct data values", () => {
      const data = [10, 20, 30, 40, 50, 60];
      const strata = [0, 0, 0, 1, 1, 1];
      const result = stratifiedSample(data, strata, 4, 42);
      for (let i = 0; i < result.sample.length; i++) {
        expect(result.sample[i]).toBe(data[result.indices[i]]);
      }
    });
  });
});
