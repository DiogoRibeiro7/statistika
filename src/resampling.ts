import { Dataset } from "./types";
import { mean } from "./utils/descriptive";
import { createRng } from "./utils/linalg";

/**
 * Result of a cross-validation procedure.
 */
export interface CrossValidationResult {
  /** Error/score for each fold. */
  foldScores: number[];
  /** Mean score across folds. */
  meanScore: number;
  /** Standard deviation of fold scores. */
  stdScore: number;
  /** Number of folds. */
  nFolds: number;
}

/**
 * K-fold cross-validation for regression models.
 *
 * Splits data into k folds, trains on k-1 folds, and evaluates on the
 * held-out fold. Reports mean squared error by default.
 *
 * @param X - Feature matrix (n x p). All rows must have the same length.
 * @param y - Response variable
 * @param fitPredict - Function that takes (trainX, trainY, testX) and returns predictions
 * @param options - Configuration
 * @returns Cross-validation results including per-fold scores, mean, and standard deviation
 * @throws {Error} If X and y have different lengths
 * @throws {Error} If k is less than 2 or exceeds the number of observations
 * @throws {Error} If rows of X have inconsistent lengths
 *
 * @example
 * ```ts
 * const X = [[1], [2], [3], [4], [5], [6]];
 * const y = [2, 4, 6, 8, 10, 12];
 * const result = kFoldCV(X, y, (trX, trY, teX) => {
 *   // simple mean predictor
 *   const m = trY.reduce((a, b) => a + b, 0) / trY.length;
 *   return teX.map(() => m);
 * }, { k: 3 });
 * // result.meanScore — average MSE across 3 folds
 * ```
 */
export function kFoldCV(
  X: number[][],
  y: Dataset,
  fitPredict: (trainX: number[][], trainY: Dataset, testX: number[][]) => number[],
  options: {
    k?: number;
    scorer?: (actual: Dataset, predicted: Dataset) => number;
    seed?: number;
  } = {},
): CrossValidationResult {
  const n = X.length;
  if (n !== y.length) throw new Error("X and y must have the same length");

  // Validate consistent row lengths
  if (n > 0) {
    const p = X[0].length;
    for (let i = 1; i < n; i++) {
      if (X[i].length !== p) {
        throw new Error(
          `Inconsistent row lengths in X: row 0 has ${p} columns but row ${i} has ${X[i].length} columns`,
        );
      }
    }
  }

  const k = options.k ?? 5;
  if (k < 2) throw new Error("k must be at least 2");
  if (k > n) throw new Error("k cannot exceed the number of observations");

  const scorer = options.scorer ?? mse;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  // Shuffle indices
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  // Create folds
  const foldSize = Math.floor(n / k);
  const foldScores: number[] = [];

  for (let fold = 0; fold < k; fold++) {
    const testStart = fold * foldSize;
    const testEnd = fold === k - 1 ? n : testStart + foldSize;
    const testIndices = indices.slice(testStart, testEnd);
    const trainIndices = [...indices.slice(0, testStart), ...indices.slice(testEnd)];

    const trainX = trainIndices.map((i) => X[i]);
    const trainY = trainIndices.map((i) => y[i]);
    const testX = testIndices.map((i) => X[i]);
    const testY = testIndices.map((i) => y[i]);

    const predicted = fitPredict(trainX, trainY, testX);
    foldScores.push(scorer(testY, predicted));
  }

  const meanScore = mean(foldScores);
  const stdScore = Math.sqrt(
    foldScores.reduce((s, v) => s + (v - meanScore) ** 2, 0) / (k - 1),
  );

  return { foldScores, meanScore, stdScore, nFolds: k };
}

/**
 * Leave-one-out cross-validation (LOOCV).
 *
 * Special case of k-fold CV where k = n. Each observation is used as
 * a test set exactly once.
 *
 * @param X - Feature matrix (n x p)
 * @param y - Response variable
 * @param fitPredict - Function that takes (trainX, trainY, testX) and returns predictions
 * @param scorer - Scoring function (default: MSE)
 * @returns Cross-validation results with n folds
 * @throws {Error} If X and y have different lengths
 */
export function loocv(
  X: number[][],
  y: Dataset,
  fitPredict: (trainX: number[][], trainY: Dataset, testX: number[][]) => number[],
  scorer: (actual: Dataset, predicted: Dataset) => number = mse,
): CrossValidationResult {
  return kFoldCV(X, y, fitPredict, { k: X.length, scorer });
}

/**
 * Jackknife estimation of bias and standard error.
 *
 * Resamples by systematically leaving out one observation at a time.
 *
 * @param data - Input dataset
 * @param statistic - Function computing the statistic of interest
 * @returns Object containing the full-sample estimate, bias, standard error, and pseudo-values
 * @throws {Error} If data has fewer than 2 observations
 */
export function jackknife(
  data: Dataset,
  statistic: (sample: Dataset) => number,
): { estimate: number; bias: number; standardError: number; pseudoValues: number[] } {
  if (data.length < 2) throw new Error("Need at least 2 observations");

  const n = data.length;
  const fullEstimate = statistic(data);

  const jackValues = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const sample = [...data.slice(0, i), ...data.slice(i + 1)];
    jackValues[i] = statistic(sample);
  }

  const jackMean = mean(jackValues);

  // Pseudo-values
  const pseudoValues = jackValues.map((jv) => n * fullEstimate - (n - 1) * jv);

  // Bias: (n-1) * (jackknife_mean - full_estimate)
  const bias = (n - 1) * (jackMean - fullEstimate);

  // Standard error
  const se = Math.sqrt(
    ((n - 1) / n) * jackValues.reduce((s, v) => s + (v - jackMean) ** 2, 0),
  );

  return {
    estimate: fullEstimate,
    bias,
    standardError: se,
    pseudoValues,
  };
}

/**
 * Stratified random sampling.
 *
 * Samples from data while maintaining the proportion of each stratum.
 * If rounding causes the total allocated samples to exceed `sampleSize`,
 * excess samples are trimmed from the largest stratum.
 *
 * @param data - Dataset values
 * @param strata - Stratum label for each observation
 * @param sampleSize - Total number of samples to draw
 * @param seed - Optional random seed
 * @returns Object containing the sampled values and their original indices
 * @throws {Error} If data and strata have different lengths
 * @throws {Error} If sampleSize is not between 1 and the dataset size
 */
export function stratifiedSample(
  data: Dataset,
  strata: number[],
  sampleSize: number,
  seed?: number,
): { sample: number[]; indices: number[] } {
  if (data.length !== strata.length) {
    throw new Error("Data and strata must have the same length");
  }
  if (sampleSize <= 0 || sampleSize > data.length) {
    throw new Error("Sample size must be between 1 and dataset size");
  }

  const rng = seed != null ? createRng(seed) : Math.random;

  // Group by stratum
  const groups = new Map<number, number[]>();
  for (let i = 0; i < data.length; i++) {
    const s = strata[i];
    if (!groups.has(s)) groups.set(s, []);
    groups.get(s)!.push(i);
  }

  const n = data.length;

  // Compute per-group sample sizes via rounding, then fix any overshoot
  const groupEntries = [...groups.entries()];
  const groupSizes = groupEntries.map(([, indices]) =>
    Math.round((indices.length / n) * sampleSize),
  );

  let totalAllocated = groupSizes.reduce((a, b) => a + b, 0);
  // Trim excess from the largest groups (by allocated size) until we match sampleSize
  while (totalAllocated > sampleSize) {
    let maxIdx = 0;
    for (let i = 1; i < groupSizes.length; i++) {
      if (groupSizes[i] > groupSizes[maxIdx]) maxIdx = i;
    }
    groupSizes[maxIdx]--;
    totalAllocated--;
  }

  const sample: number[] = [];
  const resultIndices: number[] = [];

  for (let g = 0; g < groupEntries.length; g++) {
    const [, groupIndices] = groupEntries[g];
    const groupSampleSize = groupSizes[g];
    // Shuffle group indices
    const shuffled = [...groupIndices];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    for (let i = 0; i < Math.min(groupSampleSize, shuffled.length); i++) {
      resultIndices.push(shuffled[i]);
      sample.push(data[shuffled[i]]);
    }
  }

  return { sample, indices: resultIndices };
}

// ── Scoring Functions ───────────────────────────────────────────────────

/**
 * Mean Squared Error.
 *
 * @param actual - Actual values
 * @param predicted - Predicted values
 * @returns The mean squared error between actual and predicted
 * @throws {Error} If arrays have different lengths
 */
export function mse(actual: Dataset, predicted: Dataset): number {
  if (actual.length !== predicted.length) throw new Error("Arrays must have same length");
  let sum = 0;
  for (let i = 0; i < actual.length; i++) sum += (actual[i] - predicted[i]) ** 2;
  return sum / actual.length;
}

/**
 * Root Mean Squared Error.
 *
 * @param actual - Actual values
 * @param predicted - Predicted values
 * @returns The root mean squared error between actual and predicted
 * @throws {Error} If arrays have different lengths
 */
export function rmse(actual: Dataset, predicted: Dataset): number {
  return Math.sqrt(mse(actual, predicted));
}

/**
 * Mean Absolute Error.
 *
 * @param actual - Actual values
 * @param predicted - Predicted values
 * @returns The mean absolute error between actual and predicted
 * @throws {Error} If arrays have different lengths
 */
export function mae(actual: Dataset, predicted: Dataset): number {
  if (actual.length !== predicted.length) throw new Error("Arrays must have same length");
  let sum = 0;
  for (let i = 0; i < actual.length; i++) sum += Math.abs(actual[i] - predicted[i]);
  return sum / actual.length;
}

/**
 * R² (coefficient of determination).
 *
 * @param actual - Actual values
 * @param predicted - Predicted values
 * @returns The R² score (1.0 for perfect prediction, 0.0 for mean-level prediction)
 * @throws {Error} If arrays have different lengths
 */
export function r2Score(actual: Dataset, predicted: Dataset): number {
  if (actual.length !== predicted.length) throw new Error("Arrays must have same length");
  const yMean = mean(actual);
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < actual.length; i++) {
    ssRes += (actual[i] - predicted[i]) ** 2;
    ssTot += (actual[i] - yMean) ** 2;
  }
  return 1 - ssRes / ssTot;
}
