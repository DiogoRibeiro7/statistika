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
 * @param X - Feature matrix (n x p)
 * @param y - Response variable
 * @param fitPredict - Function that takes (trainX, trainY, testX) and returns predictions
 * @param options - Configuration
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
 *
 * @param data - Dataset values
 * @param strata - Stratum label for each observation
 * @param sampleSize - Total number of samples to draw
 * @param seed - Optional random seed
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
  const sample: number[] = [];
  const indices: number[] = [];

  for (const [, groupIndices] of groups) {
    const groupSampleSize = Math.round((groupIndices.length / n) * sampleSize);
    // Shuffle group indices
    const shuffled = [...groupIndices];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    for (let i = 0; i < Math.min(groupSampleSize, shuffled.length); i++) {
      indices.push(shuffled[i]);
      sample.push(data[shuffled[i]]);
    }
  }

  return { sample, indices };
}

// ── Scoring Functions ───────────────────────────────────────────────────

/**
 * Mean Squared Error.
 */
export function mse(actual: Dataset, predicted: Dataset): number {
  if (actual.length !== predicted.length) throw new Error("Arrays must have same length");
  let sum = 0;
  for (let i = 0; i < actual.length; i++) sum += (actual[i] - predicted[i]) ** 2;
  return sum / actual.length;
}

/**
 * Root Mean Squared Error.
 */
export function rmse(actual: Dataset, predicted: Dataset): number {
  return Math.sqrt(mse(actual, predicted));
}

/**
 * Mean Absolute Error.
 */
export function mae(actual: Dataset, predicted: Dataset): number {
  if (actual.length !== predicted.length) throw new Error("Arrays must have same length");
  let sum = 0;
  for (let i = 0; i < actual.length; i++) sum += Math.abs(actual[i] - predicted[i]);
  return sum / actual.length;
}

/**
 * R² (coefficient of determination).
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
